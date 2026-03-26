# BSC-QC Implementation Plan

## Overview

Implement Binary Success Critic (BSC) on top of the Q-Chunking (QC) codebase as a new
agent `ACBSC` (`agents/acbsc.py`). BSC replaces the MSE regression critic with a binary
classifier that estimates P(success | s, action_chunk), trained with BCE loss.

**No existing files are modified** except `agents/__init__.py` (to register the new agent).
QC runs exactly as before.

---

## 1. The Robomimic Dataset Signal Problem

### What the existing dataset looks like

In `envs/robomimic_utils.py`, `get_dataset()` sets:
```
terminals = dones           # 1 at last step of EVERY trajectory (success or timeout)
masks     = 1 - dones       # 0 at last step of EVERY trajectory (success or timeout)
```

This differs from the FQL/AntMaze case where `masks=0` only at **success** steps.
With this encoding, `terminals * masks = 0` for both success and timeout — we cannot
distinguish them using `masks` and `terminals` alone.

### How we distinguish success from timeout: rewards

After `process_train_dataset()` in `main.py` applies the robomimic reward shift (`reward - 1.0`):

| Step type       | Original reward | After shift | After sparse |
|----------------|-----------------|-------------|--------------|
| Success         | 1.0             | 0.0         | 0.0          |
| All other steps | 0.0             | -1.0        | -1.0         |

Note: the sparse mode flag makes no difference for robomimic (the shifted rewards are already
binary `{0, -1}`).

In `sample_sequence()`, the returned `batch['rewards'][:, i]` is the **cumulative discounted
reward** up to step `i`. The contribution of the **last step** (step `H-1`) is:

```
last_step_contribution = batch['rewards'][:, -1] - batch['rewards'][:, -2]
                       = batch_reward[H-1] * discount^(H-1)
```

- **Success at step H-1**: contribution = `0 * discount^(H-1) = 0`
- **Timeout or non-terminal at step H-1**: contribution = `-1 * discount^(H-1) < 0`

So `last_step_contribution > -ε` (for small ε, e.g. 1e-6) is `True` **only** at a success step,
regardless of horizon length or discount value. This becomes the new BSC-internal variable.

### BSC target derivation

Define inside `critic_loss()`:

```python
# New BSC-internal variables (don't touch existing masks/terminals fields)
last_step_contribution = batch['rewards'][..., -1] - batch['rewards'][..., -2]  # H>=2
bsc_is_success  = last_step_contribution > -1e-6          # True only if success at last step
bsc_is_terminal = batch['terminals'][..., -1] > 0.5       # True for success OR timeout

target_q = (
    bsc_is_success.astype(jnp.float32) * 1.0              # success  → 1.0
    + (~bsc_is_terminal).astype(jnp.float32) * next_q     # non-term → Q_target
    # timeout: bsc_is_success=False, bsc_is_terminal=True → target = 0.0 ✓
)
```

| Case | `bsc_is_success` | `bsc_is_terminal` | target |
|------|-----------------|-------------------|--------|
| Success (last step) | True | True | 1.0 ✓ |
| Timeout (last step) | False | True | 0.0 ✓ |
| Non-terminal | False | False | Q_target ✓ |

**`discount = 1.0`** is used in the BSC config for the BSC target formula (not related to the
`FLAGS.discount=0.99` used in `sample_sequence()` for cumulative reward computation).

---

## 2. Required Changes

### 2.1 `agents/acbsc.py` — New file (copy of `acfql.py` with BSC modifications)

#### Class name and docstring

```python
class ACBSCAgent(flax.struct.PyTreeNode):
    """Action-Chunked Binary Success Critic (ACBSC).

    Critic estimates P(success | s_t, action_chunk) with BCE loss.
    Success/timeout distinction: inferred from last-step reward contribution
    (success reward = 0 after robomimic -1 shift; timeout = -1).
    No existing dataset fields (masks, terminals) are redefined.
    discount=1.0 for pure P(success) semantics.
    """
```

#### Modified `critic_loss()`

```python
def critic_loss(self, batch, grad_params, rng):
    if self.config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]

    rng, sample_rng = jax.random.split(rng)
    next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

    # Target critic outputs as probabilities (logits → sigmoid).
    next_qs = self.network.select('target_critic')(
        batch['next_observations'][..., -1, :], actions=next_actions
    )
    next_qs = jax.nn.sigmoid(next_qs)
    next_q = next_qs.mean(axis=0) if self.config['q_agg'] == 'mean' else next_qs.min(axis=0)

    # BSC-internal variables (new, don't touch existing masks/terminals semantics).
    # last_step_contribution = reward contributed by the last step of the chunk.
    # With robomimic -1 shift: success=0, all others <0.
    last_step_contribution = batch['rewards'][..., -1] - batch['rewards'][..., -2]
    bsc_is_success  = last_step_contribution > -1e-6   # True only at success terminal
    bsc_is_terminal = batch['terminals'][..., -1] > 0.5  # True at any trajectory boundary

    # BSC target: success=1.0, timeout=0.0, non-terminal=Q_target.
    target_q = (
        bsc_is_success.astype(jnp.float32) * 1.0
        + (~bsc_is_terminal).astype(jnp.float32) * self.config['discount'] * next_q
    )

    # Current critic output is an unbounded logit; BCE uses logits directly for stability.
    q_logit = self.network.select('critic')(
        batch['observations'], actions=batch_actions, params=grad_params
    )
    # q_logit: (num_ensembles, B). target_q: (B,). Apply valid mask.
    critic_loss = optax.sigmoid_binary_cross_entropy(q_logit, target_q[None])
    critic_loss = (critic_loss * batch['valid'][..., -1][None]).mean()

    return critic_loss, {
        'critic_loss': critic_loss,
        'q': jax.nn.sigmoid(q_logit).mean(),
        'target_q': target_q.mean(),
        'success_frac': bsc_is_success.astype(jnp.float32).mean(),
        'timeout_frac': (bsc_is_terminal & ~bsc_is_success).astype(jnp.float32).mean(),
    }
```

#### Modified `actor_loss()`

Same as `acfql.py` except:
- Q-loss applies `sigmoid` to critic logits before averaging
- Total loss uses `beta` to scale the Q-loss

```python
        if self.config["actor_type"] == "distill-ddpg":
            # ... (distillation loss identical to ACFQL) ...

            # Q loss: maximize P(success). Critic outputs logits → apply sigmoid.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            q_logit = self.network.select('critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(jax.nn.sigmoid(q_logit), axis=0)  # ∈ [0, 1]
            q_loss = -q.mean()

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + self.config['beta'] * q_loss
```

#### Modified `sample_actions()` for best-of-n

Apply `sigmoid` when scoring candidate action chunks:

```python
        elif self.config["actor_type"] == "best-of-n":
            # ... (same candidate generation as ACFQL) ...

            # Critic outputs logits → sigmoid to get probabilities before argmax.
            q_vals = jax.nn.sigmoid(self.network.select("critic")(observations, actions))
            q = q_vals.mean(axis=0) if self.config["q_agg"] == "mean" else q_vals.min(axis=0)
            indices = jnp.argmax(q, axis=-1)
            # ... (same selection logic as ACFQL) ...
```

#### `get_config()` — key differences from ACFQL

```python
agent_name='acbsc',
discount=1.0,           # Must be 1.0: Q = P(success), not discounted time-to-success.
q_agg='mean',           # mean preferred for probabilities; min can be overly pessimistic.
alpha=10.0,             # BC weight. May need tuning down to 1–5 (Q bounded to [-1, 0]).
beta=1.0,               # New: Q-loss scale factor (independent of alpha).
normalize_q_loss=False, # Must remain False (sigmoid near 0 would make lam explode).
actor_type="best-of-n", # Default for QC.
actor_num_samples=32,
```

### 2.2 `agents/__init__.py` — Register ACBSC

```python
from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acbsc import ACBSCAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acbsc=ACBSCAgent,
)
```

### 2.3 `runscripts/bsc/square.sh` — New runscript

```bash
#!/bin/bash
export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
cd "$(dirname "$0")/../.."
python main.py \
    --run_group=BSC \
    --project_name=BSC \
    --agent=agents/acbsc.py \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=square-mh-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --n_eval_envs=32 \
    --eval_episodes=2 \
    --eval_interval=100000
```

---

## 3. What Does NOT Change

| File | Status |
|------|--------|
| `envs/robomimic_utils.py` | **Unchanged** — `masks`, `terminals` definitions kept as-is |
| `agents/acfql.py` | **Unchanged** — QC runs identically |
| `utils/datasets.py` | **Unchanged** — `sample_sequence()` not modified |
| `main.py` | **Unchanged** — reward processing, online loop, flags unchanged |
| `evaluation.py` | **Unchanged** — success rate already the right metric |
| All network/distribution files | **Unchanged** — `Value` network unchanged; sigmoid applied in agent |

---

## 4. Key Design Decisions

### Why rewards to distinguish success from timeout

With `masks = 1 - dones` in the robomimic dataset, both success and timeout have `masks=0`.
Using `last_step_contribution = batch['rewards'][:,-1] - batch['rewards'][:,-2]` cleanly
separates them: success contributes 0 (shifted reward = 0.0), timeout contributes a negative
value. This is robust to any discount and horizon length.

### Why discount=1.0

P(success | s, chunk) is a pure probability — it should be 1.0 for all states on a
successful trajectory, regardless of how many steps away success is. `discount < 1.0`
reintroduces time-to-success bias (Q ≈ γ^t < 1 for states t steps from success).

### Why normalize_q_loss=False is mandatory

`normalize_q_loss` computes `lam = 1 / |q|.mean()`. Early in training on sparse tasks,
`q.mean()` ≈ 0 (low success rate), making `lam` blow up. Must remain False for BSC.

### Online phase correctness

`main.py`'s online collection already uses `masks = 1.0 - terminated` (0 only on success)
and `terminals = float(done)`. The `last_step_contribution` approach also works online:
success transitions have `reward = 0.0` (after robomimic shift), timeouts have `reward = -1.0`.

---

## 5. File Change Summary

| File | Change |
|------|--------|
| `agents/acbsc.py` | **Create** — new BSC agent |
| `agents/__init__.py` | **Modify** — add `acbsc=ACBSCAgent` |
| `runscripts/bsc/square.sh` | **Create** — new runscript |
