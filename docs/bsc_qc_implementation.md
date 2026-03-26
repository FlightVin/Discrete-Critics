# BSC-QC Implementation Plan

## Overview

This document specifies how to implement Binary Success Critic (BSC) on top of the Q-Chunking (QC) codebase. BSC replaces the MSE regression critic with a binary classifier that directly estimates P(success | s, action_chunk), trained with binary cross-entropy (BCE) loss.

The new agent is called **ACBSC** (Action-Chunked Binary Success Critic).

---

## 1. Background: Key Differences from FQL-BSC

The QC codebase differs from FQL in three important ways that affect how BSC is implemented:

### 1.1 Action Chunking

QC operates on action chunks of length `H` (e.g., H=5). The critic takes as input:
- First observation of the chunk: `s_t`
- Full flattened action chunk: `[a_t, ..., a_{t+H-1}]`

The BSC target must match this semantic: **P(success | s_t, [a_t, ..., a_{t+H-1}])**.

Target logic (with γ=1.0):
- Chunk contains a success step → target = 1.0
- Chunk ends at a timeout (trajectory boundary, no success) → target = 0.0
- Chunk is entirely non-terminal → target = Q_target(s_{t+H}, next_chunk)

### 1.2 Robomimic Dataset: `masks` Field is Incorrect for BSC

In AntMaze (FQL), `masks = 1 - success_flag` (0 only at goal-reaching steps). This is the correct semantics for BSC.

In the QC robomimic dataset (`envs/robomimic_utils.py`), `masks = 1 - dones` where `dones=1` at the last step of **every** trajectory—both successful and timed-out. This makes `masks=0` at timeout trajectory ends too, so BSC cannot distinguish success from timeout using `masks` alone.

**Fix required**: change `get_dataset()` to compute `masks = 1 - success_flag` (0 only at last step of a **successful** demo, where `reward > 0`).

### 1.3 ACFQL Has a Latent Timeout Bootstrapping Bug

With the current (incorrect) `masks = 1 - dones`, `masks=0` blocks bootstrapping at all trajectory boundaries—including timeouts. This is accidentally correct behavior achieved by wrong means. After fixing the dataset masks, we must explicitly prevent ACFQL from bootstrapping past timeout boundaries via `bootstrap_mask = masks * (1 - terminals)`.

---

## 2. Dataset Signal Analysis

After fixing `get_dataset()`, the three fields needed for BSC are all present:

| Transition Type | `masks` | `terminals` | `terminals * masks` | BSC target |
|---|---|---|---|---|
| Success (last step of successful demo) | 0 | 1 | 0 | **1.0** |
| Timeout (last step of failed demo) | 1 | 1 | **1** | **0.0** |
| Non-terminal | 1 | 0 | 0 | **Q_target(s', a')** |

`truncated = terminals * masks` uniquely identifies timeout steps (as in the FQL-BSC design doc).

At the sequence level (action chunk of length H), `sample_sequence()` returns:
- `masks[:, -1]` = running minimum → 0 if chunk contains any success step
- `terminals[:, -1]` = running maximum → 1 if chunk hits any trajectory boundary
- `truncated_chunk = terminals[:, -1] * masks[:, -1]` = 1 only if chunk ends at timeout

Because robomimic success always occurs at the **last step** of a trajectory, success-containing chunks always have the success step at position H-1 (not in the middle). Cross-episode sequences are already zeroed by `batch['valid'][..., -1]`.

**No new buffer fields are needed.**

---

## 3. Required Changes

### Change 1: `envs/robomimic_utils.py` — Fix Dataset Masks

In `get_dataset()`, change `masks` from `1 - dones` to `1 - success_flag`:

**Current code (lines ~125–127):**
```python
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
```

**New code:**
```python
        terminals.append(dones.astype(np.float32))
        # success_flag is 1 only at the last step of a successful demo (reward > 0).
        # This differs from dones (which is 1 at ALL trajectory ends, success or timeout).
        # BSC needs masks=0 only at success, masks=1 at timeout.
        success_flag = np.zeros(len(dones), dtype=np.float32)
        success_flag[-1] = float(r[-1] > 0)
        masks.append(1.0 - success_flag)
        next_observations.append(next_obs.astype(np.float32))
```

**Why this is safe for ACFQL:** With the companion fix in Change 2, ACFQL's behavior is identical before and after this change (see §4).

---

### Change 2: `agents/acfql.py` — Fix Timeout Bootstrapping

In `critic_loss()`, change the TD target to explicitly prevent bootstrapping past **any** trajectory boundary (success or timeout), using `bootstrap_mask = masks * (1 - terminals)`:

**Current code (lines ~40–41):**
```python
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q
```

**New code:**
```python
        # Prevent bootstrapping past any trajectory boundary (success OR timeout).
        # With fixed dataset masks (0=success, 1=timeout/non-terminal), masks alone would
        # incorrectly bootstrap through timeout steps. The (1 - terminals) term blocks that.
        bootstrap_mask = batch['masks'][..., -1] * (1 - batch['terminals'][..., -1])
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * bootstrap_mask * next_q
```

**Behavioral equivalence proof** (no regression for ACFQL):
- Success chunk: old `masks[:,-1]=0`, new `masks[:,-1]=0` → `bootstrap_mask=0` → target = reward. Same.
- Timeout chunk: old `masks[:,-1]=0`, new `masks[:,-1]=1, terminals[:,-1]=1` → `bootstrap_mask=0` → target = reward. Same.
- Non-terminal chunk: old `masks[:,-1]=1, terminals[:,-1]=0`, new same → `bootstrap_mask=1` → target = reward + bootstrap. Same.

---

### Change 3: `agents/acbsc.py` — New BSC Agent (New File)

Create `agents/acbsc.py` as a copy of `agents/acfql.py` with the following modifications:

#### 3.1 Class Name and Docstring
```python
class ACBSCAgent(flax.struct.PyTreeNode):
    """Action-Chunked Binary Success Critic (ACBSC).

    The critic is trained with BCE loss to directly estimate P(success | s, chunk).
    TD targets: 1.0 at success steps, 0.0 at timeout steps, Q_target elsewhere.
    The critic output is an unbounded logit; sigmoid is applied at inference/loss time.
    discount=1.0 ensures pure P(success) semantics across the full chunk.
    """
```

#### 3.2 Modified `critic_loss()`

Replace the entire method:

```python
    def critic_loss(self, batch, grad_params, rng):
        """Compute the ACBSC critic loss (BCE instead of MSE)."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next action chunk using the actor.
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        # Target critic outputs are logits; convert to probabilities ∈ [0,1].
        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions
        )
        next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        # BSC target: success=1.0, timeout=0.0, non-terminal=bootstrap.
        # truncated = terminals * masks identifies timeout steps uniquely
        # (masks=0 only at success, terminals=1 at all trajectory boundaries).
        truncated = batch['terminals'][..., -1] * batch['masks'][..., -1]
        target_q = (
            (1.0 - batch['masks'][..., -1]) * 1.0
            + batch['masks'][..., -1] * (1.0 - truncated) * self.config['discount'] * next_q
        )

        # Online critic outputs are raw logits. BCE uses logits directly via
        # optax.sigmoid_binary_cross_entropy for numerical stability.
        q_logit = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params
        )
        # q_logit shape: (num_ensembles, batch_size). target_q shape: (batch_size,).
        # Apply valid mask (zeros out sequences crossing trajectory boundaries).
        critic_loss = optax.sigmoid_binary_cross_entropy(q_logit, target_q[None])
        critic_loss = (critic_loss * batch['valid'][..., -1][None]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q': jax.nn.sigmoid(q_logit).mean(),
            'target_q': target_q.mean(),
            'success_frac': (1.0 - batch['masks'][..., -1]).mean(),
            'timeout_frac': truncated.mean(),
        }
```

#### 3.3 Modified `actor_loss()`

Replace only the Q-loss portion and total loss formula:

```python
    def actor_loss(self, batch, grad_params, rng):
        """Compute the ACBSC actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss (unchanged from ACFQL).
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss (unchanged from ACFQL).
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss: maximize P(success) — critic outputs logits, apply sigmoid.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            q_logit = self.network.select('critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(jax.nn.sigmoid(q_logit), axis=0)  # ∈ [0, 1]
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # beta independently scales the Q-loss relative to BC/distillation terms.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + self.config['beta'] * q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }
```

#### 3.4 Modified `sample_actions()` for best-of-n

In the `best-of-n` branch, apply sigmoid when scoring candidate chunks:

```python
        elif self.config["actor_type"] == "best-of-n":
            # ... (same noise/observation setup as ACFQL) ...

            # Apply sigmoid: critic outputs logits, convert to probabilities before argmax.
            if self.config["q_agg"] == "mean":
                q = jax.nn.sigmoid(self.network.select("critic")(observations, actions)).mean(axis=0)
            else:
                q = jax.nn.sigmoid(self.network.select("critic")(observations, actions)).min(axis=0)
            indices = jnp.argmax(q, axis=-1)
            # ... (same index selection logic as ACFQL) ...
```

#### 3.5 New `get_config()`

```python
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='acbsc',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=1.0,           # Must be 1.0: ensures Q = P(success), not P(success) * gamma^t.
            tau=0.005,
            q_agg='mean',           # 'mean' preferred for probabilities (min can be too pessimistic).
            alpha=10.0,             # BC coefficient. May need tuning down to 1-5 (Q bounded to [-1,0]).
            beta=1.0,               # Q-loss scale. New param: controls Q signal vs BC/distillation.
            num_qs=2,
            flow_steps=10,
            normalize_q_loss=False, # Must be False: sigmoid output near 0 would make lam explode.
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="best-of-n",  # best-of-n is the standard for QC.
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
        )
    )
    return config
```

---

### Change 4: `agents/__init__.py` — Register ACBSC

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

---

### Change 5: `runscripts/bsc/square.sh` — New Runscript

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

Note: `--sparse=True` is kept for consistency with QC runscript but is irrelevant for BSC (the BSC critic ignores rewards entirely).

---

## 4. What Does NOT Change

- **`main.py`**: No changes needed. BSC is selected via `--agent=agents/acbsc.py`. The reward processing pipeline (`process_train_dataset`, reward shift, sparse mode) remains but is ignored by the BSC critic. Online transition schema already computes `masks=1-terminated` correctly.
- **`evaluation.py`**: No changes. Evaluation measures success rate directly via `info['success']`, which is perfectly aligned with BSC's objective.
- **`utils/datasets.py`**: No changes. `sample_sequence()` already returns `masks`, `terminals`, and `valid` fields needed by BSC.
- **Network architecture**: No changes to `utils/networks.py`. The `Value` class output is an unbounded logit; sigmoid is applied in the agent methods (as done in FQL-BSC).
- **`utils/flax_utils.py`**, **`rlpd_networks/`**, **`rlpd_distributions/`**: No changes.

---

## 5. Online Phase Correctness

In `main.py`'s online interaction loop, transitions are stored as:
```python
transition = dict(
    observations=ob,
    actions=action,
    rewards=int_reward,
    terminals=float(done),         # 1 if done (terminated OR truncated)
    masks=1.0 - terminated,        # 0 only if success (terminated=True)
    next_observations=next_ob,
)
```

This already correctly encodes:
- Success: `terminals=1, masks=0` → BSC target = 1.0 ✓
- Timeout: `terminals=1, masks=1` → BSC target = 0.0 ✓
- Non-terminal: `terminals=0, masks=1` → BSC target = Q_target ✓

No changes needed for the online phase.

---

## 6. Key Design Decisions

### discount = 1.0
With γ=1.0, Q(s,a) = P(success | s, chunk) for all states along a successful trajectory. With γ<1.0, states farther from success get lower Q even on successful trajectories, reintroducing time-to-success bias. BSC uses γ=1.0 by default.

### q_agg = 'mean' (not 'min')
With MSE critics, `min` aggregation counters overestimation. With BSC, Q-values are probabilities — `min` underestimates success probability, which may be overly pessimistic. `mean` is the default for BSC; `min` can be swept.

### normalize_q_loss = False (mandatory)
If enabled, `lam = 1 / |q|.mean()`. Early in training on sparse tasks, `q.mean()` is near 0 (few successes seen), making `lam` blow up and destabilizing training. Must remain False.

### alpha and beta
`alpha=10.0` was tuned for unbounded Q-values in ACFQL. With BSC, Q-loss is bounded to `[-1, 0]`, so alpha may need reducing (try 1–5). `beta=1.0` independently controls the Q-loss scale without affecting BC/distillation balance.

### The valid mask in critic_loss
Unlike ACFQL (which applies valid to MSE loss), BSC applies valid to BCE loss:
```python
critic_loss = (optax.sigmoid_binary_cross_entropy(q_logit, target_q[None]) * batch['valid'][..., -1][None]).mean()
```
This zeroes out loss for sequences crossing trajectory boundaries, which would have incorrect bootstrapped next_obs.

---

## 7. File Change Summary

| File | Type | Description |
|------|------|-------------|
| `envs/robomimic_utils.py` | Modify | Fix `masks = 1 - success_flag` in `get_dataset()` |
| `agents/acfql.py` | Modify | Fix timeout bootstrapping: `bootstrap_mask = masks * (1 - terminals)` |
| `agents/acbsc.py` | Create | New BSC agent: BCE critic loss, sigmoid Q, BSC target formula |
| `agents/__init__.py` | Modify | Register `acbsc=ACBSCAgent` |
| `runscripts/bsc/square.sh` | Create | New runscript for square-mh-low_dim with BSC |

---

## 8. Recommended Validation Experiments

1. **Sanity check**: Q-values from BSC critic should lie in (0,1). Log `critic/q` and verify.
2. **Success correlation**: `critic/success_frac` (fraction of batch with target=1.0) should increase over training.
3. **ACFQL regression**: Verify ACFQL performance is unchanged after dataset masks fix.
4. **Alpha sweep**: Compare `alpha ∈ {1, 3, 5, 10}` with BSC.
5. **Beta sweep**: Compare `beta ∈ {0.1, 0.5, 1.0, 2.0}` with BSC.
6. **Discount ablation**: Compare `discount=1.0` vs `discount=0.99` for BSC.
7. **q_agg ablation**: Compare `q_agg='mean'` vs `q_agg='min'` for BSC.
