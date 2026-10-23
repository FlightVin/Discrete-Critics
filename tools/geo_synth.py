"""Synthetic context-to-scalar diagnostic for the ARB-BSC critic variants E0-E9 (inv/geo2b).

Recipe (fixed before fitting):
  --variant E0..E9 --out <json> [--updates 20000]
  Data (np.random.RandomState(123), drawn in this order): W1 (64, 16) and W2 (1, 64) ~ N(0, 1/fan_in), no biases;
    50,000 train x then 10,000 held-out x, x in R^16 ~ N(0, I). g(x) = W2 tanh(W1 x). Action input = zeros(35)
    (horizon 5 x action dim 7). u = k / 20000; s = sigmoid(3 g(x)); base k = round(19980 s); j = floor(999 s).
    Sign pattern p = sum_i [x_i > 0] 2^i over x[0:4] (0..15): p=0 -> k=0; p=1 -> k=19980; p in {2,3} -> k = 20 j
    (exact codes); p=4 -> 20 j + 1, p=5 -> 20 j - 1 (clipped to [0, 19980]) (boundary); else base k.
    Groups: base (p >= 6), k=0 (p=0), k=19980 (p=1), exact (p in {2,3}), boundary (p in {4,5}).
  Model: the real ACARBAgent critic of the variant's config (v1 arb_bsc, real sizes), 2 copies, init seed 0.
    E0: greedy readout + 'hard' labels (ordinary). E1-E4: ordinary vocabulary, sample_mean (M=10, T=0.7) with
    arb_label hard / twohot_last / twohot_last2 / twohot_all. E5-E9: redundant vocabulary, sample_mean (M=10,
    T=0.7) with (arb_label, arb_rollin) = (corrective, corrective) / (corrective, model_mix) / (corrective, fixed)
    / (noisy_fixed, fixed) / (noisy_resid, fixed).
  Training: Adam 3e-4 on the critic params only, batch 256, --updates steps. Batch indices
    np.random.RandomState(0).randint(0, 50000, (updates, 256)); the roll-in key of update i is
    fold_in(PRNGKey(0), i). Targets k are supplied directly (no TD, no target network); the loss is the agent's
    own critic_ce (labels / roll-ins exactly as its critic loss, E6 model roll-in included), averaged over
    copies and positions (all rows valid).
  Measures on the held-out set, with the variant's readout (E0: greedy, mean of the 2 copies' greedy values;
    E1-E9: integer sample_mean S / 20000, M=10, T=0.7) in chunks of 1000 with chunk c keyed
    fold_in(PRNGKey(7), c):
    bias = mean(value - u), mae = mean |value - u|, overall and per group; oor_frac = decoded values outside
    [0, 999] before the clip (per sampled code; E0: per greedy code).
    Correction under known prefixes: first token forced to c1 + e, c1 = floor(k / 2000), e in {-2, -1, +1, +2}
    (rows where c1 + e is outside the vocabulary are skipped: [0, 9] ordinary, [-22, 31] redundant); positions
    2-3 continue with the variant's readout (greedy, or M=10 samples per copy at T=0.7, key
    fold_in(fold_in(PRNGKey(7), 100 + e), c)); value = mean over copies (and samples) of clip(n, 0, 999) / 1000;
    reports mean |value - u| per e and the number of rows used.
    Readout variance: 5 repeats with base keys fold_in(PRNGKey(7), 1 + r) (chunks as above): mean per-example
    variance across repeats and the fraction of examples whose 5 values are identical.
    Timing: ms per update (mean of up to 50 timed updates after compilation) and ms per readout of
    32 x 256 (s, a) pairs (mean of 5 after compilation).
Writes everything (and the variant's agent config) to --out as JSON.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from agents import agents, geo_codes, v1_config  # noqa: E402

K_DEN, K_TOP = 20000, 19980
OBS_DIM, H, ADIM = 16, 5, 7
N_TRAIN, N_TEST, BATCH, CHUNK = 50000, 10000, 256, 1000
SM = dict(arb_readout='sample_mean', arb_readout_samples=10, arb_readout_temp=0.7)
RED = dict(SM, arb_vocab='redundant')
VARIANTS = {
    'E0': dict(arb_readout='greedy', arb_label='hard'),
    'E1': dict(SM, arb_label='hard'),
    'E2': dict(SM, arb_label='twohot_last'),
    'E3': dict(SM, arb_label='twohot_last2'),
    'E4': dict(SM, arb_label='twohot_all'),
    'E5': dict(RED, arb_label='corrective', arb_rollin='corrective'),
    'E6': dict(RED, arb_label='corrective', arb_rollin='model_mix'),
    'E7': dict(RED, arb_label='corrective', arb_rollin='fixed'),
    'E8': dict(RED, arb_label='noisy_fixed', arb_rollin='fixed'),
    'E9': dict(RED, arb_label='noisy_resid', arb_rollin='fixed'),
}
GROUPS = ('base', 'k0', 'k19980', 'exact', 'boundary')
OFFSETS = (-2, -1, 1, 2)


def make_data():
    """(x_train, k_train, group_train), (x_test, k_test, group_test) as in the module docstring."""
    rs = np.random.RandomState(123)
    w1 = rs.randn(64, OBS_DIM) / np.sqrt(OBS_DIM)
    w2 = rs.randn(1, 64) / np.sqrt(64)
    xs = [rs.randn(N_TRAIN, OBS_DIM), rs.randn(N_TEST, OBS_DIM)]
    out = []
    for x in xs:
        g = (np.tanh(x @ w1.T) @ w2.T)[:, 0]
        s = 1.0 / (1.0 + np.exp(-3.0 * g))
        k = np.round(K_TOP * s).astype(np.int64)
        j = np.floor(999 * s).astype(np.int64)
        p = sum((x[:, i] > 0).astype(np.int64) << i for i in range(4))
        group = np.full(len(x), 'base', dtype=object)
        k = np.where(p == 0, 0, k)
        k = np.where(p == 1, K_TOP, k)
        k = np.where((p == 2) | (p == 3), 20 * j, k)
        k = np.where(p == 4, np.clip(20 * j + 1, 0, K_TOP), k)
        k = np.where(p == 5, np.clip(20 * j - 1, 0, K_TOP), k)
        group[p == 0], group[p == 1] = 'k0', 'k19980'
        group[(p == 2) | (p == 3)], group[(p == 4) | (p == 5)] = 'exact', 'boundary'
        out.append((x.astype(np.float32), k.astype(np.int32), group))
    return out


def make_agent(variant):
    cfg = v1_config('arb_bsc')
    cfg.horizon_length = H
    cfg.critic_target_bon = False
    cfg.update(VARIANTS[variant])
    return agents['acarb'].create(0, np.zeros(OBS_DIM, np.float32), np.zeros(ADIM, np.float32), cfg)


def with_critic(agent, p):
    return agent.replace(network=agent.network.replace(params=dict(agent.network.params, modules_critic=p)))


def greedy(agent):
    return agent.config['arb_readout'] == 'greedy'


@jax.jit
def readout(agent, obs, key):
    """Ensemble value (n,) and oor fraction (n,) of the variant's readout."""
    act = jnp.zeros((*obs.shape[:-1], H * ADIM), jnp.float32)
    if greedy(agent):
        n = agent._decode(agent.network.select('critic')(obs, act))  # (num_qs, n, L)
        oor = ((n < 0) | (n > 999)).astype(jnp.float32).mean(0)
        return jnp.clip(n, 0, 999).astype(jnp.float32).mean(0) / 1000, oor
    s, oor = agent.critic_score(obs, act, rng=key)
    return s.astype(jnp.float32) / agent.readout_den(), oor


@jax.jit
def readout_from(agent, obs, first_classes, key):
    """Value (n,) with the first class forced, positions 2..L by the variant's readout."""
    act = jnp.zeros((*obs.shape[:-1], H * ADIM), jnp.float32)
    num_qs, L = agent.config['num_qs'], agent.config['arb_length']
    M = 1 if greedy(agent) else agent.config['arb_readout_samples']
    nc = geo_codes.NUM_CLASSES if agent._redundant() else agent.config['arb_base']
    fn = agent._percopy_critic(False)
    digits = jnp.zeros((num_qs, obs.shape[0], M, L), jnp.int32)
    digits = digits.at[..., 0].set(jnp.broadcast_to(first_classes[None, :, None], digits.shape[:-1]))
    gumbel = jax.random.gumbel(key, (*digits.shape, nc))
    for l in range(1, L):
        logits = fn(obs, act, digits)[..., l, :]
        z = jnp.argmax(logits, -1) if greedy(agent) else \
            jnp.argmax(logits / agent.config['arb_readout_temp'] + gumbel[..., l, :], -1)
        digits = digits.at[..., l].set(z.astype(jnp.int32))
    return jnp.clip(agent._decode(digits), 0, 999).astype(jnp.float32).mean(axis=(0, 2)) / 1000


def chunked(fn, n, key):
    parts = [jax.device_get(fn(slice(s, s + CHUNK), jax.random.fold_in(key, c)))
             for c, s in enumerate(range(0, n, CHUNK))]
    return [np.concatenate([np.asarray(p[i]) for p in parts]) for i in range(len(parts[0]))]


def train(agent, x, k, updates):
    tx = optax.adam(3e-4)
    params = agent.network.params['modules_critic']
    opt_state = tx.init(params)

    @jax.jit
    def step(agent, params, opt_state, obs, kk, key):
        act = jnp.zeros((obs.shape[0], H * ADIM), jnp.float32)

        def loss_fn(p):
            gp = dict(agent.network.params, modules_critic=p)
            ce, _, _, info = agent.critic_ce(obs, act, kk.astype(jnp.float32) / K_DEN, kk, gp, key,
                                             jnp.ones(obs.shape[0], jnp.float32))
            return ce.mean(), info

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, dict(info, loss=loss)

    idx = np.random.RandomState(0).randint(0, N_TRAIN, (updates, BATCH))
    xj, kj = jnp.asarray(x), jnp.asarray(k)
    base = jax.random.PRNGKey(0)
    hist, times = [], []
    for i in range(updates):
        t0 = time.perf_counter()
        params, opt_state, info = step(agent, params, opt_state, xj[idx[i]], kj[idx[i]], jax.random.fold_in(base, i))
        if i >= 1 and len(times) < 50:
            jax.block_until_ready(params)
            times.append(time.perf_counter() - t0)
        if i >= updates - 100:
            hist.append(jax.device_get(info))
        if i % 1000 == 0:
            print(f'update {i} loss {float(info["loss"]):.4f}', flush=True)
    last = {key: float(np.mean([h[key] for h in hist])) for key in hist[0]}
    return with_critic(agent, params), last, 1000 * float(np.mean(times)) if times else float('nan')


def evaluate(agent, x, k, group):
    u = k.astype(np.float64) / K_DEN
    xj = jnp.asarray(x)
    key = jax.random.PRNGKey(7)
    value, oor = chunked(lambda sl, kk: readout(agent, xj[sl], kk), len(x), key)
    err = value.astype(np.float64) - u
    out = {'bias': float(err.mean()), 'mae': float(np.abs(err).mean()), 'oor_frac': float(oor.mean()),
           'groups': {}}
    for gname in GROUPS:
        m = group == gname
        out['groups'][gname] = {'n': int(m.sum()), 'bias': float(err[m].mean()), 'mae': float(np.abs(err[m]).mean()),
                                'oor_frac': float(oor[m].mean())}

    lo, hi = (geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX) if agent._redundant() else (0, 9)
    offset = -geo_codes.TOKEN_MIN if agent._redundant() else 0
    c1 = k // 2000
    out['correction'] = {}
    for e in OFFSETS:
        t = c1 + e
        ok = (t >= lo) & (t <= hi)
        cls = jnp.asarray(np.clip(t + offset, 0, None).astype(np.int32))
        (v,) = chunked(lambda sl, kk: (readout_from(agent, xj[sl], cls[sl], kk),), len(x),
                       jax.random.fold_in(key, 100 + e))
        out['correction'][str(e)] = {'n': int(ok.sum()),
                                     'mae': float(np.abs(v[ok].astype(np.float64) - u[ok]).mean()) if ok.any()
                                     else float('nan')}

    reps = np.stack([chunked(lambda sl, kk: readout(agent, xj[sl], kk), len(x), jax.random.fold_in(key, 1 + r))[0]
                     for r in range(5)]).astype(np.float64)
    out['readout_var_mean'] = float(reps.var(0).mean())
    out['readout_identical_frac'] = float((reps == reps[0]).all(0).mean())

    obs = jnp.asarray(np.broadcast_to(x[:256, None], (256, 32, OBS_DIM)))
    jax.block_until_ready(readout(agent, obs, key))
    times = []
    for r in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(readout(agent, obs, jax.random.fold_in(key, r)))
        times.append(time.perf_counter() - t0)
    out['ms_per_readout_32x256'] = 1000 * float(np.mean(times))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--out', required=True)
    p.add_argument('--updates', type=int, default=20000)
    args = p.parse_args()

    (x_tr, k_tr, _), (x_te, k_te, g_te) = make_data()
    agent = make_agent(args.variant)
    agent, train_info, ms_update = train(agent, x_tr, k_tr, args.updates)
    res = {'variant': args.variant, 'updates': args.updates, 'ms_per_update': ms_update,
           'train_last100': train_info, **evaluate(agent, x_te, k_te, g_te),
           'config': json.loads(json.dumps(dict(agent.config), default=str))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'config'}), flush=True)


if __name__ == '__main__':
    main()
