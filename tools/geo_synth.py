"""Synthetic context-to-scalar diagnostic for the ARB-BSC critic variants E0-E11 (inv/geo2b, inv/geo3a).

Recipe (fixed before fitting):
  --variant E0..E11 --out <json> [--updates 20000]
  Data (np.random.RandomState(123), drawn in this order): W1 (64, 16) and W2 (1, 64) ~ N(0, 1/fan_in), no biases;
    50,000 train x then 10,000 held-out x, x in R^16 ~ N(0, I). g(x) = W2 tanh(W1 x). Action input = zeros(35)
    (horizon 5 x action dim 7). u = k / 20000; s = sigmoid(3 g(x)); base k = round(19980 s); j = floor(999 s).
    Sign pattern p = sum_i [x_i > 0] 2^i over x[0:4] (0..15): p=0 -> k=0; p=1 -> k=19980; p in {2,3} -> k = 20 j
    (exact codes); p=4 -> 20 j + 1, p=5 -> 20 j - 1 (clipped to [0, 19980]) (boundary); else base k.
    Groups: base (p >= 6), k=0 (p=0), k=19980 (p=1), exact (p in {2,3}), boundary (p in {4,5}).
  Model: the real ACARBAgent critic of the variant's config (v1 arb_bsc, real sizes), 2 copies, init seed 0.
    E0: greedy readout + 'hard' labels (ordinary). E1-E4: ordinary vocabulary, sample_mean (M=10, T=0.7) with
    arb_label hard / twohot_last / twohot_last2 / twohot_all. E5-E11: redundant vocabulary, sample_mean (M=10,
    T=0.7) with (arb_label, arb_rollin) = (corrective, corrective) / (corrective, model_mix) / (corrective, fixed)
    / (noisy_fixed, fixed) / (noisy_resid, fixed); E10, E11: (noisy_fixed, model_mix) / (noisy_resid, model_mix).
  Training: Adam 3e-4 on the critic params only, batch 256, --updates steps. Batch indices
    np.random.RandomState(0).randint(0, 50000, (updates, 256)); the roll-in key of update i is
    fold_in(PRNGKey(0), i). Targets k are supplied directly (no TD, no target network); the loss is the agent's
    own critic_ce (labels / roll-ins exactly as its critic loss, E6 / E10 / E11 model roll-in included), averaged over
    copies and positions (all rows valid).
  Measures on the held-out set, with the variant's readout (E0: greedy, mean of the 2 copies' greedy values;
    E1-E11: integer sample_mean S / 20000, M=10, T=0.7) in chunks of 1000 with chunk c keyed
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

geo5b: --study adds res['study'] (controlled supervised study and exact-target diagnostic; see study()): same
inputs, exact targets, matched budget (the training above is identical for every variant), no actor, no TD;
held-out label fit / token accuracy / value error by group / forced-prefix recovery, boundary-code probes
(boundary_probes()) with overshoot / undershoot prefixes and alternative redundant encodings, each under the normal
readout and the correction ablation. --save_params keeps the trained critic params.
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
    'E10': dict(RED, arb_label='noisy_fixed', arb_rollin='model_mix'),
    'E11': dict(RED, arb_label='noisy_resid', arb_rollin='model_mix'),
}
GROUPS = ('base', 'k0', 'k19980', 'exact', 'boundary')
OFFSETS = (-2, -1, 1, 2)


def make_weights(rs=None):
    rs = np.random.RandomState(123) if rs is None else rs
    return rs.randn(64, OBS_DIM) / np.sqrt(OBS_DIM), rs.randn(1, 64) / np.sqrt(64)


def recipe_targets(x, w1, w2):
    """Targets k (int32) and groups of inputs x under the recipe of the module docstring."""
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
    return k.astype(np.int32), group


def make_data():
    """(x_train, k_train, group_train), (x_test, k_test, group_test) as in the module docstring."""
    rs = np.random.RandomState(123)
    w1, w2 = make_weights(rs)
    xs = [rs.randn(N_TRAIN, OBS_DIM), rs.randn(N_TEST, OBS_DIM)]
    return [(x.astype(np.float32), *recipe_targets(x, w1, w2)) for x in xs]


# geo5b (7C): held-out exact-code probes at decimal boundaries. Fresh x ~ N(0, I) from RandomState(456)
# (PROBE_DRAWS rows, disjoint from train / test draws), recipe targets; kept: exact codes (k % 20 == 0) whose code
# n = k / 20 is in a boundary class, the first PROBE_PER_CLASS of each.
PROBE_DRAWS, PROBE_PER_CLASS = 400000, 500
PROBE_CLASSES = {'d00': lambda n: n % 100 == 0, 'd99': lambda n: n % 100 == 99,
                 'd0': lambda n: (n % 10 == 0) & (n % 100 != 0), 'd9': lambda n: (n % 10 == 9) & (n % 100 != 99)}
# Prefixes forced at the probes: overshoot / undershoot of position 1 or 2 (e = +-1, +-2, 0 = control) and other
# redundant encodings of the same code (deltas on positions 1, 2; the exact final token then is
# c3 - 100 d1 - 10 d2).
ENCODINGS = {'enc_0_0': (2, (0, 0)), 'enc_0_-1': (2, (0, -1)), 'enc_0_+1': (2, (0, 1)), 'enc_0_-2': (2, (0, -2)),
             'enc_-1_+10': (2, (-1, 10)), 'enc_+1_-10': (2, (1, -10)), 'enc_-1_+9': (2, (-1, 9))}


def boundary_probes():
    w1, w2 = make_weights()
    x = np.random.RandomState(456).randn(PROBE_DRAWS, OBS_DIM)
    k, _ = recipe_targets(x, w1, w2)
    xs, ks, cls = [], [], []
    for name, f in PROBE_CLASSES.items():
        idx = np.nonzero((k % 20 == 0) & f(k // 20))[0][:PROBE_PER_CLASS]
        xs.append(x[idx])
        ks.append(k[idx])
        cls += [name] * len(idx)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ks).astype(np.int32), np.array(cls, dtype=object)


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


def _mean_by(values, groups, names):
    """{name: mean over rows of that group (axis 0)} as lists / floats, plus 'all'."""
    out = {'all': np.nanmean(values, 0).tolist()}
    for g in names:
        m = groups == g
        out[g] = np.nanmean(values[m], 0).tolist() if m.any() else None
    return out


def study(agent, x, k, group, n_forced):
    """geo5b controlled supervised study (sections 7C / 8), on the held-out set (x, k, group) and the boundary
    probes. Keys (chunks of CHUNK rows, chunk c): label roll-in fold_in(PRNGKey(8), c), model readout
    fold_in(PRNGKey(9), c); forced prefixes PRNGKey(10) (one key for every prefix: paired). Normal readout and the
    correction ablation (tokens < 0 or > 9 masked before sampling: an inference intervention) side by side.
      labelfit[readout]: per-position CE / H / KL (method labels q vs p, target-derived prefixes 'a_' and
        model-generated prefixes 'b_'), KL(q*||p), token-mean error (and x 10^-l), clean-target mass, teacher-forced
        accuracy a_tf_acc (argmax p == argmax q), model-generated token accuracy b_token_ok (sampled token in the
        support of the clean corrective two-hot given its own prefix), by group;
      value[readout]: held-out bias / MAE by group and exact-code fraction;
      forced[readout]: forced-prefix recovery (p in {1, 2}, e in -2..2) on the first n_forced held-out rows;
      probes[readout]: per boundary class (d00 / d99 / d0 / d9 codes) unforced value MAE, exact-code fraction and
        forced recovery for the (p, e) grid and the alternative encodings (ENCODINGS), with per-token exact mass
        p(z_{p+1} = r) at the next position."""
    from utils import arb_introspect as AI
    u = k.astype(np.float64) / K_DEN
    act = lambda n: jnp.zeros((n, H * ADIM), jnp.float32)  # noqa: E731
    out = {}
    for ablate in (False, True):
        name = 'ablated' if ablate else 'normal'
        parts = []
        for c, s in enumerate(range(0, len(x), CHUNK)):
            sl = slice(s, s + CHUNK)
            kk = jnp.asarray(k[sl])
            lf = AI.label_fit_k(agent, jnp.asarray(x[sl]), act(len(x[sl])), kk.astype(jnp.float32) / K_DEN, kk,
                                jax.random.fold_in(jax.random.PRNGKey(8), c), jax.random.fold_in(jax.random.PRNGKey(9), c),
                                ablate=ablate)
            parts.append({kq: v for kq, v in lf.items() if kq != 'label_mode'})
        lf = {kq: np.concatenate([p[kq] for p in parts]) for kq in parts[0]}
        stats = {}
        for kq, v in lf.items():
            if kq.startswith(('a_', 'b_')):
                if ablate and kq.startswith('a_'):
                    continue  # target-derived prefixes do not depend on the readout
                stats[kq] = _mean_by(v, group, GROUPS)
                if kq.endswith('tok_mean_err'):
                    stats[kq + '_abs_scaled'] = _mean_by(np.abs(v) * AI.POS_SCALE, group, GROUPS)
        err = lf['readout_value'].astype(np.float64) - u
        out.setdefault('labelfit', {})[name] = stats
        out.setdefault('value', {})[name] = {'bias': _mean_by(err, group, GROUPS),
                                             'mae': _mean_by(np.abs(err), group, GROUPS),
                                             'exact_code_frac': _mean_by(lf['exact_code_frac'], group, GROUPS)}
        key = jax.random.PRNGKey(10)
        fr = AI.forced_recovery(agent, jnp.asarray(x[:n_forced]), act(n_forced), key, u[:n_forced], ablate=ablate)
        out.setdefault('forced', {})[name] = AI.forced_summary(agent, fr)

        px, pk, pcls = boundary_probes()
        pu = pk.astype(np.float64) / K_DEN
        probes = {}
        for cname in PROBE_CLASSES:
            m = pcls == cname
            if not m.any():
                continue
            o, a = jnp.asarray(px[m]), act(int(m.sum()))
            tr = AI.trace(agent, o, a, key, ablate=ablate, with_probs=False)
            n = np.asarray(tr['n'])
            spec = dict({f'p{p}_e{e:+d}': (p, (0,) * (p - 1) + (e,)) for p in (1, 2) for e in AI.OFFSETS},
                        **ENCODINGS)
            fr = AI.forced_recovery(agent, o, a, key, pu[m], ablate=ablate, prefixes=spec)
            probes[cname] = {'n': int(m.sum()),
                             'mae': float(np.abs(np.asarray(tr['value'], np.float64) - pu[m]).mean()),
                             'exact_code_frac': float((n == np.round(pu[m] * 1000)[None, :, None]).mean()),
                             'forced': AI.forced_summary(agent, fr)}
        out.setdefault('probes', {})[name] = probes
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    p.add_argument('--out', required=True)
    p.add_argument('--updates', type=int, default=20000)
    p.add_argument('--study', action='store_true', help='geo5b: add the controlled supervised study (study()).')
    p.add_argument('--study_forced', type=int, default=2000, help='held-out rows for forced-prefix recovery.')
    p.add_argument('--save_params', default=None, help='pickle the trained critic params here.')
    args = p.parse_args()
    if args.study and args.variant == 'E0':
        p.error('--study needs the sample_mean readout (E1-E11)')

    (x_tr, k_tr, _), (x_te, k_te, g_te) = make_data()
    agent = make_agent(args.variant)
    agent, train_info, ms_update = train(agent, x_tr, k_tr, args.updates)
    if args.save_params:
        import pickle
        with open(args.save_params, 'wb') as f:
            pickle.dump(jax.device_get(agent.network.params['modules_critic']), f)
    res = {'variant': args.variant, 'updates': args.updates, 'ms_per_update': ms_update,
           'train_last100': train_info, **evaluate(agent, x_te, k_te, g_te),
           'config': json.loads(json.dumps(dict(agent.config), default=str))}
    if args.study:
        t0 = time.perf_counter()
        res['study'] = study(agent, x_te, k_te, g_te, args.study_forced)
        res['study_s'] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('config', 'study')}), flush=True)


if __name__ == '__main__':
    main()
