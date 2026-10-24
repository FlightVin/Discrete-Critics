"""geo3c: ARB-QC-better targets for the sampled-readout AR-B critic (tiny agents), every label / roll-in recipe at L=3.

Scaling and the exact QC grid (k / 10^6) against Fraction references, the TD target (R + gamma^h m Qbar) on
hand-built batches, one actor next chunk, labels / roll-ins seen by the loss against the label functions called
directly on K_q, and the E6 / E10 / E11 model roll-in. Run from the project root.
"""
import os
import sys
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

import train_v1 as T  # noqa: E402
from agents import agents, geo_codes, v1_config  # noqa: E402
from agents.acarb import (ACARBAgent, GEO_COMBOS, QC_K_DEN, QC_K_TOP, ar_label_dists, canonical_digits,  # noqa: E402
                          geo_config_errors, qc_units, raw_to_u, soft_cross_entropy, u_to_raw)

OBS, ADIM = 6, 3
NC = geo_codes.NUM_CLASSES
COMBOS = {v: k for k, v in GEO_COMBOS.items()}  # 'E5' -> (label, rollin)
ORDINARY = {'E1': 'hard', 'E2': 'twohot_last'}
RECIPES = ['E1', 'E2', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11']
CAP = Fraction(999, 1000)


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', '--horizon_length=5'])
    yield


def qc_cfg(variant='E1', H=5, **over):
    cfg = v1_config('arb_qc')
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    cfg.arb_d_model = 32
    cfg.flow_steps = 2
    cfg.horizon_length = H
    cfg.batch_size = 16
    cfg.critic_target_bon = False
    cfg.arb_readout = 'sample_mean'
    cfg.arb_readout_samples = 10
    cfg.arb_readout_temp = 0.7
    if variant in ORDINARY:
        cfg.arb_label = ORDINARY[variant]
    else:
        cfg.arb_vocab = 'redundant'
        cfg.arb_label, cfg.arb_rollin = COMBOS[variant]
    cfg.update(over)
    return cfg


def q_agent(variant='E1', H=5, seed=0, **over):
    cfg = qc_cfg(variant, H, **over)
    agent = agents['acarb'].create(seed, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)
    return T.split_optimizer(agent)


def with_other_target(agent, seed=7):
    other = agents['acarb'].create(seed, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32),
                                   dict(agent.config))
    p = dict(agent.network.params)
    p['modules_target_critic'] = other.network.params['modules_critic']
    return agent.replace(network=agent.network.replace(params=p))


def make_batch(n=16, H=5, seed=0):
    rs = np.random.RandomState(seed)
    terminals = np.maximum.accumulate((rs.rand(n, H) < 0.25).astype(np.float32), axis=1)
    rewards = -rs.randint(0, H + 1, n).astype(np.float32)
    return dict(observations=rs.randn(n, OBS).astype(np.float32),
                actions=rs.uniform(-1, 1, (n, H, ADIM)).astype(np.float32),
                next_observations=rs.randn(n, H, OBS).astype(np.float32),
                terminals=terminals, masks=(rs.rand(n, H) < 0.7).astype(np.float32),
                valid=np.ones((n, H), np.float32),
                rewards=np.broadcast_to(rewards[:, None], (n, H)).copy(),
                num_chunks_to_success=rs.randn(n).astype(np.float32))


def ref_k(y):
    """Fraction reference of qc_units for one float32 y: (k, lo, hi)."""
    u = (Fraction(float(np.float32(y))) + 100) / 100
    lo, hi = u < 0, u > CAP
    return round(min(max(u, Fraction(0)), CAP) * QC_K_DEN), lo, hi  # round(): half to even


# ---- 1. scaling and the QC grid ----------------------------------------------------------------------
def test_raw_u_inverse_and_cap():
    u = np.linspace(0, 0.999, 1001, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(raw_to_u(u_to_raw(jnp.asarray(u)))), u, atol=1e-6)
    q = np.linspace(-100, -0.1, 1001, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(u_to_raw(raw_to_u(jnp.asarray(q)))), q, atol=2e-5)  # 2 ulp at 100
    # Inverses in float64 (QC_SCALE = 100.0).
    for x in (0.0, 0.999, 0.012345, 1 / 3):
        assert raw_to_u(u_to_raw(x)) == pytest.approx(x, abs=1e-13)
        assert u_to_raw(raw_to_u(x - 100)) == pytest.approx(x - 100, abs=1e-12)
    # Largest decodable value: u = 0.999 -> Qhat = -0.1 (exactly in rationals; float32 within 1 ulp of 100).
    assert QC_K_TOP == 999_000 and Fraction(QC_K_TOP, QC_K_DEN) * 100 - 100 == Fraction(-1, 10)
    assert abs(float(u_to_raw(jnp.float32(0.999))) + 0.1) < 1e-5
    # Largest readout value S / K_den with every code 999 also decodes to -0.1.
    k_den = 20000
    assert Fraction(20 * 999, k_den) * 100 - 100 == Fraction(-1, 10)


def test_cap_and_floor_targets():
    y = jnp.asarray([0.0, -0.05, -0.1, -0.0999, 3.0, -100.0, -100.5, -1000.0, -99.99995], jnp.float32)
    k, lo, hi = (np.asarray(x) for x in qc_units(y))
    assert k.dtype == np.int32
    np.testing.assert_array_equal(k[:5], QC_K_TOP)
    assert hi[:2].all() and hi[3:5].all() and not lo[:5].any()
    # float32(-0.1) is slightly below -0.1: u just under the cap (k rounds to the cap, not clipped).
    assert k[2] == QC_K_TOP and not hi[2]
    assert k[5] == 0 and not lo[5] and not hi[5]          # y = -100 -> u = 0 exactly
    assert k[6] == 0 and k[7] == 0 and lo[6] and lo[7]    # below -100: clipped
    assert k[8] == ref_k(-99.99995)[0]


def test_grid_rounding_matches_fraction_reference():
    rs = np.random.RandomState(0)
    ys = [rs.uniform(-101, 0.5, 100_000).astype(np.float32)]
    ys.append(-rs.exponential(2.0, 20_000).astype(np.float32))  # dense near the cap and in (-1, 0)
    ys.append(np.float32(-100) + np.arange(0, 3200, dtype=np.float32) / 32)  # exact ties (y + 100) * 1e4 = n + 1/2
    edges = np.array([-100, -0.1, 0, -1, -2 ** -4, -0.1000001, -0.0999999, -99.9999, -100.0001, -50.00005,
                      -64.0, -63.99999, -1e-30, -2 ** -27, -3 * 2 ** -28], np.float32)
    ys.append(np.concatenate([edges, np.nextafter(edges, np.float32(1)), np.nextafter(edges, np.float32(-1))]))
    y = np.concatenate(ys).astype(np.float32)
    k, lo, hi = (np.asarray(x) for x in jax.jit(qc_units)(jnp.asarray(y)))
    ref = [ref_k(v) for v in y]
    rk = np.array([r[0] for r in ref])
    bad = np.nonzero(rk != k)[0]
    assert len(bad) == 0, [(float(y[i]), int(k[i]), int(rk[i])) for i in bad[:10]]
    np.testing.assert_array_equal(lo, [r[1] for r in ref])
    np.testing.assert_array_equal(hi, [r[2] for r in ref])
    # |k / K_q - clip(u)| <= 5e-7 (normalised) = 5e-5 return units, exactly (rationals).
    worst = max(abs(Fraction(int(kk), QC_K_DEN) - min(max((Fraction(float(v)) + 100) / 100, Fraction(0)), CAP))
                for kk, v in zip(k, y))
    print(f'qc grid: {len(y)} values, worst |k/K_q - u| = {float(worst):.3e} (bound 5e-7), '
          f'ties tested {3200}, all equal to the Fraction reference')
    assert worst <= Fraction(1, 2 * QC_K_DEN)


# ---- 2. / 3. TD target ---------------------------------------------------------------------------------
def patched_score(monkeypatch, S):
    """critic_score returns the fixed integer S' (target critic only) and records calls."""
    calls = []

    def score(self, o, a, target=False, rng=None):
        calls.append(dict(obs=o.shape, act=a.shape, target=target))
        return jnp.asarray(S, jnp.int32), jnp.zeros(S.shape, jnp.float32)

    monkeypatch.setattr(ACARBAgent, 'critic_score', score)
    return calls


@pytest.mark.parametrize('H', [5, 8])
def test_td_target_against_fraction(H, monkeypatch):
    agent = q_agent('E1', H=H)
    k_den = agent.readout_den()
    assert k_den == 20000 and agent.config['arb_length'] == 3
    n = 12
    rs = np.random.RandomState(H)
    S = np.array([0, 19980, 10000, 12345, 1, 19979, 7777, 0, 19980, 15000, 5, 19000], np.int32)
    b = make_batch(n=n, H=H, seed=H)
    R = np.array([0, -1, -2.5, -0.3, -4.9, 0, -1, 0, -3, -0.75, 0, -2], np.float32)
    b['rewards'] = np.broadcast_to(R[:, None], (n, H)).copy()
    m = np.ones(n, np.float32)
    term = np.zeros(n, np.float32)
    m[[1, 4, 7]] = 0.0        # success / timeout / failure terminals given mask 0: y = R
    term[[1, 4, 7, 9]] = 1.0  # row 9: terminal with mask 1 (truncation): bootstraps (mask only)
    b['masks'][:, -1], b['terminals'][:, -1] = m, term
    calls = patched_score(monkeypatch, S)
    y, u, info = agent.td_target(b, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert calls == [dict(obs=(n, OBS), act=(n, ADIM * H), target=True)]
    y, u, k = np.asarray(y), np.asarray(u), np.asarray(info['target_k'])
    g = Fraction(0.99) ** H  # gamma^h with h = horizon_length (never arb_length = 3)
    for i in range(n):
        qbar = 100 * Fraction(int(S[i]), k_den) - 100
        yr = Fraction(float(R[i])) + g * Fraction(float(m[i])) * qbar
        assert abs(float(yr) - y[i]) < 2e-5, (i, float(yr), y[i])
        assert k[i] == ref_k(y[i])[0]
        uc = min(max((yr + 100) / 100, Fraction(0)), CAP)
        assert abs(Fraction(int(k[i]), QC_K_DEN) - uc) <= Fraction(1, 2 * QC_K_DEN) + Fraction(1, 10 ** 7)
        assert u[i] == np.float32(k[i] / QC_K_DEN)
    np.testing.assert_array_equal(y[[1, 4, 7]], R[[1, 4, 7]])
    # gamma^3 instead of gamma^H would differ visibly on bootstrapped rows.
    wrong = R[2] + 0.99 ** 3 * (100 * S[2] / k_den - 100)
    assert abs(wrong - y[2]) > 0.01
    assert float(info['target_clip_hi_frac']) == pytest.approx(np.mean([ref_k(v)[2] for v in y]), abs=1e-7)
    assert float(info['target_clip_lo_frac']) == pytest.approx(np.mean([ref_k(v)[1] for v in y]), abs=1e-7)


def test_td_target_draws_one_actor_chunk(monkeypatch):
    agent = q_agent('E5')
    calls = []
    orig = ACARBAgent.compute_flow_actions.__wrapped__

    def rec(self, observations, noises):
        calls.append((observations.shape, noises.shape))
        return orig(self, observations, noises)

    monkeypatch.setattr(ACARBAgent, 'compute_flow_actions', rec)
    monkeypatch.setattr(ACARBAgent, 'sample_actions', lambda *a, **k: pytest.fail('best-of-n in the TD target'))
    agent.td_target(make_batch(), jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert calls == [((16, OBS), (16, ADIM * 5))], calls  # ONE chunk per element, no N=32 candidates
    # The full loss path also draws exactly one chunk for the target.
    calls.clear()
    agent.critic_loss(make_batch(), agent.network.params, jax.random.PRNGKey(3))
    assert calls == [((16, OBS), (16, ADIM * 5))], calls


# ---- 4. labels seen by the loss, every recipe ---------------------------------------------------------
class Recorder:
    def __init__(self, monkeypatch):
        self.calls = []
        orig = geo_codes.build

        def build(k, label_mode, rollin_mode, key, model_token=None, k_den=geo_codes.K_DEN):
            out = orig(k, label_mode, rollin_mode, key, model_token, k_den)
            self.calls.append(dict(k=k, label_mode=label_mode, rollin_mode=rollin_mode, key=key, k_den=k_den,
                                   out=out))
            return out

        monkeypatch.setattr(geo_codes, 'build', build)


def loss_keys(rng):
    _, sample_rng, readout_rng, rollin_rng = jax.random.split(rng, 4)
    return sample_rng, readout_rng, rollin_rng


def label_means_ref(k, prefix_tokens, mode):
    """Fraction expected token per position given the actual prefix: r (corrective), c (noisy_fixed), r - 1/2."""
    out = []
    for l in range(3):
        P = sum(geo_codes.UNIT_WEIGHTS[j] * int(prefix_tokens[j]) for j in range(l))
        r = Fraction(int(k) - QC_K_DEN // 1000 * P, QC_K_DEN // 10 ** (l + 1))
        out.append({'corrective': r, 'noisy_fixed': Fraction(r.numerator // r.denominator),
                    'noisy_resid': r - Fraction(1, 2)}[mode])
    return out


@pytest.mark.parametrize('variant', RECIPES)
def test_loss_labels_match_label_functions_on_kq(variant, monkeypatch):
    agent = with_other_target(q_agent(variant))
    batch = make_batch(n=64, seed=1)
    rng = jax.random.PRNGKey(3)
    rec = Recorder(monkeypatch)
    loss, info = agent.critic_loss(batch, agent.network.params, rng)
    sample_rng, readout_rng, rollin_rng = loss_keys(rng)
    y, u, tinfo = agent.td_target(batch, sample_rng, readout_rng)
    k = np.asarray(tinfo['target_k'])
    assert k.dtype == np.int32 and agent.label_den() == QC_K_DEN
    np.testing.assert_array_equal(k, [ref_k(v)[0] for v in np.asarray(y)])
    act = batch['actions'].reshape(64, -1)
    w = batch['valid'][:, -1]
    if variant in ORDINARY:
        assert rec.calls == []
        d, q, clip = ar_label_dists(jnp.asarray(k), QC_K_DEN, 10, 3, ORDINARY[variant])
        np.testing.assert_array_equal(np.asarray(d), np.asarray(canonical_digits(k, QC_K_DEN, 10, 3)))
        np.testing.assert_array_equal(np.asarray(d), np.stack([k // 100000, k // 10000 % 10, k // 1000 % 10], -1))
        logits = agent.network.select('critic')(batch['observations'], act, d)
        if variant == 'E1':
            ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), d[None, ..., None], -1)[..., 0]
        else:
            ce = soft_cross_entropy(logits, q[None])
            # Label means: canonical digits, then T(r_3) on the actual (canonical) prefix: r_3 exact.
            means = np.asarray((q * jnp.arange(10)).sum(-1))
            clipped = 0
            for i in range(64):
                r3 = Fraction(int(k[i]) - 1000 * 10 * int(k[i] // 10000), 1000)  # prefix P = 100 d1 + 10 d2
                assert 0 <= r3 < 10
                clipped += r3 > 9  # ordinary clamp to the last class (as BSC)
                assert abs(means[i, 2] - float(min(r3, Fraction(9)))) < 1e-5
            assert float(info['label_clip_pos3']) == pytest.approx(clipped / 64, abs=1e-7)
        ref = (ce.mean(axis=(0, 2)) * w).mean()
        assert float(loss) == pytest.approx(float(ref), rel=1e-6, abs=1e-7)
        return
    assert len(rec.calls) == 1
    call = rec.calls[0]
    label, rollin_mode = COMBOS[variant]
    assert (call['label_mode'], call['rollin_mode'], call['k_den']) == (label, rollin_mode, QC_K_DEN)
    np.testing.assert_array_equal(np.asarray(call['k']), np.broadcast_to(k, (2, 64)))
    assert np.array_equal(np.asarray(call['key']), np.asarray(rollin_rng))
    labels, rollin, binfo = call['out']
    # Labels == the label function called directly with k on K_q, given the actual roll-in prefix.
    l2, _ = geo_codes.label_and_info(jnp.broadcast_to(k, (2, 64)), rollin, label, QC_K_DEN)
    np.testing.assert_array_equal(np.asarray(labels), np.asarray(l2))
    if rollin_mode != 'model_mix':
        l3, r3, _ = geo_codes.build(jnp.broadcast_to(k, (2, 64)), label, rollin_mode, rollin_rng, k_den=QC_K_DEN)
        np.testing.assert_array_equal(np.asarray(labels), np.asarray(l3))
        np.testing.assert_array_equal(np.asarray(rollin), np.asarray(r3))
    # Label means vs Fraction reference, every copy / element / position.
    tokens = np.arange(geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX + 1)
    means = (np.asarray(labels, np.float64) * tokens).sum(-1)
    rl = np.asarray(rollin)
    for c in range(2):
        for i in range(64):
            ref = label_means_ref(k[i], rl[c, i], label)
            for l in range(3):
                assert abs(means[c, i, l] - float(ref[l])) < 2e-6, (c, i, l, means[c, i, l], ref[l])
    assert float(info['oov_label']) == 0 and float(info['oov_rollin']) == 0
    # Loss == the protocol's loss from these labels / roll-ins.
    logits = agent._percopy_critic(False)(batch['observations'], act, geo_codes.token_to_class(rollin))
    ref = (soft_cross_entropy(logits, labels).mean(axis=(0, 2)) * w).mean()
    assert float(loss) == pytest.approx(float(ref), rel=1e-6, abs=1e-7)
    bias = float(info['label_mean_bias'])
    assert float(info['label_mean_bias_raw']) == pytest.approx(100 * bias, rel=1e-6, abs=1e-9)
    if label == 'corrective':
        assert abs(bias) < 2e-7, bias
    elif label == 'noisy_resid':
        assert bias == pytest.approx(-0.0005, abs=2e-7)
    for key in ('target_clip_hi_frac', 'target_clip_lo_frac', 'target_q', 'target_q_raw', 'readout_oor_frac',
                'label_mean_bias', 'label_mean_bias_raw', 'oov_label', 'oov_rollin'):
        assert np.isfinite(float(info[key])), key
    assert 'target_k' not in info


@pytest.mark.parametrize('label', geo_codes.LABEL_MODES)
@pytest.mark.parametrize('rollin', geo_codes.ROLLIN_MODES)
def test_label_coverage_on_kq(label, rollin):
    """0 oov over the whole QC grid (every 7th k plus both ends) for every label / roll-in pair."""
    k = jnp.concatenate([jnp.arange(0, QC_K_TOP + 1, 7, dtype=jnp.int32), jnp.asarray([QC_K_TOP], jnp.int32)])
    mt = (lambda prefix, l: jnp.full(k.shape, 31 if l else -22, jnp.int32)) if rollin == 'model_mix' else None
    _, _, info = geo_codes.build(k, label, rollin, jax.random.PRNGKey(0), mt, k_den=QC_K_DEN)
    assert int(info['oov_label']) == 0 and int(info['oov_rollin']) == 0


# ---- 5. E6 / E10 / E11 under QC ---------------------------------------------------------------------
def test_model_mix_clean_branch_independent_of_label_rule(monkeypatch):
    batch = make_batch(n=128, seed=4)
    rng = jax.random.PRNGKey(11)
    outs = {}
    for v in ('E6', 'E10', 'E11'):
        rec = Recorder(monkeypatch)
        agent = with_other_target(q_agent(v))  # same seed: same params for all three
        agent.critic_loss(batch, agent.network.params, rng)
        outs[v] = rec.calls[0]
    for v in ('E10', 'E11'):
        np.testing.assert_array_equal(np.asarray(outs[v]['k']), np.asarray(outs['E6']['k']))
        np.testing.assert_array_equal(np.asarray(outs[v]['out'][1]), np.asarray(outs['E6']['out'][1]))
        np.testing.assert_array_equal(np.asarray(outs[v]['out'][2]['model_mask']),
                                      np.asarray(outs['E6']['out'][2]['model_mask']))
    # Clean branch: the corrective two-hot draw c + Bernoulli(f) of build's own key, whatever the label rule.
    _, rollin, binfo = outs['E6']['out']
    mask = np.asarray(binfo['model_mask'])
    _, clean, _ = geo_codes.build(outs['E6']['k'], 'corrective', 'corrective', outs['E6']['key'], k_den=QC_K_DEN)
    # Position 1's clean draw depends on the actual position-0 token: compare where position 0 matches too.
    r, cl = np.asarray(rollin), np.asarray(clean)
    np.testing.assert_array_equal(r[..., 0][~mask[..., 0]], cl[..., 0][~mask[..., 0]])
    same0 = r[..., 0] == cl[..., 0]
    np.testing.assert_array_equal(r[..., 1][~mask[..., 1] & same0], cl[..., 1][~mask[..., 1] & same0])


@pytest.mark.parametrize('variant', ['E6', 'E10', 'E11'])
def test_model_branch_is_current_copy_at_temp(variant, monkeypatch):
    agent = with_other_target(q_agent(variant))
    p = dict(agent.network.params)  # sharpen the current critic so a wrong temperature shows
    crit = dict(p['modules_critic'])
    crit['head'] = dict(crit['head'], kernel=crit['head']['kernel'] * 10)
    p['modules_critic'] = crit
    agent = agent.replace(network=agent.network.replace(params=p))
    batch = make_batch(n=256, seed=4)
    rng = jax.random.PRNGKey(11)
    rec = Recorder(monkeypatch)
    _, info = agent.critic_loss(batch, agent.network.params, rng)
    _, rollin, binfo = rec.calls[0]['out']
    rollin, mask, c = (np.asarray(x) for x in (rollin, binfo['model_mask'], binfo['c']))
    model_rng = jax.random.split(loss_keys(rng)[2], 4)[3]
    act = batch['actions'].reshape(256, -1)
    f = agent._percopy_critic(False)  # CURRENT critic
    clamps = 0
    for l in range(2):
        prefix = rollin.copy()
        prefix[..., l:] = 0
        logits = np.asarray(f(batch['observations'], act, geo_codes.token_to_class(jnp.asarray(prefix))))[..., l, :]
        g = np.asarray(jax.random.gumbel(jax.random.fold_in(model_rng, l), logits.shape))
        m = np.argmax(logits / 0.7 + g, -1) + geo_codes.TOKEN_MIN
        expect = np.clip(m, c[..., l] - 2, c[..., l] + 2)
        np.testing.assert_array_equal(rollin[..., l][mask[..., l]], expect[mask[..., l]])
        assert ((np.argmax(logits + g, -1) + geo_codes.TOKEN_MIN) != m)[mask[..., l]].any()  # T matters
        clamps += int((np.abs(m - c[..., l]) > 2)[mask[..., l]].sum())
    assert int(binfo['model_clamp']) == clamps
    assert 0.44 < mask.mean() < 0.56
    assert float(info['model_clamp_frac']) == pytest.approx(clamps / max(mask.sum(), 1), abs=1e-6)


# ---- config, readout, logging -------------------------------------------------------------------------
@pytest.mark.parametrize('variant', RECIPES)
def test_qc_recipes_allowed(variant):
    cfg = qc_cfg(variant)
    assert geo_config_errors(cfg) == []
    T.check_agent_config('bc', cfg)  # no exit ('online' also needs --method: checked in the e2e test)


@pytest.mark.parametrize('over', [dict(critic_target_bon=True), dict(discount=1.0), dict(discount=0.995),
                                  dict(use_mc_target=True)])
def test_qc_sample_mean_refusals(over):
    with pytest.raises(AssertionError):
        agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), qc_cfg('E5', **over))
    with pytest.raises(SystemExit) as e:
        T.check_agent_config('bc', qc_cfg('E5', **over))
    assert e.value.code == 2


def test_bsc_sample_mean_still_needs_discount_one():
    cfg = qc_cfg('E5', arb_value='bsc', discount=0.99)
    assert geo_config_errors(cfg)
    cfg = qc_cfg('E5', arb_value='bsc', discount=1.0)
    assert geo_config_errors(cfg) == []


def test_readout_logging_and_ranking(monkeypatch):
    agent = q_agent('E6')
    d = agent.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    assert float(d['critic/q_readout_raw']) == pytest.approx(100 * float(d['critic/q_readout']) - 100, abs=1e-4)
    a1 = q_agent('E1')
    d1 = a1.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    assert float(d1['critic/q_readout_raw']) == pytest.approx(100 * float(d1['critic/q_readout']) - 100, abs=1e-4)
    # BO32 ranks by the integer S (first index on ties), as BSC.
    N = agent.config['actor_num_samples']
    cands = jnp.asarray(np.arange(2 * N * ADIM * 5, dtype=np.float32).reshape(2, N, ADIM * 5))
    S = np.zeros((2, N), np.int32)
    S[0, [3, 8]] = 19980
    S[1, 30] = 1
    monkeypatch.setattr(ACARBAgent, '_candidates', lambda self, o, rng: (jnp.repeat(o[:, None], N, 1), cands))
    monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (jnp.asarray(S), None))
    out = ACARBAgent.sample_actions.__wrapped__(agent, jnp.zeros((2, OBS)), rng=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cands[jnp.arange(2), jnp.asarray([3, 30])]))


def test_update_runs_and_target_family():
    agent = q_agent('E11')
    new, info = T.critic_update(agent, make_batch())
    assert T.opt_counts(new) == {'actor': 0, 'critic': 1}
    for k in ('critic_loss', 'target_q', 'target_q_raw', 'target_clip_hi_frac', 'target_clip_lo_frac',
              'label_mean_bias', 'label_mean_bias_raw', 'readout_oor_frac', 'oov_label', 'oov_rollin'):
        assert np.isfinite(float(info[f'critic/{k}'])), k
    assert T.target_family(agent.config) == 'ARB-QC-better'
    assert T.target_family(v1_config('arb_bsc')) == 'ARB-BSC'
    assert T.resolved_qc_args(agent.config) == dict(arb_value='qc', discount=0.99, target_family='ARB-QC-better')
    assert T.resolved_qc_args(v1_config('arb_qc')) is None  # greedy ARB-QC: args.json unchanged
