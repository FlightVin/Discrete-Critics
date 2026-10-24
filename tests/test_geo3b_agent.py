"""geo3b: five-token codes (--agent.arb_length=5) inside the ARB-BSC sampled-readout critic (tiny agents).

Init hashes, the integer TD target for every S, the executed loss path (E1 / E2 ordinary, E5-E7 / E9-E11 redundant)
against direct recomputation, the model roll-in, the sampled readout recomputed position by position, config
checks, diagnostics and the fixed-candidate probes. Run from the project root.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

import train_v1 as T  # noqa: E402
from agents import acarb, agents, geo_codes, v1_config  # noqa: E402
from agents.acarb import ACARBAgent, GEO_COMBOS, ar_label_dists, canonical_digits, soft_cross_entropy  # noqa: E402
from test_geo2b_agent import direct_loss, leaf_hashes, make_batch, sharpen  # noqa: E402
from visualization import critic_probe  # noqa: E402

H, OBS, ADIM, L = 5, 6, 3, 5
NC = geo_codes.NUM_CLASSES
K_DEN = 2 * 10 * 10 ** L  # num_qs M B^L = 2,000,000
G = K_DEN // 10 ** L      # 20
K_TOP = K_DEN - G
COMBOS = {v: k for k, v in GEO_COMBOS.items()}  # 'E5' -> (label, rollin)
ORDINARY = {'E1': 'hard', 'E2': 'twohot_last'}
REDUNDANT = ['E5', 'E6', 'E7', 'E9', 'E10', 'E11']
MODEL_MIX = ['E6', 'E10', 'E11']


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def cfg_of(variant, length=L, temp=0.7, **over):
    cfg = v1_config('arb_bsc')
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    cfg.arb_d_model = 32
    cfg.flow_steps = 2
    cfg.horizon_length = H
    cfg.batch_size = 16
    cfg.critic_target_bon = False
    cfg.arb_readout = 'sample_mean'
    cfg.arb_readout_samples = 10
    cfg.arb_readout_temp = temp
    cfg.arb_length = length
    if variant in ORDINARY:
        cfg.arb_label = ORDINARY[variant]
    else:
        cfg.arb_vocab = 'redundant'
        cfg.arb_label, cfg.arb_rollin = COMBOS[variant]
    cfg.update(over)
    return cfg


def make(variant, seed=0, split=True, other_target=False, **kw):
    agent = agents['acarb'].create(seed, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg_of(variant, **kw))
    if other_target:  # target critic with different parameters (so using the wrong network shows)
        other = agents['acarb'].create(seed + 7, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32),
                                       cfg_of(variant, **kw))
        p = dict(agent.network.params)
        p['modules_target_critic'] = other.network.params['modules_critic']
        agent = agent.replace(network=agent.network.replace(params=p))
    return T.split_optimizer(agent) if split else agent


class Recorder:
    """Wraps geo_codes.build (as called from acarb) and keeps its concrete inputs / outputs."""

    def __init__(self, monkeypatch):
        self.calls = []
        orig = geo_codes.build

        def build(k, label_mode, rollin_mode, key, model_token=None, k_den=geo_codes.K_DEN, length=geo_codes.LENGTH):
            out = orig(k, label_mode, rollin_mode, key, model_token, k_den, length)
            self.calls.append(dict(k=k, label_mode=label_mode, rollin_mode=rollin_mode, key=key, k_den=k_den,
                                   length=length, out=out))
            return out

        monkeypatch.setattr(geo_codes, 'build', build)


def loss_keys(rng):
    _, sample_rng, readout_rng, rollin_rng = jax.random.split(rng, 4)
    return sample_rng, readout_rng, rollin_rng


# ---- initialisation ---------------------------------------------------------------------------------
def test_init_hashes():
    h = {v: leaf_hashes(make(v, split=False).network.params) for v in [*ORDINARY, *REDUNDANT]}
    assert h['E2'] == h['E1']
    for v in REDUNDANT[1:]:
        assert h[v] == h['E5'], v
    h3 = {v: leaf_hashes(make(v, split=False, length=3).network.params) for v in ('E1', 'E5')}
    report = {}
    for v in ('E1', 'E5'):
        report[f'{v} L3 vs L5'] = sorted(p for p in h[v] if h[v][p] != h3[v][p])
    report['E1 vs E5 (L5)'] = sorted(p for p in h['E1'] if h['E1'][p] != h['E5'][p])
    report['E1 vs E5 (L3)'] = sorted(p for p in h3['E1'] if h3['E1'][p] != h3['E5'][p])
    for k, v in report.items():
        print(f'init differs [{k}]: {v}')
    for v in ('E1', 'E5'):
        assert set(h[v]) == set(h3[v])
        assert all('pos_embed' in p for p in report[f'{v} L3 vs L5']) and len(report[f'{v} L3 vs L5']) == 2
    token = [p for p in h['E1'] if 'digit_embed' in p or "['head']" in p]
    assert report['E1 vs E5 (L5)'] == sorted(token) and len(token) == 6
    shapes = {jax.tree_util.keystr(p): x.shape for p, x in
              jax.tree_util.tree_leaves_with_path(make('E5', split=False).network.params)}
    assert [s for p, s in shapes.items() if 'pos_embed' in p] == [(2, L, 32)] * 2


# ---- TD target: integer k = S ------------------------------------------------------------------------
def test_td_target_k_is_S_for_every_score(monkeypatch):
    """Rounding rule: none. The target numerator is the integer readout score itself, k = S (no float on the way),
    success k = k_top = K_DEN - 20, timeout k = 0; the float u = S / K_DEN is logging only. Canonical digits are
    floor(k / 20) (integer floor division) and the final-position residual is eps_5 = (k mod 20) / 20. Checked for
    every S in [0, 20 * 99999] with the real td_target / ar_label_dists code, one actor chunk per element."""
    agent = make('E2')
    n_all = 20 * (10 ** L - 1) + 1
    chunk = 1 << 18
    state = {}

    def fake_score(self, o, a, target=False, rng=None):
        assert target
        return jnp.asarray(state['S']), jnp.zeros(o.shape[0], jnp.float32)

    calls = []
    orig = ACARBAgent.compute_flow_actions.__wrapped__

    def rec(self, observations, noises):
        calls.append((observations.shape, noises.shape))
        return orig(self, observations, noises)

    monkeypatch.setattr(ACARBAgent, 'critic_score', fake_score)
    monkeypatch.setattr(ACARBAgent, 'compute_flow_actions', rec)
    labels_fn = jax.jit(lambda k: ar_label_dists(k, K_DEN, 10, L, 'twohot_last'))
    for s0 in range(0, n_all, chunk):
        S = np.arange(s0, min(s0 + chunk, n_all), dtype=np.int32)
        n = len(S)
        state['S'] = S
        batch = dict(next_observations=np.zeros((n, H, OBS), np.float32), terminals=np.zeros((n, H), np.float32),
                     masks=np.ones((n, H), np.float32), num_chunks_to_success=np.zeros(n, np.float32))
        calls.clear()
        _, u, info = agent.td_target(batch, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
        assert calls == [((n, OBS), (n, ADIM * H))], calls
        k = np.asarray(info['target_k'])
        assert k.dtype == np.int32
        np.testing.assert_array_equal(k, S)
        assert float(info['target_clip_frac']) == 0.0
        d, q, clip = (np.asarray(x) for x in labels_fn(jnp.asarray(k)))
        code = S // G
        np.testing.assert_array_equal(d, (code[:, None] // 10 ** np.arange(L - 1, -1, -1)) % 10)
        np.testing.assert_array_equal(np.asarray(canonical_digits(k, K_DEN, 10, L)), d)
        eps = (S % G) / G
        last = d[:, -1]
        top = (last == 9) & (eps > 0)  # ordinary clamp of r_5 to [0, 9] (logged as label_clip_pos5)
        np.testing.assert_array_equal(clip[:, -1], top)
        np.testing.assert_array_equal(clip[:, :-1], False)
        np.testing.assert_allclose(q[np.arange(n), -1, last], np.where(top, 1.0, 1 - eps), atol=1e-6)
        ok = ~top & (eps > 0)
        np.testing.assert_allclose(q[np.arange(n)[ok], -1, last[ok] + 1], eps[ok], atol=1e-6)
    # Terminals: success -> k_top, timeout -> 0.
    batch = make_batch(n=64, seed=3)
    state['S'] = np.full(64, 12345, np.int32)
    _, _, info = agent.td_target(batch, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    k = np.asarray(info['target_k'])
    term = batch['terminals'][:, -1] > 0.5
    succ = term & (batch['masks'][:, -1] < 0.5)
    assert succ.any() and (term & ~succ).any() and (~term).any()
    assert (k[succ] == K_TOP).all() and (k[term & ~succ] == 0).all() and (k[~term] == 12345).all()


# ---- loss path ---------------------------------------------------------------------------------------
@pytest.mark.parametrize('variant', REDUNDANT)
def test_redundant_loss_path(variant, monkeypatch):
    agent = make(variant, other_target=True)
    batch = make_batch()
    rng = jax.random.PRNGKey(3)
    rec = Recorder(monkeypatch)
    loss, info = agent.critic_loss(batch, agent.network.params, rng)
    assert len(rec.calls) == 1
    call = rec.calls[0]
    sample_rng, readout_rng, rollin_rng = loss_keys(rng)
    _, _, tinfo = agent.td_target(batch, sample_rng, readout_rng)
    k = np.broadcast_to(np.asarray(tinfo['target_k']), (2, 16))
    np.testing.assert_array_equal(np.asarray(call['k']), k)
    assert np.array_equal(np.asarray(call['key']), np.asarray(rollin_rng))
    assert (call['label_mode'], call['rollin_mode'], call['k_den'], call['length']) == (*COMBOS[variant], K_DEN, L)
    labels, rollin, binfo = call['out']
    assert labels.shape == (2, 16, L, NC) and rollin.shape == (2, 16, L)
    if variant not in MODEL_MIX:  # model_mix: test_model_mix_rollin
        l2, r2, _ = geo_codes.build(jnp.asarray(k), *COMBOS[variant], rollin_rng, k_den=K_DEN, length=L)
        np.testing.assert_array_equal(np.asarray(labels), np.asarray(l2))
        np.testing.assert_array_equal(np.asarray(rollin), np.asarray(r2))
    assert not np.array_equal(np.asarray(rollin[0, :, :-1]), np.asarray(rollin[1, :, :-1]))
    # Each copy teacher-forced on its own roll-in; loss = the direct recomputation.
    ref = direct_loss(agent, batch, agent.network.params, labels, rollin)
    assert float(loss) == pytest.approx(float(ref), rel=1e-6, abs=1e-7)
    # Final-token supervision at position 5: CE = -sum q log p recomputed.
    act = batch['actions'].reshape(16, -1)
    logits = agent._percopy_critic(False)(batch['observations'], act, geo_codes.token_to_class(rollin))
    q5 = np.asarray(labels[..., L - 1, :])
    assert (q5.sum(-1) > 0.999).all()
    ce5 = -(q5 * np.asarray(jax.nn.log_softmax(logits[..., L - 1, :], -1))).sum(-1)  # (2, 16)
    assert float(info[f'ce_digit{L}']) == pytest.approx(float(ce5.mean()), rel=1e-5)
    for l in range(1, L + 1):
        for key in (f'ce_digit{l}', f'tf_acc_digit{l}', f'greedy_acc_digit{l}'):
            assert np.isfinite(float(info[key])), key
    assert float(info['oov_label']) == 0 and float(info['oov_rollin']) == 0
    # label_mean_bias in value units: corrective 0, noisy_resid -0.5e-5, noisy_fixed -1e-5 eps_5.
    bias = float(info['label_mean_bias'])
    label = COMBOS[variant][0]
    if label == 'corrective':
        assert abs(bias) < 1e-9, bias
    elif label == 'noisy_resid':
        assert bias == pytest.approx(-0.5e-5, abs=1e-9)
    else:
        f = np.asarray(binfo['rem'][..., -1]) / np.asarray(binfo['den'][..., -1])
        assert bias == pytest.approx(-f.mean() * 1e-5, abs=1e-9)
    print(f'{variant} label_mean_bias {bias:.3e}')
    jl, _ = jax.jit(agent.critic_loss)(batch, agent.network.params, rng)
    assert float(jl) == pytest.approx(float(loss), rel=1e-5)


@pytest.mark.parametrize('variant', list(ORDINARY))
def test_ordinary_loss_path(variant):
    agent = make(variant, other_target=True)
    batch = make_batch(n=64, seed=5)
    rng = jax.random.PRNGKey(4)
    loss, info = agent.critic_loss(batch, agent.network.params, rng)
    sample_rng, readout_rng, _ = loss_keys(rng)
    tq, u, tinfo = agent.td_target(batch, sample_rng, readout_rng)
    k = np.asarray(tinfo['target_k'])
    d, q, clip = ar_label_dists(jnp.asarray(k), K_DEN, 10, L, ORDINARY[variant])
    np.testing.assert_array_equal(np.asarray(d), np.asarray(canonical_digits(jnp.asarray(k), K_DEN, 10, L)))
    act = batch['actions'].reshape(64, -1)
    logits = agent.network.select('critic')(batch['observations'], act, d)  # canonical teacher forcing
    ce = soft_cross_entropy(logits, q[None])
    np.testing.assert_allclose(float(loss), float((ce.mean(axis=(0, 2)) * batch['valid'][:, -1]).mean()), rtol=1e-5)
    assert float(info[f'ce_digit{L}']) == pytest.approx(float(ce[..., L - 1].mean()), rel=1e-5)
    if variant == 'E2':  # two-hot at position 5 only
        assert (np.asarray(q[:, :L - 1]).max(-1) == 1).all()
        assert (np.asarray(q[:, L - 1]).max(-1) < 1).any()
    else:
        assert (np.asarray(q).max(-1) == 1).all()
    for l in range(1, L + 1):
        assert float(info[f'label_clip_pos{l}']) == pytest.approx(float(np.asarray(clip[:, l - 1]).mean()))
        assert np.isfinite(float(info[f'tf_acc_digit{l}'])) and np.isfinite(float(info[f'greedy_acc_digit{l}']))


@pytest.mark.parametrize('variant', MODEL_MIX)
def test_model_mix_rollin(variant, monkeypatch):
    agent = sharpen(make(variant, other_target=True), 10)  # sharp: a wrong temperature shows
    batch = make_batch(n=256, seed=4)
    rng = jax.random.PRNGKey(11)
    rec = Recorder(monkeypatch)
    _, info = agent.critic_loss(batch, agent.network.params, rng)
    labels, rollin, binfo = rec.calls[0]['out']
    rollin, mask, c, rem = (np.asarray(x) for x in (rollin, binfo['model_mask'], binfo['c'], binfo['rem']))
    k = np.asarray(rec.calls[0]['k'])
    _, _, rollin_rng = loss_keys(rng)
    model_rng = jax.random.split(rollin_rng, 4)[3]
    key_bern, key_coin, _ = jax.random.split(rollin_rng, 3)
    shape = (2, 256, L - 1)
    dens = jnp.array([K_DEN // 10 ** (l + 1) for l in range(L - 1)], jnp.int32)
    bern_u = np.asarray(jax.random.randint(key_bern, shape, 0, dens))
    np.testing.assert_array_equal(mask, np.asarray(jax.random.bernoulli(key_coin, 0.5, shape)))
    act = batch['actions'].reshape(256, -1)
    f = agent._percopy_critic(False)  # CURRENT critic (target differs)
    clamps = 0
    for l in range(L - 1):
        prefix = rollin.copy()
        prefix[..., l:] = 0
        logits = np.asarray(f(batch['observations'], act, geo_codes.token_to_class(jnp.asarray(prefix))))[..., l, :]
        g = np.asarray(jax.random.gumbel(jax.random.fold_in(model_rng, l), logits.shape))
        m = np.argmax(logits / 0.7 + g, -1) + geo_codes.TOKEN_MIN
        expect = np.clip(m, c[..., l] - 2, c[..., l] + 2)
        np.testing.assert_array_equal(rollin[..., l][mask[..., l]], expect[mask[..., l]])
        m1 = np.argmax(logits + g, -1) + geo_codes.TOKEN_MIN
        assert (m1 != m)[mask[..., l]].any()
        clamps += int((np.abs(m - c[..., l]) > 2)[mask[..., l]].sum())
        # Clean branch: the two-hot draw c + [u < rem] with E5's uniform u (same key, same element / position).
        clean = ~mask[..., l]
        np.testing.assert_array_equal(rollin[..., l][clean], (c[..., l] + (bern_u[..., l] < rem[..., l]))[clean])
    assert int(binfo['model_clamp']) == clamps
    # Elements whose coins are all clean have exactly E5's roll-in for the same key.
    _, r5, _ = geo_codes.build(jnp.asarray(k), 'corrective', 'corrective', rollin_rng, k_den=K_DEN, length=L)
    allclean = ~mask.any(-1)
    assert allclean.sum() >= 10
    np.testing.assert_array_equal(rollin[allclean], np.asarray(r5)[allclean])
    used = mask.mean()
    assert 0.45 < used < 0.55, used
    assert float(info['model_used_frac']) == pytest.approx(used, abs=1e-6)
    assert float(info['oov_label']) == 0 and float(info['oov_rollin']) == 0
    print(f'{variant}: model_used {used:.3f}, clamp {clamps / mask.sum():.3f}')


# ---- readout -----------------------------------------------------------------------------------------
@pytest.mark.parametrize('variant', ['E1', 'E5'])
def test_sampled_readout_recomputed(variant, monkeypatch):
    """Each sampled position is argmax(logits / 0.7 + Gumbel) (an exact softmax(logits / 0.7) sample) with logits of
    the actual sampled prefix; codes decoded, clipped to [0, 99999] per code, summed to S over 2 copies x 10 samples,
    value S / 2,000,000."""
    agent = make(variant, seed=3)
    rec = []
    orig = acarb.ar_sample_codes

    def wrapped(critic_fn, observations, actions, gumbel, temp):
        out = orig(critic_fn, observations, actions, gumbel, temp)
        rec.append((gumbel, temp, out))
        return out

    monkeypatch.setattr(acarb, 'ar_sample_codes', wrapped)
    rs = np.random.RandomState(0)
    obs = jnp.asarray(rs.randn(64, OBS), jnp.float32)
    act = jnp.asarray(rs.uniform(-1, 1, (64, ADIM * H)), jnp.float32)
    rng = jax.random.PRNGKey(5)
    S, oor = agent.critic_score(obs, act, rng=rng)
    gumbel, temp, digits = rec[0]
    nc = NC if variant == 'E5' else 10
    assert temp == 0.7 and digits.shape == (2, 64, 10, L) and gumbel.shape == (2, 64, 10, L, nc)
    np.testing.assert_array_equal(np.asarray(gumbel), np.asarray(jax.random.gumbel(rng, (2, 64, 10, L, nc))))
    logits = np.asarray(agent._percopy_critic(False)(obs, act, digits))  # causal: position l sees digits[:l]
    digits, gumbel = np.asarray(digits), np.asarray(gumbel)
    for l in range(L):
        np.testing.assert_array_equal(digits[..., l], np.argmax(logits[..., l, :] / 0.7 + gumbel[..., l, :], -1))
        if l > 0:  # the prefix matters: the same position under the zero prefix gives different samples
            z = np.asarray(agent._percopy_critic(False)(obs, act, jnp.zeros_like(digits)))[..., l, :]
            assert (np.argmax(z / 0.7 + gumbel[..., l, :], -1) != digits[..., l]).any()
    tokens = digits + (geo_codes.TOKEN_MIN if variant == 'E5' else 0)
    n = (tokens * 10 ** np.arange(L - 1, -1, -1)).sum(-1)
    if variant == 'E5':
        assert ((n < 0) | (n > 10 ** L - 1)).any()  # the per-code clip is active
    np.testing.assert_array_equal(np.asarray(S), np.clip(n, 0, 10 ** L - 1).sum((0, 2)))
    np.testing.assert_allclose(np.asarray(oor), ((n < 0) | (n > 10 ** L - 1)).mean((0, 2)), atol=1e-7)
    assert np.asarray(S).dtype == np.int32 and np.asarray(S).max() <= 20 * (10 ** L - 1)
    q = agent.readout_q(obs, act, rng)
    np.testing.assert_allclose(np.asarray(q), np.asarray(S) / K_DEN, rtol=1e-6)
    v = agent.critic_value(obs, act, rng=rng)  # per-copy values: mean of S / K_DEN over copies
    np.testing.assert_allclose(np.asarray(v).mean(0), np.asarray(S) / K_DEN, rtol=1e-6)


def test_first_position_frequencies_match_tempered_softmax():
    agent = make('E5', seed=2)
    rs = np.random.RandomState(0)
    obs = jnp.asarray(np.broadcast_to(rs.randn(OBS), (4000, OBS)), jnp.float32)
    act = jnp.asarray(np.broadcast_to(rs.uniform(-1, 1, ADIM * H), (4000, ADIM * H)), jnp.float32)
    gumbel = jax.random.gumbel(jax.random.PRNGKey(0), (2, 4000, 10, L, NC))
    d = acarb.ar_sample_codes(agent._percopy_critic(False), obs, act, gumbel, 0.7)
    p = np.asarray(jax.nn.softmax(agent._percopy_critic(False)(obs[:1], act[:1], jnp.zeros((2, 1, L), jnp.int32))
                                  [:, 0, 0] / 0.7, -1))
    for c in range(2):
        freq = np.bincount(np.asarray(d[c, ..., 0]).ravel(), minlength=NC) / 40000
        assert np.abs(freq - p[c]).max() < 5 * np.sqrt(p[c].max() / 40000) + 1e-3


# ---- config, diagnostics, updates, probes -------------------------------------------------------------
@pytest.mark.parametrize('variant', ['E1', 'E2', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11'])
def test_allowed_at_l5(variant):
    cfg = cfg_of(variant)
    T.check_agent_config('bc', cfg)  # no exit
    agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)


@pytest.mark.parametrize('over', [dict(arb_length=4), dict(arb_length=6), dict(arb_base=8)])
def test_redundant_other_lengths_refused(over):
    cfg = cfg_of('E5', **over)
    with pytest.raises(AssertionError):
        agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)
    with pytest.raises(SystemExit) as e:
        T.check_agent_config('bc', cfg)
    assert e.value.code == 2


def test_int32_bound():
    assert acarb.int32_errors(cfg_of('E1')) == [] and acarb.int32_errors(cfg_of('E5')) == []
    assert acarb.int32_errors(cfg_of('E1', length=7)) == []         # k_den 2e8: bound 4.4e8
    assert acarb.int32_errors(cfg_of('E1', length=8)) != []         # k_den 2e9
    with pytest.raises(AssertionError):
        agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg_of('E1', length=8))


@pytest.mark.parametrize('variant', ['E1', 'E2', 'E6', 'E11'])
def test_diagnostics_and_updates_l5(variant):
    agent = make(variant)
    d = agent.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    for k in ('critic/q_readout', 'critic/greedy_q', 'critic/greedy_oor_frac', 'critic/digit1_maxprob',
              'critic/bon_tie_frac', 'critic/bon_greedy_spread'):
        assert np.isfinite(float(d[k])), k
    assert 'critic/expected_q' not in d  # no 10^4 / 54^4 prefix enumeration at L = 5
    if variant in ORDINARY:
        assert float(d['critic/greedy_oor_frac']) == 0.0
    new, info = T.critic_update(agent, make_batch())
    assert T.opt_counts(new) == {'actor': 0, 'critic': 1}
    keys = [f'critic/{p}{l}' for l in range(1, L + 1) for p in ('ce_digit', 'tf_acc_digit', 'greedy_acc_digit')]
    keys += [f'critic/label_clip_pos{l}' for l in range(1, L + 1)] if variant in ORDINARY else \
        ['critic/oov_label', 'critic/oov_rollin', 'critic/model_used_frac', 'critic/label_mean_bias']
    for k in keys + ['critic/critic_loss', 'critic/target_clip_frac', 'critic/readout_oor_frac']:
        assert np.isfinite(float(info[k])), k
    new, info = T.joint_update(agent, make_batch(seed=1), make_batch(seed=2))
    assert T.opt_counts(new) == {'actor': 1, 'critic': 1}


def test_sample_actions_first_index_ties_l5(monkeypatch):
    agent = make('E11')
    N = agent.config['actor_num_samples']
    cands = jnp.asarray(np.arange(3 * N * ADIM * H, dtype=np.float32).reshape(3, N, ADIM * H))
    S = np.zeros((3, N), np.int32)
    S[0, [4, 9]] = 1999980
    S[1, 31] = 1999979
    S[2] = 5
    monkeypatch.setattr(ACARBAgent, '_candidates', lambda self, o, rng: (jnp.repeat(o[:, None], N, 1), cands))
    monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (jnp.asarray(S), None))
    out = ACARBAgent.sample_actions.__wrapped__(agent, jnp.zeros((3, OBS)), rng=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cands[jnp.arange(3), jnp.asarray([4, 31, 0])]))


@pytest.mark.parametrize('variant', ['E2', 'E9'])
def test_probes_l5_integer_scores(variant):
    agent = make(variant)
    rs = np.random.RandomState(0)
    obs = rs.randn(24, OBS).astype(np.float32)
    cands = rs.uniform(-1, 1, (24, 6, ADIM * H)).astype(np.float32)
    score, value, oor = critic_probe.fixed_scores(agent, obs, cands, jax.random.PRNGKey(0))
    score, value = np.asarray(score), np.asarray(value)
    assert score.dtype == np.int32 and score.min() >= 0 and score.max() <= 20 * (10 ** L - 1)
    np.testing.assert_allclose(value, score / K_DEN, rtol=1e-6)
    out = critic_probe.probe_fixed(agent, obs, cands)
    for k, v in out.items():
        assert np.isfinite(v), k
    assert out['probe_fixed/score_mean'] == pytest.approx(float(value.mean()), rel=1e-6)
    assert (out['probe_fixed/tie_all'], out['probe_fixed/tie_max']) == critic_probe.tie_stats(score)
    # Ties are exact integer equality: two scores 1 apart (1 / 2e6 in value) are not tied.
    assert critic_probe.tie_stats(np.array([[1999979, 1999980]], np.int32)) == (0.0, 0.0)
