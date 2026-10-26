"""geo5b: utils/arb_introspect on freshly initialised real-size critics (tools/geo_synth.make_agent and the ARB-QC
counterpart): token trace == production readout, forced prefixes (e = 0 == readout conditioned on the reference
prefix, unsupported flagged not clamped), correction ablation, label q == the agent's critic_ce labels for every
label mode, H / KL against numpy, CE = H + KL, label_fit smoke. Run from the project root.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))
os.chdir(REPO)

import geo_synth as G  # noqa: E402
from agents import agents, geo_codes, v1_config  # noqa: E402
from agents.acarb import QC_K_DEN, ar_sample_codes, soft_cross_entropy  # noqa: E402
from utils import arb_introspect as AI  # noqa: E402

VARIANTS = ['E1', 'E2', 'E5', 'E6', 'E7', 'E10', 'E11']
Q = 16


def qc_agent(variant):
    cfg = v1_config('arb_qc')
    cfg.horizon_length = G.H
    cfg.critic_target_bon = False
    cfg.update(G.VARIANTS[variant])
    return agents['acarb'].create(0, np.zeros(G.OBS_DIM, np.float32), np.zeros(G.ADIM, np.float32), cfg)


_cache = {}


def agent_of(name):
    if name not in _cache:
        _cache[name] = qc_agent(name[3:]) if name.startswith('qc-') else G.make_agent(name)
    return _cache[name]


def inputs(n=Q, seed=0):
    rs = np.random.RandomState(seed)
    return jnp.asarray(rs.randn(n, G.OBS_DIM).astype(np.float32)), jnp.asarray(rs.uniform(-1, 1, (n, G.H * G.ADIM))
                                                                              .astype(np.float32))


@pytest.mark.parametrize('name', VARIANTS + ['qc-E1', 'qc-E10'])
def test_trace_equals_production_readout(name):
    agent = agent_of(name)
    obs, act = inputs()
    for s in range(2):
        key = jax.random.PRNGKey(11 + s)
        S, oor = agent.critic_score(obs, act, rng=key)
        tr = AI.trace(agent, obs, act, key)
        assert np.array_equal(np.asarray(S), np.asarray(tr['S']))
        assert np.allclose(np.asarray(tr['value']), np.asarray(agent.readout_q(obs, act, key)), atol=0, rtol=0)
        n = np.asarray(tr['n'])
        assert np.allclose(((n < 0) | (n > 999)).mean(axis=(0, 2)), np.asarray(oor))
        nc, lo, hi = AI.vocab(agent)
        assert tr['probs'].shape == (2, Q, 10, 3, nc)
        assert np.allclose(np.asarray(tr['probs']).sum(-1), 1, atol=1e-5)
        t = np.asarray(tr['tokens'])
        assert t.min() >= lo and t.max() <= hi
        assert np.array_equal(np.asarray(tr['classes']) + lo, t)
        assert np.array_equal(np.asarray(agent._decode(tr['classes'])), n)
        pu = np.asarray(tr['prefix_units'])
        assert (pu[..., 0] == 0).all() and np.array_equal(pu[..., 2], 100 * t[..., 0] + 10 * t[..., 1])
        s = AI.trace_summary(agent, tr)
        assert s['true_oov_count'] == 0
        if not agent._redundant():
            assert s['noncanonical_token_any_frac'] == 0 and s['decoded_oor_frac'] == 0


def conditioned_readout(agent, obs, act, key, prefix_classes, p):
    """Independent path: the agent's own ar_sample_codes, with the critic reading prefix_classes at positions < p."""
    fn = agent._percopy_critic(False)
    nc = AI.vocab(agent)[0]
    M = agent.config['arb_readout_samples']
    gumbel = jax.random.gumbel(key, (2, obs.shape[0], M, 3, nc))
    pc = jnp.broadcast_to(prefix_classes[None, :, None, :p], (2, obs.shape[0], M, p))
    wrapped = lambda o, a, d: fn(o, a, d.at[..., :p].set(pc))  # noqa: E731
    d = ar_sample_codes(wrapped, obs, act, gumbel, agent.config['arb_readout_temp'])
    return np.asarray(d.at[..., :p].set(pc))


@pytest.mark.parametrize('name', ['E2', 'E5', 'E10', 'qc-E11'])
def test_forced_prefix(name):
    agent = agent_of(name)
    obs, act = inputs()
    _, lo, hi = AI.vocab(agent)
    rs = np.random.RandomState(3)
    u = rs.uniform(0, 0.999, Q)
    u[:3] = [0.0123, 0.9876, 0.5]  # c1 = 0 / 9: e = -1, -2 / +1, +2 unsupported in the ordinary vocabulary
    se = np.where(np.arange(Q) % 4 == 0, 0.02, 0.0)
    key = jax.random.PRNGKey(5)
    fr = AI.forced_recovery(agent, obs, act, key, u, se)
    ref, dec = AI.reference_digits(u, se)
    assert np.array_equal(ref[:3], [[0, 1, 2], [9, 8, 7], [5, 0, 0]])
    assert not dec[0].all() and dec[1].all() and dec[3].all()  # se = 0.02 crosses the second digit boundary
    for p in (1, 2):
        r0 = fr[f'p{p}_e+0']
        assert r0['supported'].all()
        expect = conditioned_readout(agent, obs, act, key, jnp.asarray(ref - lo, jnp.int32), p) + lo
        assert np.array_equal(r0['tokens'], expect)
        v = (np.clip(geo_codes.decode_units(expect), 0, 999) / 1000).mean(axis=(0, 2))
        assert np.allclose(r0['value'], v)
        assert np.allclose(r0['err_prefix'], np.floor(u * 1000 + 1e-9) / 1000 - u)
        for e in (-2, -1, 1, 2):
            r = fr[f'p{p}_e{e:+d}']
            forced = ref[:, p - 1] + e
            sup = (forced >= lo) & (forced <= hi)
            assert np.array_equal(r['supported'], sup)
            assert np.isnan(r['value'][~sup]).all() and np.isfinite(r['value'][sup]).all()
            assert (r['tokens'][..., p - 1][:, sup] == forced[sup][None, :, None]).all()  # never clamped
            assert np.array_equal(r['digit_ok'], sup & dec[:, :p + 1].all(-1))
            for t in AI.TOLS:
                assert ((r[f'recov_{t}'][sup] >= 0) & (r[f'recov_{t}'][sup] <= 1)).all()
            assert np.all(np.isfinite(r['corrective_mass'][sup]))
    if agent._redundant():
        assert fr['p1_e-2']['supported'][0] and fr['p1_e-2']['tokens'][0, 0, 0, 0] == -2
    else:
        assert not fr['p1_e-1']['supported'][0] and not fr['p1_e+1']['supported'][1]
    s = AI.forced_summary(agent, fr)
    assert set(s) == {f'p{p}_e{e:+d}' for p in (1, 2) for e in AI.OFFSETS}


@pytest.mark.parametrize('name', ['E1', 'E5', 'E10', 'qc-E10'])
def test_ablation(name):
    agent = agent_of(name)
    obs, act = inputs(64)
    key = jax.random.PRNGKey(9)
    ab = AI.ablation(agent, obs, act, key)
    S, _ = agent.critic_score(obs, act, rng=key)
    assert np.array_equal(ab['S'], np.asarray(S))  # normal readout unchanged
    t = ab['tokens_ablated']
    assert t.min() >= 0 and t.max() <= 9
    if not agent._redundant():
        assert np.array_equal(ab['S'], ab['S_ablated'])
    else:
        assert ((ab['tokens'] < 0) | (ab['tokens'] > 9)).any() and not np.array_equal(ab['S'], ab['S_ablated'])
    cands = jnp.asarray(np.random.RandomState(1).uniform(-1, 1, (4, 32, G.H * G.ADIM)).astype(np.float32))
    ca = AI.candidate_ablation(agent, obs[:4], cands, key)
    o = jnp.repeat(obs[:4, None], 32, axis=1)
    S2, _ = agent.critic_score(o, cands, rng=key)
    assert np.array_equal(ca['S'], np.asarray(S2)) and ca['argmax'].shape == (4,)
    assert np.array_equal(ca['argmax'], np.asarray(S2).argmax(-1))


LABEL_CASES = [('E1', 'hard'), ('E2', 'twohot_last'), ('E3', 'twohot_last2'), ('E4', 'twohot_all'),
               ('E5', 'corrective'), ('E6', 'corrective'), ('E7', 'corrective'), ('E8', 'noisy_fixed'),
               ('E9', 'noisy_resid'), ('E10', 'noisy_fixed'), ('E11', 'noisy_resid'),
               ('qc-E1', 'hard'), ('qc-E2', 'twohot_last'), ('qc-E10', 'noisy_fixed')]


def np_stats(q, logits):
    logits = np.asarray(logits, np.float64)
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    q = np.asarray(q, np.float64)
    h = -np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0).sum(-1)
    kl = np.where(q > 0, q * (np.log(np.where(q > 0, q, 1)) - logp), 0).sum(-1)
    return h, kl, -(q * logp).sum(-1)


@pytest.mark.parametrize('name, mode', LABEL_CASES)
def test_labels_equal_critic_ce(name, mode):
    agent = agent_of(name)
    assert agent.config['arb_label'] == mode
    obs, act = inputs(32)
    den = agent.label_den()
    rs = np.random.RandomState(4)
    k = rs.randint(0, den - den // 1000 + 1, 32)
    k[:4] = [0, den - den // 1000, den // 1000 * 517, den // 1000 * 517 + 1]
    k = jnp.asarray(k, jnp.int32)
    u = k.astype(jnp.float32) / den
    params = agent.network.params
    for s in range(2):
        rr = jax.random.PRNGKey(20 + s)
        ce, logits, labels, _ = agent.critic_ce(obs, act, u, k, params, rr, jnp.ones(32))
        q, prefix, my_logits = AI.method_labels(agent, obs, act, u, k, rr)
        assert np.array_equal(np.asarray(my_logits), np.asarray(logits))
        my_ce = soft_cross_entropy(my_logits, q)
        if mode == 'hard':
            assert np.allclose(np.asarray(my_ce), np.asarray(ce), atol=2e-5)
            assert np.array_equal(np.asarray(jnp.argmax(q, -1)), np.broadcast_to(np.asarray(labels), q.shape[:-1]))
        else:
            assert np.array_equal(np.asarray(my_ce), np.asarray(ce))
            # critic_ce's labels: argmax q (redundant), the canonical roll-in digits (ordinary two-hot)
            mine = jnp.argmax(q, -1) if agent._redundant() else prefix
            assert np.array_equal(np.asarray(mine), np.broadcast_to(np.asarray(labels), q.shape[:-1]))
        assert np.allclose(np.asarray(q).sum(-1), 1, atol=1e-6)
        # H, KL against numpy
        nc, lo, _ = AI.vocab(agent)
        qstar, r, oov = AI.corrective_twohot(jnp.broadcast_to(k, prefix.shape[:-1]), den, prefix + lo, lo, nc)
        st = AI.dist_stats(q, jax.nn.log_softmax(my_logits, -1), qstar, r, AI.class_tokens(agent).astype(jnp.float32))
        h, kl, ce_np = np_stats(q, my_logits)
        assert np.allclose(np.asarray(st['h']), h, atol=1e-5) and np.allclose(np.asarray(st['kl']), kl, atol=1e-4)
        assert np.allclose(np.asarray(st['ce']), ce_np, atol=1e-4)
        hs, kls, _ = np_stats(qstar, my_logits)
        assert np.allclose(np.asarray(st['kl_star']), kls, atol=1e-4)
        # q* is the exact corrective two-hot: mean token == residual where nothing left the vocabulary
        mt = (np.asarray(qstar) * np.asarray(AI.class_tokens(agent))).sum(-1)
        ok = np.asarray(oov) == 0
        assert np.allclose(mt[ok], np.asarray(r)[ok], atol=1e-5)
        if mode == 'corrective':
            assert np.allclose(np.asarray(q), np.asarray(qstar), atol=1e-6)  # E5-E7: method labels are q*


def test_ce_equals_h_plus_kl():
    rs = np.random.RandomState(0)
    for C in (10, 54):
        q = rs.dirichlet(np.ones(C) * 0.3, (64,)) * (rs.rand(64, C) > 0.5)
        q = q / np.maximum(q.sum(-1, keepdims=True), 1e-12)
        q[q.sum(-1) == 0, 0] = 1.0
        logits = rs.randn(64, C) * 3
        logp = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), -1)
        st = AI.dist_stats(jnp.asarray(q, jnp.float32), logp, jnp.asarray(q, jnp.float32), jnp.zeros(64),
                           jnp.arange(C, dtype=jnp.float32))
        assert np.allclose(np.asarray(st['ce']), np.asarray(st['h']) + np.asarray(st['kl']), atol=1e-5)
        h, kl, ce = np_stats(q, logits)
        assert np.allclose(np.asarray(st['ce']), ce, atol=1e-4) and np.allclose(h + kl, ce, atol=1e-9)
        assert (np.asarray(st['kl']) > -1e-5).all()
        assert np.allclose(np.asarray(st['kl']), np.asarray(st['kl_star']), atol=1e-6)


def fake_batch(n=24, seed=0, qc=False):
    rs = np.random.RandomState(seed)
    term = (rs.rand(n) < 0.3).astype(np.float32)
    succ = term * (rs.rand(n) < 0.5)
    terminals = np.zeros((n, G.H), np.float32)
    terminals[:, -1] = term
    masks = np.ones((n, G.H), np.float32)
    masks[:, -1] = 1 - succ
    rewards = np.zeros((n, G.H), np.float32) - (0 if not qc else rs.randint(0, 5, (n, 1)))
    return dict(observations=jnp.asarray(rs.randn(n, G.OBS_DIM).astype(np.float32)),
                actions=jnp.asarray(rs.uniform(-1, 1, (n, G.H, G.ADIM)).astype(np.float32)),
                next_observations=jnp.asarray(rs.randn(n, G.H, G.OBS_DIM).astype(np.float32)),
                rewards=jnp.asarray(rewards), masks=jnp.asarray(masks), terminals=jnp.asarray(terminals),
                valid=jnp.ones((n, G.H)), num_chunks_to_success=jnp.asarray(np.where(succ > 0, 1.0, -1.0)))


@pytest.mark.parametrize('name', ['E1', 'E2', 'E6', 'E10', 'qc-E1', 'qc-E11'])
def test_label_fit(name):
    agent = agent_of(name)
    batch = fake_batch(qc=name.startswith('qc'))
    key = jax.random.PRNGKey(2)
    clean = np.random.RandomState(1).uniform(0.1, 0.9, 24)
    if name.startswith('qc'):
        clean = clean * 100 - 100
    lf = AI.label_fit(agent, batch, key, AI.to_u(agent, clean))
    # The TD target is the agent's own: same key split as critic_loss
    _, sr, rr, _ = jax.random.split(key, 4)
    y, u, info = agent.td_target(batch, sr, rr)
    assert np.array_equal(lf['y'], np.asarray(y))
    assert np.allclose(lf['y_u'], np.asarray(info['target_k']) / agent.label_den())
    for pre in ('a', 'b'):
        for s in ('ce', 'h', 'kl', 'kl_star', 'tok_mean_err', 'clean_mass', 'qstar_oov_mass'):
            v = lf[f'{pre}_{s}']
            assert v.shape == (24, 3) and np.isfinite(v).all(), (pre, s)
        assert np.allclose(lf[f'{pre}_ce'], lf[f'{pre}_h'] + lf[f'{pre}_kl'], atol=1e-4)
        assert (lf[f'{pre}_kl'] > -1e-4).all() and (lf[f'{pre}_kl_star'] > -1e-4).all()
    assert ((lf['a_qstar_oov_mass'] >= 0) & (lf['a_qstar_oov_mass'] < 1)).all()
    S =AI.label_fit_summary(agent, lf)
    assert len(S['a_kl']) == 3 and 'readout_value_abs_err_vs_clean' in S and 'tf_value_abs_err_vs_y' in S
