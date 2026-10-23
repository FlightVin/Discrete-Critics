"""geo1: ARB-BSC sample_mean readout, TD1 target, label modes, gradients and update isolation (tiny agents).

Run from the project root.
"""
import itertools
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
from agents import agents, v1_config  # noqa: E402
from agents.acarb import ACARBAgent  # noqa: E402

H, OBS, ADIM = 5, 6, 3
MODES = ['hard', 'twohot_last', 'twohot_last2', 'twohot_all']


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def e_agent(label='twohot_all', readout='sample_mean', M=10, temp=0.7, seed=0, split=True, **over):
    cfg = v1_config('arb_bsc')
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    cfg.arb_d_model = 32
    cfg.flow_steps = 2
    cfg.horizon_length = H
    cfg.batch_size = 16
    cfg.critic_target_bon = False
    cfg.arb_readout = readout
    cfg.arb_readout_samples = M
    cfg.arb_readout_temp = temp
    cfg.arb_label = label
    cfg.update(over)
    agent = agents['acarb'].create(seed, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)
    return T.split_optimizer(agent) if split else agent


def make_batch(n=16, seed=0, invalid=()):
    rs = np.random.RandomState(seed)
    terminals = (rs.rand(n, H) < 0.25).astype(np.float32)
    valid = np.ones((n, H), np.float32)
    for i in invalid:
        valid[i, -1] = 0.0
    return dict(observations=rs.randn(n, OBS).astype(np.float32),
                actions=rs.uniform(-1, 1, (n, H, ADIM)).astype(np.float32),
                next_observations=rs.randn(n, H, OBS).astype(np.float32),
                terminals=np.maximum.accumulate(terminals, axis=1), masks=(rs.rand(n, H) < 0.5).astype(np.float32),
                valid=valid, rewards=np.zeros((n, H), np.float32),
                num_chunks_to_success=rs.randn(n).astype(np.float32))


def leaves(tree):
    return jax.tree_util.tree_leaves_with_path(jax.device_get(tree))


def assert_tree_equal(a, b):
    la, lb = leaves(a), leaves(b)
    assert [p for p, _ in la] == [p for p, _ in lb]
    for (p, x), (_, y) in zip(la, lb):
        assert np.array_equal(np.asarray(x), np.asarray(y)), jax.tree_util.keystr(p)


# ---- readout ---------------------------------------------------------------------------------------
def exact_value(agent, obs, act, temp):
    """Exact E[S / K_den] under temperature-`temp` sampling: enumerate all 1000 codes per copy (incl. clip)."""
    codes = jnp.asarray(list(itertools.product(range(10), repeat=3)), jnp.int32)  # (1000, 3)
    logits = agent.network.select('critic')(obs[None], act[None], codes[None])[:, 0]  # (num_qs, 1000, 3, 10)
    logp = jax.nn.log_softmax(logits / temp, axis=-1)
    lp = jnp.take_along_axis(logp, codes[None, :, :, None], axis=-1)[..., 0].sum(-1)  # (num_qs, 1000)
    n = jnp.clip(codes[:, 0] * 100 + codes[:, 1] * 10 + codes[:, 2], 0, 999)
    p = jnp.exp(lp)
    np.testing.assert_allclose(np.asarray(p.sum(-1)), 1.0, atol=1e-4)
    return float((p * n).sum(-1).mean() / 1000), np.asarray((p * n).sum(-1) / 1000)


def test_sample_mean_matches_exact_enumeration():
    agent = e_agent(seed=3)
    rs = np.random.RandomState(0)
    obs, act = jnp.asarray(rs.randn(OBS).astype(np.float32)), jnp.asarray(rs.uniform(-1, 1, ADIM * H).astype(np.float32))
    exact, _ = exact_value(agent, obs, act, 0.7)
    R = 4000  # independent readouts, one per batch element
    S, oor = agent.critic_score(jnp.broadcast_to(obs, (R, OBS)), jnp.broadcast_to(act, (R, ADIM * H)),
                                rng=jax.random.PRNGKey(1))
    S = np.asarray(S)
    assert S.dtype == np.int32 and float(np.asarray(oor).max()) == 0.0
    v = S / 20000.0
    se = v.std() / np.sqrt(R)
    print(f'MC {v.mean():.5f} exact {exact:.5f} se {se:.2e}')
    assert abs(v.mean() - exact) < 5 * se + 1e-6, (v.mean(), exact, se)
    # The value is S / (1000 num_qs M), and readout_q agrees with it.
    q = agent.readout_q(obs[None], act[None], jax.random.PRNGKey(2))
    s2, _ = agent.critic_score(obs[None], act[None], rng=jax.random.PRNGKey(2))
    # (XLA may divide via the reciprocal: 1 ulp; ranking and labels use the integer S.)
    assert float(q[0]) == pytest.approx(int(s2[0]) / 20000, rel=1e-6)


def test_sample_counts_and_independent_copy_noise():
    agent = e_agent(seed=4, temp=5.0)  # high temperature: near-uniform digits
    # Both copies with identical parameters: any difference between their codes comes from the noise.
    p = dict(agent.network.params)
    p['modules_critic'] = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[:1], x.shape), p['modules_critic'])
    agent = agent.replace(network=agent.network.replace(params=p))
    rs = np.random.RandomState(0)
    obs, act = jnp.asarray(rs.randn(8, OBS).astype(np.float32)), jnp.asarray(rs.uniform(-1, 1, (8, ADIM * H)).astype(np.float32))
    n, oor = agent._sampled_codes(obs, act, False, jax.random.PRNGKey(0))
    assert n.shape == (2, 8, 10)  # num_qs x batch x M draws
    n = np.asarray(n)
    assert not np.array_equal(n[0], n[1])                     # copies: different noise
    assert len(np.unique(n.reshape(2, -1), axis=1).T) > 1     # samples within a copy differ
    assert all(len(np.unique(n[c, :, m])) > 1 for c in range(2) for m in range(10))  # batch elements differ
    S, _ = agent.critic_score(obs, act, rng=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(S), n.sum(axis=(0, 2)))


def test_temperature_zero_limit_is_greedy():
    """Very low temperature: every sample is the greedy code (sanity check of the AR sampling loop)."""
    agent = e_agent(seed=5, temp=1e-4)
    rs = np.random.RandomState(1)
    obs, act = jnp.asarray(rs.randn(8, OBS).astype(np.float32)), jnp.asarray(rs.uniform(-1, 1, (8, ADIM * H)).astype(np.float32))
    n, _ = agent._sampled_codes(obs, act, False, jax.random.PRNGKey(0))
    greedy = agent.network.select('critic')(obs, act)  # (num_qs, 8, 3)
    g = np.asarray(greedy[..., 0] * 100 + greedy[..., 1] * 10 + greedy[..., 2])
    np.testing.assert_array_equal(np.asarray(n), np.broadcast_to(g[..., None], n.shape))


def test_td1_draws_one_actor_chunk_per_element(monkeypatch):
    agent = e_agent()
    calls = []
    orig = ACARBAgent.compute_flow_actions.__wrapped__

    def rec(self, observations, noises):
        calls.append((observations.shape, noises.shape))
        return orig(self, observations, noises)

    monkeypatch.setattr(ACARBAgent, 'compute_flow_actions', rec)
    batch = make_batch()
    y, u, info = agent.td_target(batch, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert calls == [((16, OBS), (16, ADIM * H))], calls
    k = np.asarray(info['target_k'])
    term = batch['terminals'][:, -1] > 0.5
    succ = term & (batch['masks'][:, -1] < 0.5)
    assert (k[succ] == 19980).all() and (k[term & ~succ] == 0).all()
    np.testing.assert_allclose(np.asarray(u)[~term], np.minimum(k[~term], 19980) / 20000, rtol=1e-6)
    assert 0.0 <= float(info['readout_oor_frac']) <= 0.0


def test_sample_actions_ranks_by_integer_score_first_index(monkeypatch):
    agent = e_agent()
    N = agent.config['actor_num_samples']
    cands = jnp.asarray(np.arange(3 * N * ADIM * H, dtype=np.float32).reshape(3, N, ADIM * H))
    S = np.zeros((3, N), np.int32)
    S[0, [4, 9]] = 100   # tie at the max: first index (4)
    S[1, 31] = 7
    S[2] = 5             # all equal: index 0
    monkeypatch.setattr(ACARBAgent, '_candidates', lambda self, o, rng: (jnp.repeat(o[:, None], N, 1), cands))
    monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (jnp.asarray(S), None))
    out = ACARBAgent.sample_actions.__wrapped__(agent, jnp.zeros((3, OBS)), rng=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cands[jnp.arange(3), jnp.asarray([4, 31, 0])]))


# ---- critic network and loss ------------------------------------------------------------------------
def test_causal_alignment():
    agent = e_agent()
    rs = np.random.RandomState(0)
    obs, act = jnp.asarray(rs.randn(4, OBS).astype(np.float32)), jnp.asarray(rs.uniform(-1, 1, (4, ADIM * H)).astype(np.float32))
    digits = jnp.asarray(rs.randint(0, 10, (4, 3)))
    base = np.asarray(agent.network.select('critic')(obs, act, digits))
    for l in range(3):
        changed = digits.at[:, l:].set((digits[:, l:] + 1 + jnp.arange(3 - l)) % 10)
        new = np.asarray(agent.network.select('critic')(obs, act, changed))
        np.testing.assert_allclose(new[..., :l + 1, :], base[..., :l + 1, :], atol=1e-6, rtol=0)
        if l < 2:
            assert not np.allclose(new[..., l + 1:, :], base[..., l + 1:, :])


@pytest.mark.parametrize('mode', MODES)
def test_loss_gradients_stop_at_target_and_labels(mode):
    agent = e_agent(label=mode)
    batch = make_batch()
    params = agent.network.params

    g = jax.grad(lambda p: agent.critic_loss(batch, p, jax.random.PRNGKey(3))[0])(params)
    for k in g:
        mx = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g[k]))
        if k == 'modules_critic':
            assert mx > 0, k
        else:  # target critic and actor: no gradient
            assert mx == 0.0, (k, mx)

    # Through the labels: the loss as a function of the target critic's parameters (that build the labels).
    def via_target(tp):
        p = dict(params)
        p['modules_target_critic'] = tp
        a = agent.replace(network=agent.network.replace(params=p))
        return a.critic_loss(batch, params, jax.random.PRNGKey(3))[0]

    gt = jax.grad(via_target)(params['modules_target_critic'])
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(gt))


@pytest.mark.parametrize('mode', MODES)
def test_invalid_rows_contribute_zero(mode):
    agent = e_agent(label=mode)
    b1 = make_batch(invalid=(2, 5, 11))
    b2 = {k: np.array(v) for k, v in b1.items()}
    rs = np.random.RandomState(9)
    for i in (2, 5, 11):
        b2['observations'][i] = rs.randn(OBS)
        b2['actions'][i] = rs.uniform(-1, 1, (H, ADIM))
    l1, i1 = agent.critic_loss(b1, agent.network.params, jax.random.PRNGKey(3))
    l2, _ = agent.critic_loss(b2, agent.network.params, jax.random.PRNGKey(3))
    assert float(l1) == pytest.approx(float(l2), abs=1e-7)
    assert np.isfinite(float(l1))
    for key in ('target_clip_frac', 'target_u_mean', 'readout_oor_frac', 'label_clip_pos1', 'label_clip_pos2',
                'label_clip_pos3'):
        assert key in i1 and np.isfinite(float(i1[key])), key
    assert 'target_k' not in i1


def test_critic_only_update_keeps_actor_and_its_adam_state():
    agent = e_agent()
    batch = make_batch()
    new, info = T.critic_update(agent, batch)
    a0, a1 = jax.device_get(agent.network), jax.device_get(new.network)
    for k in a0.params:
        same = all(np.array_equal(x, y) for x, y in zip(jax.tree_util.tree_leaves(a0.params[k]),
                                                       jax.tree_util.tree_leaves(a1.params[k])))
        assert same == ('critic' not in k), k
    assert_tree_equal(a0.opt_state.inner_states['actor'], a1.opt_state.inner_states['actor'])
    assert T.opt_counts(new) == {'actor': 0, 'critic': 1}
    assert np.isfinite(float(info['critic/critic_loss']))


def test_joint_update_critic_loss_has_no_actor_gradient():
    agent = e_agent()
    mb, sb = make_batch(seed=1), make_batch(seed=2)
    _, actor_rng, critic_rng = T.joint_rngs(agent)
    g = jax.grad(lambda p: agent.critic_loss(mb, p, critic_rng)[0])(agent.network.params)
    for k in g:
        if 'critic' not in k:
            assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(g[k])), k
    ga = jax.grad(lambda p: agent.actor_loss(sb, p, actor_rng)[0])(agent.network.params)
    for k in ga:
        if 'critic' in k:
            assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(ga[k])), k
    new, info = T.joint_update(agent, mb, sb)
    assert T.opt_counts(new) == {'actor': 1, 'critic': 1}
    assert np.isfinite(float(info['critic/critic_loss'])) and np.isfinite(float(info['actor/actor_loss']))


def test_readout_diagnostics_has_q_readout():
    agent = e_agent()
    diag = agent.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    assert np.isfinite(float(diag['critic/q_readout'])) and np.isfinite(float(diag['critic/expected_q']))
    g = e_agent(readout='greedy', label='hard')
    d = g.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    assert float(d['critic/q_readout']) == pytest.approx(float(np.average(
        np.asarray(g.critic_value(make_batch()['observations'],
                                  make_batch()['actions'].reshape(16, -1), readout='greedy')).mean(0))), abs=1e-6)


def test_config_refusals():
    with pytest.raises(AssertionError):
        e_agent(q_agg='min')
    with pytest.raises(AssertionError):
        e_agent(label='twohot_all', readout='greedy')
    with pytest.raises(AssertionError):
        e_agent(label='twohot_last', arb_value='qc', discount=0.99)
    e_agent(label='hard', readout='sample_mean')
    e_agent(label='hard', readout='greedy')
