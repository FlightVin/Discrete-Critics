"""geo2b: the redundant-token vocabulary and roll-ins (E5-E9) inside the ARB-BSC critic (tiny agents).

Init hashes, the executed loss path against geo_codes.build called directly, per-copy teacher forcing, the E6
model roll-in, the redundant sample_mean readout against exact enumeration of all 54^3 sequences (with the
final [0, 999] clip), and config refusals. Run from the project root.
"""
import hashlib
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
from agents import agents, geo_codes, v1_config  # noqa: E402
from agents.acarb import ACARBAgent, GEO_COMBOS, soft_cross_entropy  # noqa: E402

H, OBS, ADIM = 5, 6, 3
NC = geo_codes.NUM_CLASSES
COMBOS = {v: k for k, v in GEO_COMBOS.items()}  # 'E5' -> (label, rollin)
TOKENS = np.arange(geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX + 1)


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not T.FLAGS.is_parsed():
        T.FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def g_agent(variant='E5', seed=0, split=True, temp=0.7, **over):
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
    if variant.startswith('label:'):  # ordinary vocabulary with a geo1 label mode
        cfg.arb_label = variant[len('label:'):]
    else:
        cfg.arb_vocab = 'redundant'
        cfg.arb_label, cfg.arb_rollin = COMBOS[variant]
    cfg.update(over)
    agent = agents['acarb'].create(seed, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)
    return T.split_optimizer(agent) if split else agent


def with_other_target(agent, seed=7):
    """Target critic with different parameters (so using the wrong network shows)."""
    other = g_agent(agent_variant(agent), seed=seed, split=False)
    p = dict(agent.network.params)
    p['modules_target_critic'] = other.network.params['modules_critic']
    return agent.replace(network=agent.network.replace(params=p))


def sharpen(agent, factor):
    """Current critic with its output head kernel scaled by `factor` (sharper token distributions)."""
    p = dict(agent.network.params)
    crit = dict(p['modules_critic'])
    crit['head'] = dict(crit['head'], kernel=crit['head']['kernel'] * factor)
    p['modules_critic'] = crit
    return agent.replace(network=agent.network.replace(params=p))


def agent_variant(agent):
    c = agent.config
    if c['arb_vocab'] == 'ordinary':
        return 'label:' + c['arb_label']
    return GEO_COMBOS[(c['arb_label'], c['arb_rollin'])]


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


def leaf_hashes(tree):
    return {jax.tree_util.keystr(p): hashlib.sha256(np.asarray(x).tobytes() + str(np.asarray(x).shape).encode())
            .hexdigest() for p, x in jax.tree_util.tree_leaves_with_path(jax.device_get(tree))}


# ---- initialisation ---------------------------------------------------------------------------------
def test_init_hashes():
    red = {v: leaf_hashes(g_agent(v, split=False).network.params) for v in ('E5', 'E6', 'E7', 'E8', 'E9')}
    for v in ('E6', 'E7', 'E8', 'E9'):
        assert red[v] == red['E5'], v
    ordinary = leaf_hashes(g_agent('label:hard', split=False).network.params)
    assert set(ordinary) == set(red['E5'])
    token = [p for p in ordinary if 'digit_embed' in p or "['head']" in p]
    assert len(token) == 2 * 3, token  # critic + target: embedding, head kernel, head bias
    for p in ordinary:
        if p in token:
            assert ordinary[p] != red['E5'][p], p
        else:
            assert ordinary[p] == red['E5'][p], p
    shapes = {jax.tree_util.keystr(p): x.shape
              for p, x in jax.tree_util.tree_leaves_with_path(g_agent('E5', split=False).network.params)}
    emb = [s for p, s in shapes.items() if 'digit_embed' in p]
    head = [s for p, s in shapes.items() if "['head']" in p and 'kernel' in p]
    assert emb == [(2, NC, 32)] * 2 and head == [(2, 32, NC)] * 2, (emb, head)


# ---- loss path --------------------------------------------------------------------------------------
class Recorder:
    """Wraps geo_codes.build (as called from acarb) and keeps its concrete inputs / outputs."""

    def __init__(self, monkeypatch):
        self.calls = []
        orig = geo_codes.build

        def build(k, label_mode, rollin_mode, key, model_token=None, k_den=geo_codes.K_DEN):
            out = orig(k, label_mode, rollin_mode, key, model_token, k_den)
            self.calls.append(dict(k=k, label_mode=label_mode, rollin_mode=rollin_mode, key=key, k_den=k_den,
                                   model_token=model_token, out=out))
            return out

        monkeypatch.setattr(geo_codes, 'build', build)


def loss_keys(rng):
    _, sample_rng, readout_rng, rollin_rng = jax.random.split(rng, 4)
    return sample_rng, readout_rng, rollin_rng


def direct_loss(agent, batch, params, labels, rollin):
    """The protocol's loss from given labels / roll-ins: per-copy teacher forcing on own roll-in, soft CE."""
    act = batch['actions'].reshape(batch['actions'].shape[0], -1)
    cls = geo_codes.token_to_class(rollin)
    logits = agent._percopy_critic(False, params=params['modules_critic'])(batch['observations'], act, cls)
    ce = soft_cross_entropy(logits, labels)
    w = batch['valid'][..., -1]
    return (ce.mean(axis=(0, 2)) * w).mean()


@pytest.mark.parametrize('variant', ['E5', 'E6', 'E7', 'E8', 'E9'])
def test_loss_uses_build_directly(variant, monkeypatch):
    agent = with_other_target(g_agent(variant))
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
    assert (call['label_mode'], call['rollin_mode'], call['k_den']) == (*COMBOS[variant], 20000)
    labels, rollin, binfo = call['out']
    if variant != 'E6':  # E6's callback is checked in test_e6_model_tokens
        l2, r2, _ = geo_codes.build(jnp.asarray(k), *COMBOS[variant], rollin_rng)
        np.testing.assert_array_equal(np.asarray(labels), np.asarray(l2))
        np.testing.assert_array_equal(np.asarray(rollin), np.asarray(r2))
    # Different copies get different roll-in paths (independent per element and copy).
    assert not np.array_equal(np.asarray(rollin[0, :, :2]), np.asarray(rollin[1, :, :2]))
    ref = direct_loss(agent, batch, agent.network.params, labels, rollin)
    assert float(loss) == pytest.approx(float(ref), rel=1e-6, abs=1e-7)
    assert float(info['oov_label']) == 0 and float(info['oov_rollin']) == 0
    # label_mean_bias: E5-E7 exact, E8 -w3 eps, E9 -0.0005 (float32 rounding only).
    bias = float(info['label_mean_bias'])
    if variant in ('E5', 'E6', 'E7'):
        assert abs(bias) < 2e-7, bias
    elif variant == 'E9':
        assert bias == pytest.approx(-0.0005, abs=2e-7)
    else:
        f = np.asarray(binfo['rem'][..., 2]) / np.asarray(binfo['den'][..., 2])
        w = np.broadcast_to(batch['valid'][:, -1], f.shape)
        assert bias == pytest.approx(-(f * w).sum() / w.sum() / 1000, abs=2e-7)
    # Jitted (as in the update): same loss.
    jl, _ = jax.jit(agent.critic_loss)(batch, agent.network.params, rng)
    assert float(jl) == pytest.approx(float(loss), rel=1e-5)


@pytest.mark.parametrize('variant', ['E5', 'E6', 'E9'])
def test_gradients_only_to_critic_not_through_labels_or_rollins(variant, monkeypatch):
    agent = with_other_target(g_agent(variant))
    batch = make_batch()
    params = agent.network.params
    rng = jax.random.PRNGKey(3)
    rec = Recorder(monkeypatch)
    agent.critic_loss(batch, params, rng)  # eager: the labels / roll-ins of this key
    labels, rollin = jax.device_get(rec.calls[0]['out'][:2])
    g = jax.grad(lambda p: agent.critic_loss(batch, p, rng)[0])(params)
    for k in g:
        mx = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(g[k]))
        assert (mx > 0) == (k == 'modules_critic'), (k, mx)
    # Same gradient as the loss with the labels / roll-ins held fixed: nothing flows through them (E6: the
    # model tokens come from the current params, stop-gradient).
    gd = jax.grad(lambda p: direct_loss(agent, batch, p, labels, rollin))(params)
    for (pa, x), (_, y) in zip(jax.tree_util.tree_leaves_with_path(g['modules_critic']),
                               jax.tree_util.tree_leaves_with_path(gd['modules_critic'])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-4, atol=1e-7,
                                   err_msg=jax.tree_util.keystr(pa))
    # Through the target critic (which builds k): zero.
    def via_target(tp):
        p = dict(params)
        p['modules_target_critic'] = tp
        a = agent.replace(network=agent.network.replace(params=p))
        return a.critic_loss(batch, params, rng)[0]

    gt = jax.grad(via_target)(params['modules_target_critic'])
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(gt))


def test_each_copy_reads_its_own_rollin():
    agent = g_agent('E7')
    rs = np.random.RandomState(0)
    obs, act = jnp.asarray(rs.randn(4, OBS), jnp.float32), jnp.asarray(rs.uniform(-1, 1, (4, ADIM * H)), jnp.float32)
    cls = jnp.asarray(rs.randint(0, NC, (2, 4, 3)))
    f = agent._percopy_critic(False)
    base = np.asarray(f(obs, act, cls))
    changed = cls.at[1, :, :2].set((cls[1, :, :2] + 7) % NC)
    new = np.asarray(f(obs, act, changed))
    np.testing.assert_array_equal(new[0], base[0])  # copy 0 unchanged
    assert not np.allclose(new[1, :, 1:], base[1, :, 1:])
    np.testing.assert_allclose(new[1, :, 0], base[1, :, 0], atol=1e-6, rtol=0)
    # The per-copy critic is the same function as the ensemble critic when both copies read the same prefix.
    shared = jnp.broadcast_to(cls[0], cls.shape)
    np.testing.assert_allclose(np.asarray(f(obs, act, shared)),
                               np.asarray(agent.network.select('critic')(obs, act, cls[0])), atol=1e-6, rtol=0)


def test_causal_alignment_redundant():
    agent = g_agent('E5')
    rs = np.random.RandomState(0)
    obs, act = jnp.asarray(rs.randn(4, OBS), jnp.float32), jnp.asarray(rs.uniform(-1, 1, (4, ADIM * H)), jnp.float32)
    cls = jnp.asarray(rs.randint(0, NC, (4, 3)))
    base = np.asarray(agent.network.select('critic')(obs, act, cls))
    assert base.shape == (2, 4, 3, NC)
    for l in range(3):
        changed = cls.at[:, l:].set((cls[:, l:] + 1 + jnp.arange(3 - l)) % NC)
        new = np.asarray(agent.network.select('critic')(obs, act, changed))
        np.testing.assert_allclose(new[..., :l + 1, :], base[..., :l + 1, :], atol=1e-6, rtol=0)
        if l < 2:
            assert not np.allclose(new[..., l + 1:, :], base[..., l + 1:, :])


@pytest.mark.parametrize('variant', ['E5', 'E6', 'E8'])
def test_invalid_rows_contribute_zero(variant):
    agent = g_agent(variant)
    b1 = make_batch(invalid=(2, 5, 11))
    b2 = {k: np.array(v) for k, v in b1.items()}
    rs = np.random.RandomState(9)
    for i in (2, 5, 11):
        b2['observations'][i] = rs.randn(OBS)
        b2['actions'][i] = rs.uniform(-1, 1, (H, ADIM))
    l1, i1 = agent.critic_loss(b1, agent.network.params, jax.random.PRNGKey(3))
    l2, _ = agent.critic_loss(b2, agent.network.params, jax.random.PRNGKey(3))
    assert np.isfinite(float(l1))
    # E6: invalid rows' observations change the model tokens of those rows only; their weight is 0.
    assert float(l1) == pytest.approx(float(l2), abs=1e-7)
    for key in ('oov_label', 'oov_rollin', 'model_used_frac', 'model_clamp_frac', 'label_mean_bias',
                'target_clip_frac', 'target_u_mean', 'readout_oor_frac', 'tf_acc_digit1', 'greedy_mae'):
        assert key in i1 and np.isfinite(float(i1[key])), key
    assert 'target_k' not in i1


# ---- E6 model roll-in --------------------------------------------------------------------------------
def test_e6_model_tokens(monkeypatch):
    # Sharpen the current critic (head x 10) so a wrong temperature would show.
    agent = sharpen(with_other_target(g_agent('E6', temp=0.7)), 10)
    batch = make_batch(n=256, seed=4)
    rng = jax.random.PRNGKey(11)
    rec = Recorder(monkeypatch)
    _, info = agent.critic_loss(batch, agent.network.params, rng)
    labels, rollin, binfo = rec.calls[0]['out']
    rollin, mask, c = (np.asarray(x) for x in (rollin, binfo['model_mask'], binfo['c']))
    _, _, rollin_rng = loss_keys(rng)
    model_rng = jax.random.split(rollin_rng, 4)[3]
    for i in range(3):  # disjoint from build's own keys
        assert not np.array_equal(np.asarray(model_rng), np.asarray(jax.random.split(rollin_rng, 3)[i]))
    act = batch['actions'].reshape(256, -1)
    f = agent._percopy_critic(False)  # CURRENT critic (target differs)
    clamps = 0
    for l in range(2):
        prefix = rollin.copy()
        prefix[..., l:] = 0
        logits = np.asarray(f(batch['observations'], act, geo_codes.token_to_class(jnp.asarray(prefix))))[..., l, :]
        g = np.asarray(jax.random.gumbel(jax.random.fold_in(model_rng, l), logits.shape))
        m = np.argmax(logits / 0.7 + g, -1) + geo_codes.TOKEN_MIN
        expect = np.clip(m, c[..., l] - 2, c[..., l] + 2)
        np.testing.assert_array_equal(rollin[..., l][mask[..., l]], expect[mask[..., l]])
        # Temperature matters: at T=1 the samples differ somewhere.
        m1 = np.argmax(logits + g, -1) + geo_codes.TOKEN_MIN
        assert (m1 != m)[mask[..., l]].any()
        # Clamp count: model tokens outside [c-2, c+2].
        assert ((rollin[..., l] - c[..., l]) >= -2).all() and ((rollin[..., l] - c[..., l]) <= 2).all()
        clamps += int((np.abs(m - c[..., l]) > 2)[mask[..., l]].sum())
    assert int(binfo['model_clamp']) == clamps and clamps > 0
    used = mask.mean()
    assert 0.44 < used < 0.56, used
    assert float(info['model_used_frac']) == pytest.approx(used, abs=1e-6)
    assert float(info['model_clamp_frac']) == pytest.approx(clamps / mask.sum(), abs=1e-6)


# ---- readout (redundant) ----------------------------------------------------------------------------
def exact_redundant(agent, obs, act, temp):
    """Exact E[clip(n) / 1000] per copy over all 54^3 sequences, the oor mass, and the unclipped mean."""
    f = agent._percopy_critic(False)
    z = jnp.arange(NC)
    p1 = jax.nn.softmax(f(obs[None], act[None], jnp.zeros((2, 1, 3), jnp.int32))[:, 0, 0] / temp, -1)  # (2, 54)
    pre1 = jnp.stack([z, jnp.zeros_like(z), jnp.zeros_like(z)], -1)  # (54, 3)
    o54, a54 = jnp.broadcast_to(obs, (NC, OBS)), jnp.broadcast_to(act, (NC, act.shape[-1]))
    p2 = jax.nn.softmax(f(o54, a54, jnp.broadcast_to(pre1, (2, NC, 3)))[:, :, 1] / temp, -1)  # (2, 54, 54)
    z1, z2 = jnp.meshgrid(z, z, indexing='ij')
    pre2 = jnp.stack([z1.ravel(), z2.ravel(), jnp.zeros(NC * NC, jnp.int32)], -1)  # (2916, 3)
    o, a = jnp.broadcast_to(obs, (NC * NC, OBS)), jnp.broadcast_to(act, (NC * NC, act.shape[-1]))
    p3 = jax.nn.softmax(f(o, a, jnp.broadcast_to(pre2, (2, NC * NC, 3)))[:, :, 2] / temp, -1).reshape(2, NC, NC, NC)
    p = np.asarray(p1[:, :, None, None] * p2[:, :, :, None] * p3, np.float64)  # (2, 54, 54, 54)
    np.testing.assert_allclose(p.sum((1, 2, 3)), 1.0, atol=1e-4)
    t = TOKENS
    n = 100 * t[:, None, None] + 10 * t[None, :, None] + t[None, None, :]
    v = (p * np.clip(n, 0, 999)).sum((1, 2, 3)) / 1000
    oor = (p * ((n < 0) | (n > 999))).sum((1, 2, 3))
    raw = (p * n).sum((1, 2, 3)) / 1000
    return v, oor, raw


def test_redundant_sample_mean_matches_exact_enumeration():
    agent = g_agent('E5', seed=3)
    rs = np.random.RandomState(0)
    obs = jnp.asarray(rs.randn(OBS).astype(np.float32))
    act = jnp.asarray(rs.uniform(-1, 1, ADIM * H).astype(np.float32))
    v, oor, raw = exact_redundant(agent, obs, act, 0.7)
    exact = float(v.mean())
    print(f'exact {exact:.5f} unclipped {raw.mean():.5f} oor {oor}')
    assert oor.min() > 0.2 and abs(raw.mean() - exact) > 0.05  # the clip is active and matters
    R = 4000
    S, o = agent.critic_score(jnp.broadcast_to(obs, (R, OBS)), jnp.broadcast_to(act, (R, ADIM * H)),
                              rng=jax.random.PRNGKey(1))
    S, o = np.asarray(S), np.asarray(o)
    assert S.dtype == np.int32 and S.min() >= 0 and S.max() <= 20 * 999
    val = S / 20000.0
    se = val.std() / np.sqrt(R)
    print(f'MC {val.mean():.5f} exact {exact:.5f} se {se:.2e}; oor MC {o.mean():.4f} exact {oor.mean():.4f}')
    assert abs(val.mean() - exact) < 5 * se, (val.mean(), exact, se)
    se_o = o.std() / np.sqrt(R)
    assert abs(o.mean() - oor.mean()) < 5 * se_o + 1e-3, (o.mean(), oor.mean())


def test_signed_decoding_and_clip():
    agent = g_agent('E5')
    cls = geo_codes.token_to_class(jnp.asarray([[-1, 5, 3], [10, -3, -22], [9, 9, 31], [0, 0, -1], [3, 12, -7]]))
    np.testing.assert_array_equal(np.asarray(agent._decode(cls)), [-47, 948, 1021, -1, 413])
    # Near-zero temperature: every sample is the greedy code; n and oor agree with the decoded greedy tokens.
    a = sharpen(g_agent('E5', seed=5, temp=1e-4), 30)  # sharp: no near-ties between the top logits
    rs = np.random.RandomState(1)
    obs, act = jnp.asarray(rs.randn(64, OBS), jnp.float32), jnp.asarray(rs.uniform(-1, 1, (64, ADIM * H)), jnp.float32)
    n, oor = a._sampled_codes(obs, act, False, jax.random.PRNGKey(0))
    g = np.asarray(a._decode(a.network.select('critic')(obs, act)))  # (2, 64)
    assert ((g < 0) | (g > 999)).any() and ((g >= 0) & (g <= 999)).any()
    np.testing.assert_array_equal(np.asarray(n), np.broadcast_to(np.clip(g, 0, 999)[..., None], n.shape))
    np.testing.assert_array_equal(np.asarray(oor), np.broadcast_to(((g < 0) | (g > 999))[..., None], n.shape))
    q = np.asarray(a.critic_value(obs, act, readout='greedy'))
    np.testing.assert_allclose(q, np.clip(g, 0, 999) / 1000, rtol=1e-6)


def test_sample_actions_first_index_ties_redundant(monkeypatch):
    agent = g_agent('E7')
    N = agent.config['actor_num_samples']
    cands = jnp.asarray(np.arange(3 * N * ADIM * H, dtype=np.float32).reshape(3, N, ADIM * H))
    S = np.zeros((3, N), np.int32)
    S[0, [4, 9]] = 100
    S[1, 31] = 7
    S[2] = 5
    monkeypatch.setattr(ACARBAgent, '_candidates', lambda self, o, rng: (jnp.repeat(o[:, None], N, 1), cands))
    monkeypatch.setattr(ACARBAgent, 'critic_score', lambda self, o, a, target=False, rng=None: (jnp.asarray(S), None))
    out = ACARBAgent.sample_actions.__wrapped__(agent, jnp.zeros((3, OBS)), rng=jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cands[jnp.arange(3), jnp.asarray([4, 31, 0])]))


def test_readout_diagnostics_and_update_redundant():
    agent = g_agent('E6')
    d = agent.readout_diagnostics(make_batch(), jax.random.PRNGKey(0))
    for k in ('critic/q_readout', 'critic/greedy_q', 'critic/greedy_oor_frac', 'critic/digit1_maxprob',
              'critic/bon_tie_frac', 'critic/bon_greedy_spread'):
        assert np.isfinite(float(d[k])), k
    assert 'critic/expected_q' not in d
    new, info = T.critic_update(agent, make_batch())
    assert T.opt_counts(new) == {'actor': 0, 'critic': 1}
    assert np.isfinite(float(info['critic/critic_loss']))
    new, info = T.joint_update(agent, make_batch(seed=1), make_batch(seed=2))
    assert T.opt_counts(new) == {'actor': 1, 'critic': 1}


# ---- config refusal ---------------------------------------------------------------------------------
BAD = [
    dict(arb_vocab='redundant', arb_label='corrective', arb_rollin='canonical'),
    dict(arb_vocab='redundant', arb_label='noisy_fixed', arb_rollin='corrective'),
    dict(arb_vocab='redundant', arb_label='noisy_resid', arb_rollin='corrective'),
    dict(arb_vocab='redundant', arb_label='hard', arb_rollin='canonical'),
    dict(arb_vocab='redundant', arb_label='twohot_all', arb_rollin='fixed'),
    dict(arb_vocab='ordinary', arb_label='corrective', arb_rollin='corrective'),
    dict(arb_vocab='ordinary', arb_label='hard', arb_rollin='fixed'),
    dict(arb_vocab='redundant', arb_label='corrective', arb_rollin='corrective', arb_readout='greedy'),
    dict(arb_vocab='redundant', arb_label='corrective', arb_rollin='fixed', arb_value='qc', discount=0.99),
    dict(arb_vocab='bogus'),
]


def raw_cfg(**over):
    cfg = v1_config('arb_bsc')
    cfg.horizon_length = H
    cfg.arb_readout = 'sample_mean'
    cfg.arb_readout_samples = 10
    cfg.update(over)
    return cfg


@pytest.mark.parametrize('over', BAD)
def test_refused_combinations(over):
    with pytest.raises(AssertionError):
        agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), raw_cfg(**over))
    with pytest.raises(SystemExit) as e:
        T.check_agent_config('bc', raw_cfg(**over))
    assert e.value.code == 2


@pytest.mark.parametrize('variant', ['E5', 'E6', 'E7', 'E8', 'E9'])
def test_allowed_combinations(variant):
    label, rollin = COMBOS[variant]
    cfg = raw_cfg(arb_vocab='redundant', arb_label=label, arb_rollin=rollin)
    T.check_agent_config('bc', cfg)  # no exit
    agents['acarb'].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)
