"""Executed-loss-path checks for the six v1 critics.

Every expected value here is rebuilt from the protocol, not from the agent's code: 32 BC-flow next chunks,
ranked by the CURRENT critic (mean of the 2 copies), the chosen chunk valued by the TARGET critic (mean of
the 2 copies), then the method's backup. The target critic is given different parameters from the critic,
so ranking by the wrong network, picking the wrong copy aggregation, or skipping best-of-n changes the result.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from agents import agents, v1_config, V1_METHODS
from agents.acarb import ar_decode
from utils.datasets import Dataset

METHODS = list(V1_METHODS)
OBS, ADIM, H, BS, N = 5, 2, 4, 8, 32
GAMMA_QC = 0.99
RAW_ATOMS = jnp.linspace(-100.0, 0.0, 101)
HL_EDGES = jnp.linspace(-100.0, 0.0, 102, dtype=jnp.float32)
HL_CENTERS = (HL_EDGES[:-1] + HL_EDGES[1:]) / 2
HL_SIGMA = 0.75 * 100.0 / 101
QC_FAMILY = ('qc', 'arb_qc', 'cat', 'hlg')


def make_agent(method, seed=0):
    name, _ = V1_METHODS[method]
    config = v1_config(method)
    # Small networks so the test runs on CPU; protocol settings (N, copies, agg, tau, gamma) untouched.
    config.update(dict(value_hidden_dims=(32, 32), actor_hidden_dims=(32, 32), horizon_length=H))
    if name == 'acarb':
        config.update(dict(arb_d_model=16, arb_num_layers=1, arb_num_heads=2, arb_diag_obs=4))
    ex_o, ex_a = np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32)
    agent = agents[name].create(seed, ex_o, ex_a, config)
    other = agents[name].create(seed + 1, ex_o, ex_a, config)
    params = dict(agent.network.params)
    params['modules_target_critic'] = other.network.params['modules_critic']
    return agent.replace(network=agent.network.replace(params=params))


def make_batch(seed=0):
    """Rows: 0-3 non-terminal, 4 success at the last step, 5 timeout at the last step,
    6 terminal inside the chunk (later steps invalid), 7 non-terminal."""
    rs = np.random.RandomState(seed)
    r = -np.ones((BS, H))
    r[4, -1] = 0.0
    terminals = np.zeros((BS, H)); terminals[4, -1] = terminals[5, -1] = 1.0; terminals[6, 1:] = 1.0
    masks = np.ones((BS, H)); masks[4, -1] = 0.0; masks[6, 1:] = 0.0
    valid = np.ones((BS, H)); valid[6, 2:] = 0.0
    rewards = np.cumsum(r * GAMMA_QC ** np.arange(H), axis=1)
    rewards[0] += rs.uniform(-30, 0)  # vary R so the backup is not identical across rows
    nts = np.full(BS, -1.0); nts[4] = 0.0
    return dict(
        observations=jnp.asarray(rs.normal(size=(BS, OBS)), jnp.float32),
        next_observations=jnp.asarray(rs.normal(size=(BS, H, OBS)), jnp.float32),
        actions=jnp.asarray(rs.uniform(-1, 1, (BS, H, ADIM)), jnp.float32),
        rewards=jnp.asarray(rewards, jnp.float32), masks=jnp.asarray(masks, jnp.float32),
        terminals=jnp.asarray(terminals, jnp.float32), valid=jnp.asarray(valid, jnp.float32),
        num_chunks_to_success=jnp.asarray(nts, jnp.float32),
    )


def values(agent, method, obs, acts, which):
    """Per-copy values (2, ...) in the units the TD backup uses."""
    net = agent.network.select(which)
    if method == 'qc':
        return net(obs, acts)
    if method == 'bsc':
        return jax.nn.sigmoid(net(obs, acts))
    if method in ('arb_bsc', 'arb_qc'):
        u = ar_decode(net(obs, acts), 10)  # greedy readout
        return u if method == 'arb_bsc' else 100.0 * u - 100.0
    probs = jax.nn.softmax(net(obs, acts), axis=-1)
    return (probs * (RAW_ATOMS if method == 'cat' else HL_CENTERS)).sum(-1)


def candidate_key(method, rng):
    """The key the agent's critic_loss hands to its best-of-n sampler for the candidate noises."""
    if V1_METHODS[method][0] == 'acarb':
        _, sample_rng, _, _ = jax.random.split(rng, 4)
        cand_rng, _ = jax.random.split(sample_rng)  # sample_actions splits off a readout key
        return cand_rng
    _, sample_rng = jax.random.split(rng)
    return sample_rng


def manual_target(agent, method, batch, rng, rank='critic', agg='mean', n=N):
    next_obs = batch['next_observations'][:, -1, :]
    noises = jax.random.normal(candidate_key(method, rng), (BS, N, ADIM * H))[:, :n]
    obs_rep = jnp.repeat(next_obs[:, None, :], n, axis=1)
    cands = jnp.clip(agent.compute_flow_actions(obs_rep, noises), -1, 1)  # (BS, n, A)
    a = getattr(jnp, agg)
    idx = jnp.argmax(a(values(agent, method, obs_rep, cands, rank), axis=0), axis=-1)
    chosen = cands[jnp.arange(BS), idx]
    qbar = a(values(agent, method, next_obs, chosen, 'target_critic'), axis=0)

    if method in QC_FAMILY:
        y = batch['rewards'][:, -1] + GAMMA_QC ** H * batch['masks'][:, -1] * qbar
    else:  # BSC backup, gamma = 1: success -> 1, timeout -> 0, otherwise qbar
        term = batch['terminals'][:, -1] > 0.5
        succ = term & (batch['masks'][:, -1] < 0.5)
        y = jnp.where(succ, 1.0, jnp.where(term, 0.0, qbar))
    return y, idx


def manual_loss(agent, method, batch, y):
    obs, acts = batch['observations'], batch['actions'].reshape(BS, -1)
    w = batch['valid'][:, -1]
    out = agent.network.select('critic')(obs, acts) if method not in ('arb_bsc', 'arb_qc') else None
    if method == 'qc':
        return (jnp.square(out - y) * w).mean()
    if method == 'bsc':
        return (optax.sigmoid_binary_cross_entropy(out, y[None]) * w[None]).mean()
    if method in ('cat', 'hlg'):
        if method == 'cat':
            yc = jnp.clip(y, -100.0, 0.0)
            lo = jnp.clip(jnp.floor(yc + 100.0), 0, 99).astype(jnp.int32)
            w_hi = yc + 100.0 - lo
            tgt = jax.nn.one_hot(lo, 101) * (1 - w_hi)[:, None] + jax.nn.one_hot(lo + 1, 101) * w_hi[:, None]
        else:
            yc = jnp.clip(y, -100.0, 0.0)
            cdf = jax.scipy.special.erf((HL_EDGES[None] - yc[:, None]) / (jnp.sqrt(2) * HL_SIGMA))
            tgt = (cdf[:, 1:] - cdf[:, :-1]) / (cdf[:, -1:] - cdf[:, :1])
        ce = -(tgt[None] * jax.nn.log_softmax(out, axis=-1)).sum(-1)
        return (ce.mean(0) * w).mean()
    # ARB: teacher-forced CE on the 3 base-10 digits of the clipped digit fraction.
    u = y if method == 'arb_bsc' else (y + 100.0) / 100.0
    code = np.floor(np.clip(np.asarray(u, np.float64), 0, 0.999) * 1000 + 1e-6).astype(np.int32)
    labels = jnp.asarray(np.stack([code // 100, (code // 10) % 10, code % 10], -1))
    logits = agent.network.select('critic')(obs, acts, labels)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels[None])
    return (ce.mean(axis=(0, 2)) * w).mean()


@pytest.fixture(scope='module', params=METHODS)
def setup(request):
    method = request.param
    return method, make_agent(method), make_batch(), jax.random.PRNGKey(7)


def test_v1_config(setup):
    method, agent, _, _ = setup
    c = agent.config
    assert c['actor_type'] == 'best-of-n' and c['actor_num_samples'] == 32
    assert c['num_qs'] == 2 and c['q_agg'] == 'mean' and c['tau'] == 0.005
    if 'critic_target_bon' in c:
        assert c['critic_target_bon']
    assert c['discount'] == (GAMMA_QC if method in QC_FAMILY else 1.0)
    out = agent.network.select('critic')(jnp.zeros((3, OBS)), jnp.zeros((3, ADIM * H)))
    assert out.shape[0] == 2  # two critic copies


def test_td_target_is_best_of_32(setup):
    """critic_loss on the executed path equals the loss built from a manual best-of-32 target."""
    method, agent, batch, rng = setup
    loss, info = agent.critic_loss(batch, agent.network.params, rng)
    y, idx = manual_target(agent, method, batch, rng)
    want = manual_loss(agent, method, batch, y)
    print(f'{method}: critic_loss {float(loss):.6f} manual best-of-32 {float(want):.6f}; picks {np.asarray(idx)}')
    np.testing.assert_allclose(float(loss), float(want), rtol=1e-5)

    if V1_METHODS[method][0] == 'acarb':
        B_ = jax.random.split(rng, 4)
        y_agent, _, _ = agent.td_target(batch, B_[1], B_[2])
        np.testing.assert_allclose(np.asarray(y_agent), np.asarray(y), atol=1e-5)
        key = 'target_q_raw' if method == 'arb_qc' else 'target_q'
        want_mean = y.mean() if method == 'arb_qc' else jnp.clip(y, 0, 0.999).mean()
        np.testing.assert_allclose(float(info[key]), float(want_mean), atol=1e-6)
    elif 'target_q_mean' in info:
        np.testing.assert_allclose(float(info['target_q_mean']), float(y.mean()), rtol=1e-5)
    elif 'target_q' in info:
        np.testing.assert_allclose(float(info['target_q']), float(y.mean()), rtol=1e-5)

    # Non-vacuous: wrong ranking network / aggregation / no best-of-n give different targets.
    for alt in (dict(rank='target_critic'), dict(agg='min'), dict(n=1)):
        y_alt, idx_alt = manual_target(agent, method, batch, rng, **alt)
        assert float(jnp.abs(y_alt - y).max()) > 1e-6, f'{method}: target insensitive to {alt}'
    assert len(set(np.asarray(idx).tolist())) > 1


def test_valid_weights(setup):
    """Rows with valid[-1] = 0 do not affect the loss; valid rows do."""
    method, agent, batch, rng = setup
    base, _ = agent.critic_loss(batch, agent.network.params, rng)
    for row, should_change in ((6, False), (0, True)):
        b = dict(batch)
        b['observations'] = batch['observations'].at[row].add(3.0)
        b['actions'] = batch['actions'].at[row].multiply(-1.0)
        b['rewards'] = batch['rewards'].at[row].add(-5.0)
        loss, _ = agent.critic_loss(b, agent.network.params, rng)
        assert (abs(float(loss) - float(base)) > 1e-7) == should_change, (method, row)


def test_target_soft_update_uses_pre_step_params(setup):
    method, agent, batch, _ = setup
    new, _ = agent.update(batch)
    p0, p1 = agent.network.params, new.network.params
    tau = 0.005
    want = jax.tree_util.tree_map(lambda c, t: tau * c + (1 - tau) * t,
                                  p0['modules_critic'], p0['modules_target_critic'])
    wrong = jax.tree_util.tree_map(lambda c, t: tau * c + (1 - tau) * t,
                                   p1['modules_critic'], p0['modules_target_critic'])
    err = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()), want, p1['modules_target_critic'])
    assert max(jax.tree_util.tree_leaves(err)) < 1e-7
    gap = jax.tree_util.tree_map(lambda a, b: float(jnp.abs(a - b).max()), wrong, p1['modules_target_critic'])
    assert max(jax.tree_util.tree_leaves(gap)) > 0  # the critic did move, so pre- vs post-step is tested


def test_loss_gradients_finite(setup):
    method, agent, batch, rng = setup
    b = dict(batch)
    b['rewards'] = batch['rewards'].at[1].set(-1e4).at[2].set(50.0)  # far outside every support
    g = jax.grad(lambda p: agent.critic_loss(b, p, rng)[0])(agent.network.params)
    assert all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(g))


# --- dataset-side R, M, D ----------------------------------------------------------------------------

def test_sample_sequence_chunk_reward_and_masks():
    T = 12
    r = -np.ones(T, np.float32); r[7] = 0.0                      # success at step 7 (shifted reward 0)
    term = np.zeros(T, np.float32); term[7] = 1.0; term[-1] = 1.0  # episode 1 ends at 7 (success), 2 at 11 (timeout)
    m = np.ones(T, np.float32); m[7] = 0.0                       # timeout keeps mask 1
    rs = np.random.RandomState(0)
    ds = Dataset.create(observations=rs.normal(size=(T, OBS)).astype(np.float32),
                        next_observations=rs.normal(size=(T, OBS)).astype(np.float32),
                        actions=rs.uniform(-1, 1, (T, ADIM)).astype(np.float32),
                        rewards=r, masks=m, terminals=term)
    np.random.seed(0)
    b = ds.sample_sequence(64, H, GAMMA_QC)
    starts = np.array([np.where((ds['observations'] == o).all(-1))[0][0] for o in b['observations']])
    for s, R, M, D, V in zip(starts, b['rewards'], b['masks'], b['terminals'], b['valid']):
        np.testing.assert_allclose(R[-1], sum(GAMMA_QC ** i * r[s + i] for i in range(H)), rtol=1e-6)
        assert M[-1] == m[s:s + H].min() and D[-1] == term[s:s + H].max()
        np.testing.assert_array_equal(V[1:], 1 - np.maximum.accumulate(term[s:s + H])[:-1])
    assert (starts + H - 1 >= 7).any() and (starts <= 7).any()  # some chunks contain the success


# --- ARB-QC: raw-unit bootstrap with a known target-critic value ---------------------------------------

def test_arb_qc_raw_unit_bootstrap():
    agent = make_agent('arb_qc')
    # Pin the target critic's greedy read-out to digits (7, 7, 7) -> u = 0.777 -> q_raw = -22.3 for every (s, a).
    params = jax.tree_util.tree_map(lambda x: x, dict(agent.network.params))
    tc = jax.tree_util.tree_map(lambda x: x, params['modules_target_critic'])
    head = tc['VmapARCritic_0']['head'] if 'VmapARCritic_0' in tc else tc['head']
    head['kernel'] = jnp.zeros_like(head['kernel'])
    head['bias'] = jnp.zeros_like(head['bias']).at[..., 7].set(50.0)
    params['modules_target_critic'] = tc
    agent = agent.replace(network=agent.network.replace(params=params))

    batch = make_batch()
    R = jnp.array([-5.0, -30.0, -0.05, -150.0, 0.0, -3.94, -2.0, -1.0], jnp.float32)
    batch['rewards'] = batch['rewards'].at[:, -1].set(R)
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    y, u, info = agent.td_target(batch, keys[1], keys[2])
    qbar = 100 * 0.777 - 100  # -22.3
    M = batch['masks'][:, -1]
    expected = R + GAMMA_QC ** H * M * qbar
    print(f'ARB-QC targets {np.round(np.asarray(y), 4)}; expected {np.round(np.asarray(expected), 4)}')
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected), atol=1e-4)
    assert abs(float(y[4]) - 0.0) < 1e-6        # success row: M = 0, no bootstrap
    assert abs(float(y[5]) - (-3.94 + GAMMA_QC ** H * qbar)) < 1e-4  # timeout row bootstraps
    np.testing.assert_allclose(np.asarray(u), np.clip((np.asarray(expected) + 100) / 100, 0, 0.999), atol=1e-6)
    # clip rates: row 3 below the support (u < 0); row 4 at q = 0 (u = 1 > 0.999).
    np.testing.assert_allclose(float(info['clip_low_frac']), 1 / BS)
    np.testing.assert_allclose(float(info['clip_high_frac']), 1 / BS)


def test_arb_readout_diagnostics_separate():
    for method in ('arb_bsc', 'arb_qc'):
        agent, batch = make_agent(method), make_batch()
        _, info = agent.critic_loss(batch, agent.network.params, jax.random.PRNGKey(0))
        assert not any('expected' in k or k.startswith('bon_') for k in info)
        diag = agent.readout_diagnostics(batch, jax.random.PRNGKey(1))
        assert {'critic/expected_q', 'critic/bon_argmax_agree', 'critic/greedy_expected_gap'} <= set(diag)
        assert all(bool(jnp.isfinite(v)) for v in diag.values())
