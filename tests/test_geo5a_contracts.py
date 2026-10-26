"""geo5a value contracts, on hand-built batches and on real env steps.

Reward per primitive step r = r_env - 1 (sparse: -1, 0 on the success step); success -> terminated, step limit
(square 400, toolhang 1000) -> truncated; replay mask = 1 - terminated. BSC targets: 1 at success, 0 at a timeout;
QC targets (acarb arb_value='qc' and acfql): a timeout bootstraps (mask 1), only success ends the sum.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import os
import sys
import types
from collections import defaultdict

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from agents import agents, v1_config  # noqa: E402
from utils import diag_v1  # noqa: E402

H, OBS, ADIM = 5, 9, 7
GAMMA = 0.99


def _flags():
    import train_v1
    if not train_v1.FLAGS.is_parsed():
        train_v1.FLAGS(['test', '--env_name=square-mh-low_dim', f'--horizon_length={H}'])
    return train_v1


# ---- env_step: reward transform, terminated / truncated, masks ------------------------------------------------------
class _StubEnv:
    def __init__(self, outs):
        self.outs = list(outs)

    def step(self, action):
        r_env, term, trunc = self.outs.pop(0)
        return np.zeros(OBS), r_env, term, trunc, {}


class _StubBuf:
    def __init__(self):
        self.rows = []

    def add_transition(self, row):
        self.rows.append(row)
        return len(self.rows) - 1


@pytest.mark.parametrize('r_env, term, trunc, r, mask, terminal', [
    (0.0, False, False, -1.0, 1.0, 0.0),   # ordinary step
    (1.0, True, False, 0.0, 0.0, 1.0),     # success step: terminated, reward 0, mask 0
    (0.0, False, True, -1.0, 1.0, 1.0),    # step limit: truncated, mask stays 1 (bootstrap)
])
def test_env_step_row(r_env, term, trunc, r, mask, terminal):
    tv = _flags()
    buf = _StubBuf()
    ended = []
    self = types.SimpleNamespace(action_queue=[np.zeros(ADIM)], env=_StubEnv([(r_env, term, trunc)]),
                                 buffers={'main': buf}, ep_rows=[], ctr=defaultdict(int), phase='warmup',
                                 max_ep_len=400, H=H, ob=np.zeros(OBS),
                                 end_episode=lambda success, info: ended.append(success))
    tv.Trainer.env_step(self, 'n1')
    row = buf.rows[0]
    assert (row['rewards'], row['masks'], row['terminals']) == (r, mask, terminal)
    assert diag_v1.primitive_reward(r_env) == r
    assert ended == ([term] if term or trunc else [])


def test_process_demos_rewards():
    tv = _flags()
    ds = dict(observations=np.zeros((6, OBS)), actions=np.zeros((6, ADIM)), rewards=np.array([0, 0, 1, 0, 0, 1.0]),
              terminals=np.array([0, 0, 1, 0, 0, 1.0]), masks=np.array([1, 1, 0, 1, 1, 0.0]),
              next_observations=np.zeros((6, OBS)))
    d = tv.process_demos(ds, 'square-mh-low_dim', H, True)
    assert d['rewards'].tolist() == [-1, -1, 0, -1, -1, 0]
    assert d['masks'].tolist() == [1, 1, 0, 1, 1, 0]


# ---- real env steps ------------------------------------------------------------------------------------------------
def _make(name, seed=0):
    from envs import robomimic_utils
    from envs.env_utils import EpisodeMonitor
    return EpisodeMonitor(robomimic_utils.make_env(name, seed=seed))


@pytest.mark.parametrize('name, limit', [('square-mh-low_dim', 400), ('tool_hang-ph-low_dim', 1000)])
def test_real_timeout(name, limit):
    env = _make(name)
    env.reset()
    rng = np.random.default_rng(0)
    outs = []
    for t in range(limit):
        _, r, te, tr, info = env.step(rng.uniform(-1, 1, ADIM) * 0.3)
        outs.append((r, te, tr))
        if te:
            pytest.skip(f'random actions succeeded at step {t + 1}')
    r, te, tr = map(np.array, zip(*outs))
    assert not te.any() and tr[-1] and not tr[:-1].any() and (r == 0).all()
    acc_b, acc_q = diag_v1.ReturnAccumulator('bsc'), diag_v1.ReturnAccumulator('qc', t_extra=0)
    for x in outs:
        acc_b.add(*x)
        acc_q.add(*x)
    assert acc_b.done and acc_b.result() == dict(G=0.0, success=0, steps=limit)
    q = acc_q.result()
    assert q['steps'] == limit and abs(q['G_cut'] + (1 - GAMMA ** limit) / (1 - GAMMA)) < 1e-9
    print(name, 'timeout at', limit, 'G_cut', q['G_cut'], 'tail bound', q['tail_bound'])


def _demo(name, i):
    import h5py
    from envs.robomimic_utils import _check_dataset_exists
    with h5py.File(_check_dataset_exists(name), 'r') as f:
        return np.array(f[f'data/demo_{i}/states']), np.array(f[f'data/demo_{i}/actions'])


def test_real_success():
    """Open-loop replay of a square demo from its initial state: success -> terminated with r_env 1 (r = 0)."""
    name = 'square-mh-low_dim'
    env = _make(name)
    for i in range(10):
        states, actions = _demo(name, i)
        env.reset()
        diag_v1.env_layers(env)[2].reset_to({'states': states[0]})
        outs = []
        for a in actions:
            _, r, te, tr, info = env.step(a)
            outs.append((r, te, tr))
            if te or tr:
                break
        if outs[-1][1]:
            break
    else:
        pytest.skip('no demo replay succeeded')
    r, te, tr = map(np.array, zip(*outs))
    assert te[-1] and not te[:-1].any() and not tr.any() and r[-1] == 1.0 and (r[:-1] == 0).all()
    assert [diag_v1.primitive_reward(x) for x in r[-2:]] == [-1.0, 0.0]
    acc_b, acc_q = diag_v1.ReturnAccumulator('bsc'), diag_v1.ReturnAccumulator('qc', t_extra=10)
    for x in outs:
        acc_b.add(*x)
        acc_q.add(*x)
    assert acc_b.result()['G'] == 1.0 and acc_q.result()['tail_bound'] == 0.0
    ref = diag_v1.qc_reference(r, te, tr, 10)
    assert abs(acc_q.result()['G_ext'] - ref['G_ext']) < 1e-9 and acc_q.result()['G_cut'] == acc_q.result()['G_ext']
    print('demo', i, 'success at step', len(outs), 'G_qc', ref['G_ext'])


# ---- agents' TD targets on a hand-built batch ------------------------------------------------------------------------
def _batch():
    """Rows: 0 success at the chunk's last step, 1 timeout at the last step, 2 non-terminal. Same next state."""
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(3, OBS)).astype(np.float32)
    nxt = np.repeat(rng.normal(size=(1, H, OBS)).astype(np.float32), 3, 0)
    r = -np.ones((3, H))
    r[0, -1] = 0.0
    disc = GAMMA ** np.arange(H)
    terminals = np.zeros((3, H))
    terminals[:2, -1] = 1
    masks = np.ones((3, H))
    masks[0, -1] = 0
    return dict(observations=obs, actions=rng.uniform(-1, 1, (3, H, ADIM)).astype(np.float32),
                rewards=np.cumsum(r * disc, 1).astype(np.float32), masks=masks.astype(np.float32),
                terminals=terminals.astype(np.float32), valid=np.ones((3, H), np.float32),
                next_observations=nxt, num_chunks_to_success=np.array([1, -1, -1], np.float32))


def _agent(method, **over):
    cfg = v1_config(method)
    cfg.update(over)
    cfg['horizon_length'] = H
    return agents[cfg['agent_name']].create(0, np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32), cfg)


ARB = dict(critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10, arb_readout_temp=0.7)


def test_arb_bsc_targets():
    a = _agent('arb_bsc', **ARB)
    b = _batch()
    y, u, info = a.td_target(b, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    y, k = np.asarray(y), np.asarray(info['target_k'])
    k_den = a.readout_den()
    assert y[0] == 1.0 and y[1] == 0.0 and 0 <= y[2] <= 1
    assert k[0] == k_den - k_den // 1000 and k[1] == 0  # success -> top code (1 clipped), timeout -> 0
    print('arb_bsc targets', y.tolist(), 'k', k.tolist())


def test_arb_qc_targets_bootstrap_at_timeout():
    a = _agent('arb_qc', **ARB)
    b = _batch()
    # One row at a time with the same keys: the next-action noise and the readout's Gumbel noise are per row.
    y = np.array([float(a.td_target({k: v[i:i + 1] for k, v in b.items()}, jax.random.PRNGKey(1),
                                    jax.random.PRNGKey(2))[0][0]) for i in range(3)], np.float32)
    R = b['rewards'][:, -1]
    assert y[0] == R[0]                       # success: R only (mask 0)
    assert y[1] == y[2] and y[1] != R[1]       # timeout bootstraps exactly like a non-terminal chunk
    assert -100.0 * GAMMA ** H - 1e-3 <= y[1] - R[1] <= 0.0  # gamma^h Qbar(s', a'), Qbar in [-100, 0]
    print('arb_qc targets', y.tolist(), 'R', R.tolist())


def _fql_expected(a, b, rng):
    rng, sample_rng = jax.random.split(rng)
    nxt = b['next_observations'][:, -1]
    next_q = np.asarray(a.network.select('target_critic')(nxt, actions=a.sample_actions(nxt, rng=sample_rng))).mean(0)
    q = np.asarray(a.network.select('critic')(b['observations'], actions=b['actions'].reshape(3, -1)))
    return q, next_q


def test_acfql_targets_bootstrap_at_timeout():
    a = _agent('qc')
    b = _batch()
    rng = jax.random.PRNGKey(3)
    loss, _ = a.critic_loss(b, a.network.params, rng)
    q, next_q = _fql_expected(a, b, rng)
    R, m = b['rewards'][:, -1], b['masks'][:, -1]
    y = R + GAMMA ** H * m * next_q
    assert m.tolist() == [0, 1, 1]
    assert abs(float(loss) - float(np.square(q - y[None]).mean())) < 1e-4 * max(1.0, float(loss))
    y_cut = R + GAMMA ** H * np.array([0, 0, 1]) * next_q   # what a timeout that did NOT bootstrap would give
    assert abs(float(loss) - float(np.square(q - y_cut[None]).mean())) > 1e-3
    print('acfql targets', y.tolist())


def test_acbsc_targets():
    a = _agent('bsc')
    b = _batch()
    rng = jax.random.PRNGKey(3)
    _, info = a.critic_loss(b, a.network.params, rng)
    rng, sample_rng = jax.random.split(rng)
    nxt = b['next_observations'][:, -1]
    next_q = np.asarray(jax.nn.sigmoid(a.network.select('target_critic')(
        nxt, actions=a.sample_actions(nxt, rng=sample_rng)))).mean(0)
    logit = a.network.select('critic')(b['observations'], actions=b['actions'].reshape(3, -1))
    for y, ok in (([1.0, 0.0, next_q[2]], True), ([1.0, next_q[1], next_q[2]], False)):
        bce = float(optax.sigmoid_binary_cross_entropy(logit, jnp.asarray(y, jnp.float32)[None]).mean())
        assert (abs(bce - float(info['bce_loss'])) < 1e-5) == ok, (y, bce, float(info['bce_loss']))
    print('acbsc targets [1, 0, next_q]', [1.0, 0.0, float(next_q[2])])
