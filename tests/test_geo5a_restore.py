"""geo5a: exact snapshot / restore of robomimic states (utils/diag_v1).

Per task: one episode of random actions run to the step limit (A), 24 random mid-episode snapshots (4 of them within
H steps of the limit); each snapshot restored into ANOTHER env instance with a different history (B) and the stored
chunk replayed: next observations, rewards, terminated / truncated must equal the original ones (max |diff| <= 1e-6
required; printed). Also: a successful demo replay (terminated reproduced), the vector-pool worker path, a long
replay (100 steps), and robomimic's own reset_to({'states'}) as the baseline (printed, not required).

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from utils import diag_v1  # noqa: E402

TASKS = [('square-mh-low_dim', 400), ('tool_hang-ph-low_dim', 1000)]
H, N_SNAP, ADIM = 8, 24, 7


def _make(name, seed=0):
    from envs import robomimic_utils
    from envs.env_utils import EpisodeMonitor
    return EpisodeMonitor(robomimic_utils.make_env(name, seed=seed))


def _rollout(env, limit, seed):
    """Smoothed random actions until done; per step (action, obs, r, terminated, truncated), and the snapshot taken
    before each step (snapshot_at[t]: state before step t)."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    a = np.zeros(ADIM)
    steps, snaps = [], {}
    pick = set(rng.choice(np.arange(1, limit - H), N_SNAP - 4, replace=False).tolist()) | \
        {limit - 1, limit - 3, limit - H, limit - H + 1}
    for t in range(limit):
        if t in pick:
            snaps[t] = diag_v1.snapshot(env, obs)
        a = np.clip(0.8 * a + 0.5 * rng.uniform(-1, 1, ADIM), -1, 1)
        obs, r, te, tr, _ = env.step(a)
        steps.append((a.copy(), obs.copy(), r, te, tr))
        if te or tr:
            break
    return steps, snaps


def _replay(env, snap, actions, restore=diag_v1.restore):
    restore(env, snap)
    return [env.step(a)[:4] for a in actions]


def _diff(ref, got):
    """(max |obs diff|, rewards equal, flags equal)."""
    d = max(float(np.abs(np.asarray(x[1]) - np.asarray(y[0])).max()) for x, y in zip(ref, got))
    return d, all(x[2] == y[1] for x, y in zip(ref, got)), all((x[3], x[4]) == (bool(y[2]), bool(y[3]))
                                                              for x, y in zip(ref, got))


@pytest.fixture(scope='module', params=TASKS, ids=[t[0].split('-')[0] for t in TASKS])
def task(request):
    name, limit = request.param
    A, B = _make(name, 0), _make(name, 1)
    B.reset()
    for _ in range(5):
        B.step(np.random.default_rng(9).uniform(-1, 1, ADIM))
    steps, snaps = _rollout(A, limit, 0)
    return name, limit, A, B, steps, snaps


def test_restore_replay_exact(task):
    name, limit, A, B, steps, snaps = task
    if len(steps) < limit:
        pytest.skip(f'random rollout ended at {len(steps)} < {limit}')
    assert len(snaps) >= 20
    worst, n_trunc = 0.0, 0
    for t, s in sorted(snaps.items()):
        ref = steps[t:t + H]
        got = _replay(B, s, [x[0] for x in ref])
        d, r_ok, f_ok = _diff(ref, got)
        assert r_ok and f_ok, (t, [x[2:] for x in ref], [y[1:] for y in got])
        worst = max(worst, d)
        n_trunc += any(x[4] for x in ref)
        assert s['t'] == t
    print(f'{name}: {len(snaps)} snapshots, replay of {H} steps into another env: max |obs diff| = {worst:.3g}; '
          f'{n_trunc} snapshots replay the truncation at t={limit}')
    assert worst <= 1e-6 and n_trunc >= 3


def test_restore_long_and_baseline(task):
    name, limit, A, B, steps, snaps = task
    ts = [t for t in sorted(snaps) if t + 100 <= len(steps)][:4]
    worst_long, worst_base = 0.0, 0.0
    for t in ts:
        ref = steps[t:t + 100]
        worst_long = max(worst_long, _diff(ref, _replay(B, snaps[t], [x[0] for x in ref]))[0])
        ref8 = steps[t:t + H]
        worst_base = max(worst_base, _diff(ref8, _replay(B, snaps[t], [x[0] for x in ref8],
                                                          diag_v1.restore_sim_only))[0])
    print(f'{name}: 100-step replays from {len(ts)} snapshots: max |obs diff| = {worst_long:.3g}; '
          f'robomimic reset_to(states) baseline over {H} steps: {worst_base:.3g}')
    assert worst_long <= 1e-6


def test_restore_success_demo():
    import h5py
    from envs.robomimic_utils import _check_dataset_exists
    name = 'square-mh-low_dim'
    env, other = _make(name, 0), _make(name, 3)
    other.reset()
    with h5py.File(_check_dataset_exists(name), 'r') as f:
        states, actions = np.array(f['data/demo_0/states']), np.array(f['data/demo_0/actions'])
    obs, _ = env.reset()
    obs = diag_v1.env_layers(env)[2].reset_to({'states': states[0]})
    obs = env.env.get_observation()
    steps, snaps = [], {}
    for t, a in enumerate(actions):
        if t >= len(actions) - 3 * H and t % 4 == 0:
            snaps[t] = diag_v1.snapshot(env, obs)
        obs, r, te, tr, _ = env.step(a)
        steps.append((a, obs.copy(), r, te, tr))
        if te or tr:
            break
    if not steps[-1][3]:
        pytest.skip('demo replay did not succeed')
    worst = 0.0
    for t, s in snaps.items():
        ref = steps[t:t + H]
        d, r_ok, f_ok = _diff(ref, _replay(other, s, [x[0] for x in ref]))
        assert r_ok and f_ok
        worst = max(worst, d)
    print(f'success demo: {len(snaps)} snapshots before the success at step {len(steps)}, max |obs diff| = {worst:.3g}')
    assert worst <= 1e-6


def test_restore_via_pool():
    """The vector-pool worker path (pickled snapshots, restore inside AsyncVectorEnv workers)."""
    name, limit = TASKS[0]
    A = _make(name, 0)
    steps, snaps = _rollout(A, limit, 1)
    envs = diag_v1.make_pool(name, 2)
    try:
        ts = sorted(snaps)[:6]
        worst = 0.0
        for t0, t1 in zip(ts[::2], ts[1::2]):
            diag_v1.pool_exec(envs, [('restore', (1, snaps[t0])), ('restore', (2, snaps[t1]))])
            for j in range(H):
                o, r, te, tr, _ = envs.step(np.stack([steps[t0 + j][0], steps[t1 + j][0]]))
                for i, t in enumerate((t0, t1)):
                    x = steps[t + j]
                    worst = max(worst, float(np.abs(o[i] - x[1]).max()))
                    assert (r[i], bool(te[i]), bool(tr[i])) == (x[2], x[3], x[4])
        snap_back = diag_v1.pool_exec(envs, [('snapshot', None), None])[0]
        assert snap_back['t'] == ts[-2] + H and snap_back['model'] == snaps[ts[0]]['model']
    finally:
        envs.close()
    print(f'pool restore: max |obs diff| = {worst:.3g}')
    assert worst <= 1e-6


def test_compactor_roundtrip(task):
    name, limit, A, B, steps, snaps = task
    ss = [snaps[t] for t in sorted(snaps)]
    bases, comp = diag_v1.compact_snapshots(ss)
    import pickle
    for s, c in zip(ss, comp):
        e = diag_v1.expand_snapshot(c, bases)
        assert pickle.dumps(e['py']) == pickle.dumps(s['py']) and np.array_equal(e['mj'], s['mj'])
    raw = len(pickle.dumps(ss))
    small = len(pickle.dumps((bases, comp)))
    print(f'{name}: snapshot {len(pickle.dumps(ss[0])) / 1e3:.1f} kB; {len(ss)} compacted {small / raw:.2f} of raw')
