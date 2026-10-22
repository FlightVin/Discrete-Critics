"""v1 methods end to end (square, tiny sizes): critic probe for every agent, exception exit, all six methods
through every phase of train_v1.py --mode=online.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set (the subprocess tests need the
square dataset and eval envs).
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents import V1_METHODS, agents, v1_config  # noqa: E402
from visualization.critic_probe import probe_batch, q_values  # noqa: E402

H = 5
METHODS = list(V1_METHODS)
COMMON = ['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--n_eval_envs=2', '--eval_episodes=1',
          '--log_interval=250', '--probe_states=256', '--seed=0']
ONLINE = ['--mode=online', '--warmup_steps=1000', '--pretrain_updates=200', '--online_steps=3000',
          '--eval_interval=1500', '--buffer_cap=90000']


# ---- critic probe ----------------------------------------------------------------------------------
def q_values_0134fe7(agent, observations, actions, rng):
    """critic_probe.q_values as of commit 0134fe7 (name dispatch; four agents)."""
    name = agent.config['agent_name']
    if name == 'acbsc':
        qs = jax.nn.sigmoid(agent.network.select('critic')(observations, actions))
    elif name == 'acfql':
        qs = agent.network.select('critic')(observations, actions)
    elif name == 'accat':
        qs = agent.critic_value(observations, actions)
    elif name == 'acarb':
        qs = agent.critic_value(observations, actions, rng=rng)
    else:
        raise ValueError(f'unsupported agent {name}')
    return qs.min(axis=0) if agent.config['q_agg'] == 'min' else qs.mean(axis=0)


def small_agent(method, obs_dim=6, action_dim=3):
    cfg = v1_config(method)
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    cfg.actor_num_samples = 4
    cfg.flow_steps = 2
    cfg.horizon_length = H
    return agents[cfg['agent_name']].create(0, np.zeros(obs_dim, np.float32), np.zeros(action_dim, np.float32), cfg)


def fixed_batch(n=16, obs_dim=6, action_dim=3):
    rs = np.random.RandomState(0)
    return (rs.randn(n, obs_dim).astype(np.float32), rs.uniform(-1, 1, (n, action_dim * H)).astype(np.float32),
            rs.standard_normal((n, 4, action_dim * H)).astype(np.float32))


@pytest.mark.parametrize('method', [m for m in METHODS if m != 'hlg'])
def test_q_values_unchanged_for_existing_agents(method):
    agent = small_agent(method)
    obs, act, _ = fixed_batch()
    rng = jax.random.PRNGKey(3)
    new, old = np.asarray(q_values(agent, obs, act, rng)), np.asarray(q_values_0134fe7(agent, obs, act, rng))
    assert new.shape == (16,) and np.isfinite(new).all()
    assert new.dtype == old.dtype and np.array_equal(new, old), (method, new, old)


def test_q_values_hlg_is_expected_value_over_bin_centres():
    agent = small_agent('hlg')
    assert agent.config['agent_name'] == 'achlg'
    obs, act, _ = fixed_batch()
    logits = agent.network.select('critic')(obs, act)
    expected = (jax.nn.softmax(logits, axis=-1) * agent.support()).sum(-1).mean(0)  # v1: q_agg mean
    got = q_values(agent, obs, act, jax.random.PRNGKey(3))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-6, atol=1e-6)
    assert float(got.min()) >= float(agent.support().min()) and float(got.max()) <= float(agent.support().max())


@pytest.mark.parametrize('method', METHODS)
def test_probe_batch_runs_for_every_method(method):
    agent = small_agent(method)
    obs, act, noises = fixed_batch()
    q_data, q_cand, q_rand = probe_batch(agent, jnp.asarray(obs), act, noises, act[::-1], jax.random.PRNGKey(0))
    assert q_data.shape == (16,) and q_cand.shape == (16, 4) and q_rand.shape == (16,)
    assert all(np.isfinite(np.asarray(x)).all() for x in (q_data, q_cand, q_rand))


# ---- train_v1.py subprocesses ---------------------------------------------------------------------
def run_env(**extra):
    return dict(os.environ, WANDB_MODE='disabled', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
                **extra)


def train(args, log_path, timeout=1200, **extra_env):
    with open(log_path, 'w') as log:
        p = subprocess.run([sys.executable, 'train_v1.py', *args], cwd=REPO, env=run_env(**extra_env),
                           stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, open(log_path).read()


@pytest.fixture(scope='module')
def init_actor(tmp_path_factory):
    d = tmp_path_factory.mktemp('bc')
    code, out = train([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n',
                       '--bc_steps=200', '--eval_interval=200', f'--run_dir={d / "run"}'], d / 'bc.log')
    assert code == 0, out[-3000:]
    return str(d / 'run' / 'final_actor.pkl')


@pytest.fixture(scope='module')
def method_runs(init_actor, tmp_path_factory):
    """All six methods, run concurrently: {method: (exit code, log, metrics rows)}."""
    root = tmp_path_factory.mktemp('methods')

    def one(method):
        run_dir = root / method
        t0 = time.time()
        code, out = train([*COMMON, *ONLINE, f'--method={method}', f'--init_actor={init_actor}',
                           f'--run_dir={run_dir}'], root / f'{method}.log')
        path = run_dir / 'metrics.jsonl'
        rows = [json.loads(l) for l in open(path)] if path.exists() else []
        print(f'{method}: exit {code} in {time.time() - t0:.0f}s', flush=True)
        return code, out, rows

    with ThreadPoolExecutor(len(METHODS)) as ex:
        return dict(zip(METHODS, ex.map(one, METHODS)))


@pytest.mark.parametrize('method', METHODS)
def test_method_runs_every_phase(method, method_runs):
    code, out, rows = method_runs[method]
    assert code == 0, out[-4000:]
    assert 'RUN_COMPLETE' in out
    cfg = [r for r in rows if r['kind'] == 'config']
    assert len(cfg) == 1 and cfg[0]['agent_name'] == V1_METHODS[method][0]
    ends = [r['phase'] for r in rows if r['kind'] == 'phase_end']
    assert ends == ['warmup', 'pretrain', 'pretrain_eval', 'joint'], ends

    def has_fields(row):
        for prefix in ('n1/', 'bon/', 'probe/'):
            keys = [k for k in row if k.startswith(prefix)]
            assert keys, (prefix, sorted(row))
            for k in keys:
                assert isinstance(row[k], (int, float)), (k, row[k])
        assert 0.0 <= row['n1/success_rate'] <= 1.0 and 0.0 <= row['bon/success_rate'] <= 1.0
        assert np.isfinite(row['probe/q_data_mean']) and np.isfinite(row['probe/cand_range_mean'])

    pe = [r for r in rows if r['kind'] == 'pretrain_eval']
    assert len(pe) == 1, len(pe)
    has_fields(pe[0])
    joint = [r for r in rows if r['kind'] == 'eval' and r['phase'] == 'joint']
    assert [r['nominal_step'] for r in joint] == [1500, 3000], [r['nominal_step'] for r in joint]
    for r in joint:
        has_fields(r)
    assert any(r['kind'] == 'joint' for r in rows) and any(r['kind'] == 'pretrain' for r in rows)


def test_exception_in_pretrain_eval_exits_nonzero_within_60s(init_actor, tmp_path):
    """An unhandled exception after the eval envs have run (where job 10888854 failed) ends the process."""
    args = [*COMMON, *ONLINE[:1], '--warmup_steps=1000', '--pretrain_updates=50', '--online_steps=3000',
            '--buffer_cap=90000', '--method=hlg', f'--init_actor={init_actor}', f'--run_dir={tmp_path / "run"}']
    p = subprocess.Popen([sys.executable, 'train_v1.py', *args], cwd=REPO,
                         env=run_env(V1_TEST_RAISE_IN_PRETRAIN_EVAL='1'),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, t_exc = [], []

    def reader():
        for line in p.stdout:
            lines.append(line)
            if line.startswith('EXIT_EXCEPTION ') and not t_exc:
                t_exc.append(time.time())

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        code = p.wait(timeout=900)
    except subprocess.TimeoutExpired:
        p.kill()
        raise
    t_exit = time.time()
    th.join(10)
    out = ''.join(lines)
    assert 'PHASE_BEGIN pretrain_eval' in out and 'ValueError: V1_TEST_RAISE_IN_PRETRAIN_EVAL' in out, out[-3000:]
    assert t_exc, out[-3000:]
    dt = t_exit - t_exc[0]
    print(f'exception exit: code {code}, {dt:.2f}s from EXIT_EXCEPTION to process exit')
    assert code != 0 and code != 143, (code, out[-3000:])
    assert dt < 60, dt
