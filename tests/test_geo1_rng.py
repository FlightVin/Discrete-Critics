"""geo1: RNG streams. Evaluation and probes do not advance any training stream; the separate replay Generator
survives a checkpoint round trip. Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import os
import random
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
from utils.replay_v1 import SeqReplay  # noqa: E402
from visualization.critic_probe import make_candidates  # noqa: E402

H = 5
FLAGS = T.FLAGS


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not FLAGS.is_parsed():
        FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def replay_trainer(run_dir, gen_seed):
    from test_v1_train import make_trainer
    tr = make_trainer(run_dir, seed=0)
    tr.replay_rng = np.random.Generator(np.random.PCG64(gen_seed))
    tr.buffers = dict(main=SeqReplay(tr.demos, 200, H, episodic=False, rng=tr.replay_rng),
                      success=SeqReplay(tr.demos, 200, H, episodic=True, rng=tr.replay_rng))
    return tr


def draws(tr, n=5):
    out = []
    for _ in range(n):
        for b in tr.buffers.values():
            out.append(b.sample_sequence(16, H, 0.99)['observations'])
    return out


def test_replay_generator_survives_checkpoint(tmp_path):
    from test_v1_train import work
    np.random.seed(3)
    random.seed(3)
    tr = replay_trainer(tmp_path, T.REPLAY_SEED_OFFSET)
    tr.store.load()
    tr.metrics = T.MetricsLog(os.path.join(tmp_path, 'metrics.jsonl'), 0)
    work(tr, np.random.RandomState(0), 5)
    tr.checkpoint('test')
    g_state = tr.replay_rng.bit_generator.state
    np_state = np.random.get_state()
    uninterrupted = draws(tr)
    # Sampling uses only the Generator: numpy's global state is untouched.
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert tr.replay_rng.bit_generator.state != g_state

    tr2 = replay_trainer(tmp_path, 12345)  # a fresh process: different Generator seed
    tr2.restore()
    assert tr2.replay_rng.bit_generator.state == g_state
    resumed = draws(tr2)
    assert all(np.array_equal(a, b) for a, b in zip(uninterrupted, resumed))


def test_global_replay_rng_is_unchanged_default(tmp_path):
    """Default: no Generator, sampling draws np.random.randint exactly as before; no replay_rng in the state."""
    from test_v1_train import make_trainer
    tr = make_trainer(tmp_path, seed=0)
    assert tr.buffers['main'].rng is None
    np.random.seed(5)
    a = tr.buffers['main'].sample_starts(16)[0]
    np.random.seed(5)
    assert np.array_equal(a, np.random.randint(tr.buffers['main'].num_valid_starts, size=16))  # demos only here
    tr.metrics = T.MetricsLog(os.path.join(tmp_path, 'm.jsonl'), 0)
    assert 'replay_rng' not in tr.state_dict()


@pytest.fixture(scope='module')
def square():
    from envs.env_utils import make_env_and_datasets
    env, eval_env, ds, _ = make_env_and_datasets('square-mh-low_dim')
    for e in (env, eval_env):
        if hasattr(e, 'close'):
            e.close()
    return T.process_demos(ds, 'square-mh-low_dim', H, True)


def test_eval_and_probe_leave_training_streams(square, tmp_path, monkeypatch):
    from envs.env_utils import make_vectorized_eval_envs
    for k, v in dict(env_name='square-mh-low_dim', n_eval_envs=2, eval_episodes=1, probe_states=256,
                     run_dir=str(tmp_path), probe_candidates=None).items():
        monkeypatch.setattr(FLAGS, k, v)
    cfg = v1_config('arb_bsc')
    cfg.update(dict(actor_hidden_dims=(32, 32), value_hidden_dims=(32, 32), arb_d_model=32, flow_steps=2,
                    horizon_length=H, critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10,
                    arb_readout_temp=0.7, arb_label='twohot_all'))
    agent = agents['acarb'].create(0, square['observations'][0], square['actions'][0], cfg)
    tr = T.Trainer()
    tr.mode, tr.phase, tr.H = 'online', 'joint', H
    tr.agent = T.split_optimizer(agent)
    tr.demos = square
    tr.action_dim = square['actions'].shape[-1]
    tr.setup_probe(cfg)
    tr.probe_cands = make_candidates(tr.agent, tr.probe_obs, tr.probe_noises)
    np_state = np.random.get_state()
    tr.eval_envs = make_vectorized_eval_envs('square-mh-low_dim', 2, seed=FLAGS.eval_seed)
    np.random.set_state(np_state)
    tr.replay_rng = np.random.Generator(np.random.PCG64(1))
    tr.action_key = jax.random.PRNGKey(7)
    np.random.seed(17)
    random.seed(17)
    before = (jax.device_get(tr.agent), np.asarray(tr.action_key), np.random.get_state(), random.getstate(),
              tr.replay_rng.bit_generator.state)
    try:
        res = tr.evaluate(['n1', 'bon'])
        row = tr.probe_row()
    finally:
        tr.close_envs()
    assert res is not None and set(res) == {'n1', 'bon'}
    for k in ('probe/tie_rate', 'probe/tie_max', 'probe_fixed/tie_all', 'probe_fixed/tie_max',
              'probe_fixed/score_mean', 'probe_fixed/cand_range_mean', 'probe_fixed/oor_frac',
              'probe_stab/score_var_mean', 'probe_stab/same_argmax_frac'):
        assert np.isfinite(row[k]), k
    assert row['probe_stab/score_var_mean'] > 0  # sampled readout: repeats differ
    after = (jax.device_get(tr.agent), np.asarray(tr.action_key), np.random.get_state(), random.getstate(),
             tr.replay_rng.bit_generator.state)
    for x, y in zip(jax.tree_util.tree_leaves(before[0]), jax.tree_util.tree_leaves(after[0])):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert np.array_equal(np.asarray(before[0].rng), np.asarray(after[0].rng))
    assert np.array_equal(before[1], after[1])
    assert before[2][0] == after[2][0] and np.array_equal(before[2][1], after[2][1]) and before[2][2:] == after[2][2:]
    assert before[3] == after[3] and before[4] == after[4]


def test_deterministic_readout_probe_stability(square, monkeypatch):
    """Greedy readout: probe_stab variance 0 and same-argmax 1."""
    monkeypatch.setattr(FLAGS, 'probe_states', 256)
    cfg = v1_config('arb_bsc')
    cfg.update(dict(actor_hidden_dims=(32, 32), value_hidden_dims=(32, 32), arb_d_model=32, flow_steps=2,
                    horizon_length=H))
    agent = agents['acarb'].create(0, square['observations'][0], square['actions'][0], cfg)
    from visualization.critic_probe import probe_fixed, probe_states
    idx, _ = probe_states(square, H, 256, 0)
    obs = square['observations'][idx]
    noises = np.random.RandomState(1).standard_normal((256, 32, square['actions'].shape[-1] * H)).astype(np.float32)
    out = probe_fixed(agent, obs, make_candidates(agent, obs, noises))
    assert out['probe_stab/score_var_mean'] == 0.0 and out['probe_stab/same_argmax_frac'] == 1.0
    assert out['probe_fixed/oor_frac'] == 0.0
