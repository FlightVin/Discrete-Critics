"""train_v1.py: update functions, checkpoint round trip, episode routing, restart of a complete run.

Uses a small acbsc agent on synthetic data (CPU is fine).
"""
import os
import pickle
import random
import subprocess
import sys
import time

import jax
import numpy as np
import optax
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

import train_v1 as T  # noqa: E402
from agents import acbsc, agents, v1_config  # noqa: E402
from test_v1_replay import make_demos, row  # noqa: E402
from utils.ckpt_v1 import CheckpointStore  # noqa: E402
from utils.replay_v1 import FIELDS, SeqReplay  # noqa: E402

H = 5
FLAGS = T.FLAGS


@pytest.fixture(scope='module', autouse=True)
def parsed_flags(tmp_path_factory):
    if not FLAGS.is_parsed():
        FLAGS(['test', f'--run_dir={tmp_path_factory.mktemp("run")}', f'--horizon_length={H}'])
    yield


def demos():
    d = make_demos([30, 40, 25, 33])
    d['observations'][:, 1:] = np.random.RandomState(5).randn(len(d['observations']), 3)
    return d


def small_agent(seed=0, weight_decay=0.0, split=True, method=None):
    """Small agent; split=True: the train_v1 per-network optimisers (as setup() builds them)."""
    cfg = acbsc.get_config() if method is None else v1_config(method)
    cfg.actor_type = 'best-of-n'
    cfg.actor_hidden_dims = (32, 32)
    cfg.value_hidden_dims = (32, 32)
    cfg.actor_num_samples = 4
    cfg.flow_steps = 2
    cfg.batch_size = 16
    cfg.horizon_length = H
    cfg.weight_decay = weight_decay
    d = demos()
    agent = agents[cfg['agent_name']].create(seed, d['observations'][0], d['actions'][0], cfg)
    return T.split_optimizer(agent) if split else agent


def leaves(tree):
    return jax.tree_util.tree_leaves_with_path(jax.device_get(tree))


def assert_tree_equal(a, b):
    la, lb = leaves(a), leaves(b)
    assert [p for p, _ in la] == [p for p, _ in lb]
    for (p, x), (_, y) in zip(la, lb):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), jax.tree_util.keystr(p)


def batches(buf, n=16):
    return buf.sample_sequence(n, H, 0.99), buf.sample_sequence(n, H, 0.99)


def filled_buffers(cap=200):
    d = demos()
    main = SeqReplay(d, cap, H, episodic=False)
    succ = SeqReplay(d, cap, H, episodic=True)
    k = 0
    for ep in range(12):
        L = 11 + ep
        rows = [row(10 ** 7 + k + i, terminal=i == L - 1, success=(ep % 3 == 0) and i == L - 1) for i in range(L)]
        for r in rows:
            main.add_transition(r)
        if ep % 3 == 0:
            succ.add_episode(rows)
        k += L
    return d, main, succ


# ---- (d) update functions --------------------------------------------------------------------------
def test_critic_update_keeps_actor_bit_identical():
    np.random.seed(0)
    _, main, succ = filled_buffers()
    agent = small_agent(weight_decay=1e-2)  # adamw would move every param if not frozen
    for _ in range(3):  # non-zero optimiser moments for the actor too
        agent, _ = T.joint_update(agent, *batches(main)[:1], batches(succ)[0])
    params = agent.network.params
    actor = {k: v for k, v in params.items() if 'critic' not in k}
    assert set(actor) == {'modules_actor_bc_flow', 'modules_actor_onestep_flow'}
    for _ in range(5):
        new, info = T.critic_update(agent, main.sample_sequence(16, H, 0.99))
        assert_tree_equal({k: new.network.params[k] for k in actor}, actor)
        assert_tree_equal(new.network.opt_state.inner_states['actor'], agent.network.opt_state.inner_states['actor'])
        assert T.opt_counts(new) == dict(actor=3, critic=T.opt_counts(agent)['critic'] + 1)
        changed = jax.tree_util.tree_map(lambda a, b: not np.array_equal(a, b),
                                         jax.device_get(params['modules_critic']),
                                         jax.device_get(new.network.params['modules_critic']))
        assert any(jax.tree_util.tree_leaves(changed)), 'critic did not move'
        assert not np.array_equal(jax.device_get(new.rng), jax.device_get(agent.rng))
        agent = new
    assert 'critic/critic_loss' in info and not any(k.startswith('actor/') for k in info)


def test_bc_update_keeps_critic_and_target_bit_identical():
    np.random.seed(0)
    _, main, _ = filled_buffers()
    agent = small_agent(weight_decay=1e-2)
    crit = {k: v for k, v in agent.network.params.items() if 'critic' in k}
    critic_opt = agent.network.opt_state.inner_states['critic']
    for _ in range(3):
        agent, info = T.bc_update(agent, main.sample_sequence(16, H, 0.99))
    assert_tree_equal({k: agent.network.params[k] for k in crit}, crit)
    assert_tree_equal(agent.network.opt_state.inner_states['critic'], critic_opt)
    assert T.opt_counts(agent) == dict(actor=3, critic=0)
    assert set(info) >= {'actor/bc_flow_loss'} and not any(k.startswith('critic/') for k in info)


def test_joint_update_moves_actor_critic_and_target():
    np.random.seed(0)
    _, main, succ = filled_buffers()
    agent = small_agent()
    new, info = T.joint_update(agent, main.sample_sequence(16, H, 0.99), succ.sample_sequence(16, H, 0.99))
    for k in ('modules_actor_bc_flow', 'modules_critic', 'modules_target_critic'):
        a = jax.tree_util.tree_leaves(jax.device_get(agent.network.params[k]))
        b = jax.tree_util.tree_leaves(jax.device_get(new.network.params[k]))
        assert any(not np.array_equal(x, y) for x, y in zip(a, b)), k
    assert 'critic/critic_loss' in info and 'actor/bc_flow_loss' in info
    assert float(info['actor/q_loss']) == 0.0 and float(info['actor/distill_loss']) == 0.0


# ---- separate optimisers ---------------------------------------------------------------------------
def _old_critic_update(agent, batch):
    """critic_update as it was with the agent's single shared Adam (before the split)."""
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        loss, info = agent.critic_loss(batch, grad_params, rng)
        return loss, info

    network, _ = agent.network.apply_loss_fn(loss_fn=loss_fn)
    params = dict(network.params)
    for k in T._modules(params, critic=False):
        params[k] = agent.network.params[k]
    network = network.replace(params=params)
    agent.target_update(network, 'critic')
    return agent.replace(network=network, rng=new_rng)


def test_pretrain_critic_updates_match_single_optimiser():
    np.random.seed(0)
    _, main, _ = filled_buffers()
    bs = [main.sample_sequence(16, H, 0.99) for _ in range(6)]
    old = small_agent(split=False)              # create(): one Adam over all modules, count 0
    new = small_agent(split=True)
    assert_tree_equal(old.network.params, new.network.params)
    old_fn = jax.jit(_old_critic_update)
    for b in bs:
        old = old_fn(old, b)
        new, _ = T.critic_update(new, b)
    assert T.opt_counts(new) == dict(actor=0, critic=len(bs))
    for k in ('modules_critic', 'modules_target_critic'):
        for x, y in zip(jax.tree_util.tree_leaves(jax.device_get(old.network.params[k])),
                        jax.tree_util.tree_leaves(jax.device_get(new.network.params[k]))):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7, err_msg=k)
    # Critic Adam moments equal the shared optimiser's moments for the critic leaves.
    old_mu = old.network.opt_state[0].mu['modules_critic']
    new_mu = new.network.opt_state.inner_states['critic'].inner_state[0].mu['modules_critic']
    for x, y in zip(jax.tree_util.tree_leaves(jax.device_get(old_mu)), jax.tree_util.tree_leaves(jax.device_get(new_mu))):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-9)


def test_first_joint_step_is_a_fresh_adam_step_for_the_actor():
    np.random.seed(0)
    _, main, succ = filled_buffers()
    agent = small_agent()
    for _ in range(20):                         # pretrain: critic Adam at count 20, actor Adam at 0
        agent, _ = T.critic_update(agent, main.sample_sequence(16, H, 0.99))
    assert T.opt_counts(agent) == dict(actor=0, critic=20)
    mb, sb = main.sample_sequence(16, H, 0.99), succ.sample_sequence(16, H, 0.99)
    new, _ = T.joint_update(agent, mb, sb)
    assert T.opt_counts(new) == dict(actor=1, critic=21)

    _, actor_rng, critic_rng = T.joint_rngs(agent)
    grads = jax.jit(jax.grad(lambda p: T.joint_loss(agent, mb, sb, p, actor_rng, critic_rng)[0]))(
        agent.network.params)
    actor = {k: agent.network.params[k] for k in T._modules(agent.network.params, critic=False)}
    g = {k: grads[k] for k in actor}
    tx = optax.adam(agent.config['lr'])
    upd, _ = tx.update(g, tx.init(actor), actor)
    expected = optax.apply_updates(actor, upd)
    max_step = 0.0
    for k in actor:
        for e, n, a in zip(jax.tree_util.tree_leaves(jax.device_get(expected[k])),
                           jax.tree_util.tree_leaves(jax.device_get(new.network.params[k])),
                           jax.tree_util.tree_leaves(jax.device_get(actor[k]))):
            # Steps (~lr * g / (|g| + eps)) agree to 2% of lr: float noise in g matters only where |g| ~ eps.
            # The shared-optimiser defect (bias correction at a large count, zero moments) gives ~3.2 lr.
            np.testing.assert_allclose(n - a, e - a, rtol=0, atol=0.02 * agent.config['lr'], err_msg=k)
            max_step = max(max_step, float(np.abs(n - a).max()))
    # A first Adam step moves each weight by at most lr (|m_hat / sqrt(v_hat)| <= 1).
    assert max_step <= agent.config['lr'] * 1.0001, max_step

    # The check discriminates: with the agent's single shared Adam the same step is far off.
    shared = small_agent(split=False)
    for _ in range(20):
        shared = jax.jit(_old_critic_update)(shared, main.sample_sequence(16, H, 0.99))
    shared = shared.replace(network=shared.network.replace(params=agent.network.params), rng=agent.rng)
    bad, _ = T.joint_update(shared, mb, sb)
    k = 'modules_actor_bc_flow'
    err = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(
        jax.tree_util.tree_leaves(jax.device_get(bad.network.params[k])),
        jax.tree_util.tree_leaves(jax.device_get(expected[k]))))
    assert err > 0.2 * agent.config['lr'], err


# ---- method presets, diagnostics, preemption notice ------------------------------------------------
def test_method_presets_and_overrides():
    for method in T.V1_METHODS:
        argv = T.argv_with_method_agent(['train_v1.py', f'--method={method}', '--agent.lr=1e-4'])
        assert argv[1].endswith(f'agents/{T.V1_METHODS[method][0]}.py'), argv
        flag_cfg = v1_config(method)            # stands in for FLAGS.agent parsed from that file
        flag_cfg.lr = 1e-4
        flag_cfg.num_qs = 10                    # not on the command line: the preset wins
        cfg = T.resolve_agent_config(method, flag_cfg, argv)
        assert cfg.agent_name == T.V1_METHODS[method][0]
        assert cfg.lr == 1e-4 and cfg.num_qs == 2 and cfg.actor_num_samples == 32
        assert cfg.actor_type == 'best-of-n' and cfg.q_agg == 'mean'
    assert T.resolve_agent_config('arb_qc', v1_config('arb_qc'), []).discount == 0.99
    assert T.resolve_agent_config('bsc', v1_config('bsc'), []).critic_target_bon
    with pytest.raises(AssertionError):
        T.resolve_agent_config('qc', acbsc.get_config(), [])
    explicit = ['x', '--method=qc', '--agent=agents/acbsc.py']
    assert T.argv_with_method_agent(explicit) == explicit


def test_online_refuses_non_v1_actor(tmp_path):
    env = dict(os.environ, WANDB_MODE='disabled', JAX_PLATFORMS='cpu')
    for extra, msg in ((['--agent.actor_num_samples=4'], 'actor_num_samples=4'),
                       (['--agent.actor_type=distill'], 'actor_type=distill')):
        p = subprocess.run([sys.executable, 'train_v1.py', '--mode=online', '--method=bsc', *extra,
                            f'--run_dir={tmp_path / extra[0][-3:]}', '--init_actor=/nonexistent.pkl'],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 2 and 'CONFIG_REFUSED' in p.stdout and msg in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]


def test_arb_readout_diagnostics_logged_without_touching_training_state(tmp_path):
    np.random.seed(0)
    _, main, _ = filled_buffers()
    tr = make_trainer(tmp_path, seed=0)
    tr.agent = small_agent(method='arb_qc')
    tr.metrics = T.MetricsLog(os.path.join(tmp_path, 'metrics.jsonl'), 0)
    import wandb
    wandb.init(mode='disabled')
    batch = main.sample_sequence(16, H, 0.99)
    before = (jax.device_get(tr.agent), np.random.get_state()[1].copy())
    tr.log_readout('pretrain', batch, 250)
    assert_tree_equal(jax.device_get(tr.agent), before[0])
    assert np.array_equal(np.random.get_state()[1], before[1])
    rows = [T.json.loads(l) for l in open(os.path.join(tmp_path, 'metrics.jsonl'))]
    assert rows[-1]['kind'] == 'readout' and rows[-1]['updates'] == 250
    for k in ('critic/expected_q', 'critic/greedy_expected_gap', 'critic/bon_argmax_agree', 'critic/digit1_maxprob'):
        assert np.isfinite(rows[-1][k]), k
    tr.agent = small_agent()                    # non-ARB agents: nothing logged
    tr.log_readout('pretrain', batch, 500)
    assert len([T.json.loads(l) for l in open(os.path.join(tmp_path, 'metrics.jsonl'))]) == len(rows)


def test_preempt_watcher_fires_once_with_file_mtime(tmp_path):
    path = tmp_path / 'preempt'
    seen = []
    w = T.PreemptWatcher(str(path), seen.append, poll_s=0.2)
    time.sleep(0.5)
    assert not seen and not w.check()
    path.write_text(str(int(time.time())))
    mtime = os.stat(path).st_mtime
    t0 = time.time()
    while not seen and time.time() - t0 < 5:
        time.sleep(0.05)
    assert seen == [mtime] and time.time() - t0 < 1.0
    assert w.check() and seen == [mtime]


# ---- (a) checkpoint round trip ---------------------------------------------------------------------
def make_trainer(run_dir, seed):
    FLAGS.run_dir = str(run_dir)
    tr = T.Trainer()
    tr.mode, tr.phase, tr.H, tr.batch_size = 'online', 'joint', H, 16
    tr.agent = small_agent(seed)
    tr.demos = demos()
    tr.buffers = dict(main=SeqReplay(tr.demos, 200, H, episodic=False),
                      success=SeqReplay(tr.demos, 200, H, episodic=True))
    tr.action_key = jax.random.PRNGKey(seed + 1)
    tr.last_eval, tr.wandb_id, tr.metrics_offset = 0, f'id{seed}', 0
    tr.ctr = T.defaultdict(int)
    tr.env_acc = T.defaultdict(list)
    return tr


def snapshot(tr):
    return dict(
        agent=jax.device_get(tr.agent), action_key=np.asarray(tr.action_key), np_rng=np.random.get_state(),
        py_rng=random.getstate(), ctr=dict(tr.ctr), phase=tr.phase, last_eval=tr.last_eval,
        env_acc={k: list(v) for k, v in tr.env_acc.items()}, wandb_id=tr.wandb_id,
        buffers={k: (b.meta(), {f: b.data[f].copy() for f in FIELDS}, b.valid_start_slots())
                 for k, b in tr.buffers.items()})


def compare(s1, s2):
    assert_tree_equal(s1['agent'], s2['agent'])  # params incl. target critic, optimiser state, agent rng
    assert np.array_equal(s1['action_key'], s2['action_key'])
    a, b = s1['np_rng'], s2['np_rng']
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    for k in ('py_rng', 'ctr', 'phase', 'last_eval', 'env_acc', 'wandb_id'):
        assert s1[k] == s2[k], k
    for name in s1['buffers']:
        m1, d1, v1 = s1['buffers'][name]
        m2, d2, v2 = s2['buffers'][name]
        assert m1 == m2, name  # pointers, k_lo, success episode index, overwrite counts
        assert np.array_equal(v1, v2), name  # valid starts
        for f in FIELDS:
            assert np.array_equal(d1[f], d2[f]) and d1[f].dtype == d2[f].dtype, (name, f)


def work(tr, rs, steps):
    """Some env-free activity touching every piece of state."""
    for i in range(steps):
        k = tr.buffers['main'].n
        L = int(rs.randint(8, 30))
        success = rs.rand() < 0.5
        rows = [row(10 ** 7 + k + j, terminal=j == L - 1, success=success and j == L - 1) for j in range(L)]
        for r in rows:
            tr.buffers['main'].add_transition(r)
        if success:
            tr.buffers['success'].add_episode(rows)
        tr.ctr['env_steps'] += L
        tr.ctr['episodes'] += 1
        tr.env_acc['success'].append(int(success))
        tr.agent, _ = T.joint_update(tr.agent, tr.buffers['main'].sample_sequence(16, H, 0.99),
                                     tr.buffers['success'].sample_sequence(16, H, 0.99))
        tr.ctr['joint_updates'] += 1
        tr.action_key, _ = jax.random.split(tr.action_key)
        random.random()
        if i % 3 == 0:
            tr.metrics.log('test', tr.ctr['env_steps'], {'i': i}, wandb_too=False)


def test_checkpoint_round_trip_bit_exact(tmp_path):
    np.random.seed(11)
    random.seed(11)
    rs = np.random.RandomState(0)
    tr = make_trainer(tmp_path, seed=0)
    tr.store.load()
    tr.metrics = T.MetricsLog(os.path.join(tmp_path, 'metrics.jsonl'), 0)
    work(tr, rs, 6)
    tr.checkpoint('test')          # several generations, so restore needs several segments
    work(tr, rs, 15)               # > capacity: overwrites in main, evictions in success
    tr.checkpoint('test')
    work(tr, rs, 4)
    tr.checkpoint('test')
    assert tr.buffers['main'].overwrites > 0 and tr.buffers['success'].overwrites > 0
    expected = snapshot(tr)
    offset = tr.metrics.offset()
    rs_state = rs.get_state()
    work(tr, rs, 3)                # work after the checkpoint is lost on restart
    after_ckpt = snapshot(tr)

    np.random.seed(999)            # a fresh process: different RNG states, different agent init
    random.seed(999)
    tr2 = make_trainer(tmp_path, seed=7)
    tr2.restore()
    tr2.metrics = T.MetricsLog(os.path.join(tmp_path, 'metrics.jsonl'), tr2.metrics_offset)
    assert tr2.metrics_offset == offset and os.path.getsize(os.path.join(tmp_path, 'metrics.jsonl')) == offset
    compare(snapshot(tr2), expected)

    # Continuing from the restored state reproduces the lost work exactly (deterministic on CPU).
    rs2 = np.random.RandomState()
    rs2.set_state(rs_state)        # work() also draws episode lengths from this test-local RNG
    work(tr2, rs2, 3)
    compare(snapshot(tr2), after_ckpt)


# ---- episode routing -------------------------------------------------------------------------------
class FakeEnv:
    """Episodes alternate: success after 7 steps, then timeout at the 10-step limit."""

    def __init__(self):
        self.k, self.ep, self.t = 0, -1, 0

    def reset(self):
        self.ep += 1
        self.t = 0
        return self.obs(), {}

    def obs(self):
        return np.array([10 ** 7 + self.k, 0.1, 0.2, 0.3], np.float32)

    def step(self, action):
        self.k += 1
        self.t += 1
        success = self.ep % 2 == 0 and self.t == 7
        trunc = not success and self.t == 10
        info = {'success': int(success)}
        if success or trunc:
            info['episode'] = {'return': float(success), 'length': self.t}
        return self.obs(), float(success), success, trunc, info


def test_episode_routing(tmp_path):
    np.random.seed(0)
    tr = make_trainer(tmp_path, seed=0)
    tr.metrics = T.MetricsLog(os.path.join(tmp_path, 'metrics.jsonl'), 0)
    tr.env, tr.max_ep_len, tr.action_dim, tr.phase = FakeEnv(), 10, 2, 'warmup'
    tr.ob, tr.action_queue, tr.ep_rows = None, [], []
    FLAGS.env_name = 'square-mh-low_dim'
    while tr.ctr['episodes'] < 6:
        if tr.ob is None:
            tr.reset_env()
        tr.env_step('n1')
    main, succ = tr.buffers['main'], tr.buffers['success']
    assert main.n == 3 * 7 + 3 * 10 == tr.ctr['env_steps']
    assert tr.ctr['warmup_successes'] == 3 and tr.ctr['warmup_failures'] == 3
    assert [L for _, L in succ.episodes] == [7, 7, 7]
    # Success buffer rows are exactly the successful episodes' rows, with chunks-to-success filled in.
    k = 0
    for ep in range(6):
        L = 7 if ep % 2 == 0 else 10
        slots = main.slot(np.arange(k, k + L))
        ncs = main.data['num_chunks_to_success'][slots]
        if ep % 2 == 0:
            assert np.array_equal(ncs, np.ceil((L - np.arange(L)) / H))
            assert main.data['masks'][slots][-1] == 0 and main.data['terminals'][slots][-1] == 1
            assert main.data['rewards'][slots][-1] == 0 and np.all(main.data['rewards'][slots][:-1] == -1)
        else:
            assert np.all(ncs == -1)
            assert main.data['masks'][slots][-1] == 1 and main.data['terminals'][slots][-1] == 1
        k += L
    for (s, L) in succ.episodes:
        ss = succ.slot(np.arange(s, s + L))
        ids = succ.data['observations'][ss, 0]
        src = main.slot(np.arange(ids[0] - 10 ** 7, ids[0] - 10 ** 7 + L).astype(int))
        for f in FIELDS:
            assert np.array_equal(succ.data[f][ss], main.data[f][src]), f


# ---- (e) a complete run exits 0 without work -------------------------------------------------------
def test_restart_of_complete_run_exits_zero(tmp_path):
    run_dir = tmp_path / 'done_run'
    store = CheckpointStore(str(run_dir / 'ckpt'))
    store.load()
    store.save({'phase': 'done'}, {}, {}, meta=dict(phase='done', ctr={'env_steps': 6000}))
    before = sorted(os.listdir(run_dir)), open(run_dir / 'ckpt' / 'LATEST').read()
    env = dict(os.environ, WANDB_MODE='disabled', JAX_PLATFORMS='cpu')
    t0 = time.time()
    p = subprocess.run([sys.executable, 'train_v1.py', '--mode=online', f'--run_dir={run_dir}',
                        '--init_actor=/nonexistent.pkl'], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=300)
    dt = time.time() - t0
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert 'RUN_ALREADY_COMPLETE' in p.stdout
    assert 'PHASE_BEGIN' not in p.stdout and 'CKPT_BEGIN' not in p.stdout
    assert (sorted(os.listdir(run_dir)), open(run_dir / 'ckpt' / 'LATEST').read()) == before
    print(f'complete-run restart: exit 0 in {dt:.1f}s (mostly imports)')
