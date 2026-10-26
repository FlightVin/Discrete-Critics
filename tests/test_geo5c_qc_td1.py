"""geo5c: scalar QC with a TD1 target (--method=qc --agent.critic_target_bon=False).

critic_target_bon=False: the TD target's next chunk is ONE BC-flow sample clip(flow(s', N(0, I)), -1, 1) from the
critic_loss sample key, no critic selection; y = R + 0.99^h m mean_copies(Q_target(s', a')) as before.
critic_target_bon=True (default): bit-identical loss / grads to the base commit. End to end (square, horizon 5): the
TD1 command shape through every phase; the default qc run's rows and args.json identical to the base commit's; final_eval
of this commit accepts a base-commit qc run dir and refuses it with critic_target_bon=False.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from agents import v1_config  # noqa: E402
from agents.acfql import ACFQLAgent  # noqa: E402
from utils.flax_utils import TrainState  # noqa: E402

BASE = os.environ.get('GEO5C_BASE_COMMIT', 'b2fdb26')  # the commit geo5c branched from
OBS, ADIM, BS, GAMMA = 5, 2, 6, 0.99


def make_agent(H, td1, cls=ACFQLAgent, seed=0):
    config = v1_config('qc')
    config.update(dict(value_hidden_dims=(32, 32), actor_hidden_dims=(32, 32), flow_steps=3, horizon_length=H))
    if td1 is not None:
        config['critic_target_bon'] = not td1
    else:
        del config['critic_target_bon']  # the base commit's config has no such key
    ex_o, ex_a = np.zeros(OBS, np.float32), np.zeros(ADIM, np.float32)
    agent = cls.create(seed, ex_o, ex_a, config)
    other = cls.create(seed + 1, ex_o, ex_a, config)
    params = dict(agent.network.params)
    params['modules_target_critic'] = other.network.params['modules_critic']  # target != online critic
    return agent.replace(network=agent.network.replace(params=params))


def make_batch(H, seed=0):
    """Rows: 0 success (m=0), 1 timeout (terminal, m=1), 2-5 non-terminal."""
    rs = np.random.RandomState(seed)
    terminals = np.zeros((BS, H)); terminals[0, -1] = terminals[1, -1] = 1.0
    masks = np.ones((BS, H)); masks[0, -1] = 0.0
    valid = np.ones((BS, H))
    rewards = np.cumsum(-np.ones((BS, H)) * GAMMA ** np.arange(H), axis=1)
    rewards[0, -1] += GAMMA ** (H - 1)  # success step reward 0
    rewards += rs.uniform(-5, 0, (BS, 1))
    return dict(
        observations=jnp.asarray(rs.normal(size=(BS, OBS)), jnp.float32),
        next_observations=jnp.asarray(rs.normal(size=(BS, H, OBS)), jnp.float32),
        actions=jnp.asarray(rs.uniform(-1, 1, (BS, H, ADIM)), jnp.float32),
        rewards=jnp.asarray(rewards, jnp.float32), masks=jnp.asarray(masks, jnp.float32),
        terminals=jnp.asarray(terminals, jnp.float32), valid=jnp.asarray(valid, jnp.float32),
        num_chunks_to_success=jnp.asarray(np.where(np.arange(BS) == 0, 0.0, -1.0), jnp.float32))


# ---- 1. one actor chunk per next state, no critic selection ---------------------------------------------
def test_td1_draws_one_flow_chunk_no_selection(monkeypatch):
    H = 5
    agent, batch = make_agent(H, td1=True), make_batch(H)
    flow_calls, selects = [], []
    orig_flow = ACFQLAgent.compute_flow_actions.__wrapped__
    orig_select = TrainState.select

    def rec_flow(self, observations, noises):
        flow_calls.append((observations.shape, noises.shape))
        return orig_flow(self, observations, noises)

    def rec_select(self, name):
        f = orig_select(self, name)

        def g(*args, **kw):
            selects.append((name, args[0].shape, kw.get('params') is not None))
            return f(*args, **kw)
        return g

    monkeypatch.setattr(ACFQLAgent, 'compute_flow_actions', rec_flow)
    monkeypatch.setattr(ACFQLAgent, 'sample_actions', lambda *a, **k: pytest.fail('best-of-n in the TD target'))
    monkeypatch.setattr(TrainState, 'select', rec_select)
    loss, _ = agent.critic_loss(batch, agent.network.params, jax.random.PRNGKey(3))
    assert np.isfinite(float(loss))
    assert flow_calls == [((BS, OBS), (BS, ADIM * H))], flow_calls
    critic = [s for s in selects if s[0] == 'critic']
    target = [s for s in selects if s[0] == 'target_critic']
    # The online critic is called once: on the batch (s, a) with grad params (the loss), never for selection.
    assert critic == [('critic', (BS, OBS), True)], selects
    assert target == [('target_critic', (BS, OBS), False)], selects


# ---- 2. TD target against numpy ---------------------------------------------------------------------------
@pytest.mark.parametrize('H', [5, 8])
def test_td1_target_numpy(H):
    agent, batch, rng = make_agent(H, td1=True), make_batch(H), jax.random.PRNGKey(11)
    loss, _ = agent.critic_loss(batch, agent.network.params, rng)

    _, sample_rng = jax.random.split(rng)
    next_obs = batch['next_observations'][:, -1, :]
    noises = jax.random.normal(sample_rng, (BS, ADIM * H))
    a_next = np.clip(np.asarray(agent.compute_flow_actions(next_obs, noises)), -1, 1)
    qt = np.asarray(agent.network.select('target_critic')(next_obs, jnp.asarray(a_next)), np.float64)  # (2, BS)
    R, m = np.asarray(batch['rewards'][:, -1], np.float64), np.asarray(batch['masks'][:, -1], np.float64)
    y = R + GAMMA ** H * m * qt.mean(0)
    assert m[0] == 0 and m[1] == 1 and batch['terminals'][1, -1] == 1  # success cut, timeout bootstraps
    assert y[0] == pytest.approx(R[0]) and y[1] != pytest.approx(R[1])
    q = np.asarray(agent.network.select('critic')(batch['observations'], batch['actions'].reshape(BS, -1)),
                   np.float64)
    want = (np.square(q - y[None]) * np.asarray(batch['valid'][:, -1])[None]).mean()
    np.testing.assert_allclose(float(loss), want, rtol=1e-5)

    # Differs from the best-of-32 target on the same key (the flag changes the target).
    bon = make_agent(H, td1=False)
    loss_bon, _ = bon.critic_loss(batch, bon.network.params, rng)
    assert abs(float(loss_bon) - float(loss)) > 1e-6


# ---- 3. default bit-identical to the base commit ----------------------------------------------------------
def _base_acfql():
    src = subprocess.run(['git', 'show', f'{BASE}:agents/acfql.py'], cwd=REPO, capture_output=True,
                         check=True).stdout
    spec = importlib.util.spec_from_loader('acfql_base', loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(src, mod.__dict__)
    return mod


def test_default_bit_identical_to_base():
    base = _base_acfql()
    assert 'critic_target_bon' not in base.get_config()
    H = 5
    batch, rng = make_batch(H), jax.random.PRNGKey(5)
    new, old = make_agent(H, td1=False), make_agent(H, td1=None, cls=base.ACFQLAgent)
    assert new.config['critic_target_bon'] is True
    for a, b in zip(jax.tree_util.tree_leaves(new.network.params), jax.tree_util.tree_leaves(old.network.params)):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def lg(agent):
        f = lambda p: agent.total_loss(batch, p, rng=rng)  # noqa: E731
        (loss, info), grads = jax.value_and_grad(f, has_aux=True)(agent.network.params)
        return jax.device_get((loss, info, grads))

    (l1, i1, g1), (l0, i0, g0) = lg(new), lg(old)
    assert np.asarray(l1).tobytes() == np.asarray(l0).tobytes()
    assert set(i1) == set(i0) and all(np.asarray(i1[k]).tobytes() == np.asarray(i0[k]).tobytes() for k in i1)
    for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g0)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_strip_is_per_agent():
    import train_v1 as T
    assert T.strip_later_agent_defaults(dict(agent_name='acfql', critic_target_bon=True, lr=1)) == dict(
        agent_name='acfql', lr=1)
    assert T.strip_later_agent_defaults(dict(agent_name='acfql', critic_target_bon=False)) == dict(
        agent_name='acfql', critic_target_bon=False)
    for name in ('acbsc', 'acarb'):  # the key predates geo5c there: kept as before
        assert T.strip_later_agent_defaults(dict(agent_name=name, critic_target_bon=True)) == dict(
            agent_name=name, critic_target_bon=True)


# ---- 4/5. end to end -------------------------------------------------------------------------------------
from e2e_v1_interrupt import launch  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, rows, run, tree_hash  # noqa: E402

QC = ['--method=qc', '--replay_rng=separate']
TD1 = ['--method=qc', '--agent.critic_target_bon=False', '--replay_rng=separate']
FE = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--seed=0', '--mode=final_eval', '--final_n_seeds=4',
      '--n_eval_envs=4', '--eval_episodes=1']


@pytest.fixture(scope='module')
def srcs(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo5c')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    tree = d / 'base_src'
    tar = subprocess.run(['git', 'archive', BASE], cwd=REPO, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(tree)
    online = [*COMMON, *ONLINE, f'--init_actor={d / "bc" / "final_actor.pkl"}']
    jobs = dict(
        td1=lambda: run([*online, *TD1, f'--run_dir={d / "td1"}'], d, 'td1'),
        qc=lambda: run([*online, *QC, f'--run_dir={d / "qc"}'], d, 'qc'),
        qc_base=lambda: run([*online, *QC, f'--run_dir={d / "qc_base"}'], d, 'qc_base', cwd=str(tree)))
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    for k in jobs:
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    return d, out


def test_td1_e2e(srcs):
    d, out = srcs
    lines = out['td1']['lines']
    for ph in ('warmup', 'pretrain', 'joint'):
        assert any(l.startswith(f'PHASE_BEGIN {ph}') for l in lines), ph
    rs = rows(d / 'td1')
    assert rs[0]['kind'] == 'config' and rs[0]['agent']['critic_target_bon'] is False
    assert rs[0]['agent']['agent_name'] == 'acfql' and rs[0]['agent']['actor_type'] == 'best-of-n'
    assert json.load(open(d / 'td1' / 'args.json'))['agent']['critic_target_bon'] is False
    evals = [r for r in rs if r['kind'] != 'config' and any('success' in k for k in r)]
    assert evals, 'no eval rows'
    for r in rs[1:]:
        for k, v in r.items():
            if isinstance(v, float):
                assert np.isfinite(v), (r['kind'], k)
    # The TD1 target changes training (different from the default qc run from the same actor / seed).
    assert rows(d / 'td1')[1:] != rows(d / 'qc')[1:]


def test_default_qc_rows_match_base_commit(srcs):
    d, _ = srcs
    new, old = rows(d / 'qc'), rows(d / 'qc_base')
    assert len(new) == len(old)
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            x, y = dict(x, init_actor=None), dict(y, init_actor=None)
            assert 'critic_target_bon' not in x['agent']
        assert x == y, (x['kind'], x.get('step'))
    a, b = json.load(open(d / 'qc' / 'args.json')), json.load(open(d / 'qc_base' / 'args.json'))
    assert {k: v for k, v in a.items() if k != 'run_dir'} == {k: v for k, v in b.items() if k != 'run_dir'}


def test_final_eval_against_base_qc_run(srcs):
    d, _ = srcs
    h = tree_hash(d / 'qc_base')
    ok = launch([*FE, '--method=qc', f'--src_run_dir={d / "qc_base"}', f'--run_dir={d / "fe_ok"}'],
                str(d / 'fe_ok.log'), str(d / 'fe_ok.flag'))
    assert ok['rc'] == 0, ok['lines'][-30:]
    fe = json.load(open(d / 'fe_ok' / 'final_eval.json'))
    assert {'n1', 'bon'} <= set(fe) and np.isfinite(fe['n1']['success_rate'])
    bad = launch([*FE, '--method=qc', '--agent.critic_target_bon=False', f'--src_run_dir={d / "qc_base"}',
                  f'--run_dir={d / "fe_bad"}'], str(d / 'fe_bad.log'), str(d / 'fe_bad.flag'))
    assert bad['rc'] == 2, bad['lines'][-30:]
    assert any('FINAL_EVAL_REFUSED' in l and 'agent.critic_target_bon' in l for l in bad['lines']), bad['lines'][-10:]
    assert tree_hash(d / 'qc_base') == h
