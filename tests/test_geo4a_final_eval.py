"""geo4a: --mode=final_eval (square, tiny sizes) and evaluate_parallel's optional per-episode return.

Source runs (tiny online runs from one tiny BC actor): sampled-readout ARB-QC E1 (this commit), the same at the base
commit (rows must match: training and in-training eval unchanged), sampled-readout ARB-BSC twohot_last at the geo1
commit (an old run dir: config row without the later agent keys), and a run stopped by the preempt flag during
warm-up (phase != done). final_eval on 8 seeds x 2 episodes (2 chunks of 4 envs): shapes / denominators, N1 and BO32
on identical initial states, two runs identical, an interrupted + resumed run identical, refusals with the source
left byte-identical. With the sq3 checkpoint present: wall time of one real-size chunk (32 envs x 4 episodes, both
rules).

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import flag, launch, starts_with  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, rows, run, tree_hash  # noqa: E402

BASE = os.environ.get('GEO4A_BASE_COMMIT', 'eb39863')  # the commit geo4a branched from
OLD = os.environ.get('GEO4A_OLD_COMMIT', '22bf78c')    # geo1 era (geo-E1..E9 runs)
SQ3 = '/data/user_data/vineethb/Discrete-Critics/geo/geo-sq3-qc-E10-model-prefix-fixed-label-s0'
QC = ['--method=arb_qc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
      '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--replay_rng=separate']
BSC = ['--method=arb_bsc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
       '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--agent.arb_label=twohot_last',
       '--replay_rng=separate']
ENV = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--seed=0']
SMALL = ['--mode=final_eval', '--final_n_seeds=8', '--n_eval_envs=4', '--eval_episodes=2']
NONDET = {'wall_s'}


def final_eval(d, name, src, agent_args, extra=(), interrupt=None):
    args = [*ENV, *SMALL, *agent_args, f'--src_run_dir={src}', f'--run_dir={d / name}', *extra]
    return launch(args, str(d / f'{name}.log'), str(d / f'{name}.flag'), interrupt)


def load(path):
    return {k: v for k, v in json.load(open(path)).items() if k not in NONDET}


@pytest.fixture(scope='module')
def srcs(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo4a')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    actor = str(d / 'bc' / 'final_actor.pkl')
    trees = {}
    for name, commit in (('base', BASE), ('old', OLD)):
        trees[name] = d / f'{name}_src'
        tar = subprocess.run(['git', 'archive', commit], cwd=REPO, capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(trees[name])
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = dict(
        qc=lambda: run([*online, *QC, f'--run_dir={d / "qc"}'], d, 'qc'),
        qc_base=lambda: run([*online, *QC, f'--run_dir={d / "qc_base"}'], d, 'qc_base', cwd=str(trees['base'])),
        bsc_old=lambda: run([*online, *BSC, f'--run_dir={d / "bsc_old"}'], d, 'bsc_old', cwd=str(trees['old'])),
        unfinished=lambda: launch([*online, *QC, f'--run_dir={d / "unfinished"}'], str(d / 'unfinished.log'),
                                  str(d / 'unfinished.flag'), flag('in warmup', starts_with('PHASE_BEGIN warmup'),
                                                                   delay=5.0)))
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    for k in ('qc', 'qc_base', 'bsc_old'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    assert out['unfinished']['rc'] == 143, out['unfinished']['lines'][-40:]
    return d, out


@pytest.fixture(scope='module')
def evals(srcs):
    d, _ = srcs
    hashes = {k: tree_hash(d / k) for k in ('qc', 'bsc_old')}
    jobs = dict(
        A=lambda: final_eval(d, 'A', d / 'qc', QC[:-1]),
        B=lambda: final_eval(d, 'B', d / 'qc', QC[:-1]),
        I=lambda: final_eval(d, 'I', d / 'qc', QC[:-1],
                             interrupt=flag('after the first part', starts_with('FINAL_EVAL_PART '))),
        bsc=lambda: final_eval(d, 'bsc', d / 'bsc_old', BSC[:-1]))
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    parts_after_interrupt = sorted(os.listdir(d / 'I' / 'final_eval_parts'))
    out['I_final_written'] = os.path.exists(d / 'I' / 'final_eval.json')
    out['I2'] = final_eval(d, 'I', d / 'qc', QC[:-1])
    return d, out, hashes, parts_after_interrupt


@pytest.mark.parametrize('name, src', [('A', 'qc'), ('bsc', 'bsc_old')])
def test_final_eval_shapes(name, src, evals):
    d, out, hashes, _ = evals
    r = out[name]
    assert r['rc'] == 0, r['lines'][-40:]
    assert tree_hash(d / src) == hashes[src]
    res = json.load(open(d / name / 'final_eval.json'))
    assert res['phase'] == 'done' and res['ctr']['env_steps'] == 3000 and res['gen'] >= 1
    assert res['seeds'] == list(range(100000, 100008)) and res['n_eval_envs'] == 4 and res['eval_episodes'] == 2
    for rule in ('n1', 'bon'):
        x = res[rule]
        S = np.array(x['per_episode_success'])
        assert S.shape == (8, 2) and set(S.ravel()) <= {0, 1}
        assert x['episodes'] == 16 and x['successes'] == S.sum() == sum(x['per_seed_successes'])
        assert x['per_seed_successes'] == S.sum(1).tolist() and x['success_rate'] == x['successes'] / 16
        lo, hi = x['wilson95']
        assert 0 <= lo <= x['success_rate'] <= hi <= 1
        assert np.array(x['per_episode_length']).shape == (8, 2) and min(np.ravel(x['per_episode_length'])) > 0
    p = res['paired']
    diff = np.array(res['bon']['per_seed_successes']) - np.array(res['n1']['per_seed_successes'])
    assert p['bon_minus_n1_successes'] == diff.sum() and abs(p['per_seed_diff_mean'] - diff.mean()) < 1e-12
    # N1 and BO32 start from identical states: first observation of every seed (and of every episode)
    assert p['same_first_obs_seeds'] == 8, p
    parts = {f: json.load(open(d / name / 'final_eval_parts' / f)) for f in os.listdir(d / name / 'final_eval_parts')}
    assert sorted(parts) == ['bon_0.json', 'bon_1.json', 'n1_0.json', 'n1_1.json']
    for c in (0, 1):
        n1, bon = parts[f'n1_{c}.json'], parts[f'bon_{c}.json']
        assert [e[0] for e in n1['first_obs_digest']] == [e[0] for e in bon['first_obs_digest']]
        assert len({e[0] for e in n1['first_obs_digest']}) == 4  # different seeds, different initial states
        assert n1['seeds'] == bon['seeds'] == list(range(100000 + 4 * c, 100004 + 4 * c))
    print(name, 'same initial obs (all episodes):', p['same_initial_obs_episodes'], '/ 16',
          {r: res[r]['successes'] for r in ('n1', 'bon')})
    line = [l for l in r['lines'] if l.startswith('FINAL_EVAL ')]
    assert len(line) == 1 and 'per_seed_successes' not in line[0] and 'per_episode' not in line[0]


def test_two_runs_identical(evals):
    d, out, _, _ = evals
    assert out['A']['rc'] == 0 and out['B']['rc'] == 0
    a, b = load(d / 'A' / 'final_eval.json'), load(d / 'B' / 'final_eval.json')
    assert a == b, {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def test_interrupted_resume_identical(evals):
    d, out, _, parts_after = evals
    assert out['I']['rc'] == 143, out['I']['lines'][-30:]
    assert any(l.startswith('EXIT_PREEMPTED source=flag') for l in out['I']['lines'])
    assert 1 <= len(parts_after) < 4 and not out['I_final_written'], parts_after
    assert not [f for f in parts_after if f.endswith('.tmp')]
    assert out['I2']['rc'] == 0, out['I2']['lines'][-30:]
    skipped = [l for l in out['I2']['lines'] if l.startswith('FINAL_EVAL_PART_SKIP')]
    assert len(skipped) == len(parts_after)
    a, i = load(d / 'A' / 'final_eval.json'), load(d / 'I' / 'final_eval.json')
    assert a == i, {k: (a.get(k), i.get(k)) for k in set(a) | set(i) if a.get(k) != i.get(k)}


def test_refusals_leave_source_unchanged(srcs, evals):
    d, out = srcs
    cases = dict(
        unfinished=(d / 'unfinished', QC[:-1], [], 'phase='),
        inside=(d / 'qc', QC[:-1], [f'--run_dir={d / "qc" / "fe"}'], 'lies inside'),
        indivisible=(d / 'qc', QC[:-1], ['--n_eval_envs=3'], 'multiple'),
        config=(d / 'qc', [*QC[:-2], '--agent.arb_readout_temp=0.5'], [], 'agent.arb_readout_temp'),
        old_config=(d / 'bsc_old', [*BSC[:-2], '--agent.arb_label=hard'], [], 'agent.arb_label'))
    for name, (src, agent_args, extra, msg) in cases.items():
        before = tree_hash(src)
        r = final_eval(d, f'refuse_{name}', src, agent_args, extra)
        lines = [l for l in r['lines'] if l.startswith('FINAL_EVAL_REFUSED')]
        assert r['rc'] == 2 and lines and msg in lines[0], (name, r['lines'][-20:])
        assert tree_hash(src) == before, name
        assert not os.path.exists(d / f'refuse_{name}' / 'final_eval.json')
        print(name, lines[0][:200])


def test_training_rows_match_base_commit(srcs):
    d, out = srcs
    new, old = rows(d / 'qc'), rows(d / 'qc_base')
    assert len(new) == len(old)
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            x, y = dict(x, init_actor=None), dict(y, init_actor=None)
        assert x == y, (x['kind'], x['step'])
    a, b = json.load(open(d / 'qc' / 'args.json')), json.load(open(d / 'qc_base' / 'args.json'))
    assert {k: v for k, v in a.items() if k != 'run_dir'} == {k: v for k, v in b.items() if k != 'run_dir'}


# ---- evaluate_parallel: default return unchanged ------------------------------------------------------------
class FakeVecEnv:
    """n envs; episode e of env i lasts 3 + (i + 2 e) % 5 steps and succeeds iff (i + e) is even."""

    def __init__(self, n, obs_dim=3):
        self.n, self.obs_dim, self.e = n, obs_dim, -1

    def reset(self):
        self.e += 1
        self.t = np.zeros(self.n, dtype=int)
        return np.arange(self.n)[:, None] * 10.0 + self.e + np.zeros((self.n, self.obs_dim)), {}

    def step(self, actions):
        self.t += 1
        lens = 3 + (np.arange(self.n) + 2 * self.e) % 5
        term = self.t >= lens
        succ = ((np.arange(self.n) + self.e) % 2 == 0) & term
        obs = np.arange(self.n)[:, None] * 10.0 + self.e + self.t[:, None] + np.zeros((self.n, self.obs_dim))
        info = dict(success=succ.astype(float), episode=dict(length=self.t.astype(float)))
        return obs, np.zeros(self.n), term, np.zeros(self.n, bool), info


class ZeroPolicy:
    def sample_actions(self, observations, rng=None):
        return np.zeros((observations.shape[0], 2 * 2))


def _base_evaluation():
    src = subprocess.run(['git', 'show', f'{BASE}:evaluation.py'], cwd=REPO, capture_output=True, check=True).stdout
    spec = importlib.util.spec_from_loader('evaluation_base', loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(src, mod.__dict__)
    return mod


def test_evaluate_parallel_default_unchanged():
    import evaluation
    base = _base_evaluation()
    kw = dict(agent=ZeroPolicy(), num_eval_envs=4, eval_episodes=3, action_dim=2)
    np.random.seed(0)
    ref = base.evaluate_parallel(envs=FakeVecEnv(4), **kw)
    np.random.seed(0)
    new = evaluation.evaluate_parallel(envs=FakeVecEnv(4), **kw)
    np.random.seed(0)
    *with_eps, eps = evaluation.evaluate_parallel(envs=FakeVecEnv(4), return_episodes=True, **kw)
    assert len(ref) == len(new) == 3 and new[1:] == ref[1:] == ([], [])
    for got in (new, with_eps):
        assert dict(got[0]) == dict(ref[0]) and type(got[0]) is type(ref[0])
    i, e = np.meshgrid(np.arange(4), np.arange(3), indexing='ij')
    assert np.array_equal(eps['success'], ((i + e) % 2 == 0).astype(float))
    assert np.array_equal(eps['length'], 3 + (i + 2 * e) % 5)
    assert eps['first_obs'].shape == (4, 3, 3) and np.array_equal(eps['first_obs'][..., 0], i * 10.0 + e)
    assert abs(eps['success'].mean() - ref[0]['success']) < 1e-12


# ---- measure: one real-size chunk on the sq3 checkpoint ------------------------------------------------------
def _ckpt_hash(src):
    h = hashlib.sha256()
    gen = open(os.path.join(src, 'ckpt', 'LATEST')).read().strip()
    for p in ('args.json', 'metrics.jsonl', 'ckpt/LATEST', f'ckpt/{gen}/manifest.json', f'ckpt/{gen}/state.pkl'):
        h.update(p.encode() + hashlib.sha256(open(os.path.join(src, p), 'rb').read()).digest())
    return h.hexdigest()


@pytest.mark.skipif(not os.path.exists(os.path.join(SQ3, 'ckpt', 'LATEST')), reason='sq3 checkpoint not present')
def test_measure_real_size_chunk(tmp_path):
    before = _ckpt_hash(SQ3)
    agent = ['--method=arb_qc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
             '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--agent.arb_label=noisy_fixed',
             '--agent.arb_vocab=redundant', '--agent.arb_rollin=model_mix']
    r = launch([*ENV, '--mode=final_eval', '--final_n_seeds=32', '--n_eval_envs=32', *agent,
                f'--src_run_dir={SQ3}', f'--run_dir={tmp_path / "fe"}'], str(tmp_path / 'fe.log'),
               str(tmp_path / 'fe.flag'))
    assert r['rc'] == 0, r['lines'][-40:]
    assert _ckpt_hash(SQ3) == before
    res = json.load(open(tmp_path / 'fe' / 'final_eval.json'))
    parts = [l for l in r['lines'] if l.startswith('FINAL_EVAL_PART ')]
    print('MEASURE', *parts, f'chunk (both rules) wall_s={res["wall_s"]:.1f}',
          {k: res[k]['successes'] for k in ('n1', 'bon')}, 'episodes', res['n1']['episodes'],
          'same_initial_obs', res['paired']['same_initial_obs_episodes'], sep='\n  ')
    assert res['n1']['episodes'] == 128 and res['eval_episodes'] == 4
