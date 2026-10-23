"""geo1 end to end (square, tiny step counts): the E1-E4 command shapes through every phase, a preempted and
resumed E4 run (pretrain and joint) against the uninterrupted one, --mode=probe_ckpt on a finished run dir
(left byte-identical), and the default arb_bsc TD1 run against the parent commit.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set. Subprocesses run with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (tests/e2e_v1_interrupt.launch).
"""
import hashlib
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

from e2e_v1_interrupt import chain, flag, launch, progress_at_least  # noqa: E402

PARENT = os.environ.get('GEO1_PARENT_COMMIT', '8f1d09e')  # the commit geo1 branched from
H = 5
COMMON = ['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--n_eval_envs=2', '--eval_episodes=1',
          '--log_interval=50', '--probe_states=256', '--seed=0', '--ckpt_minutes=0.1']
ONLINE = ['--mode=online', '--warmup_steps=1000', '--pretrain_updates=200', '--online_steps=3000',
          '--eval_interval=1500', '--buffer_cap=90000']
TD1 = ['--method=arb_bsc', '--agent.critic_target_bon=False']
LABELS = ['hard', 'twohot_last', 'twohot_last2', 'twohot_all']
NEW_KEYS = ['critic/target_clip_frac', 'critic/target_u_mean', 'critic/readout_oor_frac', 'critic/label_clip_pos1',
            'critic/label_clip_pos2', 'critic/label_clip_pos3']


def e_args(label, cands):
    return [*TD1, '--agent.arb_readout=sample_mean', '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7',
            f'--agent.arb_label={label}', '--replay_rng=separate', f'--probe_candidates={cands}']


def rows(run_dir):
    with open(os.path.join(run_dir, 'metrics.jsonl')) as f:
        return [{k: v for k, v in json.loads(l).items() if k != 'time'} for l in f]


def tree_hash(root):
    """sha256 over every path, file content and mode below root (directories included)."""
    h = hashlib.sha256()
    for d, dirs, files in sorted(os.walk(root)):
        dirs.sort()
        h.update(f'D {os.path.relpath(d, root)} {os.stat(d).st_mode}\n'.encode())
        for n in sorted(files):
            p = os.path.join(d, n)
            with open(p, 'rb') as f:
                h.update(f'F {os.path.relpath(p, root)} {os.stat(p).st_mode} '.encode() + hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def run(args, d, name, cwd=None):
    """train_v1.py once (uninterrupted), in `cwd` (default: this repo)."""
    log, flag_path = str(d / f'{name}.log'), str(d / f'{name}.flag')
    if cwd is None:
        return launch(args, log, flag_path)
    env = dict(os.environ, WANDB_MODE='disabled', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true')
    env.pop('AE_PREEMPT_FLAG', None)
    with open(log, 'w') as f:
        p = subprocess.run([sys.executable, 'train_v1.py', *args], cwd=cwd, env=env, stdout=f,
                           stderr=subprocess.STDOUT, timeout=1800)
    return dict(rc=p.returncode, lines=open(log).read().splitlines())


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo1')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    actor = str(d / 'bc' / 'final_actor.pkl')
    cands = str(d / 'cands.npz')
    r = run(['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--probe_states=256',
             '--mode=make_probe_candidates', f'--init_actor={actor}', f'--out={cands}'], d, 'cands')
    assert r['rc'] == 0, r['lines'][-30:]
    return d, actor, cands


@pytest.fixture(scope='module')
def runs(setup):
    """E1-E4 uninterrupted, an interrupted E4 chain, the default TD1 run and the same at the parent commit."""
    d, actor, cands = setup
    parent = d / 'parent_src'
    tar = subprocess.run(['git', 'archive', PARENT], cwd=REPO, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(parent)
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = {f'E_{lab}': (lambda lab=lab: run([*online, *e_args(lab, cands), f'--run_dir={d / ("E_" + lab)}'],
                                             d, f'E_{lab}'))
            for lab in LABELS}
    jobs['default'] = lambda: run([*online, *TD1, f'--run_dir={d / "default"}'], d, 'default')
    jobs['parent'] = lambda: run([*online, *TD1, f'--run_dir={d / "parent"}'], d, 'parent', cwd=str(parent))
    interrupts = [flag('during pretrain updates', progress_at_least('pretrain', 'updates', 100)),
                  flag('during joint updates', progress_at_least('joint', 'env_steps', 2000))]
    jobs['chain'] = lambda: chain('E4I', [*online, *e_args('twohot_all', cands), f'--run_dir={d / "E4I"}'],
                                  str(d / 'E4I.log'), str(d / 'E4I.flag'), interrupts)
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out


@pytest.mark.parametrize('label', LABELS)
def test_e_shapes_run_every_phase(label, runs):
    d, out = runs
    r = out[f'E_{label}']
    assert r['rc'] == 0, r['lines'][-40:]
    m = rows(d / f'E_{label}')
    cfg = [x for x in m if x['kind'] == 'config']
    assert len(cfg) == 1
    assert cfg[0]['agent']['arb_label'] == label and cfg[0]['agent']['arb_readout'] == 'sample_mean'
    assert cfg[0]['agent']['arb_readout_temp'] == 0.7 and cfg[0]['agent']['critic_target_bon'] is False
    assert len(cfg[0]['probe_candidates']['sha256']) == 64 and cfg[0]['replay_rng'] == 'separate'
    assert [x['phase'] for x in m if x['kind'] == 'phase_end'] == ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    for kind in ('pretrain', 'joint'):
        tr = [x for x in m if x['kind'] == kind]
        assert tr, kind
        for x in tr:
            assert np.isfinite(x['critic/critic_loss']), x
            for k in NEW_KEYS:
                assert np.isfinite(x[k]), k
            assert x['critic/readout_oor_frac'] == 0.0
    rd = [x for x in m if x['kind'] == 'readout']
    assert rd and all(np.isfinite(x['critic/q_readout']) for x in rd)
    evals = [x for x in m if x['kind'] in ('eval', 'pretrain_eval')]
    assert len(evals) == 3
    for x in evals:
        for k in ('probe/tie_rate', 'probe/tie_max', 'probe_fixed/tie_all', 'probe_fixed/tie_max',
                  'probe_fixed/score_mean', 'probe_fixed/cand_range_mean', 'probe_fixed/oor_frac',
                  'probe_stab/score_var_mean', 'probe_stab/same_argmax_frac'):
            assert np.isfinite(x[k]), k


def test_interrupted_resume_matches_uninterrupted(runs):
    d, out = runs
    res = out['chain']
    assert [e['rc'] for e in res] == [143, 143], res
    a, i = rows(d / 'E_twohot_all'), rows(d / 'E4I')
    assert len(a) == len(i)
    for x, y in zip(a, i):
        assert x == y, (x, y)


def test_probe_ckpt_is_read_only(runs, setup):
    d, out = runs
    _, actor, cands = setup
    src = d / 'E_twohot_all'
    before = tree_hash(src)
    r = run(['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--probe_states=256', '--seed=0',
             '--mode=probe_ckpt', f'--src_run_dir={src}', f'--run_dir={d / "probe_out"}',
             f'--probe_candidates={cands}', *e_args('twohot_all', cands)[:-2]], d, 'probe_ckpt')
    assert r['rc'] == 0, r['lines'][-40:]
    assert tree_hash(src) == before
    res = json.load(open(d / 'probe_out' / 'probe_ckpt.json'))
    assert res['phase'] == 'done' and res['step'] == 3000 and res['gen'] >= 1
    # The final eval row of the source run probed the same agent with the same keys: identical values.
    last = [x for x in rows(src) if x['kind'] == 'eval'][-1]
    for k in [k for k in last if k.startswith(('probe/', 'probe_fixed/', 'probe_stab/'))]:
        assert res[k] == last[k], (k, res[k], last[k])
    assert sorted(os.listdir(d / 'probe_out')) == ['probe_ckpt.json']


def test_defaults_match_parent_commit(runs):
    d, out = runs
    for k in ('default', 'parent'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    new, old = rows(d / 'default'), rows(d / 'parent')
    assert [(x['kind'], x['step']) for x in new] == [(x['kind'], x['step']) for x in old]
    added = set()
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            y = dict(y, init_actor=None)
            x = dict(x, init_actor=None)
        for k, v in y.items():
            assert x[k] == v, (x['kind'], x['step'], k, x[k], v)
        added |= set(x) - set(y)
    print('keys added under defaults:', sorted(added))
    assert added <= {'probe/tie_max', 'critic/q_readout'}, added
