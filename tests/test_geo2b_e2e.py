"""geo2b end to end (square, tiny step counts): the E5-E9 command shapes through every phase, a preempted and
resumed E6 run (pretrain and joint) against the uninterrupted one, and E1 / E4 runs against the parent commit
(geo1 paths unchanged). The default arb_bsc run against 8f1d09e is tests/test_geo1_e2e.py.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set. Subprocesses run with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (tests/e2e_v1_interrupt.launch).
"""
import io
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

from e2e_v1_interrupt import chain, flag, progress_at_least  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, H, e_args, rows, run  # noqa: E402

PARENT = os.environ.get('GEO2B_PARENT_COMMIT', '22bf78c')  # the commit geo2b branched from
GEO = {'E5': ('corrective', 'corrective'), 'E6': ('corrective', 'model_mix'), 'E7': ('corrective', 'fixed'),
       'E8': ('noisy_fixed', 'fixed'), 'E9': ('noisy_resid', 'fixed')}
NEW_KEYS = ['critic/oov_label', 'critic/oov_rollin', 'critic/model_used_frac', 'critic/model_clamp_frac',
            'critic/label_mean_bias', 'critic/target_clip_frac', 'critic/target_u_mean', 'critic/readout_oor_frac']


def g_args(variant, cands):
    label, rollin = GEO[variant]
    # As the generated recipes: the geo1 E-shape with the redundant vocabulary and the variant's label / roll-in.
    return [*e_args(label, cands), '--agent.arb_vocab=redundant', f'--agent.arb_rollin={rollin}']


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo2b')
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
    """E5-E9 uninterrupted, an interrupted E6 chain, E1 / E4 here and at the parent commit."""
    d, actor, cands = setup
    parent = d / 'parent_src'
    tar = subprocess.run(['git', 'archive', PARENT], cwd=REPO, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(parent)
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = {v: (lambda v=v: run([*online, *g_args(v, cands), f'--run_dir={d / v}'], d, v)) for v in GEO}
    interrupts = [flag('during pretrain updates', progress_at_least('pretrain', 'updates', 100)),
                  flag('during joint updates', progress_at_least('joint', 'env_steps', 2000))]
    jobs['chain'] = lambda: chain('E6I', [*online, *g_args('E6', cands), f'--run_dir={d / "E6I"}'],
                                  str(d / 'E6I.log'), str(d / 'E6I.flag'), interrupts)
    for name, label in (('E1', 'hard'), ('E4', 'twohot_all')):
        args = [*online, *e_args(label, cands)]
        jobs[name] = lambda n=name, a=args: run([*a, f'--run_dir={d / n}'], d, n)
        jobs[name + 'p'] = lambda n=name, a=args: run([*a, f'--run_dir={d / (n + "p")}'], d, n + 'p',
                                                      cwd=str(parent))
    with ThreadPoolExecutor(6) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out


@pytest.mark.parametrize('variant', list(GEO))
def test_geo_shapes_run_every_phase(variant, runs):
    d, out = runs
    r = out[variant]
    assert r['rc'] == 0, r['lines'][-40:]
    m = rows(d / variant)
    cfg = [x for x in m if x['kind'] == 'config']
    assert len(cfg) == 1
    agent = cfg[0]['agent']
    assert (agent['arb_label'], agent['arb_rollin'], agent['arb_vocab']) == (*GEO[variant], 'redundant')
    assert agent['arb_readout'] == 'sample_mean' and agent['arb_readout_samples'] == 10
    assert agent['arb_readout_temp'] == 0.7 and agent['critic_target_bon'] is False
    assert [x['phase'] for x in m if x['kind'] == 'phase_end'] == ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    for kind in ('pretrain', 'joint'):
        tr = [x for x in m if x['kind'] == kind]
        assert tr, kind
        for x in tr:
            assert np.isfinite(x['critic/critic_loss']), x
            for k in NEW_KEYS:
                assert np.isfinite(x[k]), k
            assert x['critic/oov_label'] == 0 and x['critic/oov_rollin'] == 0, x
            if variant == 'E6':
                assert 0.3 < x['critic/model_used_frac'] < 0.7, x
            else:
                assert x['critic/model_used_frac'] == 0, x
    rd = [x for x in m if x['kind'] == 'readout']
    assert rd and all(np.isfinite(x['critic/q_readout']) and np.isfinite(x['critic/greedy_oor_frac']) for x in rd)
    evals = [x for x in m if x['kind'] in ('eval', 'pretrain_eval')]
    assert len(evals) == 3
    for x in evals:
        for k in ('probe/tie_rate', 'probe_fixed/tie_all', 'probe_fixed/score_mean', 'probe_fixed/oor_frac',
                  'probe_stab/score_var_mean', 'probe_stab/same_argmax_frac'):
            assert np.isfinite(x[k]), k


def test_e6_interrupted_resume_matches_uninterrupted(runs):
    d, out = runs
    res = out['chain']
    assert [e['rc'] for e in res] == [143, 143], res
    a, i = rows(d / 'E6'), rows(d / 'E6I')
    assert len(a) == len(i)
    for x, y in zip(a, i):
        assert x == y, (x, y)


@pytest.mark.parametrize('name', ['E1', 'E4'])
def test_geo1_paths_match_parent_commit(name, runs):
    d, out = runs
    for k in (name, name + 'p'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    new, old = rows(d / name), rows(d / (name + 'p'))
    assert len(new) == len(old)
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            x, y = dict(x, init_actor=None), dict(y, init_actor=None)
        assert x == y, (x['kind'], x['step'], {k: (x.get(k), y.get(k)) for k in set(x) | set(y)
                                               if x.get(k) != y.get(k)})
