"""geo3c end to end (square, tiny step counts): sampled-readout ARB-QC-better with the real command shapes
(--method=arb_qc --agent.critic_target_bon=False + sample_mean flags + recipe flags) for E1, E2, E5, E8, E10, E11
(horizon 5) and E1 (horizon 8) through every phase; preempted and resumed QC E1 / E11 runs (pretrain and joint)
against the uninterrupted ones; args.json / config row carry arb_value='qc' and discount 0.99 and a resume with
another value family is refused; BSC E1 / E6 / E9 rows against the base commit and BSC E10 against geo3a's commit.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set. Subprocesses run with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (tests/e2e_v1_interrupt.launch).
"""
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

from e2e_v1_interrupt import chain, flag, progress_at_least  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, rows, run  # noqa: E402

BASE = os.environ.get('GEO3C_BASE_COMMIT', '6551f0d')    # the commit geo3c branched from
GEO3A = os.environ.get('GEO3C_GEO3A_COMMIT', '3e7789d')  # geo3a: E10 / E11 allowed (BSC), else = BASE
RECIPE = {'E1': ('hard', None), 'E2': ('twohot_last', None), 'E5': ('corrective', 'corrective'),
          'E6': ('corrective', 'model_mix'), 'E8': ('noisy_fixed', 'fixed'), 'E9': ('noisy_resid', 'fixed'),
          'E10': ('noisy_fixed', 'model_mix'), 'E11': ('noisy_resid', 'model_mix')}
QC_RUNS = {'QE1': ('E1', 5), 'QE2': ('E2', 5), 'QE5': ('E5', 5), 'QE8': ('E8', 5), 'QE10': ('E10', 5),
           'QE11': ('E11', 5), 'QE1h8': ('E1', 8)}
QC_KEYS = ['critic/critic_loss', 'critic/target_q', 'critic/target_q_raw', 'critic/target_clip_hi_frac',
           'critic/target_clip_lo_frac', 'critic/readout_oor_frac', 'critic/target_u_mean']
GEO_KEYS = ['critic/oov_label', 'critic/oov_rollin', 'critic/label_mean_bias', 'critic/label_mean_bias_raw',
            'critic/model_used_frac']


def common(H):
    return [a if not a.startswith('--horizon_length=') else f'--horizon_length={H}' for a in COMMON]


def recipe_args(variant, method, cands):
    label, rollin = RECIPE[variant]
    out = [f'--method={method}', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
           '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', f'--agent.arb_label={label}',
           '--replay_rng=separate', f'--probe_candidates={cands}']
    if rollin is not None:
        out += ['--agent.arb_vocab=redundant', f'--agent.arb_rollin={rollin}']
    return out


def setup_h(d, H):
    """BC actor and fixed probe candidates for horizon H."""
    r = run([*common(H), '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / f"bc{H}"}'], d, f'bc{H}')
    assert r['rc'] == 0, r['lines'][-30:]
    actor, cands = str(d / f'bc{H}' / 'final_actor.pkl'), str(d / f'cands{H}.npz')
    r = run(['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--probe_states=256',
             '--mode=make_probe_candidates', f'--init_actor={actor}', f'--out={cands}'], d, f'cands{H}')
    assert r['rc'] == 0, r['lines'][-30:]
    return actor, cands


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo3c')
    with ThreadPoolExecutor(2) as ex:
        s5, s8 = ex.submit(setup_h, d, 5), ex.submit(setup_h, d, 8)
        setups = {5: s5.result(), 8: s8.result()}
    srcs = {}
    for name, commit in (('base', BASE), ('geo3a', GEO3A)):
        srcs[name] = d / f'{name}_src'
        tar = subprocess.run(['git', 'archive', commit], cwd=REPO, capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(srcs[name])

    def online(H, variant, method):
        actor, cands = setups[H]
        return [*common(H), *ONLINE, f'--init_actor={actor}', *recipe_args(variant, method, cands)]

    jobs = {}
    for name, (variant, H) in QC_RUNS.items():
        jobs[name] = lambda n=name, a=online(H, variant, 'arb_qc'): run([*a, f'--run_dir={d / n}'], d, n)
    interrupts = [flag('during pretrain updates', progress_at_least('pretrain', 'updates', 100)),
                  flag('during joint updates', progress_at_least('joint', 'env_steps', 2000))]
    for variant in ('E1', 'E11'):
        name = f'Q{variant}I'
        jobs[name] = lambda n=name, a=online(5, variant, 'arb_qc'): chain(
            n, [*a, f'--run_dir={d / n}'], str(d / f'{n}.log'), str(d / f'{n}.flag'), interrupts)
    for variant, src in (('E1', 'base'), ('E6', 'base'), ('E9', 'base'), ('E10', 'geo3a')):
        args = online(5, variant, 'arb_bsc')
        jobs[f'B{variant}'] = lambda n=f'B{variant}', a=args: run([*a, f'--run_dir={d / n}'], d, n)
        jobs[f'B{variant}p'] = lambda n=f'B{variant}p', a=args, s=src: run([*a, f'--run_dir={d / n}'], d, n,
                                                                           cwd=str(srcs[s]))
    with ThreadPoolExecutor(int(os.environ.get('GEO3C_E2E_WORKERS', 6))) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out, setups


@pytest.mark.parametrize('name', list(QC_RUNS))
def test_qc_shapes_run_every_phase(name, runs):
    d, out, _ = runs
    variant, H = QC_RUNS[name]
    r = out[name]
    assert r['rc'] == 0, r['lines'][-40:]
    m = rows(d / name)
    cfg = [x for x in m if x['kind'] == 'config']
    assert len(cfg) == 1
    agent = cfg[0]['agent']
    label, rollin = RECIPE[variant]
    assert agent['arb_value'] == 'qc' and agent['discount'] == 0.99 and agent['horizon_length'] == H
    assert agent['arb_length'] == 3 and agent['critic_target_bon'] is False and agent['arb_readout'] == 'sample_mean'
    assert agent.get('arb_label', 'hard') == label and agent.get('arb_rollin', 'canonical') == (rollin or 'canonical')
    args = json.load(open(d / name / 'args.json'))
    assert args['method'] == 'arb_qc' and args['discount'] == 0.99
    assert args['resolved_agent'] == dict(arb_value='qc', discount=0.99, target_family='ARB-QC-better')
    assert [x['phase'] for x in m if x['kind'] == 'phase_end'] == ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    for kind in ('pretrain', 'joint'):
        tr = [x for x in m if x['kind'] == kind]
        assert tr, kind
        for x in tr:
            for k in QC_KEYS + (GEO_KEYS if rollin else []):
                assert np.isfinite(x[k]), (k, x)
            assert x['critic/readout_oor_frac'] == 0.0 or rollin, x
            if rollin:
                assert x['critic/oov_label'] == 0 and x['critic/oov_rollin'] == 0, x
                assert (0.3 < x['critic/model_used_frac'] < 0.7) == (rollin == 'model_mix'), x
            assert -100.0 <= x['critic/target_q_raw'] <= 0.0 and 0.0 <= x['critic/target_q'] <= 0.999, x
    rd = [x for x in m if x['kind'] == 'readout']
    assert rd and all(np.isfinite(x['critic/q_readout']) and np.isfinite(x['critic/q_readout_raw']) and
                      abs(x['critic/q_readout_raw'] - (100 * x['critic/q_readout'] - 100)) < 1e-3 for x in rd)
    evals = [x for x in m if x['kind'] in ('eval', 'pretrain_eval')]
    assert len(evals) == 3
    for x in evals:
        for k in ('probe/tie_rate', 'probe_fixed/tie_all', 'probe_fixed/score_mean', 'probe_stab/same_argmax_frac'):
            assert np.isfinite(x[k]), k


@pytest.mark.parametrize('variant', ['E1', 'E11'])
def test_qc_interrupted_resume_matches_uninterrupted(variant, runs):
    d, out, _ = runs
    res = out[f'Q{variant}I']
    assert [e['rc'] for e in res] == [143, 143], [e['rc'] for e in res]
    a, i = rows(d / f'Q{variant}'), rows(d / f'Q{variant}I')
    assert len(a) == len(i)
    for x, y in zip(a, i):
        assert x == y, (x, y)


@pytest.mark.parametrize('variant', ['E1', 'E6', 'E9', 'E10'])
def test_bsc_rows_match_base(variant, runs):
    d, out, _ = runs
    for k in (f'B{variant}', f'B{variant}p'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    new, old = rows(d / f'B{variant}'), rows(d / f'B{variant}p')
    assert len(new) == len(old)
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            x, y = dict(x, init_actor=None), dict(y, init_actor=None)
        assert x == y, (x['kind'], x['step'], {k: (x.get(k), y.get(k)) for k in set(x) | set(y)
                                               if x.get(k) != y.get(k)})
    old_args = json.load(open(d / f'B{variant}p' / 'args.json'))
    new_args = json.load(open(d / f'B{variant}' / 'args.json'))
    assert 'resolved_agent' not in new_args
    assert {k: v for k, v in new_args.items() if k not in ('run_dir', 'init_actor')} == \
        {k: v for k, v in old_args.items() if k not in ('run_dir', 'init_actor')}


def refuse_other_value_family(d, actor, cands, src):
    """A run dir holding only the QC run's args.json (a finished run exits RUN_ALREADY_COMPLETE before the args
    check): resuming it as ARB-BSC (--method=arb_bsc, or --agent.arb_value=bsc) is refused, args.json untouched."""
    out = []
    for method, extra in (('arb_bsc', ['--agent.discount=1.0']),
                          ('arb_qc', ['--agent.arb_value=bsc', '--agent.discount=1.0'])):
        rd = d / f'refuse_{method}'
        os.makedirs(rd, exist_ok=True)
        args = json.load(open(src / 'args.json'))
        args['run_dir'] = str(rd)
        with open(rd / 'args.json', 'w') as f:
            json.dump(args, f, indent=1)
        before = open(rd / 'args.json').read()
        r = run([*common(5), *ONLINE, f'--init_actor={actor}', *recipe_args('E1', method, cands), *extra,
                 f'--run_dir={rd}'], d, f'refuse_{method}')
        mism = [l for l in r['lines'] if l.startswith('ARGS_MISMATCH')]
        assert r['rc'] == 2 and mism and 'resolved_agent' in mism[0], r['lines'][-20:]
        assert open(rd / 'args.json').read() == before
        out.append(mism[0])
    return out


def test_resume_refuses_other_value_family(runs):
    d, out, setups = runs
    for line in refuse_other_value_family(d, *setups[5], d / 'QE1'):
        print(line[:300])
