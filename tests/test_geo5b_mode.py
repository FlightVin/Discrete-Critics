"""geo5b: --mode=arb_introspect end to end (square, tiny sizes). Source runs: tiny online runs from one tiny BC actor,
sampled-readout ARB-QC E10 (redundant, model_mix roll-in, noisy_fixed labels) and ARB-BSC E2 (twohot_last). Query
file from `python -m utils.arb_introspect make_queries` on the QC run's main buffer (online rows) plus synthetic
reference values. Checks: every component written, sources left byte-identical (non-mutation), a run preempted
after its first part and resumed == an uninterrupted run, refusals (run_dir inside the source, config differs,
parts written with other settings). Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import flag, launch, starts_with  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, run, tree_hash  # noqa: E402

SM = ['--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean', '--agent.arb_readout_samples=10',
      '--agent.arb_readout_temp=0.7']
QC = ['--method=arb_qc', *SM, '--agent.arb_label=noisy_fixed', '--agent.arb_vocab=redundant',
      '--agent.arb_rollin=model_mix']
BSC = ['--method=arb_bsc', *SM, '--agent.arb_label=twohot_last']
ENV = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--seed=0']
N_Q, CHUNK = 40, 16  # 3 parts per component


def introspect(d, name, src, agent_args, queries, extra=(), interrupt=None):
    args = [*ENV, '--mode=arb_introspect', *agent_args, f'--src_run_dir={src}', f'--run_dir={d / name}',
            f'--queries={queries}', f'--candidates={d / "cands.npz"}', f'--introspect_chunk={CHUNK}', *extra]
    return launch(args, str(d / f'{name}.log'), str(d / f'{name}.flag'), interrupt)


@pytest.fixture(scope='module')
def srcs(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo5b')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    online = [*COMMON, *ONLINE, f'--init_actor={d / "bc" / "final_actor.pkl"}', '--replay_rng=separate']
    jobs = dict(qc=lambda: run([*online, *QC, f'--run_dir={d / "qc"}'], d, 'qc'),
                bsc=lambda: run([*online, *BSC, f'--run_dir={d / "bsc"}'], d, 'bsc'))
    with ThreadPoolExecutor(2) as ex:
        out = {k: f.result() for k, f in {k: ex.submit(f) for k, f in jobs.items()}.items()}
    for k in out:
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
    p = subprocess.run([sys.executable, '-m', 'utils.arb_introspect', 'make_queries', f'--n={N_Q}', '--seed=3',
                        f'--src_run_dir={d / "qc"}', '--online_only', f'--out={d / "q_raw.npz"}'], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    with np.load(d / 'q_raw.npz') as z:
        q = {k: z[k] for k in z.files}
    rs = np.random.RandomState(0)
    for name, lo, hi in (('qc', -80.0, -5.0), ('bsc', 0.05, 0.95)):
        qq = dict(q, ref_value=rs.uniform(lo, hi, N_Q), ref_se=np.where(np.arange(N_Q) % 5 == 0, 0.05 * (hi - lo), 0.0),
                  clean_value=rs.uniform(lo, hi, N_Q))
        np.savez(d / f'q_{name}.npz', **qq)
    np.savez(d / 'cands.npz', observations=q['observations'][:6],
             candidates=rs.uniform(-1, 1, (6, 32, q['actions'].shape[1] * q['actions'].shape[2])).astype(np.float32))
    return d, q


@pytest.fixture(scope='module')
def evals(srcs):
    d, _ = srcs
    hashes = {k: tree_hash(d / k) for k in ('qc', 'bsc')}
    jobs = dict(
        A=lambda: introspect(d, 'A', d / 'qc', QC, d / 'q_qc.npz'),
        I=lambda: introspect(d, 'I', d / 'qc', QC, d / 'q_qc.npz',
                             interrupt=flag('after the first part', starts_with('ARB_INTROSPECT_PART '))),
        B=lambda: introspect(d, 'B', d / 'bsc', BSC, d / 'q_bsc.npz'))
    with ThreadPoolExecutor(len(jobs)) as ex:
        out = {k: f.result() for k, f in {k: ex.submit(f) for k, f in jobs.items()}.items()}
    out['I_parts'] = sorted(os.listdir(d / 'I' / 'arb_introspect_parts'))
    out['I_final'] = os.path.exists(d / 'I' / 'arb_introspect.json')
    out['I2'] = introspect(d, 'I', d / 'qc', QC, d / 'q_qc.npz')
    return d, out, hashes


N_PARTS = 4 * 3 + 1  # trace / forced / ablation / labelfit x 3 chunks + 1 candidate chunk


@pytest.mark.parametrize('name, src', [('A', 'qc'), ('B', 'bsc')])
def test_outputs(name, src, evals):
    d, out, hashes = evals
    r = out[name]
    assert r['rc'] == 0, r['lines'][-40:]
    assert tree_hash(d / src) == hashes[src]  # checkpoint and run dir not mutated
    s = json.load(open(d / name / 'arb_introspect.json'))
    assert s['n_queries'] == N_Q and s['ident']['phase'] == 'done'
    assert set(s) >= {'trace', 'forced', 'ablation', 'candidates', 'labelfit'}
    assert len(os.listdir(d / name / 'arb_introspect_parts')) == N_PARTS
    z = np.load(d / name / 'arb_introspect.npz')
    nc = 54 if src == 'qc' else 10
    assert z['trace/probs'].shape == (2, N_Q, 10, 3, nc) and z['trace/S'].shape == (N_Q,)
    assert np.array_equal(z['trace/S'], np.clip(z['trace/n'], 0, 999).sum(axis=(0, 2)))
    assert z['ablation/S'].shape == (N_Q,)
    t = z['ablation/tokens_ablated']
    assert t.min() >= 0 and t.max() <= 9
    assert z['candidates/S'].shape == (6, 32) and z['candidates/argmax'].shape == (6,)
    assert z['labelfit/a_kl'].shape == (N_Q, 3) and np.isfinite(z['labelfit/a_kl']).all()
    assert np.allclose(z['labelfit/a_ce'], z['labelfit/a_h'] + z['labelfit/a_kl'], atol=1e-4)
    if src == 'qc':
        assert (z['labelfit/y'] <= 0).all() and (z['labelfit/y'] >= -101).all()
        assert s['trace']['true_oov_count'] == 0
    else:
        assert s['trace']['noncanonical_token_any_frac'] == 0 and s['ablation']['changed_frac'] == 0
    f = s['forced']
    assert set(f) == {f'p{p}_e{e:+d}' for p in (1, 2) for e in (-2, -1, 0, 1, 2)}
    assert f['p1_e+0']['n_supported'] == N_Q and f['p1_e+0']['n_digit_ok'] < N_Q  # ref_se rows undecidable
    print(name, json.dumps({k: s[k] for k in ('trace', 'ablation', 'candidates')})[:600])


def test_interrupted_resume_identical(evals):
    d, out, _ = evals
    assert out['I']['rc'] == 143, out['I']['lines'][-30:]
    assert 1 <= len(out['I_parts']) < N_PARTS and not out['I_final'], out['I_parts']
    assert not [f for f in out['I_parts'] if f.endswith('.tmp')]
    assert out['I2']['rc'] == 0, out['I2']['lines'][-30:]
    skipped = [l for l in out['I2']['lines'] if l.startswith('ARB_INTROSPECT_PART_SKIP')]
    assert len(skipped) == len(out['I_parts'])
    a, i = np.load(d / 'A' / 'arb_introspect.npz'), np.load(d / 'I' / 'arb_introspect.npz')
    assert sorted(a.files) == sorted(i.files)
    for k in a.files:
        assert np.array_equal(a[k], i[k], equal_nan=a[k].dtype.kind == 'f'), k
    sa, si = json.load(open(d / 'A' / 'arb_introspect.json')), json.load(open(d / 'I' / 'arb_introspect.json'))
    assert sa == si


SQ3 = '/data/user_data/vineethb/Discrete-Critics/geo/geo-sq3-qc-E10-model-prefix-fixed-label-s0'


@pytest.mark.skipif(not os.path.exists(os.path.join(SQ3, 'ckpt', 'LATEST')), reason='sq3 checkpoint not present')
def test_measure_real_checkpoint(tmp_path):
    """Wall time of every component on a real-size ARB-QC E10 checkpoint, 256 demo queries (one part each)."""
    from test_geo4a_final_eval import _ckpt_hash
    before = _ckpt_hash(SQ3)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
    p = subprocess.run([sys.executable, '-m', 'utils.arb_introspect', 'make_queries', '--n=256', '--seed=0',
                        f'--out={tmp_path / "q0.npz"}'], cwd=REPO, env=env, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    q = dict(np.load(tmp_path / 'q0.npz'))
    rs = np.random.RandomState(0)
    np.savez(tmp_path / 'q.npz', **q, ref_value=rs.uniform(-60, -5, 256), ref_se=np.full(256, 0.5))
    np.savez(tmp_path / 'cands.npz', observations=q['observations'][:32],
             candidates=rs.uniform(-1, 1, (32, 32, 35)).astype(np.float32))
    import time
    t0 = time.time()
    r = launch([*ENV, '--mode=arb_introspect', *QC, f'--src_run_dir={SQ3}', f'--run_dir={tmp_path / "out"}',
                f'--queries={tmp_path / "q.npz"}', f'--candidates={tmp_path / "cands.npz"}'],
               str(tmp_path / 'm.log'), str(tmp_path / 'm.flag'))
    assert r['rc'] == 0, r['lines'][-40:]
    assert _ckpt_hash(SQ3) == before
    s = json.load(open(tmp_path / 'out' / 'arb_introspect.json'))
    print('MEASURE wall_s', round(time.time() - t0, 1), 'queries 256, candidate states 32 x 32',
          json.dumps({k: s[k] for k in ('trace', 'ablation', 'candidates')}),
          json.dumps({k: s['forced'][k] for k in ('p1_e+1', 'p1_e-1')}),
          json.dumps({k: s['labelfit'][k] for k in ('a_kl', 'b_kl', 'readout_value_abs_err_vs_y')}), sep='\n  ')


def test_refusals_leave_source_unchanged(srcs, evals):
    d, _ = srcs
    cases = dict(inside=(QC, [f'--run_dir={d / "qc" / "x"}'], 'lies inside', None),
                 config=([*QC[:-1], '--agent.arb_rollin=fixed'], [], 'agent.arb_rollin', None),
                 parts=(QC, ['--introspect_seed=1', f'--run_dir={d / "A"}'], 'other settings', 'ARB_INTROSPECT_REFUSED'))
    for name, (agent_args, extra, msg, tag) in cases.items():
        before = tree_hash(d / 'qc')
        r = introspect(d, f'refuse_{name}', d / 'qc', agent_args, d / 'q_qc.npz', extra)
        lines = [l for l in r['lines'] if l.startswith(tag or 'FINAL_EVAL_REFUSED')]
        assert r['rc'] == 2 and lines and msg in lines[0], (name, r['lines'][-20:])
        assert tree_hash(d / 'qc') == before, name
        print(name, lines[0][:200])
