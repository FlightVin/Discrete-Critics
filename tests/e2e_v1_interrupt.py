"""End-to-end preemption test for train_v1.py (square, tiny sizes), determinism cost, checkpoint benchmarks.

  python tests/e2e_v1_interrupt.py --root /tmp/v1e2e            # everything below
  python tests/e2e_v1_interrupt.py --bench_full --bench_dir DIR  # synthetic tool_hang-size checkpoint

Preemption is emulated as on Babel: the test creates the file named by AE_PREEMPT_FLAG (content: epoch
seconds) at a chosen moment, and the run must checkpoint and exit 143 within 90 s of the file's mtime
(measured here from the mtime to process exit, and inside the run). Hard kills (SIGKILL) at other moments,
one of them in the middle of a checkpoint write (writes slowed by V1_TEST_CKPT_DELAY), must resume from the
last committed generation. Every interrupted chain is restarted with the same arguments until it completes,
then started once more (must exit 0 at once).

A run that cannot reach a safe point within --preempt_budget_s of the notice exits 143 without a new
checkpoint; one interruption forces this (budget 3 s at the start of joint) and checks the resume.

All set runs use XLA_FLAGS=--xla_gpu_deterministic_ops=true: without it two identical runs differ after
a few hundred critic updates, their best-of-n actions then differ and their replay buffers cannot match.
Sets (A [and B] uninterrupted, I interrupted; I is compared with A, the A-vs-B parameter difference is
the measured tolerance; comp has no B and uses the bsc tolerance):
  bc       --mode=bc (no --method: the acbsc actor), interrupted during updates, evaluation, hard kill.
  bsc      --method=bsc from the bc actor.
  arb_qc   --method=arb_qc from the same (acbsc) bc actor; exercises the ARB readout diagnostics.
  comp     --method=bsc from a competent actor (tests/v1_actor_from_params.py on a 1M-step acbsc run):
           warm-up has real successes; the success buffer must hold >= 1 online episode and match A.
Then, alone on the machine: --method=bsc with and without XLA_FLAGS=--xla_gpu_deterministic_ops=true
(updates/s, env steps/s).
Run from the project root with MUJOCO_GL=egl, ROBOMIMIC_DATA_DIR set.
"""
import argparse
import glob
import json
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

from utils.ckpt_v1 import CheckpointStore  # noqa: E402

COMPETENT_SRC = sorted(glob.glob('/data/user_data/vineethb/Discrete-Critics/runs/dx-bsc-square/*/*/*/*/params_1000000.pkl'))
COMMON = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--n_eval_envs=2', '--eval_episodes=1',
          '--log_interval=250', '--ckpt_minutes=0.1', '--seed=0']
BC = ['--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=2000', '--eval_interval=1000']
ONLINE = ['--mode=online', '--warmup_steps=1000', '--pretrain_updates=400', '--online_steps=4000',
          '--eval_interval=1500', '--buffer_cap=90000']
BENCH = ['--mode=online', '--method=bsc', '--warmup_steps=1000', '--pretrain_updates=2000', '--online_steps=4000',
         '--eval_interval=100000', '--buffer_cap=90000']
DETERMINISTIC = '--xla_gpu_deterministic_ops=true'
DEADLINE_S = 90.0


def progress_at_least(phase, key, value):
    pat = re.compile(rf'^PROGRESS {phase} {key}=(\d+)')

    def f(line):
        m = pat.match(line)
        return m is not None and int(m.group(1)) >= value
    return f


def starts_with(prefix):
    return lambda line: line.startswith(prefix)


# An interruption: (label, trigger(line), kind 'flag' | 'kill', extra env, delay in s after the trigger,
# extra (restart-free) arguments for that process).
def flag(label, trig, env=None, delay=0.0, args=()):
    return (label, trig, 'flag', env, delay, list(args))


def kill(label, trig, env=None, delay=0.0):
    return (label, trig, 'kill', env, delay, [])


def launch(args, log_path, flag_path, interrupt=None, extra_env=None, deterministic=True):
    """Run train_v1.py once. With `interrupt`, act when its trigger first holds (after its delay).

    deterministic: XLA_FLAGS=--xla_gpu_deterministic_ops=true (without it two identical runs already
    differ after a few hundred GPU updates, so their replay buffers cannot be compared).
    Returns dict(rc, lines, flag_to_exit_s, killed, last_done_gen, ...)."""
    if os.path.exists(flag_path):
        os.remove(flag_path)  # the job preamble deletes the flag at job start
    env = dict(os.environ, WANDB_MODE='disabled', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
               AE_PREEMPT_FLAG=flag_path)
    env.pop('XLA_FLAGS', None)
    if deterministic:
        env['XLA_FLAGS'] = DETERMINISTIC
    env.update(extra_env or {})
    p = subprocess.Popen([sys.executable, 'train_v1.py', *args], cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, fired, out = [], threading.Event(), dict(flag_mtime=None, killed=False, last_done_gen=None)

    def act(kind, delay):
        time.sleep(delay)
        if p.poll() is not None:
            return
        if kind == 'flag':
            with open(flag_path + '.tmp', 'w') as f:
                f.write(f'{int(time.time())}\n')
            os.replace(flag_path + '.tmp', flag_path)
            out['flag_mtime'] = os.stat(flag_path).st_mtime
        else:
            p.send_signal(signal.SIGKILL)
            out['killed'] = True

    with open(log_path, 'a') as log:
        log.write(f'=== launch {" ".join(args)} env={extra_env} interrupt={interrupt and interrupt[:1]}\n')
        for line in p.stdout:
            log.write(line)
            lines.append(line.rstrip('\n'))
            if interrupt is not None and not fired.is_set() and interrupt[1](line):
                fired.set()
                threading.Thread(target=act, args=(interrupt[2], interrupt[4]), daemon=True).start()
        rc = p.wait()
        t_exit = time.time()
        log.write(f'=== exit {rc}\n')
    out.update(rc=rc, lines=lines, fired=fired.is_set())
    if out['killed'] or any(l.startswith('EXIT_PREEMPTED_UNSAVED') for l in lines):
        # All output printed before the exit has been read now.
        # Last committed generation: the last CKPT_DONE of this process, else the one it resumed from.
        gens = [l for l in lines if l.startswith(('CKPT_DONE', 'RESUME'))]
        m = re.search(r'gen=(\d+)', gens[-1]) if gens else None
        out['last_done_gen'] = int(m.group(1)) if m else None
        ck = [l for l in lines if l.startswith(('CKPT_BEGIN', 'CKPT_DONE'))]
        out['ckpt_in_progress'] = bool(ck) and ck[-1].startswith('CKPT_BEGIN')
    out['flag_to_exit_s'] = (t_exit - out['flag_mtime']) if out['flag_mtime'] is not None else None
    return out


def chain(name, args, log_path, flag_path, interrupts):
    """Interrupted run: one process per interruption, then one to completion, then one more (complete)."""
    res = []
    prev_kill = None  # last process ended without a checkpoint of its own (SIGKILL or unsaved exit)
    for label, trig, kind, extra_env, delay, extra_args in interrupts:
        r = launch([*args, *extra_args], log_path, flag_path, (label, trig, kind, extra_env, delay), extra_env)
        assert r['fired'], f'{name}: trigger "{label}" never fired (rc={r["rc"]})'
        resume = [l for l in r['lines'] if l.startswith('RESUME')]
        if prev_kill is not None:  # resumed from the last committed generation
            want = prev_kill['last_done_gen']
            got = int(re.search(r'gen=(\d+)', resume[0]).group(1)) if resume and 'gen=' in resume[0] else None
            assert got == want, f'{name}: after "{prev_kill["label"]}" resumed gen {got}, last committed was {want}'
            res[-1]['resumed_gen'] = got
        prev_kill = None
        if kind == 'flag':
            assert r['rc'] == 143, f'{name}: "{label}" exited {r["rc"]}, expected 143'
            internal = [l for l in r['lines'] if l.startswith('EXIT_PREEMPTED')]
            assert internal and 'source=flag' in internal[-1], internal
            unsaved = internal[-1].startswith('EXIT_PREEMPTED_UNSAVED')
            dt_in = float(re.search(r'notice_to_exit_s=([\d.]+)', internal[-1]).group(1))
            entry = dict(label=label, kind=kind, rc=r['rc'], flag_to_exit_s=round(r['flag_to_exit_s'], 2),
                         internal_s=dt_in, aborted_eval=any(l.startswith('EVAL_ABORTED') for l in r['lines']),
                         unsaved=unsaved, last_done_gen=r['last_done_gen'])
            assert r['flag_to_exit_s'] <= DEADLINE_S, entry
            print(f'[{name}] flag {label}: exit {r["rc"]}, flag-to-exit {r["flag_to_exit_s"]:.2f}s '
                  f'(internal {dt_in:.2f}s){" eval abandoned" if entry["aborted_eval"] else ""}'
                  f'{" budget exceeded: exited without new checkpoint" if unsaved else ""}', flush=True)
            if unsaved:
                prev_kill = entry
        else:
            assert r['killed'] and r['rc'] == -9, f'{name}: "{label}" rc={r["rc"]} killed={r["killed"]}'
            entry = dict(label=label, kind=kind, rc=r['rc'], last_done_gen=r['last_done_gen'],
                         ckpt_in_progress=r.get('ckpt_in_progress'))
            print(f'[{name}] SIGKILL {label}: last committed gen {r["last_done_gen"]}, '
                  f'checkpoint write in progress={r.get("ckpt_in_progress")}', flush=True)
            prev_kill = entry
        res.append(entry)
    r = launch(args, log_path, flag_path)
    assert r['rc'] == 0, f'{name}: final segment exited {r["rc"]}'
    if prev_kill is not None:
        resume = [l for l in r['lines'] if l.startswith('RESUME')]
        got = int(re.search(r'gen=(\d+)', resume[0]).group(1))
        assert got == prev_kill['last_done_gen'], (got, prev_kill)
        res[-1]['resumed_gen'] = got
    t0 = time.time()
    r = launch(args, log_path, flag_path)
    dt = time.time() - t0
    assert r['rc'] == 0 and any(l.startswith('RUN_ALREADY_COMPLETE') for l in r['lines']), f'{name}: restart of complete run'
    assert not any(l.startswith(('PHASE_BEGIN', 'CKPT_BEGIN')) for l in r['lines'])
    print(f'[{name}] restart of complete run: exit 0 in {dt:.1f}s, no work', flush=True)
    return res


def load_run(run_dir):
    store = CheckpointStore(os.path.join(run_dir, 'ckpt'))
    state, man = store.load()
    rows = {}
    for buf, meta in state.get('buffers', {}).items():
        segs = sorted(store.iter_segments(buf), key=lambda s: s[0])
        held = {}
        for a, b, r in segs:
            lo = max(a, meta['k_lo'])
            for f, v in r.items():
                held.setdefault(f, []).append(v[lo - a:])
        rows[buf] = {f: np.concatenate(v) for f, v in held.items()}
    with open(os.path.join(run_dir, 'metrics.jsonl')) as f:
        metrics = [json.loads(l) for l in f]
    return state, man, rows, metrics


def tree_maxdiff(a, b):
    import jax
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return max(float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)))) if np.size(x) else 0.0
               for x, y in zip(la, lb))


def phase_once(log_path, phases):
    """Across all processes of a chain: each phase is committed done once and never started after that."""
    with open(log_path) as f:
        lines = [l.rstrip('\n') for l in f]
    done_at = {}
    for i, l in enumerate(lines):
        if l.startswith('PHASE_DONE '):
            ph = l.split()[1]
            assert ph not in done_at, f'phase {ph} completed twice'
            done_at[ph] = i
        if l.startswith('PHASE_BEGIN '):
            ph = l.split()[1]
            assert ph not in done_at, f'phase {ph} started again after it was completed'
    assert sorted(done_at) == sorted(phases), (sorted(done_at), phases)
    return {ph: sum(l.startswith(f'PHASE_BEGIN {ph} ') for l in lines) for ph in phases}


def _rows(metrics, kinds):
    return [{k: v for k, v in r.items() if k != 'time' and 'duration' not in k}  # wall-clock fields excluded
            for r in metrics if r['kind'] in kinds]


def compare_online(name, A, B, I, log_I, competent=False, tol=None):
    """I vs A (and B vs A for the tolerance; without B, `tol` is the tolerance measured on another set)."""
    phases = ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    res = dict(phase_begin_counts_I=phase_once(log_I, phases))
    sa = A[0]
    for nm, (st, man, rows, metrics) in ((('B', B),) if B is not None else ()) + (('I', I),):
        res[f'ctr_equal_{nm}'] = st['ctr'] == sa['ctr']
        res[f'phase_{nm}'] = st['phase']
        bm = {}
        for buf in ('main', 'success'):
            ma = {k: v for k, v in sa['buffers'][buf].items() if k != 'seg_from'}
            mb = {k: v for k, v in st['buffers'][buf].items() if k != 'seg_from'}
            same_rows = rows[buf].keys() == A[2][buf].keys() and all(
                np.array_equal(rows[buf][f], A[2][buf][f]) for f in rows[buf])
            bm[buf] = dict(meta_equal=ma == mb, rows_equal=bool(same_rows), n=mb['n'], k_lo=mb['k_lo'],
                           episodes=len(mb['episodes']))
        res[f'buffers_{nm}_vs_A'] = bm
        res[f'params_maxdiff_{nm}_vs_A'] = tree_maxdiff(st['agent'], sa['agent'])
        res[f'np_rng_equal_{nm}'] = bool(np.array_equal(st['np_rng'][1], sa['np_rng'][1]))
        res[f'phase_end_rows_{nm}'] = [r['phase'] for r in metrics if r['kind'] == 'phase_end']
        res[f'eval_rows_equal_{nm}'] = _rows(metrics, ('eval', 'pretrain_eval')) == _rows(A[3], ('eval', 'pretrain_eval'))
        res[f'readout_rows_{nm}'] = len(_rows(metrics, ('readout',)))
        res[f'readout_rows_equal_{nm}'] = _rows(metrics, ('readout',)) == _rows(A[3], ('readout',))
        res[f'config_rows_equal_{nm}'] = _rows(metrics, ('config',)) == _rows(A[3], ('config',))
    if B is not None:
        tol = res['params_maxdiff_B_vs_A']
    res['tolerance_measured_A_vs_B'] = tol
    res['params_I_within_tol'] = res['params_maxdiff_I_vs_A'] <= tol
    cfg = _rows(A[3], ('config',))
    res['config_row'] = dict(method=cfg[0]['method'], agent_name=cfg[0]['agent_name'],
                             init_actor_agent=(cfg[0]['init_actor'] or {}).get('agent_name')) if cfg else None
    ctr = sa['ctr']
    res['online_successes_A'] = dict(warmup=ctr.get('warmup_successes', 0), joint=ctr.get('joint_successes', 0))
    ok = (res['ctr_equal_I'] and res['phase_I'] == 'done' and res['np_rng_equal_I']
          and all(b['meta_equal'] and b['rows_equal'] for b in res['buffers_I_vs_A'].values())
          and res['params_I_within_tol'] and res['eval_rows_equal_I'] and res['readout_rows_equal_I']
          and res['config_rows_equal_I'] and res['phase_end_rows_I'] == phases and len(cfg) == 1)
    if name == 'arb_qc':
        ok = ok and res['readout_rows_I'] > 0
    if competent:
        res['success_buffer_online_episodes'] = len(sa['buffers']['success']['episodes'])
        ok = ok and res['success_buffer_online_episodes'] >= 1 and sum(res['online_successes_A'].values()) >= 1
    res['pass'] = bool(ok)
    return res


def online_interrupts(eval_nominal, full=True):
    if not full:  # competent-actor set: fewer restarts
        return [
            flag('during warm-up collection', progress_at_least('warmup', 'env_steps', 500)),
            flag('during an evaluation (joint)', starts_with(f'EVAL_BEGIN nominal={eval_nominal}'), delay=2.0),
            kill('SIGKILL during joint updates', progress_at_least('joint', 'env_steps', 3250), delay=0.7),
        ]
    return [
        flag('during warm-up collection', progress_at_least('warmup', 'env_steps', 500)),
        flag('at a phase boundary (warmup end)', starts_with('PHASE_END warmup')),
        flag('during pretrain updates', progress_at_least('pretrain', 'updates', 250)),
        # First joint episode (compile + episode) is longer than a 3 s budget: exit without a checkpoint.
        flag('start of joint, safe point beyond --preempt_budget_s=3', starts_with('PHASE_BEGIN joint'),
             delay=1.0, args=['--preempt_budget_s=3']),
        kill('SIGKILL mid-checkpoint-write', starts_with('CKPT_BEGIN phase=joint reason=periodic'),
             {'V1_TEST_CKPT_DELAY': '4'}, delay=1.0),
        flag('during a checkpoint write', starts_with('CKPT_BEGIN phase=joint reason=periodic'),
             {'V1_TEST_CKPT_DELAY': '4'}, delay=1.0),
        flag('during joint updates', progress_at_least('joint', 'env_steps', 2250)),
        flag('during an evaluation (joint)', starts_with(f'EVAL_BEGIN nominal={eval_nominal}'), delay=2.0),
        kill('SIGKILL during joint updates', progress_at_least('joint', 'env_steps', 3250), delay=0.7),
    ]


def run_set(name, args_of, logs, flag_of, interrupts, runs='AB'):
    """Uninterrupted runs (A, and B for the tolerance) and the I chain, in parallel. Returns the chain results."""
    with ThreadPoolExecutor(3) as ex:
        futs = {n: ex.submit(launch, args_of(n), logs(n), flag_of(n)) for n in runs}
        fi = ex.submit(chain, f'{name}I', args_of('I'), logs('I'), flag_of('I'), interrupts)
        for n, f in futs.items():
            r = f.result()
            assert r['rc'] == 0, f'{name}{n} exited {r["rc"]}'
        return fi.result()


def rates(log_path):
    """Throughput from PROGRESS lines (compile time excluded: first line of each phase is the origin)."""
    out = {}
    lines = [l for l in open(log_path) if l.startswith('PROGRESS ')]
    for phase, key, label in (('warmup', 'env_steps', 'warmup_env_steps_per_s'),
                              ('pretrain', 'updates', 'pretrain_updates_per_s'),
                              ('joint', 'env_steps', 'joint_env_steps_per_s (1 update each)')):
        pts = [tuple(map(float, re.search(rf'{key}=(\d+) t=([\d.]+)', l).groups()))
               for l in lines if l.startswith(f'PROGRESS {phase} ')]
        if len(pts) >= 2:
            out[label] = round((pts[-1][0] - pts[0][0]) / (pts[-1][1] - pts[0][1]), 1)
    return out


def interruption_test(root, sets):
    os.makedirs(root, exist_ok=True)
    d = lambda n: os.path.join(root, n)  # noqa: E731
    lg = lambda n: os.path.join(root, f'{n}.log')  # noqa: E731
    fl = lambda n: os.path.join(root, f'{n}.preempt_flag')  # noqa: E731
    t0 = time.time()
    results, chains = {}, {}

    # Competent actor for the comp set (read-only source).
    from v1_actor_from_params import extract
    comp_actor = d('competent_final_actor.pkl')
    if 'comp' in sets:
        assert COMPETENT_SRC, 'competent actor checkpoint not found'
        extract(COMPETENT_SRC[0], comp_actor)

    def online_set(name, method, init):
        comp = name == 'comp'
        args_of = lambda n: [*COMMON, *ONLINE, f'--method={method}', f'--init_actor={init}',  # noqa: E731
                             f'--run_dir={d(name + n)}']
        chains[name] = run_set(name, args_of, lambda n: lg(name + n), lambda n: fl(name + n),
                               online_interrupts(3000, full=not comp), runs='A' if comp else 'AB')
        if comp:
            return  # compared below with the tolerance measured on the bsc set (same method)
        A, B, I = (load_run(d(name + n)) for n in 'ABI')
        results[name] = compare_online(name, A, B, I, lg(name + 'I'))
        print(f'[{name}] {"PASS" if results[name]["pass"] else "FAIL"}', flush=True)

    bench_args = lambda n: [*COMMON, *BENCH, f'--init_actor={os.path.join(d("bcA"), "final_actor.pkl")}',  # noqa: E731
                            f'--run_dir={d(n)}']

    with ThreadPoolExecutor(8) as ex:
        futs = []
        if 'comp' in sets:  # independent of the bc run: start at once
            futs.append(ex.submit(online_set, 'comp', 'bsc', comp_actor))

        # ---- BC: A, B uninterrupted, I interrupted.
        bc_args = lambda n: [*COMMON, *BC, f'--run_dir={d("bc" + n)}']  # noqa: E731
        bc_interrupts = [flag('bc: during updates', progress_at_least('bc', 'updates', 500)),
                         flag('bc: during an evaluation', starts_with('EVAL_BEGIN nominal=1000'), delay=1.0),
                         kill('bc: SIGKILL during updates', progress_at_least('bc', 'updates', 1250), delay=0.5)]
        chains['bc'] = run_set('bc', bc_args, lambda n: lg('bc' + n), lambda n: fl('bc' + n), bc_interrupts)
        actor = {n: pickle.load(open(os.path.join(d('bc' + n), 'final_actor.pkl'), 'rb'))['actor_params'] for n in 'ABI'}
        bc_tol = tree_maxdiff(actor['A'], actor['B'])
        bc_diff = tree_maxdiff(actor['A'], actor['I'])
        bc_cfg = _rows(load_run(d('bcA'))[3], ('config',))
        results['bc'] = dict(final_actor_maxdiff_I_vs_A=bc_diff, tolerance_measured_A_vs_B=bc_tol,
                             phase_begin_counts_I=phase_once(lg('bcI'), ['bc']),
                             config_row=dict(method=bc_cfg[0]['method'], agent_name=bc_cfg[0]['agent_name']),
                             **{'pass': bool(bc_diff <= bc_tol and len(bc_cfg) == 1)})
        print(f'[bc] {"PASS" if results["bc"]["pass"] else "FAIL"}: final_actor max|I-A| = {bc_diff:.3g} '
              f'(A-B {bc_tol:.3g})', flush=True)

        init = os.path.join(d('bcA'), 'final_actor.pkl')
        for name in ('bsc', 'arb_qc'):
            if name in sets:
                futs.append(ex.submit(online_set, name, name, init))
        if 'det' in sets:  # second run without the flag, for the divergence only (its speed is not used)
            futs.append(ex.submit(launch, bench_args('det_default_2'), lg('det_default_2'), fl('det_default_2'),
                                  deterministic=False))
        for f in futs:
            r = f.result()
            assert not isinstance(r, dict) or r['rc'] == 0, r.get('rc')
    if 'comp' in sets:
        tol = results['bsc']['tolerance_measured_A_vs_B'] if 'bsc' in results else 0.0
        A, I = (load_run(d('comp' + n)) for n in 'AI')
        results['comp'] = compare_online('comp', A, None, I, lg('compI'), competent=True, tol=tol)
        print(f'[comp] {"PASS" if results["comp"]["pass"] else "FAIL"}', flush=True)
    t_sets = time.time() - t0

    # ---- Determinism cost, alone on the machine.
    if 'det' in sets:
        det = {}
        for label, flag_on in (('default', False), ('deterministic_ops', True)):
            n = f'det_{label}'
            t = time.time()
            r = launch(bench_args(n), lg(n), fl(n), deterministic=flag_on)
            assert r['rc'] == 0, (n, r['rc'])
            det[label] = dict(**rates(lg(n)), wall_s=round(time.time() - t, 1))
        s1, s2 = load_run(d('det_default'))[0], load_run(d('det_default_2'))[0]
        det['schedule'] = ' '.join(BENCH[2:6])
        det['without_flag_two_runs'] = dict(
            params_maxdiff=tree_maxdiff(s1['agent']['network']['params'], s2['agent']['network']['params']),
            main_buffer_meta_equal=s1['buffers']['main'] == s2['buffers']['main'], ctr_equal=s1['ctr'] == s2['ctr'])
        det['with_flag_two_runs_params_maxdiff'] = {
            k: results[k]['tolerance_measured_A_vs_B'] for k in ('bc', 'bsc', 'arb_qc') if k in results}
        results['determinism'] = det

    results['interruptions'] = {k: [{kk: vv for kk, vv in c.items()} for c in v] for k, v in chains.items()}
    flag_times = [c['flag_to_exit_s'] for v in chains.values() for c in v if c['kind'] == 'flag']
    results['flag_to_exit_s_max'] = max(flag_times) if flag_times else None
    results['wall_s'] = dict(sets=round(t_sets, 1), total=round(time.time() - t0, 1))
    print('E2E_RESULT ' + json.dumps(results, indent=1, default=str), flush=True)

    print('---- summary', flush=True)
    for name in ('bc', 'bsc', 'arb_qc', 'comp'):
        if name in results:
            r = results[name]
            ft = [c['flag_to_exit_s'] for c in chains[name] if c['kind'] == 'flag']
            print(f'{"PASS" if r["pass"] else "FAIL"} {name}: tolerance(A-B)={r["tolerance_measured_A_vs_B"]:.3g} '
                  f'I-A={r.get("params_maxdiff_I_vs_A", r.get("final_actor_maxdiff_I_vs_A")):.3g} '
                  f'flag_to_exit_s={ft} kills={sum(c["kind"] == "kill" for c in chains[name])} '
                  f'unsaved_exits={sum(bool(c.get("unsaved")) for c in chains[name])}'
                  + (f' online_successes={r["online_successes_A"]} success_buffer_online_episodes='
                     f'{r.get("success_buffer_online_episodes")}' if name == 'comp' else '')
                  + (f' readout_rows={r["readout_rows_I"]}' if name == 'arb_qc' else ''), flush=True)
    if 'determinism' in results:
        print(f'DETERMINISM {json.dumps(results["determinism"])}', flush=True)
    ok = all(results[n]['pass'] for n in ('bc', 'bsc', 'arb_qc', 'comp') if n in results)
    ok = ok and all(t <= DEADLINE_S for t in flag_times)
    print('E2E_PASS' if ok else 'E2E_FAIL', flush=True)
    return ok


# ---- synthetic full-size checkpoint benchmark ----------------------------------------------------------
def bench_full(bench_dir, new_rows, obs_dim=53, act_dim=7, cap=5_000_000, demo=95_962, H=8):
    """Both buffers full at tool_hang size; time the first (full) write, an incremental (SIGTERM-path)
    write with `new_rows` new rows per buffer, and a full verified load."""
    import jax
    jax.config.update('jax_platforms', 'cpu')
    import flax
    import ml_collections
    from agents import acbsc
    from utils.replay_v1 import FIELDS, SeqReplay

    rs = np.random.default_rng(0)
    demos = {
        'observations': rs.standard_normal((demo, obs_dim), dtype=np.float32),
        'next_observations': rs.standard_normal((demo, obs_dim), dtype=np.float32),
        'actions': rs.uniform(-1, 1, (demo, act_dim)).astype(np.float32),
        'rewards': -np.ones(demo, np.float32), 'terminals': np.zeros(demo, np.float32),
        'masks': np.ones(demo, np.float32), 'num_chunks_to_timeout': np.ones(demo, np.float32),
        'num_chunks_to_success': np.ones(demo, np.float32)}
    cfg = acbsc.get_config()
    cfg.actor_type, cfg.horizon_length = 'best-of-n', H
    agent = acbsc.ACBSCAgent.create(0, demos['observations'][0], demos['actions'][0], cfg)
    agent_state = jax.device_get(flax.serialization.to_state_dict(agent))

    def fill(buf, n_rows, ep_len=700):
        """Bulk-append n_rows online rows (episodes of ep_len for the success buffer)."""
        R = buf.R
        if buf.episodic:
            n_ep = max(1, n_rows // ep_len)
            for _ in range(n_ep):
                while buf.n + ep_len - buf.k_lo > R:
                    s, L = buf.episodes.pop(0)
                    for f in FIELDS:  # as SeqReplay.add_episode: evicted slots are zeroed
                        buf.data[f][buf.slot(np.arange(s, s + L))] = 0
                    buf.k_lo += L
                    buf.overwrites += L
                buf.episodes.append([buf.n, ep_len])
                _bulk(buf, buf.n, ep_len)
                buf.n += ep_len
        else:
            _bulk(buf, buf.n, n_rows)
            buf.n += n_rows
            if buf.n - buf.k_lo > R:
                buf.overwrites += buf.n - R - buf.k_lo
                buf.k_lo = buf.n - R

    def _bulk(buf, k0, n):
        s = buf.slot(np.arange(k0, k0 + n))
        for f in FIELDS:
            shape = (n, *buf.data[f].shape[1:])
            buf.data[f][s] = rs.standard_normal(shape, dtype=np.float32) if buf.data[f].ndim > 1 else -1.0

    t = time.time()
    bufs = dict(main=SeqReplay(demos, cap, H, episodic=False), success=SeqReplay(demos, cap, H, episodic=True))
    for b in bufs.values():
        fill(b, b.R)
    print(f'BENCH built 2 x {cap} rows (obs {obs_dim}) in {time.time() - t:.1f}s; '
          f'occupancy main={bufs["main"].stats()["occupancy"]:.4f} success={bufs["success"].stats()["occupancy"]:.4f}',
          flush=True)

    root = os.path.join(bench_dir, 'ckpt')
    if os.path.exists(bench_dir):
        shutil.rmtree(bench_dir)
    store = CheckpointStore(root)
    store.load()

    def save():
        segs = {k: b.new_segment() for k, b in bufs.items()}
        state = dict(agent=agent_state, np_rng=np.random.get_state(), buffers={k: b.meta() for k, b in bufs.items()})
        t = time.time()
        info = store.save(state, segs, {k: b.k_lo for k, b in bufs.items()})
        return info, time.time() - t

    full, t_full = save()
    print(f'BENCH full write (all 2 x {cap} rows): {t_full:.1f}s, segments {full["segment_bytes"] / 1e9:.2f} GB, '
          f'state {full["state_bytes"] / 1e6:.0f} MB', flush=True)
    for b in bufs.values():
        fill(b, new_rows)
    inc, t_inc = save()
    print(f'BENCH incremental write ({new_rows} new rows per buffer, buffers full, old rows overwritten): '
          f'{t_inc:.1f}s (segments {inc["t_segments"]:.1f}s, state+commit {inc["t_state"]:.1f}s, '
          f'cleanup {inc["t_cleanup"]:.1f}s), wrote {inc["segment_bytes"] / 1e6:.0f} MB + state '
          f'{inc["state_bytes"] / 1e6:.0f} MB', flush=True)
    du = sum(os.path.getsize(p) for p in glob.glob(os.path.join(root, '**', '*'), recursive=True) if os.path.isfile(p))
    print(f'BENCH checkpoint on disk: {du / 1e9:.2f} GB', flush=True)

    t = time.time()
    store2 = CheckpointStore(root)
    state, _ = store2.load()
    fresh = {k: SeqReplay(demos, cap, H, episodic=(k == 'success')) for k in bufs}
    for k, b in fresh.items():
        b.restore(state['buffers'][k], store2.iter_segments(k))
    t_load = time.time() - t
    same = all(np.array_equal(fresh[k].data[f], bufs[k].data[f]) for k in bufs for f in FIELDS)
    print(f'BENCH verified load + restore of both buffers: {t_load:.1f}s, bit-exact={same}', flush=True)
    shutil.rmtree(bench_dir)
    return dict(t_full=t_full, t_inc=t_inc, t_load=t_load, exact=same)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/tmp/v1rec_e2e')
    p.add_argument('--sets', default='bsc,arb_qc,comp,det', help='subset of bsc,arb_qc,comp,det (bc always runs)')
    p.add_argument('--bench_full', action='store_true')
    p.add_argument('--bench_dir', default='/tmp/v1rec_bench')
    p.add_argument('--bench_new_rows', type=int, default=150_000)
    a = p.parse_args()
    if a.bench_full:
        bench_full(a.bench_dir, a.bench_new_rows)
    else:
        if os.path.exists(a.root):
            shutil.rmtree(a.root)
        sys.exit(0 if interruption_test(a.root, set(a.sets.split(','))) else 1)
