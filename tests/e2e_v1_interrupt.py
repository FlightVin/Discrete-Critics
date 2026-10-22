"""End-to-end preemption test for train_v1.py (square, tiny sizes), plus checkpoint benchmarks.

  python tests/e2e_v1_interrupt.py --root /tmp/v1e2e            # interruption test + tiny ckpt numbers
  python tests/e2e_v1_interrupt.py --bench_full --bench_dir DIR  # synthetic tool_hang-size checkpoint

Interruption test: one BC run uninterrupted and one interrupted; then online (acbsc) runs A and B
uninterrupted and run I started as a chain of processes, each sent SIGTERM at a chosen moment and
restarted with the same arguments: during warm-up collection, right at a phase boundary (twice),
during pretrain updates, during a checkpoint write (writes slowed by V1_TEST_CKPT_DELAY), during joint
updates and during an evaluation. Then I is started once more after completion (must exit 0 at once).
Checks: every interrupted process exits 143; each phase is completed once; final counters, both
buffers, metrics phase_end/eval rows equal A; params of I vs A within the tolerance measured as A vs B.
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
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.ckpt_v1 import CheckpointStore  # noqa: E402

COMMON = ['--env_name=square-mh-low_dim', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n',
          '--horizon_length=5', '--n_eval_envs=2', '--eval_episodes=1', '--log_interval=250',
          '--ckpt_minutes=0.1', '--seed=0']
BC = ['--mode=bc', '--bc_steps=2000', '--eval_interval=1000']
ONLINE = ['--mode=online', '--warmup_steps=2000', '--pretrain_updates=500', '--online_steps=6000',
          '--eval_interval=2000', '--buffer_cap=90000']


def progress_at_least(phase, key, value):
    pat = re.compile(rf'^PROGRESS {phase} {key}=(\d+)')

    def f(line):
        m = pat.match(line)
        return m is not None and int(m.group(1)) >= value
    return f


def starts_with(prefix):
    return lambda line: line.startswith(prefix)


def launch(args, log_path, trigger=None, extra_env=None):
    """Run train_v1.py; send SIGTERM when `trigger(line)` first holds. Returns (rc, sigterm_to_exit_s, lines)."""
    env = dict(os.environ, WANDB_MODE='disabled', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1')
    env.update(extra_env or {})
    p = subprocess.Popen([sys.executable, 'train_v1.py', *args], cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    t_sig, lines = None, []
    with open(log_path, 'a') as log:
        log.write(f'=== launch {" ".join(args)} env={extra_env}\n')
        for line in p.stdout:
            log.write(line)
            lines.append(line.rstrip('\n'))
            if trigger is not None and t_sig is None and trigger(line):
                p.send_signal(signal.SIGTERM)
                t_sig = time.time()
                log.write(f'=== SIGTERM sent\n')
        rc = p.wait()
        t_exit = time.time()
        log.write(f'=== exit {rc}\n')
    return rc, (t_exit - t_sig) if t_sig is not None else None, lines


def chain(name, args, log_path, triggers):
    """Interrupted run: one process per trigger, then one to completion, then one more (already complete)."""
    out = []
    for label, trig, extra_env in triggers:
        rc, dt, lines = launch(args, log_path, trig, extra_env)
        assert dt is not None, f'{name}: trigger "{label}" never fired (rc={rc})'
        assert rc == 143, f'{name}: "{label}" exited {rc}, expected 143'
        internal = [l for l in lines if l.startswith('EXIT_PREEMPTED')]
        out.append(dict(label=label, rc=rc, sigterm_to_exit_s=round(dt, 2), internal=internal[-1] if internal else None,
                        aborted_eval=any(l.startswith('EVAL_ABORTED') for l in lines)))
        print(f'[{name}] {label}: exit {rc}, SIGTERM-to-exit {dt:.2f}s', flush=True)
    rc, _, lines = launch(args, log_path)
    assert rc == 0, f'{name}: final segment exited {rc}'
    t0 = time.time()
    rc, _, lines = launch(args, log_path)
    dt = time.time() - t0
    assert rc == 0 and any(l.startswith('RUN_ALREADY_COMPLETE') for l in lines), f'{name}: restart of complete run'
    assert not any(l.startswith(('PHASE_BEGIN', 'CKPT_BEGIN')) for l in lines)
    print(f'[{name}] restart of complete run: exit 0 in {dt:.1f}s, no work', flush=True)
    return out


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


def interruption_test(root):
    os.makedirs(root, exist_ok=True)
    d = lambda n: os.path.join(root, n)  # noqa: E731
    lg = lambda n: os.path.join(root, f'{n}.log')  # noqa: E731
    t0 = time.time()

    # ---- BC: uninterrupted and interrupted, in parallel.
    bc_triggers = [('bc: during updates', progress_at_least('bc', 'updates', 500), None),
                   ('bc: during eval', starts_with('EVAL_BEGIN nominal=1000'), None),
                   ('bc: during updates after eval', progress_at_least('bc', 'updates', 1500), None)]
    with ThreadPoolExecutor(2) as ex:
        fa = ex.submit(launch, [*COMMON, *BC, f'--run_dir={d("bcA")}'], lg('bcA'))
        fi = ex.submit(chain, 'bcI', [*COMMON, *BC, f'--run_dir={d("bcI")}'], lg('bcI'), bc_triggers)
        rc, _, _ = fa.result()
        assert rc == 0
        bc_chain = fi.result()
    actor_a = pickle.load(open(os.path.join(d('bcA'), 'final_actor.pkl'), 'rb'))['actor_params']
    actor_i = pickle.load(open(os.path.join(d('bcI'), 'final_actor.pkl'), 'rb'))['actor_params']
    bc_diff = tree_maxdiff(actor_a, actor_i)
    bc_begins = phase_once(lg('bcI'), ['bc'])
    print(f'BC: final_actor max|I-A| = {bc_diff:.3g}; bc phase begun {bc_begins} times', flush=True)

    # ---- Online: A, B uninterrupted; I interrupted.
    init = f'--init_actor={os.path.join(d("bcA"), "final_actor.pkl")}'
    on = lambda n: [*COMMON, *ONLINE, init, f'--run_dir={d(n)}']  # noqa: E731
    triggers = [
        ('during warm-up collection', progress_at_least('warmup', 'env_steps', 1000), None),
        ('at phase boundary (end of warmup)', starts_with('PHASE_END warmup'), None),
        ('during pretrain updates', progress_at_least('pretrain', 'updates', 250), None),
        ('at phase boundary (pretrain_eval done)', starts_with('PHASE_DONE pretrain_eval'), None),
        ('during a checkpoint write', starts_with('CKPT_BEGIN phase=joint reason=periodic'),
         {'V1_TEST_CKPT_DELAY': '3'}),
        ('during joint updates', progress_at_least('joint', 'env_steps', 3250), None),
        ('during an evaluation', starts_with('EVAL_BEGIN nominal=4000'), None),
        ('during joint updates (late)', progress_at_least('joint', 'env_steps', 5000), None),
    ]
    with ThreadPoolExecutor(3) as ex:
        fa = ex.submit(launch, on('onA'), lg('onA'))
        fb = ex.submit(launch, on('onB'), lg('onB'))
        fi = ex.submit(chain, 'onI', on('onI'), lg('onI'), triggers)
        for f in (fa, fb):
            rc, _, _ = f.result()
            assert rc == 0
        on_chain = fi.result()

    A, B, I = (load_run(d(n)) for n in ('onA', 'onB', 'onI'))
    phases = ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    begins = phase_once(lg('onI'), phases)
    res = dict(phase_begin_counts_I=begins)
    for name, (st, man, rows, metrics) in (('B', B), ('I', I)):
        sa = A[0]
        res[f'ctr_equal_{name}'] = st['ctr'] == sa['ctr']
        res[f'phase_{name}'] = st['phase']
        bm = {}
        for buf in ('main', 'success'):
            ma = {k: v for k, v in sa['buffers'][buf].items() if k != 'seg_from'}
            mb = {k: v for k, v in st['buffers'][buf].items() if k != 'seg_from'}
            same_rows = rows[buf].keys() == A[2][buf].keys() and all(
                np.array_equal(rows[buf][f], A[2][buf][f]) for f in rows[buf])
            bm[buf] = dict(meta_equal=ma == mb, rows_equal=bool(same_rows), n=mb['n'], k_lo=mb['k_lo'],
                           episodes=len(mb['episodes']))
        res[f'buffers_{name}_vs_A'] = bm
        res[f'params_maxdiff_{name}_vs_A'] = tree_maxdiff(st['agent'], sa['agent'])
        res[f'np_rng_equal_{name}'] = bool(np.array_equal(st['np_rng'][1], sa['np_rng'][1]))
        pe = lambda m: [(r['phase']) for r in m if r['kind'] == 'phase_end']  # noqa: E731
        ev = lambda m: [{k: v for k, v in r.items() if k != 'time' and 'duration' not in k}  # noqa: E731
                        for r in m if r['kind'] in ('eval', 'pretrain_eval')]  # wall-clock fields excluded
        res[f'phase_end_rows_{name}'] = pe(metrics)
        res[f'eval_rows_equal_{name}'] = ev(metrics) == ev(A[3])
        res[f'metrics_rows_{name}_vs_A'] = (len(metrics), len(A[3]))
    tol = res['params_maxdiff_B_vs_A']
    res['tolerance_measured_A_vs_B'] = tol
    res['params_I_within_tol'] = res['params_maxdiff_I_vs_A'] <= tol
    res['sigterm_to_exit'] = [(c['label'], c['sigterm_to_exit_s']) for c in bc_chain + on_chain]
    res['eval_aborted_on_sigterm'] = [c['label'] for c in bc_chain + on_chain if c['aborted_eval']]

    # Tiny-run checkpoint size / write time (from run A's log).
    ck = [l for l in open(lg('onA')) if l.startswith('CKPT_DONE')]
    secs = [float(re.search(r'seconds=([\d.]+)', l).group(1)) for l in ck]
    st_mb = [float(re.search(r'state_mb=([\d.]+)', l).group(1)) for l in ck]
    sg_mb = [float(re.search(r'segment_mb=([\d.]+)', l).group(1)) for l in ck]
    du = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d('onA'), 'ckpt', '**', '*'), recursive=True)
             if os.path.isfile(p))
    res['tiny_ckpt'] = dict(n=len(ck), write_s_mean=round(float(np.mean(secs)), 3), write_s_max=max(secs),
                            state_mb=max(st_mb), segment_mb_max=max(sg_mb), ckpt_dir_mb=round(du / 1e6, 1))
    prog = [l for l in open(lg('onA')) if l.startswith('PROGRESS joint')]
    if len(prog) >= 2:
        s0, t0_ = map(float, re.search(r'env_steps=(\d+) t=([\d.]+)', prog[0]).groups())
        s1, t1_ = map(float, re.search(r'env_steps=(\d+) t=([\d.]+)', prog[-1]).groups())
        res['joint_steps_per_s'] = round((s1 - s0) / (t1_ - t0_), 1)
    res['bc_final_actor_maxdiff_I_vs_A'] = bc_diff
    res['wall_s'] = round(time.time() - t0, 1)
    print('E2E_RESULT ' + json.dumps(res, indent=1, default=str), flush=True)

    ok = (res['ctr_equal_I'] and res['phase_I'] == 'done'
          and all(b['meta_equal'] and b['rows_equal'] for b in res['buffers_I_vs_A'].values())
          and res['params_I_within_tol'] and res['eval_rows_equal_I']
          and res['phase_end_rows_I'] == phases and bc_diff <= tol)
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
    p.add_argument('--bench_full', action='store_true')
    p.add_argument('--bench_dir', default='/tmp/v1rec_bench')
    p.add_argument('--bench_new_rows', type=int, default=150_000)
    a = p.parse_args()
    if a.bench_full:
        bench_full(a.bench_dir, a.bench_new_rows)
    else:
        if os.path.exists(a.root):
            shutil.rmtree(a.root)
        sys.exit(0 if interruption_test(a.root) else 1)
