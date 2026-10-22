"""BC actor selection (--bc_select_threshold) and --mode=eval_actor of train_v1.py (square, tiny sizes).

Run from the project root with MUJOCO_GL=egl, ROBOMIMIC_DATA_DIR set and
XLA_FLAGS=--xla_gpu_deterministic_ops=true (the resume tests compare bytes of an interrupted and an
uninterrupted run). The selection logic is driven by V1_TEST_BC_EVAL_RATES (the rates the selection sees;
the logged eval rows stay real), so the tests do not depend on what a tiny BC run achieves.
"""
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

BASE_COMMIT = 'c5f2d84'  # train_v1.py before actor selection: the disabled flag must reproduce it
I, N = 100, 400          # eval interval, bc budget: evaluations at 100, 200, 300, 400
COMMON = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--n_eval_envs=2', '--eval_episodes=1',
          '--log_interval=50', '--seed=0']
BC = ['--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', f'--bc_steps={N}',
      f'--eval_interval={I}', *COMMON]
SEL = [*BC, '--bc_select_threshold=0.45']
NO_EXCEED = '0.2,0.3,0.3,0.1'   # best_within_budget: the 2nd evaluation (update 200), ties keep the earlier
VOLATILE = ('time', 'n1/episode.duration')  # wall-clock fields of metrics rows


def launch(args, log_path, env=None, trigger=None, action=None, script='train_v1.py'):
    """Run train_v1.py; when a stdout line starts with `trigger`, do `action` ('flag' | 'kill')."""
    flag_path = str(log_path) + '.preempt_flag'
    if os.path.exists(flag_path):
        os.remove(flag_path)
    e = dict(os.environ, WANDB_MODE='disabled', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
             AE_PREEMPT_FLAG=flag_path, PYTHONPATH=REPO, **(env or {}))
    p = subprocess.Popen([sys.executable, script, *args], cwd=REPO, env=e, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, fired = [], []
    killer = threading.Timer(1200, p.kill)
    killer.start()
    with open(log_path, 'a') as log:
        log.write(f'=== launch {" ".join(args)} env={env} trigger={trigger} action={action}\n')
        for line in p.stdout:
            log.write(line)
            lines.append(line.rstrip('\n'))
            if trigger and not fired and line.startswith(trigger):
                fired.append(line)
                if action == 'flag':
                    with open(flag_path + '.tmp', 'w') as f:
                        f.write(f'{int(time.time())}\n')
                    os.replace(flag_path + '.tmp', flag_path)
                else:
                    p.send_signal(signal.SIGKILL)
        rc = p.wait()
        log.write(f'=== exit {rc}\n')
    killer.cancel()
    return dict(rc=rc, lines=lines, fired=bool(fired), out='\n'.join(lines))


def rows(run_dir):
    return [json.loads(l) for l in open(os.path.join(run_dir, 'metrics.jsonl'))]


def stable(rs):
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rs]


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def assert_tree_equal(a, b):
    if isinstance(a, dict):
        assert sorted(a) == sorted(b)
        for k in a:
            assert_tree_equal(a[k], b[k])
    else:
        assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """Every subprocess run of this module, at most three at a time (4 CPUs, 2 eval workers each)."""
    root = tmp_path_factory.mktemp('bcsel')
    d = lambda n: str(root / n)  # noqa: E731
    lg = lambda n: str(root / f'{n}.log')  # noqa: E731
    base_script = d('train_v1_base.py')
    subprocess.run(['git', 'show', f'{BASE_COMMIT}:train_v1.py'], cwd=REPO, check=True,
                   stdout=open(base_script, 'w'))
    hook = lambda rates: {'V1_TEST_BC_EVAL_RATES': rates}  # noqa: E731
    # Interrupted runs: new best written at update 200, held there (V1_TEST_BEST_ACTOR_DELAY) so the
    # preemption flag / SIGKILL lands between best_actor.pkl and the checkpoint that commits it.
    held = dict(hook(NO_EXCEED), V1_TEST_BEST_ACTOR_DELAY='4')

    def chain(name, action):
        args = [*SEL, f'--run_dir={d(name)}']
        first = launch(args, lg(name), held, trigger=f'BEST_ACTOR_WRITTEN update={2 * I}', action=action)
        rest = launch(args, lg(name), hook(NO_EXCEED))
        again = launch(args, lg(name), hook(NO_EXCEED))
        return [first, rest, again]

    jobs = {
        'ref': lambda: launch([*SEL, f'--run_dir={d("ref")}'], lg('ref'), hook(NO_EXCEED)),
        'stop': lambda: launch([*SEL, f'--run_dir={d("stop")}'], lg('stop'), hook('0.2,0.5')),
        # The generated command shape (no hook): real rates.
        'real': lambda: launch(['--mode=bc', '--agent.actor_type=best-of-n', '--agent.actor_num_samples=32',
                                '--bc_select_threshold=0.45', f'--bc_steps={N}', f'--eval_interval={I}',
                                *COMMON, f'--run_dir={d("real")}'], lg('real')),
        'off': lambda: launch([*BC, f'--run_dir={d("off")}'], lg('off'), hook(NO_EXCEED)),
        'base': lambda: launch([*BC, f'--run_dir={d("base")}'], lg('base'), script=base_script),
        'flag': lambda: chain('flag', 'flag'),
        'kill': lambda: chain('kill', 'kill'),
    }
    with ThreadPoolExecutor(3) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    out['dir'] = d
    out['log'] = lg
    return out


def test_disabled_flag_matches_previous_train_v1(runs):
    off, base = runs['off'], runs['base']
    assert off['rc'] == 0 and base['rc'] == 0, (off['out'][-2000:], base['out'][-2000:])
    d_off, d_base = runs['dir']('off'), runs['dir']('base')
    for f in ('selection.json', 'selected_actor.pkl', 'best_actor.pkl'):
        assert not os.path.exists(os.path.join(d_off, f)), f
    assert sorted(os.listdir(d_off)) == sorted(os.listdir(d_base))
    assert stable(rows(d_off)) == stable(rows(d_base))
    assert read(os.path.join(d_off, 'final_actor.pkl')) == read(os.path.join(d_base, 'final_actor.pkl'))
    args = [json.load(open(os.path.join(x, 'args.json'))) for x in (d_off, d_base)]
    assert args[0].pop('run_dir') != args[1].pop('run_dir') and args[0] == args[1]  # no bc_select_threshold
    assert 'BEST_ACTOR_WRITTEN' not in off['out']


def selection(run_dir):
    sel = json.load(open(os.path.join(run_dir, 'selection.json')))
    blob = pickle.load(open(os.path.join(run_dir, 'selected_actor.pkl'), 'rb'))
    return sel, blob


def test_no_eval_above_threshold_selects_best_earliest_on_ties(runs):
    r = runs['ref']
    assert r['rc'] == 0, r['out'][-3000:]
    run_dir = runs['dir']('ref')
    sel, blob = selection(run_dir)
    assert sel['selection_rule'] == 'best_within_budget' and sel['threshold'] == 0.45
    assert sel['selected_update'] == 2 * I and sel['n1_success_rate'] == 0.3 and sel['bc_updates'] == N
    assert [(e['update'], e['success_rate']) for e in sel['evaluations']] == \
        [(I, 0.2), (2 * I, 0.3), (3 * I, 0.3), (4 * I, 0.1)]
    assert blob['selected_update'] == 2 * I and blob['bc_updates'] == 2 * I
    assert blob['selection_rule'] == 'best_within_budget' and blob['threshold'] == 0.45
    best = pickle.load(open(os.path.join(run_dir, 'best_actor.pkl'), 'rb'))
    final = pickle.load(open(os.path.join(run_dir, 'final_actor.pkl'), 'rb'))
    assert_tree_equal(best['actor_params'], blob['actor_params'])
    assert {k: v for k, v in blob.items() if k not in ('selection_rule', 'threshold', 'selected_update')} \
        .keys() == best.keys()
    assert set(final) == {'actor_params', 'agent_name', 'env_name', 'horizon_length', 'bc_steps', 'seed'}
    rs = rows(run_dir)
    # Logged evals are real (the hook only changes what the selection sees); one bc_selection row.
    evals = [x for x in rs if x['kind'] == 'eval']
    assert [x['nominal_step'] for x in evals] == [I, 2 * I, 3 * I, 4 * I]
    assert all(x['n1/success_rate'] in (0.0, 0.5, 1.0) for x in evals)
    srow = [x for x in rs if x['kind'] == 'bc_selection']
    assert len(srow) == 1 and srow[0]['selected_update'] == 2 * I and srow[0]['n_evaluations'] == 4
    assert [x for x in rs if x['kind'] == 'phase_end'][0]['bc_updates'] == N
    assert r['out'].count('BEST_ACTOR_WRITTEN') == 2  # 0.2 (first), 0.3; the tie at 300 keeps 200


def test_first_eval_above_threshold_stops_bc(runs):
    r = runs['stop']
    assert r['rc'] == 0, r['out'][-3000:]
    run_dir = runs['dir']('stop')
    sel, blob = selection(run_dir)
    assert sel['selection_rule'] == 'first_above_threshold' and sel['selected_update'] == 2 * I
    assert sel['n1_success_rate'] == 0.5 and sel['bc_updates'] == 2 * I
    assert [e['update'] for e in sel['evaluations']] == [I, 2 * I]
    rs = rows(run_dir)
    assert [x['nominal_step'] for x in rs if x['kind'] == 'eval'] == [I, 2 * I]
    assert [x for x in rs if x['kind'] == 'phase_end'][0]['bc_updates'] == 2 * I
    assert max(x['step'] for x in rs if x['kind'] == 'bc') <= 2 * I
    final = pickle.load(open(os.path.join(run_dir, 'final_actor.pkl'), 'rb'))
    assert_tree_equal(final['actor_params'], blob['actor_params'])  # stopped right at the selected update
    # The stop is identical to the budget run up to update 200 (same evaluations, same actor there).
    ref_best = pickle.load(open(os.path.join(runs['dir']('ref'), 'best_actor.pkl'), 'rb'))
    assert_tree_equal(ref_best['actor_params'], blob['actor_params'])


@pytest.mark.parametrize('name', ['flag', 'kill'])
def test_resume_after_interrupt_between_best_actor_and_checkpoint(runs, name):
    first, rest, again = runs[name]
    assert first['fired'], first['out'][-3000:]
    lines = first['lines']
    i = max(k for k, l in enumerate(lines) if l.startswith(f'BEST_ACTOR_WRITTEN update={2 * I}'))
    if name == 'flag':
        assert first['rc'] == 143, first['out'][-3000:]
        assert any(l.startswith('EXIT_PREEMPTED source=flag') for l in lines)
    else:
        assert first['rc'] == -9, first['out'][-3000:]
        # Killed before the checkpoint after the new best: the resume starts from the previous one and
        # puts the committed best (update 100) back before redoing update 200.
        assert not any(l.startswith('CKPT_DONE') for l in lines[i:]), lines[i:]
        assert any(l.startswith(f'BEST_ACTOR_RESTORED update={I}') for l in rest['lines']), rest['out'][-3000:]
    assert any(l.startswith('RESUME gen=') for l in rest['lines'])
    assert rest['rc'] == 0, rest['out'][-3000:]
    assert again['rc'] == 0 and any(l.startswith('RUN_ALREADY_COMPLETE') for l in again['lines'])
    a, b = runs['dir']('ref'), runs['dir'](name)
    for f in ('selection.json', 'selected_actor.pkl', 'best_actor.pkl', 'final_actor.pkl'):
        assert read(os.path.join(a, f)) == read(os.path.join(b, f)), f
    assert stable(rows(a)) == stable(rows(b))


def test_generated_command_shape_writes_selection(runs):
    r = runs['real']
    assert r['rc'] == 0, r['out'][-3000:]
    assert 'actor_num_samples=32' in r['out']
    sel, blob = selection(runs['dir']('real'))
    evals = [x for x in rows(runs['dir']('real')) if x['kind'] == 'eval']
    rates = [x['n1/success_rate'] for x in evals]
    assert [e['success_rate'] for e in sel['evaluations']] == rates
    if sel['selection_rule'] == 'first_above_threshold':
        assert rates[-1] > 0.45 and all(x <= 0.45 for x in rates[:-1])
    else:
        assert len(rates) == N // I and max(rates) <= 0.45
        assert sel['selected_update'] == evals[int(np.argmax(rates))]['nominal_step']  # argmax: earliest


def test_eval_actor_reproduces_selected_eval_and_online_loads_it(runs, tmp_path):
    assert runs['real']['rc'] == 0
    run_dir = runs['dir']('real')
    sel, _ = selection(run_dir)
    selected = os.path.join(run_dir, 'selected_actor.pkl')
    logged = [x for x in rows(run_dir) if x['kind'] == 'eval' and x['nominal_step'] == sel['selected_update']]
    assert len(logged) == 1
    r = launch(['--mode=eval_actor', '--agent.actor_type=best-of-n', f'--init_actor={selected}', *COMMON,
                f'--run_dir={tmp_path / "ev"}'], str(tmp_path / 'ev.log'))
    assert r['rc'] == 0, r['out'][-3000:]
    ev = json.load(open(tmp_path / 'ev' / 'eval_actor.json'))
    assert sorted(os.listdir(tmp_path / 'ev')) == ['eval_actor.json']  # no checkpoints, args, metrics
    assert ev['node'] == socket.gethostname()
    assert ev['successes'] == logged[0]['n1/successes'] and ev['episodes'] == logged[0]['n1/episodes']
    assert ev['success_rate'] == logged[0]['n1/success_rate']
    for k in ('episode.return', 'episode.length', 'episode.final_reward'):
        assert ev[k] == logged[0][f'n1/{k}'], k
    assert ev['init_actor']['selected_update'] == sel['selected_update']

    # --mode=online loads selected_actor.pkl with the same loader; stopped by a preemption notice once set up.
    on = tmp_path / 'on'
    r = launch(['--mode=online', '--method=bsc', f'--init_actor={selected}', '--warmup_steps=1000',
                '--pretrain_updates=10', '--online_steps=2000', '--buffer_cap=90000', *COMMON, f'--run_dir={on}'],
               str(tmp_path / 'on.log'), trigger='CONFIG ', action='flag')
    assert r['rc'] == 143, r['out'][-3000:]
    cfg = [x for x in rows(on) if x['kind'] == 'config']
    assert len(cfg) == 1 and cfg[0]['init_actor']['selected_update'] == sel['selected_update']
    assert cfg[0]['init_actor']['selection_rule'] == sel['selection_rule']
