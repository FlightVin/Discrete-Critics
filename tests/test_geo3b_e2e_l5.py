"""geo3b end to end at L = 5 (square, tiny step counts): the E1, E2, E5, E6, E7, E9, E11 command shapes of the recipes
plus --agent.arb_length=5 through every phase, and E2 / E11 preempted (flag) in pretrain and joint against the
uninterrupted runs. L = 3 against 6551f0d: tests/test_geo3b_e2e_l3.py.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set. Subprocesses run with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (tests/e2e_v1_interrupt.launch). At most 3 runs at a time.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import chain, flag, progress_at_least  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, H, e_args, rows, run  # noqa: E402

L5 = ['--agent.arb_length=5']
LABEL = {'E1': 'hard', 'E2': 'twohot_last', 'E5': 'corrective', 'E6': 'corrective', 'E7': 'corrective',
         'E9': 'noisy_resid', 'E11': 'noisy_resid'}
ROLLIN = {'E5': 'corrective', 'E6': 'model_mix', 'E7': 'fixed', 'E9': 'fixed', 'E11': 'model_mix'}
WORKERS = 3


def v_args(variant, cands):
    """The recipe's command shape of the variant (geo1 E-shape; redundant: vocabulary and roll-in)."""
    args = e_args(LABEL[variant], cands)
    if variant in ROLLIN:
        args += ['--agent.arb_vocab=redundant', f'--agent.arb_rollin={ROLLIN[variant]}']
    return args


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo3b_l5')
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
    d, actor, cands = setup
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = {v: (lambda v=v: run([*online, *v_args(v, cands), *L5, f'--run_dir={d / v}'], d, v)) for v in LABEL}
    interrupts = [flag('during pretrain updates', progress_at_least('pretrain', 'updates', 100)),
                  flag('during joint updates', progress_at_least('joint', 'env_steps', 2000))]
    for v in ('E2', 'E11'):
        jobs[v + 'I'] = lambda v=v: chain(v + 'I', [*online, *v_args(v, cands), *L5, f'--run_dir={d / (v + "I")}'],
                                          str(d / f'{v}I.log'), str(d / f'{v}I.flag'), interrupts)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out


@pytest.mark.parametrize('variant', list(LABEL))
def test_l5_runs_every_phase(variant, runs):
    d, out = runs
    r = out[variant]
    assert r['rc'] == 0, r['lines'][-40:]
    m = rows(d / variant)
    cfg = [x for x in m if x['kind'] == 'config']
    assert len(cfg) == 1
    agent = cfg[0]['agent']
    assert agent['arb_length'] == 5 and agent.get('arb_label', 'hard') == LABEL[variant]
    assert agent.get('arb_rollin', 'canonical') == ROLLIN.get(variant, 'canonical')
    assert agent.get('arb_vocab', 'ordinary') == ('redundant' if variant in ROLLIN else 'ordinary')
    assert agent['arb_readout'] == 'sample_mean' and agent['arb_readout_samples'] == 10
    assert agent['arb_readout_temp'] == 0.7 and agent['critic_target_bon'] is False
    assert [x['phase'] for x in m if x['kind'] == 'phase_end'] == ['warmup', 'pretrain', 'pretrain_eval', 'joint']
    for kind in ('pretrain', 'joint'):
        tr = [x for x in m if x['kind'] == kind]
        assert tr, kind
        for x in tr:
            assert np.isfinite(x['critic/critic_loss']), x
            for l in range(1, 6):
                for k in (f'critic/ce_digit{l}', f'critic/tf_acc_digit{l}', f'critic/greedy_acc_digit{l}'):
                    assert np.isfinite(x[k]), k
            assert 'critic/ce_digit6' not in x
            if variant in ROLLIN:
                assert x['critic/oov_label'] == 0 and x['critic/oov_rollin'] == 0, x
                assert np.isfinite(x['critic/label_mean_bias'])
                if ROLLIN[variant] == 'model_mix':
                    assert 0.3 < x['critic/model_used_frac'] < 0.7, x
                else:
                    assert x['critic/model_used_frac'] == 0, x
            else:
                assert x['critic/readout_oor_frac'] == 0.0
                for l in range(1, 6):
                    assert np.isfinite(x[f'critic/label_clip_pos{l}'])
    rd = [x for x in m if x['kind'] == 'readout']
    assert rd and all(np.isfinite(x['critic/q_readout']) and 'critic/expected_q' not in x for x in rd)
    evals = [x for x in m if x['kind'] in ('eval', 'pretrain_eval')]
    assert len(evals) == 3
    for x in evals:
        for k in ('probe/tie_rate', 'probe/tie_max', 'probe_fixed/tie_all', 'probe_fixed/tie_max',
                  'probe_fixed/score_mean', 'probe_fixed/cand_range_mean', 'probe_fixed/oor_frac',
                  'probe_stab/score_var_mean', 'probe_stab/same_argmax_frac'):
            assert np.isfinite(x[k]), k


@pytest.mark.parametrize('variant', ['E2', 'E11'])
def test_l5_interrupted_resume_matches_uninterrupted(variant, runs):
    d, out = runs
    res = out[variant + 'I']
    assert [e['rc'] for e in res] == [143, 143], res
    a, i = rows(d / variant), rows(d / (variant + 'I'))
    assert len(a) == len(i)
    for x, y in zip(a, i):
        assert x == y, (x, y)
