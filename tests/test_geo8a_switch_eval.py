"""geo8a: --mode=switch_eval (square, tiny sizes) and evaluate_parallel's optional on_reset hook.

One tiny online ARB-QC run (from one tiny BC actor); final_eval and switch_eval on 8 seeds x 2 episodes (2 chunks of
4 envs). bon_first k=0 must reproduce final_eval's n1 parts and k=10**6 its bon parts episode by episode (success and
length); a mixed k uses bon exactly for decisions d < k; a second invocation skips every part and writes the same
switch_eval.json.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import launch  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, run, tree_hash  # noqa: E402
from test_geo4a_final_eval import ENV, QC, SMALL, FakeVecEnv, ZeroPolicy  # noqa: E402

H = 5  # ENV's --horizon_length
BIG = 10 ** 6
KS = [0, 2, BIG]


def switch_eval(d, name, src, extra=()):
    args = [*ENV, *SMALL[1:], '--mode=switch_eval', *QC[:-1], f'--src_run_dir={src}', f'--run_dir={d / name}', *extra]
    return launch(args, str(d / f'{name}.log'), str(d / f'{name}.flag'))


@pytest.fixture(scope='module')
def evals(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo8a')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    r = run([*COMMON, *ONLINE, f'--init_actor={d / "bc" / "final_actor.pkl"}', *QC, f'--run_dir={d / "qc"}'], d, 'qc')
    assert r['rc'] == 0, r['lines'][-40:]
    src = d / 'qc'
    before = tree_hash(src)
    out = dict(fe=launch([*ENV, *SMALL, *QC[:-1], f'--src_run_dir={src}', f'--run_dir={d / "fe"}'],
                         str(d / 'fe.log'), str(d / 'fe.flag')))
    ks = '--switch_ks=' + ','.join(map(str, KS))
    out['S'] = switch_eval(d, 'S', src, ['--switch_order=bon_first', ks])
    with open(d / 'S' / 'switch_eval.json', 'rb') as f:
        out['S_json'] = f.read()
    out['S2'] = switch_eval(d, 'S', src, ['--switch_order=bon_first', ks])
    out['N'] = switch_eval(d, 'N', src, ['--switch_order=n1_first', '--switch_ks=2'])
    out['src_unchanged'] = tree_hash(src) == before
    for k in ('fe', 'S', 'S2', 'N'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    return d, out


def _part(d, name, sub, f):
    return json.load(open(d / name / sub / f))


# ---- evaluate_parallel: on_reset ----------------------------------------------------------------------------
def test_evaluate_parallel_on_reset():
    import evaluation
    kw = dict(agent=ZeroPolicy(), num_eval_envs=4, eval_episodes=3, action_dim=2, return_episodes=True)
    np.random.seed(0)
    *ref, ref_eps = evaluation.evaluate_parallel(envs=FakeVecEnv(4), **kw)
    n = []
    np.random.seed(0)
    *got, eps = evaluation.evaluate_parallel(envs=FakeVecEnv(4), on_reset=lambda: n.append(1), **kw)
    assert len(n) == 3
    assert dict(got[0]) == dict(ref[0]) and got[1:] == ref[1:]
    assert all(np.array_equal(eps[k], ref_eps[k]) for k in eps)


def test_alignment_check_raises():
    """The policy-call check: passes when called for exactly the active envs, raises otherwise."""
    from absl import flags
    import train_v1
    flags.FLAGS(['x'], known_only=True)
    envs = train_v1._AlignedEnvs(FakeVecEnv(4), 4)
    obs, _ = envs.reset()
    envs.check_call(obs, 0, 3)
    with pytest.raises(RuntimeError):
        envs.check_call(obs, 1, 3)  # wrong decision index for t = 0
    with pytest.raises(RuntimeError):
        envs.check_call(obs[::-1].copy(), 0, 3)  # not the active envs' observations
    for _ in range(3):
        obs, *_ = envs.step(np.zeros((4, 2)))  # env 0 (length 3) is done at t = 3
    assert envs.active.tolist() == [False, True, True, True]
    padded = np.concatenate([obs[1:], np.zeros((1, obs.shape[1]))])
    envs.check_call(padded, 1, 3)
    with pytest.raises(RuntimeError):
        envs.check_call(obs, 1, 3)  # the finished env asked for a chunk too


# ---- switch_eval on a tiny finished run ---------------------------------------------------------------------
def test_reproduces_final_eval(evals):
    d, out = evals
    assert out['src_unchanged']
    for c in (0, 1):
        n1, bon = (_part(d, 'fe', 'final_eval_parts', f'{r}_{c}.json') for r in ('n1', 'bon'))
        s0, sbig = (_part(d, 'S', 'switch_eval_parts', f'bon_first_{k}_{c}.json') for k in (0, BIG))
        assert s0['seeds'] == n1['seeds'] == sbig['seeds']
        assert s0['success'] == n1['success'] and s0['length'] == n1['length'], (s0, n1)
        assert sbig['success'] == bon['success'] and sbig['length'] == bon['length'], (sbig, bon)
        assert s0['n_bon_calls'] == 0 and s0['n_n1_calls'] > 0
        assert sbig['n_n1_calls'] == 0 and sbig['n_bon_calls'] > 0
    fe, se = json.load(open(d / 'fe' / 'final_eval.json')), json.load(open(d / 'S' / 'switch_eval.json'))
    by_k = {r['k']: r for r in se['settings']}
    assert [r['k'] for r in se['settings']] == KS
    for k, rule in ((0, 'n1'), (BIG, 'bon')):
        for key in ('successes', 'episodes', 'per_seed_successes', 'per_episode_success', 'per_episode_length',
                    'wilson95'):
            assert by_k[k][key] == fe[rule][key], (k, key)
    for key in ('src_run_dir', 'gen', 'ctr', 'seeds', 'agent'):
        assert se[key] == fe[key], key
    print('n1', fe['n1']['successes'], 'bon', fe['bon']['successes'], 'switch',
          {r['k']: r['successes'] for r in se['settings']})


@pytest.mark.parametrize('name, order, k', [('S', 'bon_first', 2), ('N', 'n1_first', 2)])
def test_mixed_setting_counts(evals, name, order, k):
    d, out = evals
    first, then = ('b', 'n') if order == 'bon_first' else ('n', 'b')
    for c in (0, 1):
        p = _part(d, name, 'switch_eval_parts', f'{order}_{k}_{c}.json')
        L = np.array(p['length'])  # [envs, episodes]
        n_dec = -(-L.max(0) // H)  # policy calls per episode: until the longest env is done
        assert [len(r) for r in p['call_rules']] == n_dec.tolist()
        assert p['call_rules'] == [first * min(k, n) + then * max(0, n - k) for n in n_dec]
        n_first = int(np.minimum(k, n_dec).sum())
        n_then = int(np.maximum(0, n_dec - k).sum())
        bon, n1 = (n_first, n_then) if order == 'bon_first' else (n_then, n_first)
        assert p['n_bon_calls'] == bon and p['n_n1_calls'] == n1, p
        if L.min() > k * H:  # no env ends before decision k
            assert (bon if order == 'bon_first' else n1) == k * 2
    lines = [l for l in out[name]['lines'] if l.startswith('SWITCH_EVAL ')]
    assert len(lines) == (len(KS) if name == 'S' else 1)
    assert all('per_episode' not in l for l in lines)


def test_resume_skips_and_identical(evals):
    d, out = evals
    skipped = [l for l in out['S2']['lines'] if l.startswith('SWITCH_EVAL_PART_SKIP')]
    assert len(skipped) == len(KS) * 2, out['S2']['lines'][-20:]
    assert not [l for l in out['S2']['lines'] if l.startswith('SWITCH_EVAL_PART ')]
    with open(d / 'S' / 'switch_eval.json', 'rb') as f:
        assert f.read() == out['S_json']
