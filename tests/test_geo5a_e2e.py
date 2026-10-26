"""geo5a end to end on a tiny real checkpoint (square; built as tests/test_geo4a_final_eval.py builds one).

BC actor (200 updates) + a tiny sampled-readout ARB-QC online run. Then, all exit 0: final_eval (4 seeds x 2 episodes),
diag_anchors with the BC actor (common panel) and with the run's own n1 / bo32 actor, diag_select on the panel
(--mc_candidates=2), diag_mc (K=2, QC contract past the limit) three times (A; B with another pool size and part
size; I interrupted by the preempt flag after the first part, then resumed), diag_mc under the BSC contract,
diag_score, a second diag_select. Checks: the replays start from final_eval's initial states, MC independence over k
and reproducibility per (q, k), resume == uninterrupted, selection unaffected by MC, refusals, source unchanged.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import json
import os
import pickle
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

QC = ['--method=arb_qc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
      '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7']
ENV = ['--env_name=square-mh-low_dim', '--horizon_length=5', '--seed=0']
REPLAY = ['--n_eval_envs=4', '--diag_n_seeds=4', '--diag_episodes=2']
T_EXTRA = 10


def go(d, name, args, interrupt=None):
    return launch([*ENV, *args, f'--run_dir={d / name}'], str(d / f'{name}.log'), str(d / f'{name}.flag'), interrupt)


def jload(p):
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope='module')
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo5a')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    r = run([*COMMON, *ONLINE, *QC, '--replay_rng=separate', f'--init_actor={d / "bc" / "final_actor.pkl"}',
             f'--run_dir={d / "qc"}'], d, 'qc')
    assert r['rc'] == 0, r['lines'][-30:]
    return d


@pytest.fixture(scope='module')
def outs(src):
    d = src
    qc, actor = d / 'qc', d / 'bc' / 'final_actor.pkl'
    before = {k: tree_hash(d / k) for k in ('qc', 'bc')}
    S = [*QC, f'--src_run_dir={qc}']
    out = {}

    def stage(jobs):
        with ThreadPoolExecutor(min(3, len(jobs))) as ex:
            futs = {k: ex.submit(f) for k, f in jobs.items()}
            out.update({k: f.result() for k, f in futs.items()})

    stage(dict(
        fe=lambda: go(d, 'fe', [*QC, '--mode=final_eval', f'--src_run_dir={qc}', '--final_n_seeds=4', '--n_eval_envs=4',
                                '--eval_episodes=2']),
        panel=lambda: go(d, 'panel', ['--mode=diag_anchors', *REPLAY, '--anchors=4', f'--policy_ckpt={actor}']),
        anchors=lambda: go(d, 'anchors', ['--mode=diag_anchors', *S, *REPLAY, '--anchors=6']),
    ))
    stage(dict(
        anchors_bo=lambda: go(d, 'anchors_bo', ['--mode=diag_anchors', *S, '--diag_rule=bo32', '--n_eval_envs=2',
                                                '--diag_n_seeds=2', '--diag_episodes=1', '--anchors=3']),
        select=lambda: go(d, 'select', ['--mode=diag_select', *S, f'--queries={d / "panel" / "anchors.pkl"}',
                                        '--mc_candidates=2']),
    ))
    sq = f'--queries={d / "select" / "select_queries.pkl"}'
    mc = ['--mode=diag_mc', *S, sq, '--mc_k=2', f'--mc_t_extra={T_EXTRA}']
    stage(dict(
        mcA=lambda: go(d, 'mcA', [*mc, '--n_eval_envs=4', '--mc_part_size=4']),
        mcB=lambda: go(d, 'mcB', [*mc, '--n_eval_envs=3', '--mc_part_size=5']),
        mcI=lambda: go(d, 'mcI', [*mc, '--n_eval_envs=4', '--mc_part_size=4'],
                       interrupt=flag('after the first part', starts_with('DIAG_PART mc'))),
    ))
    out['mcI_parts'] = sorted(os.listdir(d / 'mcI' / 'parts'))
    out['mcI_final'] = os.path.exists(d / 'mcI' / 'mc.json')
    stage(dict(
        mcI2=lambda: go(d, 'mcI', [*mc, '--n_eval_envs=4', '--mc_part_size=4']),
        mc_bsc=lambda: go(d, 'mc_bsc', ['--mode=diag_mc', *S, f'--queries={d / "anchors" / "anchors.pkl"}', '--mc_k=2',
                                        '--diag_contract=bsc', '--n_eval_envs=4']),
        score=lambda: go(d, 'score', ['--mode=diag_score', *S, sq]),
    ))
    stage(dict(
        select2=lambda: go(d, 'select2', ['--mode=diag_select', *S, f'--queries={d / "panel" / "anchors.pkl"}',
                                          '--mc_candidates=2', '--mc_seed=5']),
        score_again=lambda: go(d, 'score', ['--mode=diag_score', *S, sq]),
    ))
    out['before'] = before
    return d, out


ALL = ['fe', 'panel', 'anchors', 'anchors_bo', 'select', 'mcA', 'mcB', 'mcI2', 'mc_bsc', 'score', 'select2',
       'score_again']


def test_all_exit_0(outs):
    d, out = outs
    for k in ALL:
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-30:])
    for k in ('panel', 'anchors', 'anchors_bo'):
        print(k, [l for l in out[k]['lines'] if l.startswith('DIAG_ANCHORS')][0][:200])


def test_source_unchanged(outs):
    d, out = outs
    assert {k: tree_hash(d / k) for k in ('qc', 'bc')} == out['before']


def test_replays_start_from_final_eval_states(outs):
    d, out = outs
    fe = jload(d / 'fe' / 'final_eval_parts' / 'n1_0.json')
    for k in ('panel', 'anchors'):
        a = jload(d / k / 'anchors.json')
        assert a['first_obs_digest'] == fe['first_obs_digest'], k
        assert np.array(a['per_episode_success']).shape == (4, 2)
    print('replay initial states == final_eval (4 seeds x 2 episodes):', fe['first_obs_digest'])


def test_anchor_outputs(outs):
    d, out = outs
    for k, n, critic in (('panel', 4, False), ('anchors', 6, True), ('anchors_bo', 3, True)):
        a = jload(d / k / 'anchors.json')
        man = jload(d / k / 'manifest.json')
        assert man['complete'] and man['starts'][0]['code']['sha'] and man['counts']['anchors'] == n
        assert man['ident']['label'].startswith('diagnostic replays')
        assert len(a['anchors']) == n and all(x['weight'] > 0 for x in a['anchors'])
        assert all((x['critic'] is not None) == critic for x in a['anchors'])
        q = pickle.load(open(d / k / 'anchors.pkl', 'rb'))
        assert len(q['queries']) == len(q['states']) == n
        for s, x in zip(q['states'], q['queries']):
            assert s['t'] == x['meta']['t'] and x['chunk'].shape == (5 * 7,)
        print(k, 'strata', a['strata'], 'episodes', a['episodes'], 'successes', a['successes'])
    assert jload(d / 'anchors_bo' / 'manifest.json')['ident']['rule'] == 'bo32'
    src = jload(d / 'anchors' / 'manifest.json')['ident']['source']
    assert src['kind'] == 'checkpoint' and len(src['state_sha256']) == 64


def _mc(d, name):
    return {q['qid']: q for q in jload(d / name / 'mc.json')['queries']}


RUN_FIELDS = ('steps', 'success', 'decisions', 'first_actor_digest', 'final_obs_digest', 'G_cut', 'G_ext',
              'tail_bound', 'steps_past_limit')


def test_mc_independent_over_k_and_reproducible(outs):
    d, out = outs
    A, B = _mc(d, 'mcA'), _mc(d, 'mcB')
    assert sorted(A) == sorted(B) == list(range(len(A)))
    for q in A:
        assert A[q]['first_actor_digest'][0] != A[q]['first_actor_digest'][1], q  # k = 0, 1 differ
        for f in RUN_FIELDS:  # same (q, k): identical under another pool size / part arrangement
            assert A[q][f] == B[q][f], (q, f)
    print('mc A == B per (q, k) for', len(A), 'queries x K=2; k=0 != k=1 for every query')


def test_mc_resume_equals_uninterrupted(outs):
    d, out = outs
    I = out['mcI']
    assert I['rc'] == 143 and any(l.startswith('EXIT_PREEMPTED source=flag') for l in I['lines']), I['lines'][-20:]
    n_parts = len(os.listdir(d / 'mcA' / 'parts'))
    assert 1 <= len(out['mcI_parts']) < n_parts and not out['mcI_final'], out['mcI_parts']
    assert not [f for f in out['mcI_parts'] if f.endswith('.tmp')]
    skipped = [l for l in out['mcI2']['lines'] if l.startswith('DIAG_PART_SKIP')]
    assert len(skipped) == len(out['mcI_parts'])
    a, i = jload(d / 'mcA' / 'mc.json'), jload(d / 'mcI' / 'mc.json')
    assert a['queries'] == i['queries'] and a['ident'] == i['ident']
    assert len(jload(d / 'mcI' / 'manifest.json')['starts']) == 2


def test_mc_qc_contract(outs):
    d, out = outs
    m = jload(d / 'mcA' / 'mc.json')
    assert m['ident']['contract'] == 'qc' and m['ident']['t_extra'] == T_EXTRA
    for q in m['queries']:
        for k in range(2):
            n, s = q['steps'][k], q['success'][k]
            if not s:
                assert q['steps_past_limit'][k] == T_EXTRA and abs(q['tail_bound'][k] - 100 * 0.99 ** n) < 1e-9
                assert q['G_ext'][k] < q['G_cut'][k] <= 0
            else:
                assert q['tail_bound'][k] == 0
        assert abs(q['mean'] - np.mean(q['G_ext'])) < 1e-12 and q['se'] is not None
    print('QC queries', len(m['queries']), 'mean G_ext', np.mean([q['mean'] for q in m['queries']]))


def test_mc_bsc_contract(outs):
    d, out = outs
    m = jload(d / 'mc_bsc' / 'mc.json')
    assert m['ident']['contract'] == 'bsc' and m['ident']['t_extra'] == 0
    anchors = pickle.load(open(d / 'anchors' / 'anchors.pkl', 'rb'))
    for q in m['queries']:
        t = anchors['states'][q['state']]['t']
        for k in range(2):
            assert q['G'][k] in (0.0, 1.0) and q['G'][k] == q['success'][k]
            assert q['steps'][k] <= 400 - t and (q['success'][k] or q['steps'][k] == 400 - t)


def test_select_and_score(outs):
    d, out = outs
    sel = jload(d / 'select' / 'select.json')
    sq = pickle.load(open(d / 'select' / 'select_queries.pkl', 'rb'))
    assert len(sel['states']) == 4 and len(sq['queries']) == 4 * 4
    for r in sel['states']:
        s = np.array(r['score'])
        assert len(s) == 32 and r['chosen'] == int(np.argmax(s)) and r['chosen'] in r['subset']
        assert len(r['subset']) == 2 and 0 <= r['uniform'] < 32
    assert jload(d / 'select2' / 'select.json') == sel  # --mc_seed does not touch the selection
    sc = {q['qid']: q for q in jload(d / 'score' / 'scores.json')['queries']}
    for q in sq['queries']:
        assert len(sc[q['qid']]['keys']['score']) == 6 and sc[q['qid']]['keys']['score'][0] == sc[q['qid']]['score']
        assert abs(sc[q['qid']]['q_return'] - (100 * sc[q['qid']]['value'] - 100)) < 1e-9
        assert q['meta']['critic']['score'] == sel['states'][q['state']]['score'][q['meta']['candidate']]
    assert any(l.startswith('DIAG_SCORE_SKIP') for l in out['score_again']['lines'])

    # Re-derive in process from the checkpoint: select = the agent's best-of-32 readout layout for one observation
    # (Gumbel shape (2, 1, 32, M, L, C)) with key PRNGKey(0); diag_score = each chunk read out alone with PRNGKey(0).
    import flax
    import jax
    import jax.numpy as jnp
    import train_v1
    from agents import agents, v1_config
    state, _ = train_v1.read_committed_state(str(d / 'qc'))
    cfg = v1_config('arb_qc')
    cfg.update(critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10, arb_readout_temp=0.7,
               horizon_length=5)
    obs = np.stack([s['obs'] for s in sq['states']]).astype(np.float32)
    agent = agents['acarb'].create(0, obs[0], np.zeros(7, np.float32), cfg)
    params = flax.serialization.from_state_dict(agent.network.params, state['agent']['network']['params'])
    agent = agent.replace(network=agent.network.replace(params=params))
    from utils import diag_v1
    noise = np.stack([np.asarray(jax.random.normal(diag_v1.selection_noise_key(0, i), (32, 35))) for i in range(4)])
    for i in range(4):
        o = jnp.repeat(jnp.asarray(obs[i])[None, None], 32, 1)                     # (1, 32, obs_dim)
        cands = jnp.clip(agent.compute_flow_actions(o, jnp.asarray(noise[i])[None]), -1, 1)
        s, _ = agent.critic_score(o, cands, rng=jax.random.PRNGKey(0))
        assert np.asarray(s)[0].tolist() == sel['states'][i]['score'], i
        assert int(np.argmax(np.asarray(s)[0])) == sel['states'][i]['chosen']
    for q in sq['queries'][:6]:
        s, _ = agent.critic_score(jnp.asarray(obs[q['state']])[None], jnp.asarray(q['chunk'])[None],
                                  rng=jax.random.PRNGKey(0))
        assert int(np.asarray(s)[0]) == sc[q['qid']]['score']
    print('chosen == uniform at', sum(r['chosen'] == r['uniform'] for r in sel['states']), 'of 4 states; '
          'scores re-derived from the checkpoint')


def test_refusals_leave_source_unchanged(outs):
    d, out = outs
    qc = d / 'qc'
    sq = f'--queries={d / "select" / "select_queries.pkl"}'
    cases = dict(
        inside=([*QC, f'--src_run_dir={qc}', '--mode=diag_mc', sq, f'--run_dir={qc / "diag"}'], 'lies inside'),
        config=([*QC[:-1], '--agent.arb_readout_temp=0.5', f'--src_run_dir={qc}', '--mode=diag_score', sq],
                'agent.arb_readout_temp'),
        changed=([*QC, f'--src_run_dir={qc}', '--mode=diag_mc', sq, '--mc_k=3', '--mc_t_extra=10',
                  f'--run_dir={d / "mcA"}'], 'other inputs'),
    )
    for name, (args, msg) in cases.items():
        before = tree_hash(qc)
        a = list(args) if any(x.startswith('--run_dir') for x in args) else [*args, f'--run_dir={d / ("refuse_" + name)}']
        r = launch([*ENV, *a], str(d / f'refuse_{name}.log'), str(d / f'refuse_{name}.flag'))
        lines = [l for l in r['lines'] if l.startswith('DIAG_REFUSED')]
        assert r['rc'] == 2 and lines and msg in lines[0], (name, r['lines'][-20:])
        assert tree_hash(qc) == before
        print(name, lines[0][:160])


def test_n1_rows_match_eval_n1():
    """diag_v1.row_policy('n1') per row == train_v1.n1_actions on that row with the row's key (batch 1), and the
    batched call agrees with the one-row calls up to GPU float32 matmul differences between batch shapes (measured
    ~4e-5; n1_actions itself differs by as much between eval batch sizes)."""
    import jax
    import train_v1
    from agents import agents, v1_config
    from utils import diag_v1
    cfg = v1_config('arb_qc')
    cfg.update(critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10, arb_readout_temp=0.7,
               horizon_length=5)
    agent = agents['acarb'].create(0, np.zeros(23, np.float32), np.zeros(7, np.float32), cfg)
    obs = np.random.default_rng(0).normal(size=(6, 23)).astype(np.float32)
    keys = np.stack([np.asarray(jax.random.fold_in(jax.random.PRNGKey(4), i)) for i in range(6)])
    f = diag_v1.row_policy(agent, 'n1')
    batched = f(obs, keys)
    worst = 0.0
    for i in range(6):
        one = f(obs[i:i + 1], keys[i:i + 1])[0]
        ref = np.asarray(train_v1.n1_actions(agent, obs[i:i + 1], keys[i]))[0]
        assert np.array_equal(one, ref)
        worst = max(worst, float(np.abs(batched[i] - ref).max()))
    print(f'n1 rows: batch-1 rows bit-equal to train_v1.n1_actions; batched vs batch-1 max |diff| {worst:.2g}')
    assert worst < 1e-3
    g = diag_v1.row_policy(agent, 'bo32')
    b = g(obs[:2], keys[:2])
    assert np.array_equal(b[0], np.asarray(agent.sample_actions(obs[:1], rng=keys[0]))[0])
