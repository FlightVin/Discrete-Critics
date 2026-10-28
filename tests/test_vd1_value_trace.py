"""vd1: --mode=value_trace (square, tiny sizes).

One tiny BC actor and its fixed probe candidates; tiny online runs of each traced agent type (sampled-readout ARB-QC
acarb, scalar BSC acbsc, scalar QC acfql). final_eval and value_trace on 8 seeds x 2 episodes (2 chunks of 4 envs):
value_trace reproduces final_eval episode by episode with no bon argmax mismatch; scalar repeat reads equal the first
read, acarb's differ; --trace_repeats does not change the episodes; a second invocation skips every part and writes
identical outputs. In process: the bon replica equals each agent's own best-of-N (candidates, scores, choice), the
rounding helper and the action-identity counts on hand-made arrays, and the summary size at the 1,024-episode scale.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import launch  # noqa: E402
from test_geo1_e2e import COMMON, ONLINE, run, tree_hash  # noqa: E402
from test_geo4a_final_eval import ENV, QC, SMALL  # noqa: E402

AGENTS = dict(arb=QC, bsc=['--method=bsc', '--replay_rng=separate'], qc=['--method=qc', '--replay_rng=separate'])
SCALAR = {'arb': False, 'bsc': True, 'qc': True}
R = 2
SUMMARY_LIMIT = 3.5e6


def eval_args(d, name, src, agent):
    return [*ENV, *SMALL[1:], *AGENTS[agent][:-1], f'--src_run_dir={src}', f'--run_dir={d / name}']


def value_trace(d, name, src, agent, fe, repeats=R):
    args = [*eval_args(d, name, src, agent), '--mode=value_trace', f'--trace_repeats={repeats}',
            f'--ref_final_eval={d / fe / "final_eval.json"}', f'--probe_candidates={d / "cands.npz"}',
            '--probe_states=256']
    return launch(args, str(d / f'{name}.log'), str(d / f'{name}.flag'))


@pytest.fixture(scope='module')
def traces(tmp_path_factory):
    d = tmp_path_factory.mktemp('vd1')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    actor = str(d / 'bc' / 'final_actor.pkl')
    r = run(['--env_name=square-mh-low_dim', '--horizon_length=5', '--probe_states=256',
             '--mode=make_probe_candidates', f'--init_actor={actor}', f'--out={d / "cands.npz"}'], d, 'cands')
    assert r['rc'] == 0, r['lines'][-30:]
    with ThreadPoolExecutor(len(AGENTS)) as ex:
        futs = {a: ex.submit(run, [*COMMON, *ONLINE, f'--init_actor={actor}', *args, f'--run_dir={d / a}'], d, a)
                for a, args in AGENTS.items()}
        for a, f in futs.items():
            r = f.result()
            assert r['rc'] == 0, (a, r['lines'][-40:])
    hashes = {a: tree_hash(d / a) for a in AGENTS}

    def one(a):
        out = dict(fe=launch([*eval_args(d, f'fe_{a}', d / a, a), '--mode=final_eval'], str(d / f'fe_{a}.log'),
                             str(d / f'fe_{a}.flag')))
        out['vt'] = value_trace(d, f'vt_{a}', d / a, a, f'fe_{a}')
        if a == 'arb':
            with open(d / 'vt_arb' / 'value_trace.json', 'rb') as f:
                out['json'] = f.read()
            with open(d / 'vt_arb' / 'value_trace_summary.npz', 'rb') as f:
                out['summary'] = f.read()
            with open(d / 'vt_arb' / 'probe_trace.npz', 'rb') as f:
                out['probe'] = f.read()
            out['vt2'] = value_trace(d, 'vt_arb', d / a, a, 'fe_arb')
            out['vt0'] = value_trace(d, 'vt0_arb', d / a, a, 'fe_arb', repeats=0)
        return out

    with ThreadPoolExecutor(len(AGENTS)) as ex:
        out = dict(zip(AGENTS, ex.map(one, AGENTS)))
    for a, o in out.items():
        for k in ('fe', 'vt', 'vt2', 'vt0'):
            if k in o:
                assert o[k]['rc'] == 0, (a, k, o[k]['lines'][-40:])
        assert tree_hash(d / a) == hashes[a], a
    return d, out


def _parts(d, name, rule):
    return [(json.load(open(d / name / 'value_trace_parts' / f'{rule}_{c}.json')),
             dict(np.load(d / name / 'value_trace_parts' / f'{rule}_{c}.npz'))) for c in (0, 1)]


# ---- 1. reproduction ---------------------------------------------------------------------------------------
@pytest.mark.parametrize('agent', list(AGENTS))
def test_reproduces_final_eval(traces, agent):
    d, out = traces
    fe = json.load(open(d / f'fe_{agent}' / 'final_eval.json'))
    vt = json.load(open(d / f'vt_{agent}' / 'value_trace.json'))
    assert vt['ok'] and not vt['failures'], vt['failures']
    assert vt['n_episodes'] == 32 and vt['n_episodes_match'] == 32
    assert vt['first_obs_digests_checked'] == 32 and vt['first_obs_digest_mismatches'] == 0
    assert vt['bon_argmax_mismatches'] == 0
    for rule in ('n1', 'bon'):
        for c, (part, z) in enumerate(_parts(d, f'vt_{agent}', rule)):
            ref = json.load(open(d / f'fe_{agent}' / 'final_eval_parts' / f'{rule}_{c}.json'))
            for k in ('seeds', 'success', 'length', 'first_obs_digest'):
                assert part[k] == ref[k], (rule, c, k)
            L = np.array(part['length'])  # [envs, episodes]: one row per active env per decision
            n_dec = sum(int(np.ceil(l / 5)) for l in L.ravel())
            assert len(z['env']) == part['n_decisions'] == n_dec
            assert z['sel_value'].shape == z['sel_score'].shape == z['sel_oor'].shape == (n_dec, 32)
            assert z['rep_value'].shape == (n_dec, R, 32)
            assert z['episode'].max() == 1 and (z['decision'] < np.ceil(L.max() / 5)).all()
            if rule == 'bon':
                assert not z['argmax_mismatch'].any()
            else:
                assert z['exec_value'].shape == (n_dec,) and z['exec_rep_value'].shape == (n_dec, R)
            if agent == 'qc':
                np.testing.assert_array_equal(z['sel_u'], (z['sel_value'].astype(np.float64) + 100) / 100)
                np.testing.assert_array_equal(z['sel_q_return'], z['sel_value'].astype(np.float64))
            if agent == 'arb':
                assert 'sel_q_return' in z and 'sel_u' not in z
        print(agent, rule, fe[rule]['successes'], vt['rules'][rule])
    vt_wall = sum(p['wall_s'] for r in ('n1', 'bon') for p, _ in _parts(d, f'vt_{agent}', r))
    print(f'COST {agent} final_eval_wall_s={fe["wall_s"]:.1f} value_trace_wall_s={vt_wall:.1f} '
          f'ratio={vt_wall / fe["wall_s"]:.2f}')
    line = [l for l in out[agent]['vt']['lines'] if l.startswith('VALUE_TRACE ')]
    assert len(line) == 1 and 'per_episode' not in line[0]


# ---- 2. replication is exact; shadow reads do not touch the executed stream -----------------------------------
def _tiny_agent(method, overrides):
    from absl import flags
    from agents import agents, v1_config
    flags.FLAGS(['x'], known_only=True)
    config = v1_config(method)
    with config.unlocked():
        for k, v in overrides.items():
            config[k] = v
        config['horizon_length'] = 5
    return agents[config['agent_name']].create(0, np.zeros(9, np.float32), np.zeros(7, np.float32), config)


@pytest.mark.parametrize('method, overrides', [
    ('arb_qc', dict(critic_target_bon=False, arb_readout='sample_mean', arb_readout_samples=10, arb_readout_temp=0.7)),
    ('bsc', {}), ('qc', {})])
def test_bon_replica_equals_agent(method, overrides):
    import jax
    import jax.numpy as jnp
    import train_v1 as T
    agent = _tiny_agent(method, overrides)
    obs = jnp.asarray(np.random.RandomState(0).standard_normal((6, 9)), jnp.float32)
    for seed in range(3):
        rng = jax.random.PRNGKey(seed)
        a = np.asarray(T.bon_actions(agent, obs, rng))
        ck, rk = T.bon_trace_keys(agent, rng)
        obs_rep, cands = T._TRACE_CANDIDATES(agent, obs, ck)
        score = np.asarray(T._BATCH_SCORES(agent, obs_rep, cands, rk)[0])
        # the agent's own best-of-N scores on the same key
        if agent.config['agent_name'] == 'acarb':
            own_rng, own_rk = jax.random.split(rng)
            own_obs, own_cands = agent._candidates(obs, own_rng)
            own = np.asarray(agent.critic_score(own_obs, own_cands, rng=own_rk)[0])
        else:
            noises = jax.random.normal(rng, (6, 32, 35))
            own_obs = jnp.repeat(obs[:, None], 32, axis=1)
            own_cands = jnp.clip(agent.compute_flow_actions(own_obs, noises), -1, 1)
            q = agent.network.select('critic')(own_obs, own_cands)
            own = np.asarray((jax.nn.sigmoid(q) if agent.config['agent_name'] == 'acbsc' else q).mean(0))
        np.testing.assert_array_equal(np.asarray(cands), np.asarray(own_cands))
        np.testing.assert_array_equal(score, own)
        idx = score.argmax(-1)
        np.testing.assert_array_equal(np.asarray(cands)[np.arange(6), idx], a)


def test_trace_repeats_do_not_change_episodes(traces):
    d, _ = traces
    for rule in ('n1', 'bon'):
        for (p2, z2), (p0, z0) in zip(_parts(d, 'vt_arb', rule), _parts(d, 'vt0_arb', rule)):
            for k in ('success', 'length', 'first_obs_digest'):
                assert p2[k] == p0[k], (rule, k)
            assert z0['rep_value'].shape[1] == 0
            for k in ('env', 'episode', 'decision', 'sel_score', 'sel_value'):
                np.testing.assert_array_equal(z2[k], z0[k])
    vt0 = json.load(open(d / 'vt0_arb' / 'value_trace.json'))
    assert vt0['ok'] and vt0['n_episodes_match'] == 32


# ---- 3. repeated reads -------------------------------------------------------------------------------------
@pytest.mark.parametrize('agent', list(AGENTS))
def test_repeat_reads(traces, agent):
    d, _ = traces
    vt = json.load(open(d / f'vt_{agent}' / 'value_trace.json'))
    diffs = []
    for rule in ('n1', 'bon'):
        for _, z in _parts(d, f'vt_{agent}', rule):
            diffs.append(np.abs(z['rep_value'] - z['sel_value'][:, None]).max())
            if rule == 'n1':
                diffs.append(np.abs(z['exec_rep_value'] - z['exec_value'][:, None]).max())
    if SCALAR[agent]:
        assert max(diffs) == 0 and vt['scalar_repeat_max_abs_diff'] == 0
    else:
        assert max(diffs) > 0 and vt['scalar_repeat_max_abs_diff'] is None and vt['repeat_max_abs_diff'] > 0
    s = np.load(d / f'vt_{agent}' / 'value_trace_summary.npz')
    for rule in ('n1', 'bon'):
        if SCALAR[agent]:
            assert (s[f'{rule}/readout_sd_mean'] == 0).all() and (s[f'{rule}/argmax_agree'] == R).all()
            np.testing.assert_array_equal(s[f'{rule}/exec_value'], s[f'{rule}/exec_value_reread'])
        else:
            assert (s[f'{rule}/readout_sd_mean'] > 0).any() and 'bon/argmax_change_q1e3' not in s
    np.testing.assert_array_equal(s['bon/exec_value'], s['bon/cand_max'])  # bon executes the selection argmax
    assert s['probe/cand_max'].shape == (256,) and s['probe/demo_value'].shape == (256,)
    for g in ('n1', 'bon', 'probe'):
        nu, nt, ntd, sat = (s[f'{g}/{k}'] for k in ('n_unique_actions', 'n_ties_at_max', 'n_ties_at_max_distinct',
                                                    'cand_sat_frac'))
        assert nu.dtype == ntd.dtype == np.int8 and sat.dtype == np.float32
        assert nu.shape == ntd.shape == sat.shape == nt.shape
        assert (1 <= nu).all() and (nu <= 32).all() and (1 <= ntd).all() and (ntd <= nt).all() and (ntd <= nu).all()
        assert ((0 <= sat) & (sat <= 1)).all()
        print(f'ACTION_IDENTITY {agent} {g} rows={len(nu)} n_unique_actions<32={int((nu < 32).sum())} '
              f'ties={int((nt > 1).sum())} identical_chunk_ties={int((ntd < nt).sum())} '
              f'cand_sat_frac_mean={sat.mean():.3f}')
    for rule in ('n1', 'bon'):
        for _, z in _parts(d, f'vt_{agent}', rule):
            assert z['n_unique_actions'].shape == z['cand_sat_frac'].shape == (len(z['env']),)


# ---- 4. rounding helper ------------------------------------------------------------------------------------
def test_argmax_changes_after_rounding():
    from train_v1 import argmax_changes_after_rounding as f
    g = 1e-3
    u = np.array([
        [0.1, 0.2, 0.3],                     # clear winner: no change
        [0.5, 0.5, 0.4],                     # exact tie: first index both ways
        [0.3000, 0.3004, 0.1],               # both round to 0.300 -> argmax moves to index 0
        [0.3004, 0.3006, 0.1],               # 0.300 vs 0.301: no change
        [0.30049, 0.30051, 0.1],             # just below / above the half-grid boundary: 0.300 vs 0.301
        [0.3006, 0.30049, 0.1],              # 0.301 vs 0.300: no change
        [1.0004, 1.0001, 0.2],               # above 1, no clipping: both 1.000 -> index 0 (was 0 already)
        [1.0001, 1.0004, 0.2],               # above 1: tie after rounding -> moves to 0
        [-0.0003, -0.0001, -0.5],            # below 0: both round to 0 -> moves to 0
        [-0.0003, -0.0006, -0.5],            # 0 vs -0.001: no change
    ])
    want = [False, False, True, False, False, False, False, True, True, False]
    assert f(u, g).tolist() == want
    assert f(u[2:3], 5e-5).tolist() == [False]                             # 0.30000 vs 0.30040 on the finer grid
    assert f(np.array([[0.30001, 0.30004]]), 5e-5).tolist() == [False]    # 0.30000 vs 0.30005
    assert f(np.array([[0.30003, 0.30007]]), 5e-5).tolist() == [True]     # both 0.30005 -> index 0
    assert f(np.array([[0.300024, 0.300026]]), 5e-5).tolist() == [False]  # just below / above 0.300025
    assert f(np.array([[0.300026, 0.300074]]), 5e-5).tolist() == [True]   # both 0.30005 -> index 0
    assert f(np.array([[0.2, 0.7, 0.7], [0.7, 0.2, 0.7]]), g).tolist() == [False, False]  # ties: first index


def test_action_identity_stats():
    from train_v1 import action_identity_stats as f
    rs = np.random.RandomState(0)
    c = rs.uniform(-0.9, 0.9, (2, 6, 5, 7)).astype(np.float32)  # rows, N, H, adim
    score = np.array([[0.1, 0.2, 0.2, 0.2, 0.2, 0.0], [0.3, 0.1, 0.3, 0.3, 0.0, 0.2]], np.float32)
    # row 0: candidates 1 and 2 identical (both saturated) and tied at max; 3 and 4 distinct, also tied at max
    c[0, 1] = np.clip(c[0, 1] * 10, -1, 1)
    c[0, 2] = c[0, 1]
    c[0, 5, 0, :3] = [1.0, -1.0, 1.0]
    # row 1: candidate 4 (not tied) equals tied candidate 0; -0.0 == 0.0 exactly
    c[1, 4] = c[1, 0]
    c[1, 0, 0, 0], c[1, 4, 0, 0] = 0.0, -0.0
    out = f(c, score)
    assert out['n_unique_actions'].tolist() == [5, 5]
    assert out['n_ties_at_max_distinct'].tolist() == [3, 3]
    n_ties = (score == score.max(-1, keepdims=True)).sum(-1)
    assert n_ties.tolist() == [4, 3]  # row 0: one tie from an identical chunk; row 1: none
    sat0 = (2 * (np.abs(c[0, 1]) == 1).sum() + 3) / c[0].size
    np.testing.assert_allclose(out['cand_sat_frac'], [sat0, 0.0], rtol=1e-6)
    assert out['n_unique_actions'].dtype == out['n_ties_at_max_distinct'].dtype == np.int8
    assert out['cand_sat_frac'].dtype == np.float32
    # all 32 identical: 1 distinct chunk, 1 distinct tie, all tied
    same = np.ones((1, 32, 35), np.float32)
    o = f(same, np.zeros((1, 32)))
    assert (o['n_unique_actions'][0], o['n_ties_at_max_distinct'][0], o['cand_sat_frac'][0]) == (1, 1, 1.0)


# ---- 5. resume ---------------------------------------------------------------------------------------------
def test_resume_skips_and_identical(traces):
    d, out = traces
    lines = out['arb']['vt2']['lines']
    assert len([l for l in lines if l.startswith('VALUE_TRACE_PART_SKIP')]) == 4, lines[-20:]
    assert not [l for l in lines if l.startswith('VALUE_TRACE_PART ')]
    for f, key in (('value_trace.json', 'json'), ('value_trace_summary.npz', 'summary'), ('probe_trace.npz', 'probe')):
        with open(d / 'vt_arb' / f, 'rb') as fh:
            assert fh.read() == out['arb'][key], f


# ---- 6. summary size ---------------------------------------------------------------------------------------
@pytest.mark.parametrize('name, scalar', [('acarb', False), ('acbsc', True), ('acfql', True)])
def test_summary_size_1024_episodes(name, scalar, traces):
    """1,024 episodes per rule with the decisions per episode of tool hang at a mixed outcome (mean 80 policy calls,
    H = 8, 1000-step limit = 125 calls) and incompressible candidate values: the summary must stay < 3.5 MB.
    Also reports the tiny run's bytes per decision."""
    import train_v1 as T
    rs = np.random.RandomState(1)
    n_rep, N = 4, 32
    dpe = rs.randint(40, 126, 1024)
    dpe = np.clip(np.round(dpe * 80 / dpe.mean()).astype(int), 1, 125)

    def synth(n):
        if scalar:
            v = rs.uniform(0, 1, (n, N)) if name == 'acbsc' else rs.uniform(-100, 0, (n, N))
            v = v.astype(np.float32)
            rv = np.repeat(v[:, None], n_rep, 1)
            return v, v, rv, rv, np.zeros((n, N), np.float32)
        K = 32 * 2 * 10
        sc, rsc = rs.randint(0, K, (n, N)).astype(np.int32), rs.randint(0, K, (n, n_rep, N)).astype(np.int32)
        return (sc / K).astype(np.float32), sc, (rsc / K).astype(np.float32), rsc, \
            (rs.rand(n, N) < 0.01).astype(np.float32) / 20
    rules = {}
    for rule in ('n1', 'bon'):
        v, sc, rv, rsc, oor = synth(int(dpe.sum()))
        p = dict(sel_value=v, sel_score=sc, rep_value=rv, rep_score=rsc, sel_oor=oor,
                 episode=np.repeat(np.arange(1024), dpe), decision=np.concatenate([np.arange(k) for k in dpe]))
        if rule == 'bon':
            p['sel_index'] = np.argmax(sc, -1)
        else:
            p['exec_value'], p['exec_rep_value'] = v[:, 0], rv[:, :, 0]
        rules[rule] = p
    v, sc, rv, rsc, oor = synth(1024)
    def ident(n):
        """Action-identity counts as in the tiny runs (32 distinct chunks, 1 distinct chunk at max), 1% arbitrary;
        saturation a multiple of 1 / (N H adim) around the tiny runs' 0.075."""
        odd = rs.rand(n) < 0.01
        return dict(n_unique_actions=np.where(odd, rs.randint(1, 33, n), 32).astype(np.int8),
                    n_ties_at_max_distinct=np.where(odd, rs.randint(1, 3, n), 1).astype(np.int8),
                    cand_sat_frac=(rs.binomial(N * 8 * 7, 0.075, n) / (N * 8 * 7)).astype(np.float32))
    for p in rules.values():
        p.update(ident(len(p['episode'])))
    probe = dict(indices=np.arange(1024), value=v, score=sc, oor=oor, rep_value=rv.transpose(1, 0, 2), **ident(1024),
                 rep_score=rsc.transpose(1, 0, 2), demo_value=v[:, 0], demo_rep_value=rv[:, :, 0].T)
    size = len(T.npz_bytes(T.value_trace_summary(name, scalar, rules, probe, n_rep), lzma=True))
    d, _ = traces
    agent = {'acarb': 'arb', 'acbsc': 'bsc', 'acfql': 'qc'}[name]
    vt = json.load(open(d / f'vt_{agent}' / 'value_trace.json'))
    n_dec = sum(r['n_decisions'] for r in vt['rules'].values())
    print(f'SUMMARY_SIZE {name} synthetic_decisions={2 * int(dpe.sum())} bytes={size} '
          f'tiny_run bytes={vt["summary_bytes"]} decisions={n_dec}')
    assert size < SUMMARY_LIMIT, size
