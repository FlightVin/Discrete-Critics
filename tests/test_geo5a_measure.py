"""geo5a measure (skipped if absent): wall times on the real toolhang checkpoint geo-th3-E11 (arb_bsc, H=8), read-only.

  1. diag_anchors: 4 seeds x 1 episode (n1 diagnostic replays), 4 anchors.
  2. diag_mc, BSC contract (the checkpoint's own): those 4 queries x K=8 continuations on a 32-env pool.
  3. diag_mc, QC contract (--diag_contract=qc): one query x K=8 continuations past the step limit (t_extra 757).
Outputs go to the tinker scratch ($GEO5A_SCRATCH, default exp/scratch/geo5a_measure, git-ignored); the source is
hashed before and after. Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import hashlib
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from e2e_v1_interrupt import launch  # noqa: E402

TH = '/data/user_data/vineethb/Discrete-Critics/geo/geo-th3-E11-model-prefix-resid-label-s0'
AGENT = ['--method=arb_bsc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
         '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--agent.arb_label=noisy_resid',
         '--agent.arb_vocab=redundant', '--agent.arb_rollin=model_mix']
ENV = ['--env_name=tool_hang-ph-low_dim', '--horizon_length=8', '--seed=0']
SCRATCH = os.environ.get('GEO5A_SCRATCH', os.path.join(REPO, 'exp', 'scratch', 'geo5a_measure'))


def _ckpt_hash(src):
    h = hashlib.sha256()
    gen = open(os.path.join(src, 'ckpt', 'LATEST')).read().strip()
    for p in ('args.json', 'metrics.jsonl', 'ckpt/LATEST', f'ckpt/{gen}/manifest.json', f'ckpt/{gen}/state.pkl'):
        h.update(p.encode() + hashlib.sha256(open(os.path.join(src, p), 'rb').read()).digest())
    return h.hexdigest()


def go(name, args):
    d = os.path.join(SCRATCH, name)
    r = launch([*ENV, *AGENT, f'--src_run_dir={TH}', *args, f'--run_dir={d}'], d + '.log', d + '.flag')
    assert r['rc'] == 0, (name, r['lines'][-40:])
    return r


@pytest.mark.skipif(not os.path.exists(os.path.join(TH, 'ckpt', 'LATEST')), reason='th3 checkpoint not present')
def test_measure_toolhang():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)
    before = _ckpt_hash(TH)
    r = go('anchors', ['--mode=diag_anchors', '--n_eval_envs=4', '--diag_n_seeds=4', '--diag_episodes=1',
                       '--anchors=4'])
    a = json.load(open(os.path.join(SCRATCH, 'anchors', 'anchors.json')))
    lines = [l for l in r['lines'] if l.startswith(('DIAG_PART', 'DIAG_ANCHORS'))]

    r = go('mc_bsc', ['--mode=diag_mc', f'--queries={SCRATCH}/anchors/anchors.pkl', '--mc_k=8', '--n_eval_envs=32',
                      '--mc_part_size=4'])
    m = json.load(open(os.path.join(SCRATCH, 'mc_bsc', 'mc.json')))
    assert m['ident']['contract'] == 'bsc' and len(m['queries']) == 4 and all(q['K'] == 8 for q in m['queries'])
    lines += [l for l in r['lines'] if l.startswith('DIAG_PART')]

    q = pickle.load(open(os.path.join(SCRATCH, 'anchors', 'anchors.pkl'), 'rb'))
    one = dict(q, states=[q['states'][0]], queries=[dict(q['queries'][0], qid=0, state=0)])
    with open(os.path.join(SCRATCH, 'one_query.pkl'), 'wb') as f:
        pickle.dump(one, f)
    r = go('mc_qc', ['--mode=diag_mc', f'--queries={SCRATCH}/one_query.pkl', '--mc_k=8', '--n_eval_envs=8',
                     '--diag_contract=qc'])
    mq = json.load(open(os.path.join(SCRATCH, 'mc_qc', 'mc.json')))
    lines += [l for l in r['lines'] if l.startswith('DIAG_PART')]
    assert _ckpt_hash(TH) == before
    qq = mq['queries'][0]
    print('MEASURE (cpus=%d)' % os.cpu_count(), *lines,
          f'anchors: episodes={a["episodes"]} successes={a["successes"]} decisions={a["decisions"]} '
          f'replay wall_s={a["wall_s"]:.1f}',
          f'mc bsc 4 queries x K=8: wall_s={m["wall_s"]:.1f}, mean G per query '
          f'{[round(x["mean"], 3) for x in m["queries"]]}, steps {[x["steps"] for x in m["queries"]]}',
          f'mc qc 1 query x K=8 (t={q["states"][0]["t"]}, t_extra={mq["ident"]["t_extra"]}): wall_s={mq["wall_s"]:.1f}, '
          f'steps {qq["steps"]}, G_cut mean {qq["mean_cut"]:.3f}, G_ext mean {qq["mean"]:.3f} (se {qq["se"]}), '
          f'max tail bound {qq["max_tail_bound"]:.4f}', sep='\n  ')
