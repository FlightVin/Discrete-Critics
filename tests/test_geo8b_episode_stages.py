"""geo8b: --mode=episode_stages on the real tool hang E9 checkpoint (1 seed x 2 rules, read-only) and make_env's
optional render_offscreen.

The run writes episode_stages.json (per-episode fields), one video per rule and the side-by-side video; the stage
codes are consistent with the reported first-reach steps; the initial state equals final_eval's first episode of
the same seed (when the E9 final_eval parts exist); a second invocation skips both parts and writes the same json.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set.
"""
import inspect
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

DATA = '/data/user_data/vineethb/Discrete-Critics/geo'
E9 = f'{DATA}/geo-th3-E9-residual-label-noise-s0'
E9_FINAL = f'{DATA}/final/geo-th3-E9-residual-label-noise-s0/final_eval_parts/n1_0.json'
# As exp/geo-final-geo-th3-E9-residual-label-noise-s0.sh (the source run's flags).
E9_ARGS = ['--method=arb_bsc', '--agent.critic_target_bon=False', '--agent.arb_readout=sample_mean',
           '--agent.arb_readout_samples=10', '--agent.arb_readout_temp=0.7', '--agent.arb_vocab=redundant',
           '--agent.arb_label=noisy_resid', '--agent.arb_rollin=fixed', '--env_name=tool_hang-ph-low_dim',
           '--horizon_length=8']
SEED = 100000
H = 8


# ---- make_env -------------------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


@pytest.mark.parametrize('kw, want', [({}, False), ({'render_offscreen': True}, True)])
def test_make_env_render_offscreen(monkeypatch, kw, want):
    from envs import robomimic_utils as RU
    assert inspect.signature(RU.make_env).parameters['render_offscreen'].default is False
    got = {}

    def fake(**k):
        got.update(k)
        raise _Stop()
    monkeypatch.setattr(RU.EnvUtils, 'create_env_from_metadata', fake)
    with pytest.raises(_Stop):
        RU.make_env('square-mh-low_dim', seed=0, **kw)
    assert got['render'] is False and got['render_offscreen'] is want


# ---- episode_stages on the E9 tool hang checkpoint ------------------------------------------------------------
@pytest.fixture(scope='module')
def stages(tmp_path_factory):
    if not os.path.exists(os.path.join(E9, 'ckpt', 'LATEST')):
        pytest.skip(f'no committed checkpoint in {E9}')
    d = tmp_path_factory.mktemp('geo8b')
    latest = os.stat(os.path.join(E9, 'ckpt', 'LATEST')).st_mtime_ns
    args = ['--mode=episode_stages', *E9_ARGS, f'--src_run_dir={E9}', f'--run_dir={d / "es"}',
            '--stage_n_seeds=1', '--video_n_seeds=1']
    r1 = launch(args, str(d / 'es.log'), str(d / 'es.flag'))
    assert r1['rc'] == 0, r1['lines'][-40:]
    with open(d / 'es' / 'episode_stages.json', 'rb') as f:
        j1 = f.read()
    r2 = launch(args, str(d / 'es.log'), str(d / 'es.flag'))
    assert r2['rc'] == 0, r2['lines'][-40:]
    assert os.stat(os.path.join(E9, 'ckpt', 'LATEST')).st_mtime_ns == latest
    return d / 'es', r1, r2, j1


def test_json_and_videos(stages):
    d, r1, _, _ = stages
    out = json.load(open(d / 'episode_stages.json'))
    eps = {e['rule']: e for e in out['episodes']}
    assert sorted(eps) == ['bon', 'n1'] and len(out['episodes']) == 2
    for rule, e in eps.items():
        for k in ('seed', 'rule', 'success', 'length', 'decisions', 'first_stage1_step', 'first_stage2_step',
                  'stage_per_decision'):
            assert k in e, (rule, k)
        assert e['seed'] == SEED
        assert 1 <= e['length'] <= 1000 and e['decisions'] == -(-e['length'] // H)
        assert len(e['stage_per_decision']) == e['decisions'] and set(e['stage_per_decision']) <= {0, 1, 2}
        assert sum(e['steps_at_stage']) == e['length']
        s1, s2 = e['first_stage1_step'], e['first_stage2_step']
        # success <=> stage 2 (frame assembled + tool on frame = robosuite success) at the last step
        assert (s2 == e['length']) == bool(e['success']) and (s2 is None) == (not e['success'])
        if s2 is not None:
            assert s1 is not None and s1 <= s2 and e['stage_per_decision'][-1] == 2
        if s1 is None:
            assert e['steps_at_stage'][1] == e['steps_at_stage'][2] == 0 and max(e['stage_per_decision']) == 0
        else:
            # the per-decision code is 0 before the decision holding step s1
            assert max(e['stage_per_decision'][:(s1 - 1) // H], default=0) == 0
        v = d / 'videos' / f'{rule}_seed{SEED}.mp4'
        assert e['video'] == str(v) and 0 < os.path.getsize(v) <= 400_000
        print(f'{rule}: success={e["success"]} length={e["length"]} stage1={s1} stage2={s2} '
              f'video={os.path.getsize(v)}B steps_at_stage={e["steps_at_stage"]}')
    sbs = d / 'videos' / f'side_by_side_seed{SEED}.mp4'
    assert 0 < os.path.getsize(sbs) <= 400_000
    import imageio
    fr = imageio.v2.mimread(sbs, memtest=False)
    assert fr[0].shape[:2] == (160, 320)
    # a rendered scene, not noise: mean |horizontal pixel difference| ~5 for agentview, ~60 for corrupted frames
    rough = [float(np.abs(np.diff(f[20:, :, :3].astype(int), axis=1)).mean()) for f in (fr[0], fr[len(fr) // 2])]
    assert max(rough) < 20, rough
    n_frames = max(eps[r]['length'] // 4 + 1 + (eps[r]['length'] % 4 > 0) for r in eps)
    assert abs(len(fr) - n_frames) <= 1, (len(fr), n_frames)
    print(f'side_by_side={os.path.getsize(sbs)}B frames={len(fr)}')
    assert [l for l in r1['lines'] if l.startswith('EPISODE_STAGES ')]


def test_initial_state_as_final_eval(stages):
    if not os.path.exists(E9_FINAL):
        pytest.skip(f'{E9_FINAL} not found')
    d = stages[0]
    part = json.load(open(E9_FINAL))
    assert part['seeds'][0] == SEED
    for e in json.load(open(d / 'episode_stages.json'))['episodes']:
        assert e['first_obs_digest'] == part['first_obs_digest'][0][0], e['rule']


def test_resume_skips_parts(stages):
    d, _, r2, j1 = stages
    skips = [l for l in r2['lines'] if l.startswith('EPISODE_STAGES_PART_SKIP')]
    assert len(skips) == 2 and not [l for l in r2['lines'] if l.startswith('EPISODE_STAGES_PART ')]
    with open(d / 'episode_stages.json', 'rb') as f:
        assert f.read() == j1


def test_e9_n1_stage1_reported(stages):
    e = [e for e in json.load(open(stages[0] / 'episode_stages.json'))['episodes'] if e['rule'] == 'n1'][0]
    assert e['first_stage1_step'] is not None or e['success'] == 0
    print(f'E9 n1 seed {SEED}: first_stage1_step={e["first_stage1_step"]} success={e["success"]} '
          f'length={e["length"]} stage_per_decision={"".join(map(str, e["stage_per_decision"]))}')
