"""geo3b: L = 3 unchanged. Tiny end-to-end runs of E1, E2, E5, E6, E7, E9 (recipe command shapes, default arb_length)
here and at the parent commit 6551f0d: every metrics row identical except wall-clock fields. The default arb_bsc run
against 8f1d09e is tests/test_geo1_e2e.py.

Run from the project root with MUJOCO_GL=egl and ROBOMIMIC_DATA_DIR set. At most 3 runs at a time.
"""
import io
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.chdir(REPO)

from test_geo1_e2e import COMMON, ONLINE, H, rows, run  # noqa: E402
from test_geo3b_e2e_l5 import v_args  # noqa: E402

PARENT = os.environ.get('GEO3B_PARENT_COMMIT', '6551f0d')  # the commit geo3b branched from
VARIANTS = ['E1', 'E2', 'E5', 'E6', 'E7', 'E9']


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp('geo3b_l3')
    r = run([*COMMON, '--mode=bc', '--agent=agents/acbsc.py', '--agent.actor_type=best-of-n', '--bc_steps=200',
             '--eval_interval=200', f'--run_dir={d / "bc"}'], d, 'bc')
    assert r['rc'] == 0, r['lines'][-30:]
    actor = str(d / 'bc' / 'final_actor.pkl')
    cands = str(d / 'cands.npz')
    r = run(['--env_name=square-mh-low_dim', f'--horizon_length={H}', '--probe_states=256',
             '--mode=make_probe_candidates', f'--init_actor={actor}', f'--out={cands}'], d, 'cands')
    assert r['rc'] == 0, r['lines'][-30:]
    parent = d / 'parent_src'
    tar = subprocess.run(['git', 'archive', PARENT], cwd=REPO, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(parent)
    online = [*COMMON, *ONLINE, f'--init_actor={actor}']
    jobs = {}
    for v in VARIANTS:
        args = [*online, *v_args(v, cands)]
        jobs[v] = lambda v=v, a=args: run([*a, f'--run_dir={d / v}'], d, v)
        jobs[v + 'p'] = lambda v=v, a=args: run([*a, f'--run_dir={d / (v + "p")}'], d, v + 'p', cwd=str(parent))
    with ThreadPoolExecutor(3) as ex:
        futs = {k: ex.submit(f) for k, f in jobs.items()}
        out = {k: f.result() for k, f in futs.items()}
    return d, out


@pytest.mark.parametrize('variant', VARIANTS)
def test_l3_rows_match_parent_commit(variant, runs):
    d, out = runs
    for k in (variant, variant + 'p'):
        assert out[k]['rc'] == 0, (k, out[k]['lines'][-40:])
    new, old = rows(d / variant), rows(d / (variant + 'p'))
    assert len(new) == len(old) and len(new) > 10
    for x, y in zip(new, old):
        if x['kind'] == 'config':
            x, y = dict(x, init_actor=None), dict(y, init_actor=None)
        assert x == y, (x['kind'], x['step'], {k: (x.get(k), y.get(k)) for k in set(x) | set(y)
                                               if x.get(k) != y.get(k)})
