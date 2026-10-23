"""geo2b: tools/geo_synth.py --updates 200 smoke for all ten variants (exit 0, finite, JSON keys present), and
the data recipe's groups. Run from the project root.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

VARIANTS = [f'E{i}' for i in range(10)]
GROUPS = ('base', 'k0', 'k19980', 'exact', 'boundary')


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    d = tmp_path_factory.mktemp('synth')
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')

    def one(v):
        out = d / f'{v}.json'
        with open(d / f'{v}.log', 'w') as f:
            p = subprocess.run([sys.executable, 'tools/geo_synth.py', '--variant', v, '--updates', '200',
                                '--out', str(out)], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=1800)
        return v, p.returncode, out, d / f'{v}.log'

    with ThreadPoolExecutor(5) as ex:
        return {v: (rc, out, log) for v, rc, out, log in ex.map(one, VARIANTS)}


def finite(x):
    return np.isfinite(x) if isinstance(x, float) else True


@pytest.mark.parametrize('variant', VARIANTS)
def test_smoke(variant, results):
    rc, out, log = results[variant]
    assert rc == 0, open(log).read()[-3000:]
    r = json.load(open(out))
    assert r['variant'] == variant and r['updates'] == 200
    for k in ('bias', 'mae', 'oor_frac', 'readout_var_mean', 'readout_identical_frac', 'ms_per_update',
              'ms_per_readout_32x256'):
        assert finite(r[k]) and isinstance(r[k], float), k
    assert set(r['groups']) == set(GROUPS)
    for g in GROUPS:
        assert r['groups'][g]['n'] > 0 and all(finite(v) for v in r['groups'][g].values()), g
    assert set(r['correction']) == {'-2', '-1', '1', '2'}
    assert all(c['n'] > 0 and finite(c['mae']) for c in r['correction'].values())
    assert np.isfinite(r['train_last100']['loss'])
    cfg = r['config']
    if variant == 'E0':
        assert cfg['arb_readout'] == 'greedy' and r['readout_identical_frac'] == 1.0 and r['oor_frac'] == 0.0
    else:
        assert cfg['arb_readout'] == 'sample_mean' and cfg['arb_readout_samples'] == 10
    if variant >= 'E5':
        assert cfg['arb_vocab'] == 'redundant'
        assert r['train_last100']['oov_label'] == 0 and r['train_last100']['oov_rollin'] == 0
    else:
        assert cfg['arb_vocab'] == 'ordinary' and r['oor_frac'] == 0.0


def test_data_recipe():
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    import geo_synth as G
    (x, k, g), (xt, kt, gt) = G.make_data()
    assert x.shape == (50000, 16) and xt.shape == (10000, 16) and not np.array_equal(x[:10000], xt)
    assert k.min() >= 0 and k.max() <= 19980
    for kk, gg in ((k, g), (kt, gt)):
        assert (kk[gg == 'k0'] == 0).all() and (kk[gg == 'k19980'] == 19980).all()
        assert (kk[gg == 'exact'] % 20 == 0).all()
        b = kk[gg == 'boundary']
        assert (np.isin(b % 20, (1, 19)) | (b == 0) | (b == 19980)).all()
        assert abs((gg == 'base').mean() - 10 / 16) < 0.02
