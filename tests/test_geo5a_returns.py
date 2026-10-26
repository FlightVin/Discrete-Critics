"""geo5a: Monte-Carlo return accounting (diag_v1.ReturnAccumulator) against numpy references, the QC tail bound and
t_extra, the stratified anchor rule and the selection-test RNG helpers. No env, no GPU work."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import diag_v1  # noqa: E402

G = 0.99


def _run(contract, seq, t_extra=0):
    acc = diag_v1.ReturnAccumulator(contract, t_extra)
    for x in seq:
        if acc.add(*x):
            break
    return acc


def _seq(n, success_at=None, limit_at=None):
    """Raw env outputs (r_env, terminated, truncated) per step; limit_at: 1-based step where t hits the limit (the
    wrapper reports truncated on that step and every later one); success_at: 1-based success step."""
    out = []
    for t in range(1, n + 1):
        succ = success_at == t
        out.append((1.0 if succ else 0.0, succ, (not succ) and limit_at is not None and t >= limit_at))
    return out


def test_qc_success_before_limit():
    a = _run('qc', _seq(50, success_at=37, limit_at=40), t_extra=100).result()
    ref = -(1 - G ** 36) / (1 - G)  # 36 steps of -1, then 0
    assert abs(a['G_cut'] - ref) < 1e-12 and abs(a['G_ext'] - ref) < 1e-12
    assert a['steps'] == 37 and a['success'] == 1 and a['success_before_limit'] == 1 and a['tail_bound'] == 0.0


def test_qc_timeout_continues_past_limit():
    T = 25
    a = _run('qc', _seq(200, limit_at=40), t_extra=T).result()
    assert a['steps'] == 40 + T and a['steps_past_limit'] == T and a['success'] == 0
    assert abs(a['G_cut'] + (1 - G ** 40) / (1 - G)) < 1e-12        # stopped at the limit (step 40 included)
    assert abs(a['G_ext'] + (1 - G ** (40 + T)) / (1 - G)) < 1e-12  # continued T more steps
    assert abs(a['tail_bound'] - 100 * G ** (40 + T)) < 1e-12
    # the bound holds for any continuation: |G_true - G_ext| <= 100 * 0.99^n
    assert abs(-100.0 - a['G_ext']) <= a['tail_bound'] + 1e-12


def test_qc_success_past_limit():
    a = _run('qc', _seq(200, success_at=55, limit_at=40), t_extra=100).result()
    assert a['steps'] == 55 and a['success'] == 1 and a['success_before_limit'] == 0 and a['tail_bound'] == 0.0
    assert abs(a['G_ext'] + (1 - G ** 54) / (1 - G)) < 1e-12 and abs(a['G_cut'] + (1 - G ** 40) / (1 - G)) < 1e-12


def test_qc_matches_numpy_reference_random():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 120))
        limit = int(rng.integers(1, n + 1)) if rng.random() < 0.7 else None
        succ = int(rng.integers(1, n + 1)) if rng.random() < 0.5 else None
        t_extra = int(rng.integers(0, 30))
        seq = _seq(n, succ, limit)
        a = _run('qc', seq, t_extra)
        r, te, tr = map(np.array, zip(*seq))
        ref = diag_v1.qc_reference(r, te, tr, t_extra)
        res = a.result()
        if not a.done:  # the sequence ran out first (no limit, no success): compare the steps given
            assert ref['steps'] == n
        for k in ('G_cut', 'G_ext', 'steps', 'success', 'tail_bound'):
            assert abs(res[k] - ref[k]) < 1e-9, (k, res, ref, n, limit, succ, t_extra)


def test_bsc():
    assert _run('bsc', _seq(50, success_at=30, limit_at=40)).result() == dict(G=1.0, success=1, steps=30)
    assert _run('bsc', _seq(50, limit_at=40)).result() == dict(G=0.0, success=0, steps=40)
    # success after the episode's own limit does not count (the clock is kept)
    assert _run('bsc', _seq(60, success_at=45, limit_at=40)).result() == dict(G=0.0, success=0, steps=40)


def test_t_extra_and_bound():
    n = diag_v1.qc_t_extra(0.05)
    assert n == 757 and diag_v1.qc_tail_bound(n) <= 0.05 < diag_v1.qc_tail_bound(n - 1)
    print(f'QC t_extra = {n}: tail bound 100 * 0.99^{n} = {diag_v1.qc_tail_bound(n):.4f} return units')


def test_primitive_reward():
    assert diag_v1.primitive_reward(0.0) == -1.0 and diag_v1.primitive_reward(1.0) == 0.0


# ---- anchor rule -------------------------------------------------------------------------------------------------
def _meta(rng, n_seeds=8, eps=4):
    meta = []
    for s in range(n_seeds):
        for e in range(eps):
            succ = int(rng.random() < 0.4)
            L = int(rng.integers(60, 400)) if succ else 400
            for step in range(0, L, 8):
                meta.append(dict(seed=100000 + s, episode=e, step=step, length=L, success=succ))
    return meta


def test_select_anchors():
    rng = np.random.default_rng(0)
    meta = _meta(rng)
    a = diag_v1.select_anchors(meta, 32, 7)
    assert a == diag_v1.select_anchors(meta, 32, 7)  # deterministic
    assert a != diag_v1.select_anchors(meta, 32, 8)
    assert len(a) == 32 and len({x['index'] for x in a}) == 32
    strata = {}
    for x in a:
        strata[x['stratum']] = strata.get(x['stratum'], 0) + 1
        m = meta[x['index']]
        assert x['third'] == min(2, 3 * m['step'] // m['length']) and x['success'] == m['success']
        assert abs(x['weight'] - x['S_h'] * x['n_hs'] / x['n_h']) < 1e-12
    assert sorted(strata.values()) == [5, 5, 5, 5, 6, 6] and len(strata) == 6
    # balanced over seeds within a stratum
    for h in strata:
        seeds = [meta[x['index']]['seed'] for x in a if x['stratum'] == h]
        assert max(seeds.count(s) for s in set(seeds)) - min(seeds.count(s) for s in set(seeds)) <= 1
    # weights estimate the stratum sizes: sum of weights per stratum == N_h when every seed has equal counts
    print('strata', strata)


def test_select_anchors_empty_stratum():
    rng = np.random.default_rng(1)
    meta = [m for m in _meta(rng) if not m['success']]  # no successes
    a = diag_v1.select_anchors(meta, 32, 0)
    assert len(a) == 32 and {x['success'] for x in a} == {0}
    # quotas 6, 6, 5, 5, 5, 5 (STRATA order); each success quota moves to the failures of the same third
    assert [sum(x['third'] == t for x in a) for t in range(3)] == [12, 10, 10]


# ---- selection helpers ---------------------------------------------------------------------------------------------
def test_selection_rngs():
    import jax
    k1, k2 = diag_v1.selection_noise_key(0, 3), diag_v1.selection_noise_key(0, 4)
    assert not np.array_equal(np.asarray(k1), np.asarray(k2))
    assert np.array_equal(np.asarray(k1), np.asarray(diag_v1.selection_noise_key(0, 3)))
    u = [diag_v1.uniform_index(0, s, 32) for s in range(2000)]
    assert min(u) == 0 and max(u) == 31 and abs(np.mean(u) - 15.5) < 1.0
    for s in range(50):
        sub = diag_v1.candidate_subset(0, s, 32, 6, s % 32)
        assert len(set(sub)) == 6 and s % 32 in sub and sub == sorted(sub)
    # MC keys: distinct per (q, k), reproducible, independent of the selection keys
    keys = {(q, k): tuple(diag_v1.mc_job_key(0, q, k).tolist()) for q in range(4) for k in range(4)}
    assert len(set(keys.values())) == 16 and keys[1, 2] == tuple(diag_v1.mc_job_key(0, 1, 2).tolist())
    seeds = {diag_v1.mc_np_seed(0, q, k) for q in range(4) for k in range(4)}
    assert len(seeds) == 16
    del jax
