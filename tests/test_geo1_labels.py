"""geo1: exact two-hot digit labels on the sample_mean target grid u = k / 20000, against a rational reference.

Base 10, 3 digits, 2 copies x 10 samples: K_den = 20000. Run from the project root.
"""
import os
import sys
from fractions import Fraction
from math import floor

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from agents.acarb import ar_bin, ar_decode, ar_label_dists, soft_cross_entropy  # noqa: E402

B, L, K_DEN = 10, 3, 20000
K_TOP = K_DEN - K_DEN // B ** L  # 19980: u = 0.999
MODES = ['hard', 'twohot_last', 'twohot_last2', 'twohot_all']
FIRST_SOFT = {'hard': 3, 'twohot_last': 2, 'twohot_last2': 1, 'twohot_all': 0}


def all_ks():
    rs = np.random.RandomState(0)
    exact = [20 * j for j in range(1000)]
    bounds = [20 * j + s for j in range(1000) for s in (-1, 1) if 0 <= 20 * j + s <= K_TOP]
    return sorted(set([0, K_TOP, *exact, *bounds, *rs.randint(0, K_TOP + 1, 2000).tolist()]))


def reference(k, mode):
    """Rational labels: (digits, [(a, f, clipped)] per position) with u = k / K_DEN exactly."""
    u = Fraction(k, K_DEN)
    n = floor(u * B ** L)
    d = [(n // B ** (L - 1 - l)) % B for l in range(L)]
    out = []
    for l in range(L):
        if l < FIRST_SOFT[mode]:
            out.append((d[l], Fraction(0), False))
            continue
        v_prefix = sum(Fraction(d[j], B ** (j + 1)) for j in range(l))
        r = (u - v_prefix) / Fraction(1, B ** (l + 1))
        clipped = r < 0 or r > B - 1
        r = min(max(r, Fraction(0)), Fraction(B - 1))
        a = floor(r)
        out.append((a, r - a, clipped))
    return d, out


@pytest.fixture(scope='module')
def ks():
    return all_ks()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('jit', [False, True])
def test_labels_match_rational_reference(mode, jit, ks):
    fn = jax.jit(ar_label_dists, static_argnums=(1, 2, 3, 4)) if jit else ar_label_dists  # as inside the update
    d, q, clip = jax.device_get(fn(jnp.asarray(ks, jnp.int32), K_DEN, B, L, mode))
    q = np.asarray(q, np.float64)
    assert q.shape == (len(ks), L, B)
    assert (q >= 0).all()
    np.testing.assert_allclose(q.sum(-1), 1.0, atol=1e-6)
    n_clip = np.zeros(L, int)
    for i, k in enumerate(ks):
        d_ref, pos = reference(k, mode)
        assert list(d[i]) == d_ref, (k, d[i], d_ref)
        for l, (a, f, clipped) in enumerate(pos):
            expect = np.zeros(B)
            if f == 0:
                expect[a] = 1.0
                assert np.count_nonzero(q[i, l]) == 1, (k, l, q[i, l])  # f = 0: one class, a+1 never touched
            else:
                expect[a], expect[a + 1] = 1 - float(f), float(f)
            np.testing.assert_allclose(q[i, l], expect, atol=1e-6, err_msg=f'k={k} l={l} mode={mode}')
            nz = np.flatnonzero(q[i, l])
            assert set(nz) <= {a, a + 1}
            # expected token after the clamp
            np.testing.assert_allclose((q[i, l] * np.arange(B)).sum(), float(a + f), atol=1e-5)
            assert bool(clip[i, l]) == clipped, (k, l)
            n_clip[l] += clipped
    # clamp counts: only two-hot positions can clamp; on the exact grid positions >= 2 clamp only when the
    # digit is 9 with a remainder; position 1 (twohot_all) clamps for u > 0.9.
    assert (clip[:, :FIRST_SOFT[mode]] == 0).all()
    np.testing.assert_array_equal(clip.sum(0), n_clip)
    if mode == 'twohot_all':
        assert n_clip[0] == sum(1 for k in ks if Fraction(k, K_DEN) * B > B - 1)


def test_exact_codes_have_zero_fraction():
    """Every ordinary code k = 20 j is exact: r_3 = d_3 exactly (f = 0), so position 3 is one-hot at d_3 in every
    mode (earlier two-hot positions carry the later digits as their fraction, e.g. r_2 = d_2 + d_3 / 10)."""
    j = np.arange(1000)
    digits = np.stack([j // 100, (j // 10) % 10, j % 10], -1)
    for mode in MODES:
        d, q, clip = jax.device_get(ar_label_dists(jnp.asarray(20 * j, jnp.int32), K_DEN, B, L, mode))
        np.testing.assert_array_equal(d, digits)
        np.testing.assert_array_equal(np.asarray(q[:, 2]), np.asarray(jax.nn.one_hot(digits[:, 2], B)))
        assert not clip[:, 2].any()
        if mode == 'hard':
            np.testing.assert_array_equal(np.asarray(q), np.asarray(jax.nn.one_hot(digits, B)))


def test_example_off_grid_value():
    """u = 0.6374 (= 12748 / 20000): twohot_all labels T(6.374), T(3.74) | (6), T(7.4) | (6, 3)."""
    _, q, _ = ar_label_dists(jnp.asarray([12748], jnp.int32), K_DEN, B, L, 'twohot_all')
    q = np.asarray(q[0], np.float64)
    for l, (a, f) in enumerate([(6, 0.374), (3, 0.74), (7, 0.4)]):
        expect = np.zeros(B)
        expect[a], expect[a + 1] = 1 - f, f
        np.testing.assert_allclose(q[l], expect, atol=1e-6)
    _, pos = reference(12748, 'twohot_all')
    assert [(a, f) for a, f, _ in pos] == [(6, Fraction(374, 1000)), (3, Fraction(74, 100)), (7, Fraction(4, 10))]


def test_ar_bin_on_grid_and_round_trip():
    k = np.arange(K_TOP + 1)
    d = np.asarray(ar_bin(jnp.asarray(k, jnp.float32) / K_DEN, B, L))
    n = d[:, 0] * 100 + d[:, 1] * 10 + d[:, 2]
    np.testing.assert_array_equal(n, k // 20)
    codes = np.stack(np.unravel_index(np.arange(1000), (10, 10, 10)), -1)
    back = np.asarray(ar_bin(ar_decode(jnp.asarray(codes), B), B, L))
    np.testing.assert_array_equal(back, codes)


def test_hard_soft_ce_equals_integer_ce():
    rs = np.random.RandomState(1)
    logits = jnp.asarray(rs.randn(2, 64, L, B).astype(np.float32) * 3)
    k = jnp.asarray(rs.randint(0, K_TOP + 1, 64), jnp.int32)
    d, q, _ = ar_label_dists(k, K_DEN, B, L, 'hard')
    np.testing.assert_array_equal(np.asarray(d), np.asarray(ar_bin(k.astype(jnp.float32) / K_DEN, B, L)))
    w = jnp.asarray((rs.rand(64) > 0.2).astype(np.float32))
    soft = (soft_cross_entropy(logits, q[None]).mean(axis=(0, 2)) * w).mean()
    hard = (optax.softmax_cross_entropy_with_integer_labels(logits, d[None]).mean(axis=(0, 2)) * w).mean()
    assert abs(float(soft) - float(hard)) <= 1e-6, (float(soft), float(hard))
