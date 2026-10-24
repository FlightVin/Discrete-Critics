"""geo3b: five-token (L = 5) code arithmetic against an independent rational reference (no JAX in the reference).

Canonical digits, residuals c / rem / eps at every position, label distributions (ordinary modes of
agents/acarb.ar_label_dists and the redundant modes of agents/geo_codes) and label means, on a deterministic grid at
L = 5 and on the full grid at L = 3; integer round trips; and the coverage proof by exhaustion of every remainder at
every denominator. Run from the project root.
"""
import os
import sys
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from agents import geo_codes  # noqa: E402
from agents.acarb import ar_bin, ar_code, ar_label_dists, canonical_digits  # noqa: E402

B = 10
TOKENS = np.arange(geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX + 1)
ORD_MODES = ['hard', 'twohot_last', 'twohot_last2', 'twohot_all']


def kden(L):
    return 10 ** L * 2 * 10  # B^L num_qs M (num_qs = 2, M = 10)


# ---- independent reference (python ints / Fractions) ------------------------------------------------
def ref_canonical(k, k_den, L):
    n = int(Fraction(k, k_den) * 10 ** L)  # floor (non-negative)
    n = min(max(n, 0), 10 ** L - 1)
    return [(n // 10 ** (L - 1 - i)) % 10 for i in range(L)]


def ref_residual(k, k_den, prefix, l):
    """r_l = (u - sum_{j<l} z_j 10^-(j+1)) 10^(l+1) as a Fraction; (c, rem, eps) with rem = eps D_l (an integer)."""
    v = sum(Fraction(z, 10 ** (j + 1)) for j, z in enumerate(prefix[:l]))
    r = (Fraction(k, k_den) - v) * 10 ** (l + 1)
    c = r.numerator // r.denominator
    eps = r - c
    rem = eps * (k_den // 10 ** (l + 1))
    assert rem.denominator == 1
    return r, c, int(rem), eps


def ref_ordinary_q(k, k_den, L, mode):
    """(q (L, 10) Fractions, clip (L,)) of ar_label_dists: canonical prefix, two-hot T(r) clamped to [0, 9]."""
    first_soft = {'hard': L, 'twohot_last': L - 1, 'twohot_last2': L - 2, 'twohot_all': 0}[mode]
    d = ref_canonical(k, k_den, L)
    q, clips = [], []
    for l in range(L):
        row = [Fraction(0)] * 10
        if l < first_soft:
            row[d[l]] = Fraction(1)
            q.append(row), clips.append(False)
            continue
        r, c, rem, eps = ref_residual(k, k_den, d, l)
        if r < 0 or r > 9:
            row[min(max(c, 0), 9)] = Fraction(1)
            clips.append(True)
        else:
            row[c] += 1 - eps
            if eps:
                row[c + 1] += eps
            clips.append(False)
        q.append(row)
    return q, clips


def ref_redundant(k, k_den, prefix, mode):
    """{token: mass} per position for the redundant label modes, plus (c, rem) and the reference label mean."""
    out = []
    for l in range(len(prefix)):
        r, c, rem, eps = ref_residual(k, k_den, prefix, l)
        if mode == 'corrective':
            m = {c: 1 - eps, c + 1: eps}
            mean = r
        elif mode == 'noisy_fixed':
            m = {c - 1: Fraction(1, 3), c: Fraction(1, 3), c + 1: Fraction(1, 3)}
            mean = Fraction(c)
        else:
            m = {c - 1: (1 - eps) / 2, c: Fraction(1, 2), c + 1: eps / 2}
            mean = r - Fraction(1, 2)
        out.append(({t: v for t, v in m.items() if v != 0}, c, rem, mean))
    return out


# ---- grids ------------------------------------------------------------------------------------------
def l5_grid():
    L, k_den = 5, kden(5)
    g = k_den // 10 ** L
    k_top = k_den - g
    ks = {0, 1, g - 1, g, k_top - 1, k_top}
    ns = set(range(0, 10 ** L, 97)) | {10 ** L - 1}
    for j in range(L):
        for d in range(1, 10):
            ns |= {d * 10 ** j, d * 10 ** j - 1}
    ks |= {g * n for n in ns}
    ks |= set(np.random.RandomState(0).randint(0, k_top + 1, 20000).tolist())
    return sorted(ks)


def forced_prefixes(k, k_den, L, rs):
    """Prefix paths z_l = c_l + delta_l (delta in [-2, 2]): all +2, all -2 (negative / upper residuals at every
    later position), alternating, and one random path."""
    paths = []
    for deltas in ([2] * (L - 1), [-2] * (L - 1), [2, -2] * L, rs.randint(-2, 3, L - 1).tolist()):
        z = []
        for l in range(L - 1):
            z.append(ref_residual(k, k_den, z + [0] * L, l)[1] + deltas[l])
        paths.append(z + [0])
    return paths


def check_ordinary(ks, L, chunk=4096):
    k_den = kden(L)
    for mode in ORD_MODES:
        f = jax.jit(lambda k: ar_label_dists(k, k_den, B, L, mode))
        for s in range(0, len(ks), chunk):
            kk = ks[s:s + chunk]
            d, q, clip = (np.asarray(x) for x in f(jnp.asarray(kk, jnp.int32)))
            refs = [ref_ordinary_q(k, k_den, L, mode) for k in kk]
            np.testing.assert_array_equal(d, [ref_canonical(k, k_den, L) for k in kk])
            np.testing.assert_array_equal(clip, [r[1] for r in refs])
            np.testing.assert_allclose(q, np.array([[[float(x) for x in row] for row in r[0]] for r in refs]),
                                       atol=2e-7, rtol=0, err_msg=mode)


def check_redundant(ks, prefixes, L):
    """geo_codes.label_and_info for given prefixes (n, L) vs the reference: c / rem exact, masses, label means."""
    k_den = kden(L)
    ks = np.asarray(ks, np.int32)
    prefixes = np.asarray(prefixes, np.int32)
    for mode in geo_codes.LABEL_MODES:
        labels, info = jax.jit(lambda k, p: geo_codes.label_and_info(k, p, mode, k_den))(ks, prefixes)
        labels, c, rem, den = (np.asarray(x) for x in (labels, info['c'], info['rem'], info['den']))
        assert int(info['oov_label']) == 0
        mean = (labels.astype(np.float64) * TOKENS).sum(-1)
        want = np.zeros(labels.shape)
        rc, rrem, rmean = np.zeros(c.shape, np.int64), np.zeros(c.shape, np.int64), np.zeros(c.shape)
        for i, k in enumerate(ks.tolist()):
            for l, (m, c_, rem_, mean_) in enumerate(ref_redundant(k, k_den, prefixes[i].tolist(), mode)):
                rc[i, l], rrem[i, l], rmean[i, l] = c_, rem_, float(mean_)
                for t, v in m.items():
                    want[i, l, t - geo_codes.TOKEN_MIN] = float(v)
        np.testing.assert_array_equal(c, rc)
        np.testing.assert_array_equal(rem, rrem)
        np.testing.assert_array_equal(den, np.broadcast_to([k_den // 10 ** (l + 1) for l in range(L)], den.shape))
        np.testing.assert_allclose(labels, want, atol=2e-7, rtol=0, err_msg=mode)
        np.testing.assert_allclose(mean, rmean, atol=4e-6, rtol=0, err_msg=mode)


# ---- 1. reference comparison --------------------------------------------------------------------------
def test_l5_ordinary_labels_match_reference():
    check_ordinary(l5_grid(), 5)


def test_l5_redundant_labels_match_reference():
    L, k_den = 5, kden(5)
    ks = l5_grid()
    canon = [ref_canonical(k, k_den, L) for k in ks]
    check_redundant(ks, canon, L)
    rs = np.random.RandomState(1)
    sub = ks[::7]
    fk, fp = [], []
    for k in sub:
        for p in forced_prefixes(k, k_den, L, rs):
            fk.append(k), fp.append(p)
    check_redundant(fk, fp, L)
    # The forced paths reach negative c at positions 2..5 and residuals above 9 (upper caps of the vocabulary).
    c = np.asarray(geo_codes.label_and_info(jnp.asarray(fk), jnp.asarray(fp), 'corrective', k_den)[1]['c'])
    print('forced-prefix c min per position', c.min(0).tolist(), 'max', c.max(0).tolist())
    assert (c[:, 1:].min(0) <= -15).all() and (c[:, 1:].max(0) >= 25).all()


def test_l5_residual_function_matches_reference():
    """geo_codes.residual (prefix in units of 10^-5) at every position, with canonical and forced prefixes."""
    L, k_den = 5, kden(5)
    rs = np.random.RandomState(2)
    ks = l5_grid()[::5]
    for k in ks[:300]:
        for p in [ref_canonical(k, k_den, L)] + forced_prefixes(k, k_den, L, rs)[:2]:
            units = sum(z * 10 ** (L - 1 - j) for j, z in enumerate(p[:L - 1]))
            for l in range(L):
                pu = sum(z * 10 ** (L - 1 - j) for j, z in enumerate(p[:l]))
                c, rem, den = geo_codes.residual(k, pu, l, k_den, L)
                _, rc, rrem, _ = ref_residual(k, k_den, p, l)
                assert (int(c), int(rem), int(den)) == (rc, rrem, k_den // 10 ** (l + 1)), (k, p, l)
            assert units < 2 ** 31


def test_l3_full_grid_matches_reference():
    L, k_den = 3, kden(3)
    ks = list(range(0, k_den - k_den // 1000 + 1))
    check_ordinary(ks, L)
    canon = [ref_canonical(k, k_den, L) for k in ks]
    check_redundant(ks, canon, L)
    rs = np.random.RandomState(3)
    fk, fp = [], []
    for k in ks[::3]:
        p = forced_prefixes(k, k_den, L, rs)[rs.randint(4)]
        fk.append(k), fp.append(p)
    check_redundant(fk, fp, L)


def test_l3_canonical_digits_equal_ar_bin_on_grid():
    """At L = 3 the integer canonical digits are exactly the float ar_bin of the clipped target (L = 3 unchanged),
    including success (u = 1 -> clipped 0.999 float32 vs k_top)."""
    k_den = kden(3)
    k = jnp.arange(0, k_den - 20 + 1, dtype=jnp.int32)
    u = jnp.clip(k.astype(jnp.float32) / k_den, 0.0, 1.0 - B ** -3)
    np.testing.assert_array_equal(np.asarray(canonical_digits(k, k_den, B, 3)), np.asarray(ar_bin(u, B, 3)))
    np.testing.assert_array_equal(np.asarray(ar_bin(jnp.clip(jnp.float32(1.0), 0.0, 1.0 - B ** -3), B, 3)), [9, 9, 9])


# ---- 2. round trips -----------------------------------------------------------------------------------
def test_l5_round_trip_every_code():
    """All 10^5 codes n at L = 5: k = 20 n -> canonical digits -> integer decode == n (ordinary ar_code and the
    redundant decode_units); with the canonical prefix c_l = d_l at every position and eps_5 = 0."""
    L, k_den = 5, kden(5)
    n = jnp.arange(10 ** L, dtype=jnp.int32)
    d = canonical_digits(n * (k_den // 10 ** L), k_den, B, L)
    np.testing.assert_array_equal(np.asarray(ar_code(d, B)), np.asarray(n))
    np.testing.assert_array_equal(np.asarray(geo_codes.decode_units(d)), np.asarray(n))
    _, info = geo_codes.label_and_info(n * 20, d, 'corrective', k_den)
    np.testing.assert_array_equal(np.asarray(info['rem'][:, -1]), 0)  # exact code: eps_5 = 0 (one class)
    np.testing.assert_array_equal(np.asarray(info['c']), np.asarray(d))
    # Off-code k: floor (k // 20) at every k of one code's cell.
    for base in (0, 12345, 99999):
        kk = jnp.arange(20 * base, 20 * base + 20, dtype=jnp.int32)
        np.testing.assert_array_equal(np.asarray(ar_code(canonical_digits(kk, k_den, B, L), B)), base)
    # Why not the float: a float32 k / k_den misses exact codes at L = 5 for some n.
    fl = np.asarray(ar_code(ar_bin(n.astype(jnp.float32) * 20 / k_den, B, L), B))
    print('float32 ar_bin mismatches over the 10^5 exact codes at L=5:', int((fl != np.asarray(n)).sum()))


def test_decode_units_l5_signed():
    t = jnp.asarray([[-22, 31, -22, 31, -22], [3, -2, 14, 0, -7], [9, 9, 9, 9, 9]])
    want = [sum(z * 10 ** (4 - i) for i, z in enumerate(row)) for row in np.asarray(t).tolist()]
    np.testing.assert_array_equal(np.asarray(geo_codes.decode_units(t)), want)
    assert geo_codes.unit_weights(5) == (10000, 1000, 100, 10, 1) and geo_codes.unit_weights() == geo_codes.UNIT_WEIGHTS


# ---- 3. coverage by exhaustion ------------------------------------------------------------------------
@pytest.mark.parametrize('law', ['corrective', 'fixed_or_model'])
def test_l5_coverage_by_exhaustion(law):
    """For l >= 1, N_l = N_{l-1} - z_{l-1} D_{l-1} = rem_{l-1} - delta D_{l-1}: the next residual depends only on
    (rem_{l-1} / D_{l-1}, delta = z_{l-1} - c_{l-1}). Enumerate every rem at every denominator D_0..D_3 and every
    delta of the prefix law (corrective: delta in {0, 1}, taken only with rem > 0 for 1; fixed / clamped model
    tokens: delta in [-2, 2]) through geo_codes.residual itself, and check roll-ins c +- 2 (positions 1..4) and every
    positive-mass label token (all three modes) lie in [-22, 31]. First position: c in [0, 9] for k in [0, k_top]."""
    L, k_den = 5, kden(5)
    deltas = [0, 1] if law == 'corrective' else list(range(-2, 3))
    lo, hi = {}, {}
    for l in range(1, L):
        D_prev = k_den // 10 ** l
        rem = jnp.arange(D_prev, dtype=jnp.int32)
        for delta in deltas:
            # k = rem with an all-zero canonical prefix gives c_{l-1} = 0 and rem_{l-1} = rem; z_{l-1} = delta.
            prefix = jnp.zeros((D_prev, L), jnp.int32).at[:, l - 1].set(delta)
            pu = sum(geo_codes.unit_weights(L)[j] * prefix[:, j] for j in range(l))
            c, r, den = geo_codes.residual(rem, pu, l, k_den, L)
            c, r = np.asarray(c), np.asarray(r)
            np.testing.assert_array_equal(c, np.floor_divide(np.asarray(rem) - delta * D_prev, D_prev // 10))
            live = np.asarray(rem) > 0 if (law == 'corrective' and delta == 1) else np.ones(D_prev, bool)
            c, r = c[live], r[live]
            lo[l], hi[l] = min(lo.get(l, 99), int(c.min())), max(hi.get(l, -99), int(c.max()))
            if l < L - 1:  # roll-ins at positions 1..4 (0-indexed 0..3): z in [c - 2, c + 2]
                assert c.min() - 2 >= geo_codes.TOKEN_MIN and c.max() + 2 <= geo_codes.TOKEN_MAX
            for mode in geo_codes.LABEL_MODES:
                tok, mass = geo_codes.label_support(jnp.asarray(c), jnp.asarray(r), jnp.full(c.shape, D_prev // 10), mode)
                tok, mass = np.asarray(tok), np.asarray(mass)
                assert tok[mass > 0].min() >= geo_codes.TOKEN_MIN and tok[mass > 0].max() <= geo_codes.TOKEN_MAX, mode
    # Independent histories with the same (rem, delta) give the same next residual.
    rs = np.random.RandomState(4)
    k = jnp.asarray(rs.randint(0, k_den - 20 + 1, 5000), jnp.int32)
    z = jnp.zeros((5000, L), jnp.int32)
    for l in range(L - 1):
        c, rem, _ = geo_codes.residual(k, sum(geo_codes.unit_weights(L)[j] * z[:, j] for j in range(l)), l, k_den, L)
        delta = jnp.asarray(rs.choice(deltas, 5000), jnp.int32)
        z = z.at[:, l].set(c + delta)
        c2, _, _ = geo_codes.residual(k, sum(geo_codes.unit_weights(L)[j] * z[:, j] for j in range(l + 1)), l + 1,
                                      k_den, L)
        D = k_den // 10 ** (l + 1)
        np.testing.assert_array_equal(np.asarray(c2), np.floor_divide(np.asarray(rem - delta * D), D // 10))
    c0 = np.asarray(geo_codes.residual(jnp.arange(0, k_den - 20 + 1, dtype=jnp.int32), 0, 0, k_den, L)[0])
    lo[0], hi[0] = int(c0.min()), int(c0.max())
    assert (lo[0], hi[0]) == (0, 9)
    print(f'coverage L=5 law={law}: c min/max per position', {l: (lo[l], hi[l]) for l in sorted(lo)})
    if law == 'fixed_or_model':
        assert all((lo[l], hi[l]) == (-20, 29) for l in range(1, L))
    else:
        assert all((lo[l], hi[l]) == (-10, 9) for l in range(1, L))


def test_build_l5_rollins_and_labels_in_vocab():
    """geo_codes.build at L = 5 over many k and every roll-in mode: no OOV roll-ins or labels, roll-ins only at
    positions 1..4, roll-in offsets in the law's range."""
    L, k_den = 5, kden(5)
    k = jnp.asarray(l5_grid(), jnp.int32)
    for label, rollin in [('corrective', 'corrective'), ('corrective', 'fixed'), ('noisy_fixed', 'fixed'),
                          ('noisy_resid', 'fixed')]:
        labels, ro, info = geo_codes.build(k, label, rollin, jax.random.PRNGKey(0), k_den=k_den, length=L)
        assert labels.shape == (len(k), L, geo_codes.NUM_CLASSES) and ro.shape == (len(k), L)
        assert int(info['oov_label']) == 0 and int(info['oov_rollin']) == 0
        assert (np.asarray(ro[:, -1]) == 0).all()
        off = np.asarray(ro[:, :-1] - info['c'][:, :-1])
        assert off.min() >= (0 if rollin == 'corrective' else -2) and off.max() <= (1 if rollin == 'corrective' else 2)
        np.testing.assert_allclose(np.asarray(labels.sum(-1)), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        geo_codes.build(k, 'corrective', 'fixed', jax.random.PRNGKey(0), k_den=20000, length=L)  # not a multiple of 10^5
