"""Exact (fractions.Fraction) checks of agents/geo_codes.py: labels and roll-ins for E5-E9.

Run: env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests/test_geo_codes.py -q -s
(-s shows the reported token ranges / biases).
"""

import functools
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents import geo_codes as gc

K_DEN = 20000
K_MAX = 19980
WEIGHTS = (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000))
MODEL_KEY = jax.random.PRNGKey(123)


def random_model(prefix, position):
    """Deterministic (given shape and position) 'model' tokens in [-10, 40], ignoring the prefix."""
    return jax.random.randint(jax.random.fold_in(MODEL_KEY, position), prefix.shape[:-1], -10, 41)


def ref_residual(k, prefix_tokens, position, k_den=K_DEN):
    """Pure-Fraction reference: r, c = floor(r), f = r - c."""
    u = Fraction(int(k), k_den)
    v = sum(WEIGHTS[j] * int(prefix_tokens[j]) for j in range(position))
    r = (u - v) / WEIGHTS[position]
    c = math.floor(r)
    return r, c, r - c


def ref_masses(mode, f):
    """Fraction masses on offsets (-1, 0, +1)."""
    if mode == 'corrective':
        return (Fraction(0), 1 - f, f)
    if mode == 'noisy_fixed':
        return (Fraction(1, 3),) * 3
    return (Fraction(1, 2) * (1 - f), Fraction(1, 2), Fraction(1, 2) * f)


def label_k_set(rng):
    exact = [20 * j for j in range(1000)]
    near = [x for j in range(1000) for x in (20 * j - 1, 20 * j + 1) if 0 <= x <= K_MAX]
    rand = rng.integers(0, K_MAX + 1, size=1000).tolist()
    return np.array(sorted(set([0, K_MAX] + exact + near + rand)), np.int32)


def path_info(k, z1, z2, mode='corrective', k_den=K_DEN):
    prefix = jnp.stack([z1, z2, jnp.zeros_like(z1)], axis=-1)
    labels, info = gc.label_and_info(k, prefix, mode, k_den)
    return np.asarray(labels), {key: np.asarray(v) for key, v in info.items()}


def row_dict(row, tol=0.0):
    return {int(t): float(row[t - gc.TOKEN_MIN]) for t in range(gc.TOKEN_MIN, gc.TOKEN_MAX + 1)
            if row[t - gc.TOKEN_MIN] > tol}


# ---------------------------------------------------------------- 1. labels against Fraction references
@pytest.mark.parametrize('rollin_mode', gc.ROLLIN_MODES)
@pytest.mark.parametrize('label_mode', gc.LABEL_MODES)
def test_labels_exact(label_mode, rollin_mode):
    ks = label_k_set(np.random.default_rng(0))
    model = random_model if rollin_mode == 'model_mix' else None
    labels, rollin, info = gc.build(jnp.asarray(ks), label_mode, rollin_mode, jax.random.PRNGKey(1), model)
    labels, rollin = np.asarray(labels), np.asarray(rollin)
    c_mod, rem_mod, den_mod = (np.asarray(info[x]) for x in ('c', 'rem', 'den'))
    assert int(info['oov_label']) == 0 and int(info['oov_rollin']) == 0
    assert labels.dtype == np.float32 and labels.shape == ks.shape + (3, gc.NUM_CLASSES)
    assert rollin.dtype == np.int32 and (rollin[:, 2] == 0).all()
    assert (labels >= 0).all()
    assert np.abs(labels.sum(-1) - 1).max() <= 1e-6
    for i, k in enumerate(ks):
        for l in range(3):
            r, c, f = ref_residual(k, rollin[i], l)
            assert c == c_mod[i, l] and Fraction(int(rem_mod[i, l]), int(den_mod[i, l])) == f, (k, l)
            expected = np.zeros(gc.NUM_CLASSES)
            for off, m in zip((-1, 0, 1), ref_masses(label_mode, f)):
                expected[c + off - gc.TOKEN_MIN] += float(m)
            nz = np.flatnonzero(labels[i, l])
            assert set(nz) <= {c - 1 - gc.TOKEN_MIN, c - gc.TOKEN_MIN, c + 1 - gc.TOKEN_MIN}
            if label_mode == 'corrective':
                assert len(nz) == (1 if f == 0 else 2)
            if label_mode != 'noisy_fixed' and f == 0:
                assert labels[i, l, c + 1 - gc.TOKEN_MIN] == 0
            np.testing.assert_allclose(labels[i, l], expected, atol=1e-6)
            if l < 2:  # roll-in token relative to the label's c at that position
                d = int(rollin[i, l]) - c
                if rollin_mode == 'corrective':
                    assert d in ((0,) if f == 0 else (0, 1))
                else:
                    assert -2 <= d <= 2


# ---------------------------------------------------------------- 2. E5 exactness over the prefix tree
@pytest.mark.parametrize('k_den', [K_DEN, 1000, 40000])
def test_e5_tree_exact(k_den):
    rng = np.random.default_rng(1)
    scale = k_den // 1000
    k_max = 999 * scale
    exact = [scale * j for j in range(1000)]
    near = [x for e in exact for x in (e - 1, e + 1) if 0 <= x <= k_max]
    ks = np.array(sorted(set(exact + near + rng.integers(0, k_max + 1, 500).tolist())), np.int32)
    k = jnp.asarray(ks)
    _, base = path_info(k, jnp.zeros_like(k), jnp.zeros_like(k), k_den=k_den)
    c1 = jnp.asarray(base['c'][:, 0])
    paths = {}
    for b1 in (0, 1):
        z1 = c1 + b1
        _, i1 = path_info(k, z1, jnp.zeros_like(k), k_den=k_den)
        for b2 in (0, 1):
            z2 = jnp.asarray(i1['c'][:, 1]) + b2
            paths[b1, b2] = (np.asarray(z1), np.asarray(z2), path_info(k, z1, z2, k_den=k_den)[1])
    for i, kk in enumerate(ks):
        total = Fraction(0)
        for (b1, b2), (z1, z2, info) in paths.items():
            for l in range(3):
                prefix = (z1[i], z2[i], 0)
                _, c_ref, f_ref = ref_residual(kk, prefix, l, k_den)
                assert info['c'][i, l] == c_ref and Fraction(int(info['rem'][i, l]), int(info['den'][i, l])) == f_ref
            f1 = Fraction(int(info['rem'][i, 0]), int(info['den'][i, 0]))
            f2 = Fraction(int(info['rem'][i, 1]), int(info['den'][i, 1]))
            f3 = Fraction(int(info['rem'][i, 2]), int(info['den'][i, 2]))
            p = (f1 if b1 else 1 - f1) * (f2 if b2 else 1 - f2)
            ez3 = int(info['c'][i, 2]) + f3
            total += p * (WEIGHTS[0] * int(z1[i]) + WEIGHTS[1] * int(z2[i]) + WEIGHTS[2] * ez3)
        assert total == Fraction(int(kk), k_den), kk


# ---------------------------------------------------------------- 3. vocabulary coverage
def _all_prefixes(delta_range):
    k = jnp.arange(K_MAX + 1, dtype=jnp.int32)
    _, base = path_info(k, jnp.zeros_like(k), jnp.zeros_like(k))
    ks, z1s, z2s = [], [], []
    for d1 in delta_range:
        z1 = jnp.asarray(base['c'][:, 0]) + d1
        _, i1 = path_info(k, z1, jnp.zeros_like(k))
        for d2 in delta_range:
            ks.append(k)
            z1s.append(z1)
            z2s.append(jnp.asarray(i1['c'][:, 1]) + d2)
    return jnp.concatenate(ks), jnp.concatenate(z1s), jnp.concatenate(z2s)


def test_coverage_within_vocab():
    k, z1, z2 = _all_prefixes(range(-2, 3))
    assert k.shape[0] == 25 * (K_MAX + 1)
    rollin = np.concatenate([np.asarray(z1), np.asarray(z2)])
    lo, hi = int(rollin.min()), int(rollin.max())
    print(f'\n[coverage] roll-in tokens over all k and |dz|<=2: [{lo}, {hi}]')
    assert gc.TOKEN_MIN <= lo and hi <= gc.TOKEN_MAX
    for mode in gc.LABEL_MODES:
        _, info = path_info(k, z1, z2, mode)
        assert int(info['oov_label']) == 0, mode
        tokens, masses = gc.label_support(jnp.asarray(info['c']), jnp.asarray(info['rem']),
                                          jnp.asarray(info['den']), mode)
        live = np.asarray(tokens)[np.asarray(masses) > 0]
        print(f'[coverage] {mode:12s} label support tokens: [{live.min()}, {live.max()}]')
        assert gc.TOKEN_MIN <= live.min() and live.max() <= gc.TOKEN_MAX


def test_prefix_error_three_leaves_vocab():
    k, z1, z2 = _all_prefixes((-3, 0, 3))
    for mode in gc.LABEL_MODES:
        _, info = path_info(k, z1, z2, mode)
        tokens, masses = gc.label_support(jnp.asarray(info['c']), jnp.asarray(info['rem']),
                                          jnp.asarray(info['den']), mode)
        live = np.asarray(tokens)[np.asarray(masses) > 0]
        print(f'\n[coverage] prefix error 3, {mode}: label tokens [{live.min()}, {live.max()}], '
              f'oov_label={int(info["oov_label"])}')
        assert int(info['oov_label']) > 0 and (live.min() < gc.TOKEN_MIN or live.max() > gc.TOKEN_MAX)
    labels, _, info = gc.build(jnp.array([0], jnp.int32), 'corrective', 'model_mix', jax.random.PRNGKey(0),
                               lambda p, l: jnp.full(p.shape[:-1], 99, jnp.int32))
    # model_mix clamps, so even a wild model stays in-vocab; the clamp count says it happened.
    assert int(info['oov_label']) == 0


# ---------------------------------------------------------------- 4. E8 / E9 expectations and decoded bias
def test_noisy_expectations_and_bias():
    ks = label_k_set(np.random.default_rng(2))
    k = jnp.asarray(ks)
    _, base = path_info(k, jnp.zeros_like(k), jnp.zeros_like(k))
    for mode in ('noisy_fixed', 'noisy_resid'):
        biases = []
        c1 = jnp.asarray(base['c'][:, 0])
        for o1 in (-1, 0, 1):
            z1 = c1 + o1
            _, i1 = path_info(k, z1, jnp.zeros_like(k), mode)
            for o2 in (-1, 0, 1):
                z2 = jnp.asarray(i1['c'][:, 1]) + o2
                labels, info = path_info(k, z1, z2, mode)
                tok = np.arange(gc.TOKEN_MIN, gc.TOKEN_MAX + 1)
                for i in range(0, len(ks), 7):
                    for l in range(3):
                        c = int(info['c'][i, l])
                        f = Fraction(int(info['rem'][i, l]), int(info['den'][i, l]))
                        e = sum(m * (c + off) for off, m in zip((-1, 0, 1), ref_masses(mode, f)))
                        assert e == (c if mode == 'noisy_fixed' else c + f - Fraction(1, 2))
                        assert abs(float((labels[i, l] * tok).sum()) - float(e)) < 1e-5
        # Decoded bias of sampling a full code from the labels (exhaustive 3x3x3 tree, exact).
        for i, kk in enumerate(ks):
            total = Fraction(0)
            _, c1r, f1 = ref_residual(kk, (0, 0, 0), 0)
            for o1, m1 in zip((-1, 0, 1), ref_masses(mode, f1)):
                z1 = c1r + o1
                _, c2r, f2 = ref_residual(kk, (z1, 0, 0), 1)
                for o2, m2 in zip((-1, 0, 1), ref_masses(mode, f2)):
                    z2 = c2r + o2
                    r3, c3r, f3 = ref_residual(kk, (z1, z2, 0), 2)
                    ez3 = sum(m * (c3r + o) for o, m in zip((-1, 0, 1), ref_masses(mode, f3)))
                    total += m1 * m2 * (WEIGHTS[0] * z1 + WEIGHTS[1] * z2 + WEIGHTS[2] * ez3)
            bias = total - Fraction(int(kk), K_DEN)
            expected = (-Fraction(1, 1000) * Fraction(int(kk) % 20, 20) if mode == 'noisy_fixed'
                        else -Fraction(1, 2000))
            assert bias == expected, (mode, kk, bias)
            biases.append(bias)
        grid = (sum(-Fraction(1, 1000) * Fraction(kk % 20, 20) for kk in range(K_MAX + 1)) / (K_MAX + 1)
                if mode == 'noisy_fixed' else -Fraction(1, 2000))
        print(f'\n[bias] {mode}: decoded bias per k in [{min(biases)}, {max(biases)}]; grid mean = {grid} '
              f'= {float(grid):.3e}')


# ---------------------------------------------------------------- 5. worked examples
def test_worked_examples():
    k = jnp.array([11100], jnp.int32)
    lab, _ = path_info(k, jnp.array([0]), jnp.array([0]))
    assert row_dict(lab[0, 0]) == pytest.approx({5: 0.45, 6: 0.55})
    lab, info = path_info(k, jnp.array([6]), jnp.array([0]))
    assert (info['c'][0, 1], info['rem'][0, 1], info['den'][0, 1]) == (-5, 100, 200)  # r = -4.5
    assert row_dict(lab[0, 1]) == pytest.approx({-5: 0.5, -4: 0.5})
    lab, info = path_info(k, jnp.array([6]), jnp.array([-4]))
    assert (info['c'][0, 2], info['rem'][0, 2]) == (-5, 0)
    assert row_dict(lab[0, 2]) == {-5: 1.0}
    assert int(gc.decode_units(jnp.array([6, -4, -5]))) == 555
    lab, _ = path_info(k, jnp.array([4]), jnp.array([0]))
    assert row_dict(lab[0, 1]) == pytest.approx({15: 0.5, 16: 0.5})
    lab, info = path_info(jnp.array([12748], jnp.int32), jnp.array([0]), jnp.array([0]))
    assert (info['c'][0, 0], info['rem'][0, 0], info['den'][0, 0]) == (6, 748, 2000)  # r = 6.374
    assert row_dict(lab[0, 0]) == pytest.approx({6: 0.626, 7: 0.374})
    assert np.asarray(gc.two_hot(jnp.array(6), jnp.array(748), jnp.array(2000))) == pytest.approx(lab[0, 0])
    assert int(gc.units_from_value(jnp.float32(0.6374))) == 12748
    c, rem, den = gc.residual(jnp.array(-900), 0, 1)
    assert (int(c), int(rem), int(den)) == (-5, 100, 200)  # integer floor toward -inf
    with pytest.raises(ValueError):
        gc.denominator(0, 1500)


# ---------------------------------------------------------------- 6. sampler statistics
N = 250_000


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel())[0, 1])


def _tol(p, n):
    return 5 * math.sqrt(p * (1 - p) / n) + 1e-4


def test_corrective_rollin_frequencies():
    k = jnp.full((2, N // 2), 12748, jnp.int32)
    _, rollin, info = gc.build(k, 'corrective', 'corrective', jax.random.PRNGKey(7))
    rollin, c = np.asarray(rollin), np.asarray(info['c'])
    b1 = rollin[..., 0] - 6
    assert set(np.unique(b1)) == {0, 1}
    assert abs(b1.mean() - 0.374) < _tol(0.374, N)
    for z1 in (6, 7):
        sel = rollin[..., 0] == z1
        _, c2, f2 = ref_residual(12748, (z1, 0, 0), 1)
        b2 = rollin[..., 1][sel] - c2
        assert set(np.unique(b2)) <= {0, 1} and (c[..., 1][sel] == c2).all()
        assert abs(b2.mean() - float(f2)) < _tol(float(f2), sel.sum()), (z1, b2.mean(), f2)
    assert abs(_corr(b1[0], b1[1])) < 0.01  # across elements
    assert abs(_corr(b1, rollin[..., 1] - c[..., 1])) < 0.01  # across positions


def test_fixed_rollin_uniform_delta():
    ks = np.random.default_rng(3).integers(0, K_MAX + 1, size=(2, N // 2)).astype(np.int32)
    _, rollin, info = gc.build(jnp.asarray(ks), 'noisy_fixed', 'fixed', jax.random.PRNGKey(8))
    d = np.asarray(rollin)[..., :2] - np.asarray(info['c'])[..., :2]
    for l in range(2):
        freq = np.bincount(d[..., l].ravel() + 2, minlength=5) / d[..., l].size
        assert len(freq) == 5 and np.abs(freq - 0.2).max() < _tol(0.2, N), freq
    assert abs(_corr(d[..., 0], d[..., 1])) < 0.01
    assert abs(_corr(d[0, :, 0], d[1, :, 0])) < 0.01
    assert abs(_corr(d[0, :-1, 1], d[0, 1:, 1])) < 0.01


def test_model_mix_rollin():
    ks = np.random.default_rng(4).integers(0, K_MAX + 1, size=N).astype(np.int32)
    k = jnp.asarray(ks)
    _, rollin, info = gc.build(k, 'corrective', 'model_mix', jax.random.PRNGKey(9), random_model)
    rollin, c, rem = np.asarray(rollin), np.asarray(info['c']), np.asarray(info['rem'])
    used = np.asarray(info['model_mask'])
    assert int(info['model_used']) == used.sum()
    assert abs(used.mean() - 0.5) < 0.01 and abs(used[:, 0].mean() - 0.5) < 0.01 and abs(used[:, 1].mean() - 0.5) < 0.01
    assert abs(_corr(used[:, 0], used[:, 1])) < 0.01
    clamps = 0
    for l in range(2):
        m = np.asarray(random_model(jnp.zeros((N, 3), jnp.int32), l))
        u = used[:, l]
        assert (rollin[u, l] == np.clip(m[u], c[u, l] - 2, c[u, l] + 2)).all()
        clamps += ((m[u] < c[u, l] - 2) | (m[u] > c[u, l] + 2)).sum()
        b = rollin[~u, l] - c[~u, l]
        assert set(np.unique(b)) <= {0, 1} and (b[rem[~u, l] == 0] == 0).all()
        f = rem[~u, l] / info['den'][0, l]
        assert abs(b.mean() - f.mean()) < 0.01
    assert int(info['model_clamp']) == clamps > 0
    assert int(info['oov_rollin']) == 0 and int(info['oov_label']) == 0


def test_model_mix_sees_actual_prefix():
    seen = []

    def model(prefix, position):
        seen.append((position, np.asarray(prefix)))
        return prefix[..., 0] + 3 if position == 1 else jnp.full(prefix.shape[:-1], -50, jnp.int32)

    ks = jnp.asarray(np.random.default_rng(5).integers(0, K_MAX + 1, size=20000).astype(np.int32))
    _, rollin, info = gc.build(ks, 'corrective', 'model_mix', jax.random.PRNGKey(10), model)
    rollin, c, used = np.asarray(rollin), np.asarray(info['c']), np.asarray(info['model_mask'])
    assert [p for p, _ in seen] == [0, 1]
    assert (seen[0][1] == 0).all()
    assert (seen[1][1][:, 0] == rollin[:, 0]).all() and (seen[1][1][:, 1:] == 0).all()
    u = used[:, 1]
    assert (rollin[u, 1] == np.clip(rollin[u, 0] + 3, c[u, 1] - 2, c[u, 1] + 2)).all()
    assert (rollin[used[:, 0], 0] == c[used[:, 0], 0] - 2).all()  # m = -50 clamps to c - 2
    with pytest.raises(ValueError):
        gc.build(ks, 'corrective', 'model_mix', jax.random.PRNGKey(0))


# ---------------------------------------------------------------- 7. jit and stop-gradient
def test_jit_and_no_gradient():
    build = jax.jit(gc.build, static_argnames=('label_mode', 'rollin_mode', 'model_token', 'k_den'))
    ks = jnp.asarray(np.random.default_rng(6).integers(0, K_MAX + 1, size=(3, 64)).astype(np.int32))
    for lm in gc.LABEL_MODES:
        for rm in gc.ROLLIN_MODES:
            model = random_model if rm == 'model_mix' else None
            a = build(ks, label_mode=lm, rollin_mode=rm, key=jax.random.PRNGKey(0), model_token=model)
            b = gc.build(ks, lm, rm, jax.random.PRNGKey(0), model)
            for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
                x, y = np.asarray(x), np.asarray(y)
                if np.issubdtype(x.dtype, np.floating):  # XLA fusion may differ by an ulp
                    np.testing.assert_allclose(x, y, atol=1e-6)
                else:  # tokens, residuals, counts, masks: identical
                    np.testing.assert_array_equal(x, y)
            assert a[0].shape == (3, 64, 3, gc.NUM_CLASSES) and a[1].shape == (3, 64, 3)

    weights = jnp.arange(gc.NUM_CLASSES, dtype=jnp.float32)

    def loss(x, via_units):
        k = gc.units_from_value(x) if via_units else x * K_DEN
        labels, rollin, _ = gc.build(k, 'noisy_resid', 'corrective', jax.random.PRNGKey(0))
        return jnp.sum(labels * weights) + jnp.sum(rollin.astype(jnp.float32) * 0.0)

    x = jnp.asarray(np.random.default_rng(7).uniform(0, 0.999, size=(4, 16)).astype(np.float32))
    for via_units in (True, False):
        g = jax.jit(jax.grad(functools.partial(loss, via_units=via_units)))(x)
        assert g.shape == x.shape and (np.asarray(g) == 0).all()
