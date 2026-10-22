"""Value encodings of the v1 critics: ARB digit codes, CAT two-hot, HL-Gauss."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.acarb import ar_bin, ar_decode
from agents.accat import two_hot
from agents.achlg import hl_gauss

B, L = 10, 3


# --- ARB ---------------------------------------------------------------------------------------------

def _codes():
    n = np.arange(B ** L)
    return n, np.stack([n // 100, (n // 10) % 10, n % 10], -1).astype(np.int32)


def test_arb_roundtrip_decode_encode():
    n, digits = _codes()
    back = np.asarray(ar_bin(ar_decode(jnp.asarray(digits), B), B, L))
    fails = int((back != digits).any(-1).sum())
    print(f'ARB ar_bin(ar_decode(z)) failures: {fails}/1000')
    assert fails == 0


def test_arb_roundtrip_from_fraction():
    n, digits = _codes()
    back = np.asarray(ar_bin(jnp.asarray(n / 1000, dtype=jnp.float32), B, L))
    fails = int((back != digits).any(-1).sum())
    print(f'ARB ar_bin(n/1000) failures: {fails}/1000')
    assert fails == 0


def test_arb_floor_semantics_for_non_codes():
    y = jnp.array([0.5835, 0.0159, 0.9999, 0.0004, 0.12349, 0.1239], jnp.float32)
    want = [[5, 8, 3], [0, 1, 5], [9, 9, 9], [0, 0, 0], [1, 2, 3], [1, 2, 3]]
    np.testing.assert_array_equal(np.asarray(ar_bin(y, B, L)), want)
    # Out of [0, 1) clips to the end codes.
    np.testing.assert_array_equal(np.asarray(ar_bin(jnp.array([-0.3, 1.0, 7.0]), B, L)),
                                  [[0, 0, 0], [9, 9, 9], [9, 9, 9]])


# --- CAT two-hot -------------------------------------------------------------------------------------

ATOMS = jnp.linspace(-100.0, 0.0, 101)


def test_cat_support_spacing():
    assert ATOMS.shape == (101,)
    np.testing.assert_allclose(np.diff(np.asarray(ATOMS)), 1.0, atol=1e-5)


def test_two_hot_properties():
    y = jnp.asarray(np.random.RandomState(0).uniform(-100, 0, 4096), jnp.float32)
    p = two_hot(y, ATOMS)
    assert float(p.min()) >= 0.0
    np.testing.assert_allclose(np.asarray(p.sum(-1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray((p * ATOMS).sum(-1)), np.asarray(y), atol=2e-5)
    assert int(((p > 0).sum(-1) <= 2).all())


def test_two_hot_exact_at_atoms():
    p = two_hot(ATOMS, ATOMS)
    np.testing.assert_allclose(np.asarray(p), np.eye(101), atol=1e-5)


def test_two_hot_interpolation_direction():
    # -41.25 sits between atoms 58 (-42) and 59 (-41), 3/4 of the way to -41.
    p = np.asarray(two_hot(jnp.array([-41.25]), ATOMS))[0]
    assert abs(p[59] - 0.75) < 1e-5 and abs(p[58] - 0.25) < 1e-5  # float32 linspace atoms are inexact
    assert p.sum() - p[58] - p[59] == 0.0


def test_two_hot_clips_outside_support():
    p = np.asarray(two_hot(jnp.array([-250.0, 30.0, -100.0, 0.0]), ATOMS))
    np.testing.assert_allclose(p[0], np.eye(101)[0], atol=1e-6)
    np.testing.assert_allclose(p[1], np.eye(101)[100], atol=1e-6)
    np.testing.assert_allclose(p[2], np.eye(101)[0], atol=1e-6)
    np.testing.assert_allclose(p[3], np.eye(101)[100], atol=1e-6)


# --- HL-Gauss ----------------------------------------------------------------------------------------

NUM_BINS, VMIN, VMAX = 101, -100.0, 0.0
SIGMA = 0.75 * (VMAX - VMIN) / NUM_BINS
EDGES = jnp.linspace(VMIN, VMAX, NUM_BINS + 1, dtype=jnp.float32)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def ref_transform_to_probs(target):
    """Stop Regressing (arXiv 2403.03950v1) Fig. A.1, verbatim."""
    support = EDGES
    cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * SIGMA))
    z = cdf_evals[-1] - cdf_evals[0]
    bin_probs = cdf_evals[1:] - cdf_evals[:-1]
    return bin_probs / z


def ref_transform_from_probs(probs):
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    return jnp.sum(probs * centers)


def test_hlg_agent_constants():
    from agents.achlg import get_config, ACHLGAgent
    cfg = get_config()
    agent = ACHLGAgent(rng=None, network=None, config=dict(cfg))
    np.testing.assert_allclose(np.asarray(agent.edges()), np.asarray(EDGES), atol=1e-5)
    np.testing.assert_allclose(agent.sigma(), SIGMA, rtol=1e-9)


def test_hlg_matches_reference():
    rs = np.random.RandomState(0)
    y = np.concatenate([
        rs.uniform(VMIN, VMAX, 2000),                       # interior
        [VMIN, VMAX, VMIN + 1e-3, VMAX - 1e-3],              # boundary
        np.asarray(EDGES), np.asarray(CENTERS),              # on edges / centres
        rs.uniform(VMIN, VMIN + 3 * SIGMA, 200), rs.uniform(VMAX - 3 * SIGMA, VMAX, 200),  # near boundary
    ]).astype(np.float32)
    ours = hl_gauss(jnp.asarray(y), EDGES, SIGMA)
    ref = jax.vmap(ref_transform_to_probs)(jnp.asarray(y))
    diff = float(jnp.abs(ours - ref).max())
    print(f'HL-Gauss vs reference max abs diff: {diff:.3e} over {len(y)} targets')
    assert diff <= 1e-6
    np.testing.assert_allclose(np.asarray(ours.sum(-1)), 1.0, atol=1e-5)
    # float32 erf differences: the reference itself gives ~-1e-7 in far tails.
    print(f'HL-Gauss min prob: {float(ours.min()):.2e} (reference: {float(ref.min()):.2e})')
    assert float(ours.min()) >= -1e-6


def test_hlg_decode_bias():
    y = jnp.linspace(VMIN, VMAX, 10001)
    dec = jax.vmap(ref_transform_from_probs)(hl_gauss(y, EDGES, SIGMA))
    bias = np.asarray(dec - y)
    interior = np.asarray((y > VMIN + 4 * SIGMA) & (y < VMAX - 4 * SIGMA))
    print(f'HL-Gauss decode bias: max |bias| overall {np.abs(bias).max():.4f} '
          f'(at y={float(y[np.abs(bias).argmax()]):.3f}; bias at -100: {bias[0]:+.4f}, at 0: {bias[-1]:+.4f}); '
          f'interior (>4 sigma from edges) max {np.abs(bias[interior]).max():.2e}')
    assert np.abs(bias[interior]).max() < 1e-3
    assert np.abs(bias).max() < SIGMA  # truncation bias at the boundary is below one sigma


def test_hlg_out_of_support_finite():
    y = jnp.array([-1e6, -1e3, -150.0, -100.5, 0.5, 30.0, 1e3, 1e6], jnp.float32)
    p = hl_gauss(y, EDGES, SIGMA)
    assert bool(jnp.isfinite(p).all())
    np.testing.assert_allclose(np.asarray(p.sum(-1)), 1.0, atol=1e-5)
    # Clipping to the support: same as the boundary targets.
    np.testing.assert_allclose(np.asarray(p[:4]), np.asarray(jnp.broadcast_to(hl_gauss(jnp.array([VMIN]), EDGES, SIGMA), (4, NUM_BINS))), atol=1e-7)
    np.testing.assert_allclose(np.asarray(p[4:]), np.asarray(jnp.broadcast_to(hl_gauss(jnp.array([VMAX]), EDGES, SIGMA), (4, NUM_BINS))), atol=1e-7)
    # Without clipping the reference itself is 0/0 far outside the support.
    assert not bool(jnp.isfinite(ref_transform_to_probs(jnp.float32(-1e3))).all())


def test_hlg_loss_gradients_finite():
    y = jnp.array([-1e6, -100.0, -63.2, -0.01, 0.0, 1e6], jnp.float32)
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, len(y), NUM_BINS))

    def loss(logits, y):
        return optax.softmax_cross_entropy(logits, hl_gauss(y, EDGES, SIGMA)).mean()

    gl, gy = jax.grad(loss, argnums=(0, 1))(logits, y)
    assert bool(jnp.isfinite(gl).all()) and bool(jnp.isfinite(gy).all())
