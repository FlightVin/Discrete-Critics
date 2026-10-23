"""Redundant signed-token labels and roll-ins for the autoregressive value critic (experiments E5-E9).

Pure, jit-compatible functions; nothing here touches an agent or a training loop.

Code: base 10, 3 positions, weights w = (0.1, 0.01, 0.001). Targets live on an integer grid u = k / k_den
(default K_DEN = 20000, must be a multiple of 1000; k in [0, 0.999 * k_den]). Tokens are SIGNED integers
t in [TOKEN_MIN, TOKEN_MAX] = [-22, 31] (54 classes); the class index of token t is t - TOKEN_MIN (= t + 22).
Token values are always int32 "tokens"; class indices only appear as the last axis of `labels`.

Exact integer arithmetic (int32). With prefix units P = 100 z1 + 10 z2 (units of 0.001) before position l,
  N = k - (k_den / 1000) * P,   D_l = k_den * w_l = (k_den/10, k_den/100, k_den/1000),   r_l = N / D_l,
  c = N // D_l (floor toward -inf),   rem = N - c * D_l in [0, D_l),   f = eps = rem / D_l.

Label modes (per position, distribution over tokens):
  'corrective'  (E5-E7) two-hot T(r): 1 - f on c, f on c + 1 (f = 0 -> single mass on c; c + 1 never indexed).
  'noisy_fixed' (E8)    1/3 each on c - 1, c, c + 1 (expected token c).
  'noisy_resid' (E9)    0.5 (1 - f), 0.5, 0.5 f on c - 1, c, c + 1 (expected token r - 0.5; c + 1 not indexed if f = 0).
Labels at a position depend on the ACTUAL prefix (earlier roll-in tokens), never on that position's own roll-in.

Roll-in modes (context tokens fed at positions 1 and 2; nothing is fed after position 3):
  'corrective' (E5)  z ~ T(r), i.e. z = c + Bernoulli(f) (drawn exactly as randint[0, D) < rem).
  'model_mix'  (E6)  with prob. 0.5 z ~ T(r), else z = clip(m, c - 2, c + 2) with m = model_token(prefix, position).
  'fixed'      (E7-E9) z = c + delta, delta uniform on {-2, ..., 2}.
Randomness is independent per element, position and purpose (Bernoulli / mixture coin / delta), derived from one key.

API
  build(k, label_mode, rollin_mode, key, model_token=None, k_den=K_DEN) -> (labels, rollin, info)
    k:        int (...)  grid units (any leading shape, e.g. (num_qs, batch)); floats are rounded (see units_from_value).
    labels:   float32 (..., 3, 54), rows sum to 1, stop-gradient.
    rollin:   int32 (..., 3) TOKEN values; rollin[..., 2] is always 0 (the final position's token is never fed).
              Not clipped: callers must clip / check info['oov_rollin'] before embedding.
    info:     counts (int32 scalars): 'oov_label' (label entries with positive mass on a token outside [-22, 31],
              counted BEFORE the class index is clipped), 'oov_rollin', 'model_clamp' (model token outside
              [c - 2, c + 2] when the model branch is used), 'model_used';
              float32 'oov_label_mass' (total mass that was outside, before clipping);
              plus per-element arrays 'c', 'rem', 'den' (int32 (..., 3)) and 'model_mask' (bool (..., 2)).
    model_token(prefix_tokens, position) -> int32 (...): called for position 0 and 1 (0-indexed) with prefix_tokens
              int32 (..., 3) holding the actual roll-in tokens before `position` and 0 elsewhere. The callback owns
              its randomness (fold `position` into its key so the two calls are independent).
  label_and_info(k, prefix_tokens, label_mode, k_den) -> labels / info for a fixed prefix (used by build and tests).
  residual(k, prefix_units, position, k_den) -> (c, rem, den);  label_support(c, rem, den, mode) -> (tokens, masses)
  two_hot(c, rem, den) -> (..., 54);  decode_units(tokens) = 100 z1 + 10 z2 + z3 (units of 0.001).
  units_from_value(u, k_den) -> int32 round(u * k_den), stop-gradient.
"""

import jax
import jax.numpy as jnp

K_DEN = 20000
LENGTH = 3
TOKEN_MIN = -22
TOKEN_MAX = 31
NUM_CLASSES = TOKEN_MAX - TOKEN_MIN + 1  # 54
UNIT_WEIGHTS = (100, 10, 1)  # w_l in units of 0.001
ROLLIN_POSITIONS = 2
DELTA_MAX = 2

LABEL_MODES = ('corrective', 'noisy_fixed', 'noisy_resid')
ROLLIN_MODES = ('corrective', 'model_mix', 'fixed')


def token_to_class(tokens):
    return tokens - TOKEN_MIN


def class_to_token(classes):
    return classes + TOKEN_MIN


def _check_k_den(k_den):
    if k_den % 1000 != 0 or k_den <= 0:
        raise ValueError(f'k_den must be a positive multiple of 1000, got {k_den}')


def denominator(position, k_den=K_DEN):
    """D_l = k_den * w_l (python int)."""
    _check_k_den(k_den)
    return k_den // 10 ** (position + 1)


def units_from_value(u, k_den=K_DEN):
    """Value u in [0, 1) -> int32 grid units round(u * k_den). Not differentiable (stop-gradient)."""
    return jnp.round(jax.lax.stop_gradient(u) * k_den).astype(jnp.int32)


def decode_units(tokens):
    """Signed tokens (..., 3) -> value in units of 0.001: 100 z1 + 10 z2 + z3."""
    tokens = jnp.asarray(tokens, jnp.int32)
    return 100 * tokens[..., 0] + 10 * tokens[..., 1] + tokens[..., 2]


def residual(k, prefix_units, position, k_den=K_DEN):
    """Exact residual at `position` (0-indexed) given the prefix value in 0.001 units.

    Returns int32 (c, rem, den) with r = c + rem / den, c = floor(r), 0 <= rem < den.
    """
    den = denominator(position, k_den)
    num = jnp.asarray(k, jnp.int32) - (k_den // 1000) * jnp.asarray(prefix_units, jnp.int32)
    c = jnp.floor_divide(num, den)
    rem = num - c * den
    return c, rem, jnp.full_like(c, den)


def label_support(c, rem, den, mode):
    """Token values and float32 masses on offsets (-1, 0, +1) around c, shape (..., 3) each."""
    f = rem.astype(jnp.float32) / den.astype(jnp.float32)
    zero = jnp.zeros_like(f)
    if mode == 'corrective':
        masses = (zero, 1.0 - f, f)
    elif mode == 'noisy_fixed':
        third = jnp.full_like(f, 1.0 / 3.0)
        masses = (third, third, third)
    elif mode == 'noisy_resid':
        masses = (0.5 * (1.0 - f), jnp.full_like(f, 0.5), 0.5 * f)
    else:
        raise ValueError(f'unknown label_mode {mode!r}')
    tokens = c[..., None] + jnp.arange(-1, 2, dtype=jnp.int32)
    masses = jnp.stack(masses, axis=-1)
    # f = 0 -> exactly zero mass on c + 1 (never index it).
    masses = masses.at[..., 2].set(jnp.where(rem > 0, masses[..., 2], 0.0))
    return tokens, masses


def _scatter(tokens, masses):
    """(..., S) token values / masses -> (..., 54) distribution plus OOV entry count and mass (pre-clip)."""
    live = masses > 0
    oov = live & ((tokens < TOKEN_MIN) | (tokens > TOKEN_MAX))
    classes = jnp.clip(token_to_class(tokens), 0, NUM_CLASSES - 1)
    onehot = jax.nn.one_hot(classes, NUM_CLASSES, dtype=jnp.float32) * live[..., None]
    dist = jnp.sum(onehot * masses[..., None], axis=-2)
    return dist, jnp.sum(oov, dtype=jnp.int32), jnp.sum(jnp.where(oov, masses, 0.0))


def two_hot(c, rem, den):
    """T(r) as a (..., 54) distribution over classes."""
    return _scatter(*label_support(c, rem, den, 'corrective'))[0]


def _prefix_units(prefix_tokens, position):
    return sum(UNIT_WEIGHTS[j] * prefix_tokens[..., j] for j in range(position))


def label_and_info(k, prefix_tokens, label_mode, k_den=K_DEN):
    """Labels (..., 3, 54) for a fixed prefix path prefix_tokens (..., 3) (the last token is ignored)."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f'unknown label_mode {label_mode!r}')
    prefix_tokens = jnp.asarray(prefix_tokens, jnp.int32)
    cs, rems, dens = zip(*(residual(k, _prefix_units(prefix_tokens, l), l, k_den) for l in range(LENGTH)))
    c, rem, den = (jnp.stack(x, axis=-1) for x in (cs, rems, dens))
    labels, oov, oov_mass = _scatter(*label_support(c, rem, den, label_mode))
    info = {'oov_label': oov, 'oov_label_mass': oov_mass, 'c': c, 'rem': rem, 'den': den}
    return jax.lax.stop_gradient(labels), info


def build(k, label_mode, rollin_mode, key, model_token=None, k_den=K_DEN):
    """Sample one roll-in path per element and return (labels, rollin, info). See the module docstring."""
    if rollin_mode not in ROLLIN_MODES:
        raise ValueError(f'unknown rollin_mode {rollin_mode!r}')
    if rollin_mode == 'model_mix' and model_token is None:
        raise ValueError("rollin_mode='model_mix' needs a model_token callback")
    _check_k_den(k_den)
    k = jax.lax.stop_gradient(jnp.asarray(k))
    if jnp.issubdtype(k.dtype, jnp.floating):
        k = jnp.round(k)
    k = k.astype(jnp.int32)
    shape = k.shape + (ROLLIN_POSITIONS,)
    key_bern, key_coin, key_delta = jax.random.split(key, 3)
    dens = jnp.array([denominator(l, k_den) for l in range(ROLLIN_POSITIONS)], jnp.int32)
    bern_u = jax.random.randint(key_bern, shape, 0, dens)  # Bernoulli(rem/den) == bern_u < rem, exactly
    coin = jax.random.bernoulli(key_coin, 0.5, shape)  # True -> model branch
    delta = jax.random.randint(key_delta, shape, -DELTA_MAX, DELTA_MAX + 1)

    rollin = jnp.zeros(k.shape + (LENGTH,), jnp.int32)
    model_mask = jnp.zeros(shape, bool)
    clamp = jnp.zeros(shape, bool)
    for l in range(ROLLIN_POSITIONS):
        c, rem, _ = residual(k, _prefix_units(rollin, l), l, k_den)
        corrective = c + (bern_u[..., l] < rem).astype(jnp.int32)
        if rollin_mode == 'corrective':
            z = corrective
        elif rollin_mode == 'fixed':
            z = c + delta[..., l]
        else:
            m = jnp.asarray(model_token(jax.lax.stop_gradient(rollin), l), jnp.int32)
            m = jnp.broadcast_to(m, k.shape)
            use = coin[..., l]
            z = jnp.where(use, jnp.clip(m, c - DELTA_MAX, c + DELTA_MAX), corrective)
            model_mask = model_mask.at[..., l].set(use)
            clamp = clamp.at[..., l].set(use & ((m < c - DELTA_MAX) | (m > c + DELTA_MAX)))
        rollin = rollin.at[..., l].set(z)
    rollin = jax.lax.stop_gradient(rollin)

    labels, info = label_and_info(k, rollin, label_mode, k_den)
    fed = rollin[..., :ROLLIN_POSITIONS]
    info.update(
        oov_rollin=jnp.sum((fed < TOKEN_MIN) | (fed > TOKEN_MAX), dtype=jnp.int32),
        model_clamp=jnp.sum(clamp, dtype=jnp.int32),
        model_used=jnp.sum(model_mask, dtype=jnp.int32),
        model_mask=model_mask,
    )
    return labels, rollin, info
