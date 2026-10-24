import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents import geo_codes
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ARCritic, ensemblize

GEO1_LABELS = ('hard', 'twohot_last', 'twohot_last2', 'twohot_all')
VOCABS = ('ordinary', 'redundant')
ROLLINS = ('canonical',) + geo_codes.ROLLIN_MODES
# geo2b: the allowed (arb_label, arb_rollin) pairs of the redundant vocabulary.
GEO_COMBOS = {('corrective', 'corrective'): 'E5', ('corrective', 'model_mix'): 'E6', ('corrective', 'fixed'): 'E7',
              ('noisy_fixed', 'fixed'): 'E8', ('noisy_resid', 'fixed'): 'E9',
              ('noisy_fixed', 'model_mix'): 'E10', ('noisy_resid', 'model_mix'): 'E11'}
# geo3b: code lengths of the redundant vocabulary (coverage of [-22, 31] is proven for these, tests/test_geo3b_*).
REDUNDANT_LENGTHS = (geo_codes.LENGTH, 5)


def geo_config_errors(config):
    """Errors (list of str) for arb_vocab / arb_label / arb_rollin combinations outside the allowed set."""
    get = lambda key, default: config[key] if key in config else default
    vocab, label, rollin = get('arb_vocab', 'ordinary'), get('arb_label', 'hard'), get('arb_rollin', 'canonical')
    if vocab not in VOCABS or label not in GEO1_LABELS + geo_codes.LABEL_MODES or rollin not in ROLLINS:
        return [f'unknown arb_vocab={vocab} / arb_label={label} / arb_rollin={rollin}']
    errors = []
    if vocab == 'ordinary':
        if label in geo_codes.LABEL_MODES or rollin != 'canonical':
            errors.append(f'arb_label={label} arb_rollin={rollin} needs arb_vocab=redundant')
        return errors
    if (label, rollin) not in GEO_COMBOS:
        errors.append(f'arb_vocab=redundant allows (arb_label, arb_rollin) in {sorted(GEO_COMBOS)}, '
                      f'got ({label}, {rollin})')
    if get('arb_readout', None) != 'sample_mean' or get('arb_value', 'bsc') != 'bsc':
        errors.append('arb_vocab=redundant needs arb_readout=sample_mean and arb_value=bsc')
    if get('arb_base', None) != 10 or get('arb_length', None) not in REDUNDANT_LENGTHS:
        errors.append(f'arb_vocab=redundant needs arb_base=10 and arb_length in {REDUNDANT_LENGTHS}')
    return errors


def int32_errors(config):
    """Errors (list of str) if the sample_mean integer label arithmetic can leave int32.

    k_den = B^L num_qs M; every target k is in [0, k_den). The largest intermediates are the residual numerators
    N = k - (k_den / B^L) P_L (P_L: prefix in units of B^-L), |P_L| <= t_max (B^L - 1) / (B - 1) with t_max the
    largest token magnitude (B - 1 ordinary, 31 redundant), so |N| < k_den (1 + t_max / (B - 1)); c D_l and
    rem are bounded by |N| + D_l. The score S <= num_qs M (B^L - 1) < k_den."""
    get = lambda key, default: config[key] if key in config else default
    if get('arb_readout', None) != 'sample_mean':
        return []
    B, L = config['arb_base'], config['arb_length']
    k_den = B ** L * config['num_qs'] * config['arb_readout_samples']
    t_max = max(-geo_codes.TOKEN_MIN, geo_codes.TOKEN_MAX) if get('arb_vocab', 'ordinary') == 'redundant' else B - 1
    bound = k_den + k_den * t_max // (B - 1) + 2 * k_den // B
    if bound >= 2 ** 31:
        return [f'int32 label arithmetic overflows (k_den={k_den}, bound {bound} >= 2^31)']
    return []


def ar_bin(y, base, length):
    """Base-B expansion of y in [0, 1) into `length` digits, rounding down (Eq. 2.2).

    Uses the integer code floor(y B^L) to avoid float drift from repeated residuals. A plain float32
    floor drops exact codes by one (ar_decode([0, 1, 6]) = 0.01599999 -> 15), so values within
    1e-3 code units of an integer snap to it; everything else keeps floor (0.5835 -> 583).
    """
    s = y * base ** length
    r = jnp.round(s)
    s = jnp.where(jnp.abs(s - r) < 1e-3, r, jnp.floor(s))
    n = jnp.clip(s, 0, base ** length - 1).astype(jnp.int32)
    powers = base ** jnp.arange(length - 1, -1, -1, dtype=jnp.int32)
    return (n[..., None] // powers) % base


def ar_decode(digits, base):
    """dec_B(z_{1:L}) = sum_l z_l B^{-l}."""
    scales = base ** -jnp.arange(1, digits.shape[-1] + 1, dtype=jnp.float32)
    return (digits * scales).sum(axis=-1)


def ar_code(digits, base):
    """Integer code sum_l z_l B^(L-l) of digit sequences (..., L) (int32)."""
    L = digits.shape[-1]
    powers = base ** jnp.arange(L - 1, -1, -1, dtype=jnp.int32)
    return (digits.astype(jnp.int32) * powers).sum(-1)


def ar_sample_codes(critic_fn, observations, actions, gumbel, temp):
    """Sample codes autoregressively with per-copy noise: z_l ~ Categorical(softmax(logits_l / temp)).

    critic_fn maps (observations, actions, digits (num_qs, ..., M, L)) to teacher-forced logits
    (num_qs, ..., M, L, B), each copy reading its own digits (see ACARBAgent._percopy_critic).
    gumbel: standard Gumbel noise (num_qs, ..., M, L, B), one draw per copy, sample, element and
    position; argmax(logits / temp + gumbel) is an exact categorical sample. Each sequence's digit l
    is conditioned on its own sampled prefix (L teacher-forced passes). Returns digits (num_qs, ..., M, L).
    """
    L = gumbel.shape[-2]
    digits = jnp.zeros(gumbel.shape[:-1], dtype=jnp.int32)
    for l in range(L):
        logits = critic_fn(observations, actions, digits)[..., l, :]
        digits = digits.at[..., l].set(jnp.argmax(logits / temp + gumbel[..., l, :], axis=-1).astype(jnp.int32))
    return digits


def canonical_digits(k, k_den, base, length):
    """Canonical digits of u = k / k_den from the integer k alone: n = k // (k_den / B^L) = floor(B^L u), clipped
    to [0, B^L - 1], then its L base-B digits (..., L). Equals ar_bin(k / k_den) on the grid at L = 3; at L = 5 a
    float32 k / k_den is off by up to ~0.006 code units, so digits are never taken from the float."""
    n = jnp.clip(jnp.asarray(k, jnp.int32) // (k_den // base ** length), 0, base ** length - 1)
    powers = base ** jnp.arange(length - 1, -1, -1, dtype=jnp.int32)
    return (n[..., None] // powers) % base


def ar_label_dists(k, k_den, base, length, mode):
    """Exact digit labels for targets on the grid u = k / k_den (k int32).

    Returns (d, q, clip): d (..., L) canonical digits (the roll-in, canonical_digits), q (..., L, B) label
    distributions and clip (..., L) whether the position's residual was clamped to [0, B-1].
    Positions before `first soft` are one-hot d_l; the others are two-hot T(r_l) with
    r_l = (u - v_prefix) / w_l, v_prefix = dec(d_1..d_{l-1}), w_l = B^-l, computed in integers:
    with D_l = k_den / B^(l+1) and P_l the integer prefix code (units of B^-l), N = k - B D_l P_l,
    c = N // D_l (floor toward -inf), rem = N - c D_l, f = rem / D_l (f = 0 exactly for exact codes, then one
    class). No intermediate exceeds |N| + D_l (int32-safe for k_den up to ~10^9, see int32_errors).
    """
    first_soft = {'hard': length, 'twohot_last': length - 1, 'twohot_last2': length - 2, 'twohot_all': 0}[mode]
    k = jnp.asarray(k, jnp.int32)
    d = canonical_digits(k, k_den, base, length)
    qs, clips = [], []
    for l in range(length):
        onehot = jax.nn.one_hot(d[..., l], base)
        if l < first_soft:
            qs.append(onehot)
            clips.append(jnp.zeros(k.shape, bool))
            continue
        prefix = ar_code(d[..., :l], base) if l > 0 else jnp.zeros_like(k)
        den = k_den // base ** (l + 1)
        num = k - base * den * prefix
        a = num // den
        rem = num - a * den  # in [0, den)
        low = num < 0
        high = (a > base - 1) | ((a == base - 1) & (rem > 0))
        clip = low | high
        a = jnp.clip(a, 0, base - 1)
        f = jnp.where(clip, 0.0, rem.astype(jnp.float32) / den)
        lo = jax.nn.one_hot(a, base)
        hi = jax.nn.one_hot(jnp.minimum(a + 1, base - 1), base)
        q = jnp.where((f > 0)[..., None], (1.0 - f)[..., None] * lo + f[..., None] * hi, lo)
        qs.append(q)
        clips.append(clip)
    return d, jnp.stack(qs, axis=-2), jnp.stack(clips, axis=-1)


def soft_cross_entropy(logits, q):
    """-sum_t q(t) log softmax(logits)(t) (temperature 1); q is stop-gradient."""
    return -(jax.lax.stop_gradient(q) * jax.nn.log_softmax(logits, axis=-1)).sum(-1)


def ar_expected(critic_fn, observations, actions, base, length):
    """Exact E[dec(z)] under the AR digit distribution.

    Enumerates all B^(L-1) prefixes in one teacher-forced pass: their probabilities come from the
    first L-1 positions, and the last digit's expectation from position L.
    """
    powers = base ** jnp.arange(length - 2, -1, -1, dtype=jnp.int32)
    prefixes = (jnp.arange(base ** (length - 1), dtype=jnp.int32)[:, None] // powers) % base  # (K, L-1)
    digits = jnp.concatenate([prefixes, jnp.zeros_like(prefixes[:, :1])], axis=-1)  # (K, L)
    digits = jnp.broadcast_to(digits, (*observations.shape[:-1], *digits.shape))
    logp = jax.nn.log_softmax(critic_fn(observations, actions, digits), axis=-1)  # (num_qs, ..., K, L, B)

    log_prefix = (logp[..., :-1, :] * jax.nn.one_hot(prefixes, base)).sum((-2, -1))
    last = (jnp.exp(logp[..., -1, :]) * jnp.arange(base)).sum(-1)
    values = ar_decode(prefixes, base) + last * base ** -length  # (num_qs, ..., K)
    return (jnp.exp(log_prefix) * values).sum(-1)


# ARB-QC (arb_value='qc') maps raw QC values q in [-100, 0] to u = (q + 100) / 100 in [0, 1). With base 10
# and 3 digits, u has resolution 1e-3, i.e. 0.1 Q-units; q = 0 (u = 1) clips to the top code 0.999.
QC_SCALE = 100.0


def u_to_raw(u):
    return QC_SCALE * u - QC_SCALE


def raw_to_u(q):
    return (q + QC_SCALE) / QC_SCALE


def _wmean(x, w):
    return (x * w).sum() / jnp.maximum(w.sum(), 1.0)


class ACARBAgent(flax.struct.PyTreeNode):
    """Action-Chunked Autoregressive Binning critic (ACARB).

    Same targets as ACBSC (Q = P(success)), but the critic predicts Q as L base-B digits with a
    causal transformer, trained by teacher-forced cross-entropy. Q is read out by greedy decoding,
    a mean of sampled codes, or the exact expectation (arb_readout). Only the best-of-n actor is
    supported.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_value(self, observations, actions, target=False, rng=None, readout=None):
        """Q of each ensemble member, shape (num_qs, ...), via the configured readout."""
        critic_fn = self.network.select('target_critic' if target else 'critic')
        B, L = self.config['arb_base'], self.config['arb_length']
        readout = readout or self.config['arb_readout']
        if self._redundant():
            if readout == 'greedy':
                n = self._decode(critic_fn(observations, actions))
                return jnp.clip(n, 0, B ** L - 1).astype(jnp.float32) / B ** L
            if readout != 'sample_mean':
                raise ValueError(f'arb_readout={readout} is not implemented for arb_vocab=redundant')
        if readout == 'greedy':
            return ar_decode(critic_fn(observations, actions), B)
        if readout == 'sample':
            keys = jax.random.split(rng, self.config['arb_readout_samples'])
            return ar_decode(critic_fn(observations, actions, None, keys), B).mean(-1)
        if readout == 'expected':
            return ar_expected(critic_fn, observations, actions, B, L)
        if readout == 'sample_mean':
            n, _ = self._sampled_codes(observations, actions, target, rng)
            return n.sum(-1) / (B ** L * self.config['arb_readout_samples'])
        raise ValueError(f'unknown arb_readout {readout}')

    def _cfg(self, key, default):
        return self.config[key] if key in self.config else default

    def _redundant(self):
        return self._cfg('arb_vocab', 'ordinary') == 'redundant'

    def _decode(self, classes):
        """Class indices (..., L) -> integer value n (int32, units of B^-L), NOT clipped. Redundant vocabulary:
        tokens = classes - 22, n = 100 z1 + 10 z2 + z3 (signed); ordinary: the digit code."""
        if self._redundant():
            return geo_codes.decode_units(geo_codes.class_to_token(classes))
        return ar_code(classes, self.config['arb_base'])

    def readout_den(self):
        """K_den = B^L * num_qs * M: the sample_mean ensemble value is S / K_den."""
        return self.config['arb_base'] ** self.config['arb_length'] * self.config['num_qs'] * \
            self.config['arb_readout_samples']

    def _percopy_critic(self, target, params=None):
        """Teacher-forced critic whose digits argument has a leading num_qs axis (one prefix per copy).

        Same module and parameters as the 'critic' / 'target_critic' ensemble (nn.vmap over the same
        params); only the digits are mapped over the copies instead of broadcast. `params` (the critic
        module's params, e.g. grad_params['modules_critic']) overrides the stored ones."""
        c = self.config
        encoder = encoder_modules[c['encoder']]() if c['encoder'] is not None else None
        module = ensemblize(ARCritic, c['num_qs'], in_axes=(None, None, 0))(
            hidden_dims=c['value_hidden_dims'], base=c['arb_base'], length=c['arb_length'],
            d_model=c['arb_d_model'], num_layers=c['arb_num_layers'], num_heads=c['arb_num_heads'],
            layer_norm=c['layer_norm'], encoder=encoder, **self._vocab_kwargs(c),
        )
        if params is None:
            params = self.network.params['modules_target_critic' if target else 'modules_critic']
        return lambda o, a, d: module.apply({'params': params}, o, a, d)

    @staticmethod
    def _vocab_kwargs(config):
        """ARCritic kwargs of the vocabulary (none for the ordinary one: module unchanged)."""
        if (config['arb_vocab'] if 'arb_vocab' in config else 'ordinary') == 'redundant':
            return dict(num_classes=geo_codes.NUM_CLASSES)
        return {}

    def _sampled_codes(self, observations, actions, target, rng):
        """sample_mean readout: M codes per copy, (n (num_qs, ..., M) int32 clipped to [0, B^L-1], oor bool).

        oor marks decoded values outside [0, B^L - 1] before the clip (only possible for the redundant vocabulary)."""
        B, L, M = self.config['arb_base'], self.config['arb_length'], self.config['arb_readout_samples']
        num_classes = geo_codes.NUM_CLASSES if self._redundant() else B
        batch_shape = actions.shape[:-1]
        gumbel = jax.random.gumbel(rng, (self.config['num_qs'], *batch_shape, M, L, num_classes))
        digits = ar_sample_codes(self._percopy_critic(target), observations, actions, gumbel,
                                 self._cfg('arb_readout_temp', 1.0))
        n = self._decode(digits)
        oor = (n < 0) | (n > B ** L - 1)  # never for ordinary tokens
        return jnp.clip(n, 0, B ** L - 1), oor

    def critic_score(self, observations, actions, target=False, rng=None):
        """sample_mean: integer ensemble score S = sum over copies and samples of the codes (int32), and the
        fraction of decoded codes outside [0, B^L - 1] before the clip, both of shape (...)."""
        n, oor = self._sampled_codes(observations, actions, target, rng)
        return n.sum(axis=(0, -1)), oor.astype(jnp.float32).mean(axis=(0, -1))

    def td_target(self, batch, sample_rng, readout_rng):
        """TD target as (y, u, info): y in value units (P(success) or raw QC), u = clipped digit fraction."""
        B, L = self.config['arb_base'], self.config['arb_length']
        next_obs = batch['next_observations'][..., -1, :]
        if self.config['critic_target_bon']:
            next_actions = self.sample_actions(next_obs, rng=sample_rng)
        else:
            action_dim = self.config['action_dim'] * (
                self.config['horizon_length'] if self.config['action_chunking'] else 1
            )
            noises = jax.random.normal(sample_rng, (next_obs.shape[0], action_dim))
            next_actions = self.compute_flow_actions(next_obs, noises)
            next_actions = jnp.clip(next_actions, -1, 1)

        sample_mean = self.config['arb_readout'] == 'sample_mean'
        if sample_mean:
            # Integer ensemble score of the target critic; the value is S / K_den (q_agg = mean).
            next_s, oor = self.critic_score(next_obs, next_actions, target=True, rng=readout_rng)
        else:
            next_qs = self.critic_value(next_obs, next_actions, target=True, rng=readout_rng)  # digit fractions

        # Same terminal handling as ACBSC.
        bsc_is_terminal = batch['terminals'][..., -1] > 0.5
        bsc_is_success = bsc_is_terminal & (batch['masks'][..., -1] < 0.5)
        trajectory_success = batch['num_chunks_to_success'] >= 0.0
        info = {
            'success_frac': bsc_is_success.astype(jnp.float32).mean(),
            'timeout_frac': (bsc_is_terminal & ~bsc_is_success).astype(jnp.float32).mean(),
        }
        if sample_mean:
            # Exact grid: every target is u = k / K_den, k int32 (success: k = K_den (1 - B^-L); timeout: 0;
            # non-terminal: S of the target readout, discount 1).
            assert self.config['arb_value'] == 'bsc'
            k_den = self.readout_den()
            k_top = k_den - k_den // B ** L
            next_q = next_s.astype(jnp.float32) / k_den
            if self.config['use_mc_target']:
                target_q = trajectory_success.astype(jnp.float32)
                k = jnp.where(trajectory_success, k_top, 0)
            else:
                assert self.config['discount'] == 1.0, 'sample_mean targets need discount 1 (integer grid)'
                target_q = (
                    (bsc_is_terminal & bsc_is_success).astype(jnp.float32) * 1.0
                    + (~bsc_is_terminal).astype(jnp.float32) * next_q
                )
                k = jnp.where(bsc_is_terminal, jnp.where(bsc_is_success, k_den, 0), next_s)
            info['target_clip_frac'] = (k > k_top).astype(jnp.float32).mean()
            info['readout_oor_frac'] = oor.mean()
            info['target_k'] = jnp.minimum(k, k_top).astype(jnp.int32)  # popped by critic_loss
            u = jnp.clip(target_q, 0.0, 1.0 - B ** -L)
            info['target_u_mean'] = u.mean()
            return target_q, u, info

        if self.config['arb_value'] == 'qc':
            # Bellman backup in raw QC units, exactly as ACFQL: y = R + gamma^H * M * qbar(s', a').
            next_q = self._agg(u_to_raw(next_qs))
            target_q = batch['rewards'][..., -1] + \
                (self.config['discount'] ** self.config['horizon_length']) * batch['masks'][..., -1] * next_q
            u = raw_to_u(target_q)
            info['clip_low_frac'] = (u < 0.0).astype(jnp.float32).mean()
            info['clip_high_frac'] = (u > 1.0 - B ** -L).astype(jnp.float32).mean()
        else:
            next_q = self._agg(next_qs)
            if self.config['use_mc_target']:
                target_q = trajectory_success.astype(jnp.float32)
            else:
                target_q = (
                    (bsc_is_terminal & bsc_is_success).astype(jnp.float32) * 1.0
                    + (~bsc_is_terminal).astype(jnp.float32) * self.config['discount'] * next_q
                )
            u = target_q
        # u = 1 is not representable in [0, 1); clip to the largest code.
        return target_q, jnp.clip(u, 0.0, 1.0 - B ** -L), info

    def critic_ce(self, observations, actions, target_q, target_k, grad_params, rollin_rng, w):
        """Teacher-forced cross-entropy of the configured labels: (ce (num_qs, batch, L), logits, labels, info).

        labels are class indices broadcastable to (num_qs, batch, L) for the accuracy diagnostics; info holds
        label statistics (w: row weights, only used for those). target_q: clipped target value (batch,);
        target_k: its int32 grid numerator (sample_mean readout only); rollin_rng: the roll-in key."""
        B, L = self.config['arb_base'], self.config['arb_length']
        label_mode = self._cfg('arb_label', 'hard')
        if label_mode in geo_codes.LABEL_MODES:
            return self._geo_ce(observations, actions, target_k, grad_params, rollin_rng, w)
        info = {}
        if label_mode == 'hard':
            # sample_mean: digits of the integer target (== ar_bin(target_q) at L = 3, exact at any L).
            labels = ar_bin(target_q, B, L) if target_k is None else \
                canonical_digits(target_k, self.readout_den(), B, L)  # (batch, L)
        else:
            labels, label_q, label_clip = ar_label_dists(target_k, self.readout_den(), B, L, label_mode)

        # Positional args only: nn.vmap (ensemblize) silently drops kwargs.
        logits = self.network.select('critic')(
            observations, actions, labels, params=grad_params
        )  # (num_qs, batch, L, B)
        if label_mode == 'hard':
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels[None])  # (num_qs, batch, L)
        else:
            ce = soft_cross_entropy(logits, label_q[None])  # (num_qs, batch, L)
            for l in range(L):
                info[f'label_clip_pos{l + 1}'] = label_clip[..., l].astype(jnp.float32).mean()
        if target_k is not None and label_mode == 'hard':
            for l in range(L):
                info[f'label_clip_pos{l + 1}'] = jnp.zeros((), jnp.float32)
        return ce, logits, labels[None], info

    def _geo_ce(self, observations, actions, target_k, grad_params, rollin_rng, w):
        """Redundant vocabulary (E5-E9): one geo_codes.build prefix path per element and per copy.

        k is broadcast to (num_qs, batch); each copy is teacher-forced on its own roll-in classes with
        grad_params. E6 (model_mix): the model token at a position is ONE sample at arb_readout_temp from the
        current copy (stop-gradient params) given that copy's actual prefix, with noise
        fold_in(model_rng, position). model_rng = split(rollin_rng, 4)[3]: build uses split(rollin_rng, 3)
        (with partitionable threefry split(k, n)[i] does not depend on n, so index 3 is disjoint from it)."""
        num_qs = self.config['num_qs']
        k = jnp.broadcast_to(target_k, (num_qs, *target_k.shape))
        sg = jax.lax.stop_gradient
        to_classes = lambda t: jnp.clip(geo_codes.token_to_class(t), 0, geo_codes.NUM_CLASSES - 1)
        model_token = None
        if self._cfg('arb_rollin', 'canonical') == 'model_mix':
            model_rng = jax.random.split(rollin_rng, 4)[3]
            current = self._percopy_critic(False, params=sg(grad_params['modules_critic']))
            temp = self._cfg('arb_readout_temp', 1.0)

            def model_token(prefix_tokens, position):
                logits = sg(current(observations, actions, to_classes(prefix_tokens)))[..., position, :]
                gumbel = jax.random.gumbel(jax.random.fold_in(model_rng, position), logits.shape)
                return geo_codes.class_to_token(jnp.argmax(logits / temp + gumbel, axis=-1).astype(jnp.int32))

        L = self.config['arb_length']
        q, rollin, binfo = geo_codes.build(k, self.config['arb_label'], self.config['arb_rollin'], rollin_rng,
                                           model_token, k_den=self.readout_den(), length=L)
        q, rollin = sg(q), sg(rollin)
        logits = self._percopy_critic(False, params=grad_params['modules_critic'])(
            observations, actions, to_classes(rollin))  # (num_qs, batch, L, 54)
        ce = soft_cross_entropy(logits, q)  # (num_qs, batch, L)

        # Label bias: prefix value of the roll-ins + w_L E_{q_L}[token] - u, in units of 10^-L then scaled.
        tokens = geo_codes.class_to_token(jnp.arange(geo_codes.NUM_CLASSES)).astype(jnp.float32)
        last = (q[..., -1, :] * tokens).sum(-1)
        prefix = geo_codes.decode_units(rollin.at[..., -1].set(0))
        g = self.readout_den() // 10 ** L
        if L == geo_codes.LENGTH:
            bias = (prefix.astype(jnp.float32) + last - k.astype(jnp.float32) / g) / 10 ** L
        else:  # values ~10^5 code units: subtract the integer part of k / g exactly before going to float32
            bias = ((prefix - k // g).astype(jnp.float32) + last - (k % g).astype(jnp.float32) / g) / 10 ** L
        n_model = num_qs * target_k.shape[0] * (L - 1)
        info = {
            'oov_label': binfo['oov_label'].astype(jnp.float32),
            'oov_rollin': binfo['oov_rollin'].astype(jnp.float32),
            'model_used_frac': binfo['model_used'].astype(jnp.float32) / n_model,
            'model_clamp_frac': binfo['model_clamp'].astype(jnp.float32) / jnp.maximum(binfo['model_used'], 1),
            'label_mean_bias': _wmean(bias, jnp.broadcast_to(w, bias.shape)),
        }
        return ce, logits, jnp.argmax(q, axis=-1), info

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ACARB critic loss."""
        B, L = self.config['arb_base'], self.config['arb_length']

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Keys: TD next-action noise, target readout (training code stream), training roll-in (the 4th key:
        # unused while the roll-in is the canonical digits; the geo2b roll-ins of the redundant vocabulary).
        rng, sample_rng, readout_rng, rollin_rng = jax.random.split(rng, 4)
        raw_target, target_q, target_info = self.td_target(batch, sample_rng, readout_rng)
        target_k = target_info.pop('target_k', None)  # int32 grid numerator (sample_mean readout only)
        w = batch['valid'][..., -1]
        ce, logits, labels, label_info = self.critic_ce(
            batch['observations'], batch_actions, target_q, target_k, grad_params, rollin_rng, w)
        target_info.update(label_info)
        critic_loss = (ce.mean(axis=(0, 2)) * w).mean()

        # Diagnostics: teacher-forced vs greedy digits, and how errors propagate from digit 1. labels: class
        # indices broadcastable to (num_qs, batch, L) (redundant vocabulary: the argmax of each copy's labels).
        tf_correct = jnp.argmax(logits, axis=-1) == labels
        greedy = jax.lax.stop_gradient(self.network.select('critic')(batch['observations'], batch_actions))
        if self._redundant():
            greedy_q = jnp.clip(self._decode(greedy), 0, B ** L - 1).astype(jnp.float32) / B ** L
        else:
            greedy_q = ar_decode(greedy, B)
        greedy_correct = greedy == labels
        abs_err = jnp.abs(greedy_q - target_q[None])
        first_wrong = ~greedy_correct[..., 0]
        wb = jnp.broadcast_to(w, abs_err.shape)

        info = {
            'critic_loss': critic_loss,
            'q': _wmean(greedy_q, wb),
            'target_q': target_q.mean(),
            'greedy_mae': _wmean(abs_err, wb),
            'first_digit_wrong_frac': _wmean(first_wrong, wb),
            'greedy_mae_first_wrong': _wmean(abs_err, wb * first_wrong),
            'greedy_mae_first_right': _wmean(abs_err, wb * ~first_wrong),
            **target_info,
        }
        if self.config['arb_value'] == 'qc':
            # q / target_q / greedy_mae above are digit fractions; these are raw QC units.
            info['q_raw'] = _wmean(u_to_raw(greedy_q), wb)
            info['target_q_raw'] = raw_target.mean()
        for l in range(L):
            info[f'ce_digit{l + 1}'] = _wmean(ce[..., l], wb)
            info[f'tf_acc_digit{l + 1}'] = _wmean(tf_correct[..., l], wb)
            info[f'greedy_acc_digit{l + 1}'] = _wmean(greedy_correct[..., l], wb)
        return critic_loss, info

    @jax.jit
    def readout_diagnostics(self, batch, rng):
        """Greedy vs exact expected readout, on batch (s, a) and on best-of-n candidates.

        If the digit distribution is sharp, the two agree and greedy is harmless. Costly (exact
        expectation over B^(L-1) prefixes), so it is not part of the update; call it at log intervals.
        Values are digit fractions; keys carry the 'critic/' prefix used by the update info.
        """
        B, L = self.config['arb_base'], self.config['arb_length']
        sg = jax.lax.stop_gradient
        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"][..., 0, :]
        observations = batch['observations']
        greedy_q = self.critic_value(observations, actions, readout='greedy')
        wb = jnp.broadcast_to(batch['valid'][..., -1], greedy_q.shape)
        if self._redundant() or L > geo_codes.LENGTH:
            return self._redundant_readout_diagnostics(observations, actions, greedy_q, wb, batch, rng)
        expected_q = sg(self.critic_value(observations, actions, readout='expected'))
        digit1_p = jax.nn.softmax(sg(self.network.select('critic')(
            observations, actions, jnp.zeros((*actions.shape[:-1], L), jnp.int32)))[..., 0, :], axis=-1)

        # Best-of-n candidates on a slice of the batch.
        obs, cands = self._candidates(observations[:self.config['arb_diag_obs']], rng)
        cand_greedy = self._agg(sg(self.critic_value(obs, cands, readout='greedy')))  # (n_obs, N)
        cand_expected = self._agg(sg(self.critic_value(obs, cands, readout='expected')))
        top = cand_greedy.max(-1, keepdims=True)
        spread = lambda q: (q.max(-1) - q.min(-1)).mean()
        # The run's own readout on the batch (s, a), with a key derived from the diagnostics stream (the
        # candidates above keep `rng` itself).
        q_readout = sg(self.readout_q(observations, actions, jax.random.fold_in(rng, 1)))

        info = {
            'expected_q': _wmean(expected_q, wb),
            'greedy_expected_gap': _wmean(jnp.abs(greedy_q - expected_q), wb),
            'greedy_minus_expected': _wmean(greedy_q - expected_q, wb),
            'digit1_maxprob': _wmean(digit1_p.max(-1), wb),
            'bon_tie_frac': ((cand_greedy == top).sum(-1) > 1).mean(),
            'bon_argmax_agree': (cand_greedy.argmax(-1) == cand_expected.argmax(-1)).mean(),
            'bon_greedy_spread': spread(cand_greedy),
            'bon_expected_spread': spread(cand_expected),
            'bon_greedy_expected_gap': jnp.abs(cand_greedy - cand_expected).mean(),
            'q_readout': _wmean(q_readout, batch['valid'][..., -1]),
        }
        return {f'critic/{k}': v for k, v in info.items()}

    def _redundant_readout_diagnostics(self, observations, actions, greedy_q, wb, batch, rng):
        """readout_diagnostics for the redundant vocabulary or L > 3: greedy (decoded tokens, final value clipped)
        and the run's readout; no exact expectation (54^(L-1) or B^(L-1) prefixes per element)."""
        sg = jax.lax.stop_gradient
        L = self.config['arb_length']
        greedy_n = self._decode(self.network.select('critic')(observations, actions))
        digit1_p = jax.nn.softmax(sg(self.network.select('critic')(
            observations, actions, jnp.zeros((*actions.shape[:-1], L), jnp.int32)))[..., 0, :], axis=-1)
        obs, cands = self._candidates(observations[:self.config['arb_diag_obs']], rng)
        cand_greedy = self._agg(sg(self.critic_value(obs, cands, readout='greedy')))  # (n_obs, N)
        top = cand_greedy.max(-1, keepdims=True)
        q_readout = sg(self.readout_q(observations, actions, jax.random.fold_in(rng, 1)))
        info = {
            'greedy_q': _wmean(greedy_q, wb),
            'greedy_oor_frac': _wmean((greedy_n < 0) | (greedy_n > 10 ** L - 1), wb),
            'digit1_maxprob': _wmean(digit1_p.max(-1), wb),
            'bon_tie_frac': ((cand_greedy == top).sum(-1) > 1).mean(),
            'bon_greedy_spread': (cand_greedy.max(-1) - cand_greedy.min(-1)).mean(),
            'q_readout': _wmean(q_readout, batch['valid'][..., -1]),
        }
        return {f'critic/{k}': v for k, v in info.items()}

    def actor_loss(self, batch, grad_params, rng):
        """BC flow loss (best-of-n uses the critic only at sampling time)."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        return bc_flow_loss, {
            'actor_loss': bc_flow_loss,
            'bc_flow_loss': bc_flow_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Soft-update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @staticmethod
    def _offline_update(agent, batch):
        """Offline update: trains actor always; trains critic only if train_critic_offline=True."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            info = {}
            rng_ = rng
            rng_, actor_rng, critic_rng = jax.random.split(rng_, 3)

            actor_loss, actor_info = agent.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v

            if agent.config['train_critic_offline']:
                critic_loss, critic_info = agent.critic_loss(batch, grad_params, critic_rng)
                for k, v in critic_info.items():
                    info[f'critic/{k}'] = v
                loss = actor_loss + critic_loss
            else:
                loss = actor_loss

            return loss, info

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        if agent.config['train_critic_offline']:
            agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def offline_update(self, batch):
        return self._offline_update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Vectorised update over a leading batch dimension (for UTD > 1)."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Best-of-n over BC flow samples, ranked by Q under the configured readout."""
        action_dim = self.config['action_dim'] * \
                     (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        rng, readout_rng = jax.random.split(rng)
        observations, actions = self._candidates(observations, rng)
        if self.config['arb_readout'] == 'sample_mean':
            q, _ = self.critic_score(observations, actions, rng=readout_rng)  # integer S; first index on ties
        else:
            q = self._agg(self.critic_value(observations, actions, rng=readout_rng))
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))

        return actions

    def _candidates(self, observations, rng):
        """actor_num_samples BC flow actions per observation: (..., N, obs_dim), (..., N, action_dim)."""
        action_dim = self.config['action_dim'] * \
                     (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config["actor_num_samples"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
        actions = jnp.clip(self.compute_flow_actions(observations, noises), -1, 1)
        return observations, actions

    def readout_q(self, observations, actions, rng):
        """Ensemble value under the configured readout: S / K_den for sample_mean, else _agg(critic_value)."""
        if self.config['arb_readout'] == 'sample_mean':
            return self.critic_score(observations, actions, rng=rng)[0].astype(jnp.float32) / self.readout_den()
        return self._agg(self.critic_value(observations, actions, rng=rng))

    def _agg(self, q_vals):
        return q_vals.mean(axis=0) if self.config["q_agg"] == "mean" else q_vals.min(axis=0)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        assert config['actor_type'] == 'best-of-n', 'greedy ARB readout has no action gradient'
        if 'arb_value' not in config:  # configs saved before ARB-QC existed
            config['arb_value'] = 'bsc'
        assert config['arb_value'] in ('bsc', 'qc'), config['arb_value']
        if config['arb_value'] == 'qc':
            # Raw QC values span [-1/(1-gamma), 0] = [-QC_SCALE, 0] only for gamma = 0.99.
            assert abs(config['discount'] - (1 - 1 / QC_SCALE)) < 1e-9, 'arb_value=qc needs discount=0.99'
            assert not config['use_mc_target'], 'use_mc_target is a BSC-only option'
        assert config['arb_readout'] in ('greedy', 'sample', 'expected', 'sample_mean'), config['arb_readout']
        label = config['arb_label'] if 'arb_label' in config else 'hard'
        geo_errors = geo_config_errors(config)
        assert not geo_errors, '; '.join(geo_errors)
        if config['arb_readout'] == 'sample_mean':
            assert config['q_agg'] == 'mean', 'arb_readout=sample_mean needs q_agg=mean'
            int_errors = int32_errors(config)
            assert not int_errors, '; '.join(int_errors)
        if label != 'hard':
            assert config['arb_value'] == 'bsc' and config['arb_readout'] == 'sample_mean', \
                f'arb_label={label} needs arb_value=bsc and arb_readout=sample_mean (integer target grid)'
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Define networks.
        critic_def = ensemblize(ARCritic, config['num_qs'])(
            hidden_dims=config['value_hidden_dims'],
            base=config['arb_base'],
            length=config['arb_length'],
            d_model=config['arb_d_model'],
            num_layers=config['arb_num_layers'],
            num_heads=config['arb_num_heads'],
            layer_norm=config['layer_norm'],
            encoder=encoders.get('critic'),
            **cls._vocab_kwargs(config),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        network.params['modules_target_critic'] = network.params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='acarb',
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Critic trunk hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=1.0,  # Q = P(success), as in ACBSC. arb_value='qc' needs 0.99.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            num_qs=2,  # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int),  # will be set
            action_chunking=True,  # False means n-step return
            actor_type="best-of-n",  # Only best-of-n is supported.
            actor_num_samples=32,
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            critic_target_bon=True,  # If False, use a single BC flow sample for TD targets instead of best-of-n.
            use_mc_target=False,  # If True, use the trajectory outcome (MC) as target instead of TD.
            train_critic_offline=True,  # If False, skip critic training during offline phase.
            arb_base=10,  # B: digits per position.
            arb_length=3,  # L: number of digits.
            arb_d_model=128,  # Transformer width.
            arb_num_layers=2,  # Transformer blocks.
            arb_num_heads=4,  # Attention heads.
            arb_readout='greedy',  # Q readout for best-of-n and TD targets: 'greedy', 'sample' (mean of samples), 'expected' (exact),
                                   # 'sample_mean' (per-copy sampled codes at arb_readout_temp, integer ensemble score S).
            arb_readout_samples=32,  # Codes averaged per (s, a) (and per copy for sample_mean) when arb_readout='sample'/'sample_mean'.
            arb_readout_temp=1.0,  # sample_mean: softmax temperature of the sampled digits.
            arb_label='hard',  # 'hard' (one-hot digits) or 'twohot_last' / 'twohot_last2' / 'twohot_all' (needs sample_mean);
                               # redundant vocabulary: 'corrective' / 'noisy_fixed' / 'noisy_resid' (agents/geo_codes.py).
            arb_vocab='ordinary',  # 'ordinary' (tokens 0..9) or 'redundant' (signed tokens -22..31, 54 classes).
            arb_rollin='canonical',  # Teacher-forcing prefix: 'canonical' digits, or (redundant) 'corrective' /
                                     # 'model_mix' / 'fixed' (allowed pairs with arb_label: GEO_COMBOS).
            arb_diag_obs=32,  # Observations used for the best-of-n readout diagnostics.
            arb_value='bsc',  # 'bsc': digits encode P(success); 'qc': digits encode (q + 100) / 100 of the raw QC value.
        )
    )
    return config
