from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class CategoricalValue(nn.Module):
    """Critic with logits over `num_bins` fixed bins instead of a scalar."""

    hidden_dims: Sequence[int]
    num_bins: int
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class((*self.hidden_dims, self.num_bins), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions):
        """Return logits (num_ensembles, ..., num_bins)."""
        if self.encoder is not None:
            observations = self.encoder(observations)
        return self.value_net(jnp.concatenate([observations, actions], axis=-1))


class CausalBlock(nn.Module):
    """Pre-LN causal self-attention block."""

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        L = x.shape[-2]
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y, mask=mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(4 * self.d_model, kernel_init=default_init())(y)
        y = nn.Dense(self.d_model, kernel_init=default_init())(nn.gelu(y))
        return x + y


class ARCritic(nn.Module):
    """Autoregressive binning critic: Q in [0, 1) as L base-B digits.

    Token 0 is the (s, a) embedding from an MLP trunk; tokens 1..L-1 are the prefix digits.
    The output at position l gives logits for digit l+1.
    num_classes (default: base) sets the size of the token embedding and output head only (e.g. 54 for
    the redundant signed vocabulary of agents/geo_codes.py); digits in and out are class indices.
    """

    hidden_dims: Sequence[int]
    base: int
    length: int
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    layer_norm: bool = True
    encoder: nn.Module = None
    num_classes: Optional[int] = None

    def setup(self):
        num_classes = self.num_classes or self.base
        self.trunk = MLP((*self.hidden_dims, self.d_model), activate_final=False, layer_norm=self.layer_norm)
        self.digit_embed = nn.Embed(num_classes, self.d_model)
        self.pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (self.length, self.d_model))
        self.blocks = [CausalBlock(self.d_model, self.num_heads) for _ in range(self.num_layers)]
        self.out_ln = nn.LayerNorm()
        self.head = nn.Dense(num_classes, kernel_init=default_init())

    def _logits(self, h, digits):
        tokens = jnp.concatenate([h[..., None, :], self.digit_embed(digits[..., :-1])], axis=-2)
        x = tokens + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.out_ln(x))

    def __call__(self, observations, actions, digits=None, rng=None):
        """Returns one of:
        - teacher-forced logits (..., L, B) if `digits` (..., L) is given. `digits` may carry an extra
          code axis (..., K, L), giving logits (..., K, L, B) with the trunk shared across codes.
        - K sampled codes (..., K, L) if `rng` is given, with K = rng.shape[0] (keys are only used for K).
        - greedy digits (..., L) otherwise.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        h = self.trunk(jnp.concatenate([observations, actions], axis=-1))

        if digits is not None:
            if digits.ndim == h.ndim + 1:
                h = jnp.broadcast_to(h[..., None, :], (*digits.shape[:-1], h.shape[-1]))
            return self._logits(h, digits)

        if rng is not None:
            h = jnp.broadcast_to(h[..., None, :], (*h.shape[:-1], rng.shape[0], h.shape[-1]))
            keys = jax.random.split(rng[0], self.length)
        digits = jnp.zeros((*h.shape[:-1], self.length), dtype=jnp.int32)
        for l in range(self.length):
            logits = self._logits(h, digits)[..., l, :]
            z = jax.random.categorical(keys[l], logits) if rng is not None else jnp.argmax(logits, axis=-1)
            digits = digits.at[..., l].set(z)
        return digits


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v
