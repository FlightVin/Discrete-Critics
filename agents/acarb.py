import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ARCritic, ensemblize


def ar_bin(y, base, length):
    """Base-B expansion of y in [0, 1) into `length` digits, rounding down (Eq. 2.2).

    Uses the integer code floor(y B^L) to avoid float drift from repeated residuals.
    """
    n = jnp.clip(jnp.floor(y * base ** length), 0, base ** length - 1).astype(jnp.int32)
    powers = base ** jnp.arange(length - 1, -1, -1, dtype=jnp.int32)
    return (n[..., None] // powers) % base


def ar_decode(digits, base):
    """dec_B(z_{1:L}) = sum_l z_l B^{-l}."""
    scales = base ** -jnp.arange(1, digits.shape[-1] + 1, dtype=jnp.float32)
    return (digits * scales).sum(axis=-1)


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
        if readout == 'greedy':
            return ar_decode(critic_fn(observations, actions), B)
        if readout == 'sample':
            keys = jax.random.split(rng, self.config['arb_readout_samples'])
            return ar_decode(critic_fn(observations, actions, None, keys), B).mean(-1)
        if readout == 'expected':
            return ar_expected(critic_fn, observations, actions, B, L)
        raise ValueError(f'unknown arb_readout {readout}')

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ACARB critic loss."""
        B, L = self.config['arb_base'], self.config['arb_length']

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng, readout_rng, diag_rng = jax.random.split(rng, 4)
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

        next_qs = self.critic_value(next_obs, next_actions, target=True, rng=readout_rng)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        # Same terminal handling as ACBSC.
        bsc_is_terminal = batch['terminals'][..., -1] > 0.5
        bsc_is_success = bsc_is_terminal & (batch['masks'][..., -1] < 0.5)
        trajectory_success = batch['num_chunks_to_success'] >= 0.0

        if self.config['use_mc_target']:
            target_q = trajectory_success.astype(jnp.float32)
        else:
            target_q = (
                (bsc_is_terminal & bsc_is_success).astype(jnp.float32) * 1.0
                + (~bsc_is_terminal).astype(jnp.float32) * self.config['discount'] * next_q
            )
        # y = 1 is not representable in [0, 1); clip to the largest code.
        target_q = jnp.clip(target_q, 0.0, 1.0 - B ** -L)
        labels = ar_bin(target_q, B, L)  # (batch, L)

        # Positional args only: nn.vmap (ensemblize) silently drops kwargs.
        logits = self.network.select('critic')(
            batch['observations'], batch_actions, labels, params=grad_params
        )  # (num_qs, batch, L, B)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels[None])  # (num_qs, batch, L)
        w = batch['valid'][..., -1]
        critic_loss = (ce.mean(axis=(0, 2)) * w).mean()

        # Diagnostics: teacher-forced vs greedy digits, and how errors propagate from digit 1.
        tf_correct = jnp.argmax(logits, axis=-1) == labels[None]
        greedy = jax.lax.stop_gradient(self.network.select('critic')(batch['observations'], batch_actions))
        greedy_q = ar_decode(greedy, B)
        greedy_correct = greedy == labels[None]
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
            'success_frac': bsc_is_success.astype(jnp.float32).mean(),
            'timeout_frac': (bsc_is_terminal & ~bsc_is_success).astype(jnp.float32).mean(),
        }
        for l in range(L):
            info[f'ce_digit{l + 1}'] = _wmean(ce[..., l], wb)
            info[f'tf_acc_digit{l + 1}'] = _wmean(tf_correct[..., l], wb)
            info[f'greedy_acc_digit{l + 1}'] = _wmean(greedy_correct[..., l], wb)
        info.update(self._readout_diagnostics(batch['observations'], batch_actions, greedy_q, wb, diag_rng))
        return critic_loss, info

    def _readout_diagnostics(self, observations, actions, greedy_q, wb, rng):
        """Greedy vs exact expected readout, on batch (s, a) and on best-of-n candidates.

        If the digit distribution is sharp, the two agree and greedy is harmless.
        """
        B, L = self.config['arb_base'], self.config['arb_length']
        sg = jax.lax.stop_gradient
        expected_q = sg(self.critic_value(observations, actions, readout='expected'))
        digit1_p = jax.nn.softmax(sg(self.network.select('critic')(
            observations, actions, jnp.zeros((*actions.shape[:-1], L), jnp.int32)))[..., 0, :], axis=-1)

        # Best-of-n candidates on a slice of the batch.
        obs, cands = self._candidates(observations[:self.config['arb_diag_obs']], rng)
        cand_greedy = self._agg(sg(self.critic_value(obs, cands, readout='greedy')))  # (n_obs, N)
        cand_expected = self._agg(sg(self.critic_value(obs, cands, readout='expected')))
        top = cand_greedy.max(-1, keepdims=True)
        spread = lambda q: (q.max(-1) - q.min(-1)).mean()

        return {
            'expected_q': _wmean(expected_q, wb),
            'greedy_expected_gap': _wmean(jnp.abs(greedy_q - expected_q), wb),
            'greedy_minus_expected': _wmean(greedy_q - expected_q, wb),
            'digit1_maxprob': _wmean(digit1_p.max(-1), wb),
            'bon_tie_frac': ((cand_greedy == top).sum(-1) > 1).mean(),
            'bon_argmax_agree': (cand_greedy.argmax(-1) == cand_expected.argmax(-1)).mean(),
            'bon_greedy_spread': spread(cand_greedy),
            'bon_expected_spread': spread(cand_expected),
            'bon_greedy_expected_gap': jnp.abs(cand_greedy - cand_expected).mean(),
        }

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
            discount=1.0,  # Q = P(success), as in ACBSC.
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
            arb_readout='greedy',  # Q readout for best-of-n and TD targets: 'greedy', 'sample' (mean of samples), 'expected' (exact).
            arb_readout_samples=32,  # Codes averaged per (s, a) when arb_readout='sample'.
            arb_diag_obs=32,  # Observations used for the best-of-n readout diagnostics.
        )
    )
    return config
