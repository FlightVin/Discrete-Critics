import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class ACBSCAgent(flax.struct.PyTreeNode):
    """Action-Chunked Binary Success Critic (ACBSC).

    The critic is trained with BCE loss to estimate P(success | s_t, action_chunk).
    TD targets: 1.0 at chunks containing success, 0.0 at timeout chunks, Q_target elsewhere.

    Success vs timeout distinction (without changing existing mask/terminal definitions):
    With the robomimic dataset, both success and timeout steps have masks=0 (masks = 1 - dones).
    We distinguish them via the last step's reward contribution:
      - Success step reward = 0.0  (original 1.0, shifted by -1.0)
      - All other steps     = -1.0
    So: last_step_contribution = batch['rewards'][:,-1] - batch['rewards'][:,-2]
        is 0 at success and negative at timeout/non-terminal.

    The critic network output is an unbounded logit; sigmoid is applied at loss/inference time.
    discount=1.0 ensures Q = P(success), not a discounted time-to-success proxy.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ACBSC critic loss (BCE instead of MSE)."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next action chunk from the actor.
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        # Target critic outputs are logits; convert to probabilities ∈ [0, 1].
        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions
        )
        next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        # BSC-internal variables — does not redefine existing masks/terminals fields.
        #
        # With robomimic + -1 reward shift: success step reward = 0.0, all others = -1.0.
        # sample_sequence() accumulates: rewards[:,i] = sum_{j<=i} r_j * discount^j.
        # The last step's individual contribution:
        #   success terminal  → 0.0  * discount^(H-1) = 0   (> -1e-6)
        #   timeout/non-term  → -1.0 * discount^(H-1) < 0   (< -1e-6)
        last_step_contribution = batch['rewards'][..., -1] - batch['rewards'][..., -2]
        bsc_is_success  = last_step_contribution > -1e-6   # True only at success terminal
        bsc_is_terminal = batch['terminals'][..., -1] > 0.5  # True at any trajectory boundary

        # BSC target:
        #   success  → 1.0
        #   timeout  → 0.0  (bsc_is_terminal=True, bsc_is_success=False → neither term fires)
        #   non-term → discount * Q_target(s', next_chunk)
        target_q = (
            bsc_is_success.astype(jnp.float32) * 1.0
            + (~bsc_is_terminal).astype(jnp.float32) * self.config['discount'] * next_q
        )

        # Critic output is an unbounded logit. BCE uses logits directly for numerical stability.
        # q_logit shape: (num_ensembles, B). target_q shape: (B,).
        q_logit = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params
        )
        # Apply valid mask to zero out loss for sequences crossing trajectory boundaries.
        critic_loss = optax.sigmoid_binary_cross_entropy(q_logit, target_q[None])
        critic_loss = (critic_loss * batch['valid'][..., -1][None]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q': jax.nn.sigmoid(q_logit).mean(),
            'target_q': target_q.mean(),
            'success_frac': bsc_is_success.astype(jnp.float32).mean(),
            'timeout_frac': (bsc_is_terminal & ~bsc_is_success).astype(jnp.float32).mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the ACBSC actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2, 
                    (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss (identical to ACFQL).
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(
                batch['observations'], noises, params=grad_params
            )
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss: maximize P(success). Critic outputs logits → sigmoid to get probabilities.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            q_logit = self.network.select('critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(jax.nn.sigmoid(q_logit), axis=0)  # ∈ [0, 1]
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # beta independently scales the Q-loss (Q is now bounded to [-1, 0]).
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + self.config['beta'] * q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
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
        
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim'] *
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
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
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)

            # Critic outputs logits → sigmoid to get success probabilities before argmax.
            q_vals = jax.nn.sigmoid(self.network.select("critic")(observations, actions))
            if self.config["q_agg"] == "mean":
                q = q_vals.mean(axis=0)
            else:
                q = q_vals.min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[
                jnp.arange(bsize), indices, :
            ].reshape(bshape + (action_dim,))

        return actions

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
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
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
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        
        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
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
            agent_name='acbsc',
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=1.0,           # Must be 1.0: Q = P(success) with pure probability semantics.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            beta=1.0,               # Q-loss scale (new vs ACFQL). Controls Q signal vs BC/distillation.
        )
    )
    return config
