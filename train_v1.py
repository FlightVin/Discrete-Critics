"""v1 protocol: resumable BC preparation and online training with two replay buffers.

  --mode=bc      actor-only BC on the demos for --bc_steps updates; n1 eval every --eval_interval and at
                 the end; writes <run_dir>/final_actor.pkl.
  --mode=online  --method (qc | bsc | arb_bsc | arb_qc | cat | hlg) selects agents.v1_config(method);
                 --agent.* overrides apply on top. Agent created with --seed, actor replaced by --init_actor,
                 fresh optimisers; then phases
                 warmup (n1 actions, no updates) -> pretrain (critic-only updates on main replay)
                 -> pretrain_eval (n1 + best-of-n eval, critic probe) -> joint (best-of-n actions; per env
                 step one critic loss on main replay + actor BC loss on success replay, one gradient step).

Optimisers: actor and critic modules each have their own Adam (own moments and step count; same lr,
betas, eps and weight decay as the agent's single optimiser), so the actor's first joint step is a first
Adam step and critic-only / actor-only updates leave the other optimiser's state untouched.

Preemption: if env AE_PREEMPT_FLAG names a file, its appearance (written by the job preamble about two
minutes before the kill) stops the run at the next safe point (episode boundary during env interaction,
any update boundary otherwise; a running evaluation is abandoned), which writes a checkpoint and exits
with code 143. A background thread polls the file every second; safe points also check it. If no safe
point is reached within --preempt_budget_s of the file's mtime (a long episode, a first-time compile),
the run exits 143 without a new checkpoint and resumes from the last committed one. SIGTERM is
treated the same way, best effort only (on Babel the kill follows it within a second). Started again
with the same arguments the run resumes from the committed checkpoint; a finished run exits 0 at once.
Checkpoints are also written every --ckpt_minutes, after each evaluation and at each phase end. See
utils/ckpt_v1.py and utils/replay_v1.py.

Notes on the schedule: warm-up ends at the first episode boundary at or after --warmup_steps (so no
episode is cut). Joint evaluations are due at multiples of --eval_interval online steps and run at the
next episode boundary (a safe point), logged at the actual step with eval/nominal_step; the final
evaluation runs when the online counter reaches --online_steps.
"""
import json
import os
import pickle
import random
import re
import signal
import sys
import threading
import time
from collections import defaultdict

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ml_collections import config_flags
from scipy.stats import spearmanr

from agents import V1_METHODS, agents, v1_config
from envs.env_utils import make_env_and_datasets, make_vectorized_eval_envs
from envs.robomimic_utils import get_max_episode_length, is_robomimic_env
from evaluation import evaluate_parallel
from log_utils import setup_wandb, get_flag_dict
from utils.ckpt_v1 import CheckpointStore
from utils.datasets import Dataset
from utils.replay_v1 import FIELDS, SeqReplay
from visualization.critic_probe import probe_batch, probe_states
from visualization.eval_checkpoint import Policy

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'online', ['bc', 'online'], 'bc: shared BC preparation run; online: one method run.')
flags.DEFINE_enum('method', None, list(V1_METHODS), 'v1 method preset (agents.v1_config); required online.')
flags.DEFINE_string('run_dir', None, 'Run directory (checkpoints, metrics log, final_actor.pkl).')
flags.DEFINE_string('init_actor', None, 'online: path to final_actor.pkl from the bc run.')
flags.DEFINE_string('env_name', 'square-mh-low_dim', 'Environment (dataset) name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length H.')
flags.DEFINE_bool('sparse', True, 'Sparse reward.')
flags.DEFINE_float('discount', 0.99, 'Discount used by sample_sequence for cumulative rewards.')

flags.DEFINE_integer('bc_steps', 1000000, 'bc: actor BC updates.')
flags.DEFINE_integer('warmup_steps', 50000, 'online: warm-up env steps (n1 actions, no updates).')
flags.DEFINE_integer('pretrain_updates', 50000, 'online: critic-only updates after warm-up.')
flags.DEFINE_integer('online_steps', 1000000, 'online: total online env steps, warm-up included.')
flags.DEFINE_integer('buffer_cap', 2000000, 'Capacity of each replay buffer (demos included).')

flags.DEFINE_integer('eval_interval', 100000, 'Eval every this many updates (bc) / online env steps (joint).')
flags.DEFINE_integer('n_eval_envs', 32, 'Parallel eval envs.')
flags.DEFINE_integer('eval_episodes', 2, 'Eval episodes per env and rule.')
flags.DEFINE_integer('eval_seed', 1000, 'Eval env i is reseeded with eval_seed + i before every rule.')
flags.DEFINE_integer('probe_states', 1024, 'Fixed demo states for the critic probe.')
flags.DEFINE_integer('log_interval', 5000, 'Log interval (updates / env steps).')
flags.DEFINE_float('ckpt_minutes', 20.0, 'Periodic checkpoint interval (minutes) at safe points.')
flags.DEFINE_float('preempt_budget_s', 60.0, 'After a preemption notice: if no safe point has been reached '
                   'this many seconds after it, exit without a new checkpoint (resume from the last one).')

flags.DEFINE_string('project_name', 'qc', 'W&B project.')
flags.DEFINE_string('run_group', 'v1', 'W&B group.')
flags.DEFINE_string('wandb_name', None, 'W&B run name.')


def argv_with_method_agent(argv):
    """With --method and no --agent file: use the method's agent file, so --agent.* overrides parse."""
    method = [a.split('=', 1)[1] for a in argv if a.startswith('--method=')]
    if method and method[-1] in V1_METHODS and not any(a == '--agent' or a.startswith('--agent=') for a in argv):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents', f'{V1_METHODS[method[-1]][0]}.py')
        return [argv[0], f'--agent={path}', *argv[1:]]
    return argv


if __name__ == '__main__':
    sys.argv[:] = argv_with_method_agent(sys.argv)  # before DEFINE_config_file: it reads sys.argv
config_flags.DEFINE_config_file('agent', 'agents/acbsc.py', lock_config=False)

# Flags that may change between restarts of the same run.
RESTART_FREE_FLAGS = {'ckpt_minutes', 'project_name', 'run_group', 'wandb_name', 'log_interval', 'preempt_budget_s'}
EXIT_PREEMPTED = 143
ONLINE_PHASES = ['warmup', 'pretrain', 'pretrain_eval', 'joint', 'done']
BC_PHASES = ['bc', 'done']


# ---- update functions (generic over agents with critic_loss / actor_loss / target_update) -------------
def _modules(params, critic):
    return [k for k in params if ('critic' in k) == critic]


def _group(module):
    return 'target' if 'target' in module else 'critic' if 'critic' in module else 'actor'


def split_optimizer(agent):
    """Rebuild the agent's TrainState with one Adam for the actor modules and one for the critic.

    Same lr / betas / eps / weight decay as the agent's create(); target modules get no optimiser (they
    are set by target_update). Optimiser state starts fresh (step counts 0)."""
    cfg = agent.config

    def adam():
        if cfg['weight_decay'] > 0.:
            return optax.adamw(learning_rate=cfg['lr'], weight_decay=cfg['weight_decay'])
        return optax.adam(learning_rate=cfg['lr'])

    params = agent.network.params
    tx = optax.multi_transform({'actor': adam(), 'critic': adam(), 'target': optax.set_to_zero()},
                               {k: _group(k) for k in params})
    return agent.replace(network=agent.network.replace(tx=tx, opt_state=tx.init(params)))


def opt_counts(agent):
    """{group: Adam step count} of a split optimiser."""
    out = {}
    for g in ('actor', 'critic'):
        st = agent.network.opt_state.inner_states[g]
        counts = [int(x) for x in jax.tree_util.tree_leaves(st) if np.ndim(x) == 0]
        out[g] = counts[0]
    return out


def _keep(new_net, old_net, names, group):
    """Restore modules `names` and optimiser group `group` from old_net (that group was not updated)."""
    params = dict(new_net.params)
    for k in names:
        params[k] = old_net.params[k]
    opt = new_net.opt_state
    if isinstance(opt, optax.MultiTransformState):
        inner = dict(opt.inner_states)
        inner[group] = old_net.opt_state.inner_states[group]
        opt = opt._replace(inner_states=inner)
    return new_net.replace(params=params, opt_state=opt)


@jax.jit
def bc_update(agent, batch):
    """Actor BC loss only; critic, target critic and the critic optimiser untouched."""
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        loss, info = agent.actor_loss(batch, grad_params, rng)
        return loss, {f'actor/{k}': v for k, v in info.items()}

    network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    network = _keep(network, agent.network, _modules(network.params, critic=True), 'critic')
    return agent.replace(network=network, rng=new_rng), info


@jax.jit
def critic_update(agent, batch):
    """Critic loss only, then the target update; actor params and actor optimiser kept bit-identical."""
    new_rng, rng = jax.random.split(agent.rng)

    def loss_fn(grad_params):
        loss, info = agent.critic_loss(batch, grad_params, rng)
        return loss, {f'critic/{k}': v for k, v in info.items()}

    network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
    network = _keep(network, agent.network, _modules(network.params, critic=False), 'actor')
    agent.target_update(network, 'critic')
    return agent.replace(network=network, rng=new_rng), info


def joint_rngs(agent):
    """(next agent rng, actor loss rng, critic loss rng) of joint_update."""
    new_rng, rng = jax.random.split(agent.rng)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    return new_rng, actor_rng, critic_rng


def joint_loss(agent, main_batch, success_batch, grad_params, actor_rng, critic_rng):
    critic_loss, critic_info = agent.critic_loss(main_batch, grad_params, critic_rng)
    actor_loss, actor_info = agent.actor_loss(success_batch, grad_params, actor_rng)
    info = {f'critic/{k}': v for k, v in critic_info.items()}
    info.update({f'actor/{k}': v for k, v in actor_info.items()})
    return critic_loss + actor_loss, info


@jax.jit
def joint_update(agent, main_batch, success_batch):
    """Critic loss on main replay + actor BC loss on success replay, one gradient step, then target update."""
    new_rng, actor_rng, critic_rng = joint_rngs(agent)
    network, info = agent.network.apply_loss_fn(
        loss_fn=lambda p: joint_loss(agent, main_batch, success_batch, p, actor_rng, critic_rng))
    agent.target_update(network, 'critic')
    return agent.replace(network=network, rng=new_rng), info


@jax.jit
def n1_actions(agent, observations, rng):
    """One actor sample per decision, no critic (eval_checkpoint.make_bon with n=1)."""
    adim = agent.config['action_dim'] * agent.config['horizon_length']
    noises = jax.random.normal(rng, (*observations.shape[:-1], 1, adim))
    obs = observations[..., None, :]
    return jnp.clip(agent.compute_flow_actions(obs, noises), -1, 1)[..., 0, :]


def bon_actions(agent, observations, rng):
    return agent.sample_actions(observations=observations, rng=rng)


def _seed_numpy(seed):
    """Runs inside an eval env worker (via AsyncVectorEnv.call): robomimic resets draw from numpy's RNG."""
    np.random.seed(seed)


def _agent_override_keys(argv):
    """Keys given on the command line as --agent.<key>=... or --agent.<key> <value>."""
    keys = []
    for a in argv:
        m = re.match(r'^--agent\.([A-Za-z0-9_.]+)(=|$)', a)
        if m:
            keys.append(m.group(1))
    return keys


def resolve_agent_config(method, flag_config, argv):
    """v1_config(method) with the command line's --agent.* overrides applied on top.

    flag_config is FLAGS.agent (the --agent file's get_config() with the overrides already applied; with
    --method the --agent file defaults to the method's agent, see argv_with_method_agent)."""
    if method is None:
        return flag_config
    config = v1_config(method)
    assert flag_config['agent_name'] == config['agent_name'], \
        f'--agent is {flag_config["agent_name"]} but --method={method} uses {config["agent_name"]}'
    with config.unlocked():
        for k in _agent_override_keys(argv):
            config[k] = flag_config[k]
    return config


def check_agent_config(mode, config):
    """Refuse configurations outside the v1 protocol (exit code 2)."""
    errors = []
    if config['actor_type'] != 'best-of-n':  # bc too: best-of-n has the pure BC actor loss
        errors.append(f"actor_type={config['actor_type']} (need best-of-n)")
    if mode == 'online':
        if FLAGS.method is None:
            errors.append('--method is required for --mode=online')
        if config['actor_num_samples'] != 32:
            errors.append(f"actor_num_samples={config['actor_num_samples']} (need 32)")
    if errors:
        print('CONFIG_REFUSED ' + '; '.join(errors), flush=True)
        sys.exit(2)


def actor_params(agent):
    return {k: jax.device_get(v) for k, v in agent.network.params.items() if 'critic' not in k}


def process_demos(ds, env_name, H, sparse):
    """main.py process_train_dataset for robomimic (dataset_proportion 1): rewards - 1, sparse, chunk counts."""
    d = {k: np.asarray(v) for k, v in ds.items()}
    if is_robomimic_env(env_name):
        d['rewards'] = d['rewards'] - 1.0
    if sparse:
        d['rewards'] = ((d['rewards'] != 0.0) * -1.0).astype(d['rewards'].dtype)
    max_ep_len = get_max_episode_length(env_name) if is_robomimic_env(env_name) else 0
    terminal_locs = np.nonzero(d['terminals'] > 0)[0]
    ends = np.concatenate([[0], terminal_locs + 1])
    success = d['masks'][terminal_locs] < 0.5
    nct = np.zeros(len(d['masks']), dtype=np.float32)
    ncs = np.full(len(d['masks']), -1.0, dtype=np.float32)
    for i, (s, e) in enumerate(zip(ends[:-1], ends[1:])):
        local = np.arange(e - s)
        if max_ep_len > 0:
            nct[s:e] = np.ceil((max_ep_len - local) / H)
        if success[i]:
            ncs[s:e] = np.ceil((e - s - local) / H)
    nct = np.maximum(nct, 0)
    m = ncs >= 0
    ncs[m] = np.minimum(ncs[m], nct[m])
    d['num_chunks_to_timeout'] = nct
    d['num_chunks_to_success'] = ncs
    return d


class MetricsLog:
    """Authoritative JSONL log in run_dir; mirrored to W&B with an explicit global step."""

    def __init__(self, path, offset):
        mode = 'r+b' if os.path.exists(path) else 'w+b'
        self.f = open(path, mode)
        self.f.truncate(offset)  # drop rows written after the checkpoint we resume from
        self.f.seek(offset)

    def offset(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        return self.f.tell()

    def log(self, kind, step, data, wandb_too=True):
        row = dict(kind=kind, step=int(step), time=time.time())
        row.update({k: _py(v) for k, v in data.items()})
        self.f.write((json.dumps(row) + '\n').encode())
        self.f.flush()
        if wandb_too:
            import wandb
            payload = {f'{kind}/{k}': _py(v) for k, v in data.items()}
            payload['global_step'] = int(step)
            wandb.log(payload)


def _py(v):
    v = np.asarray(v) if not isinstance(v, (str, list, dict)) else v
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v.tolist()
    return v


class EvalAborted(Exception):
    pass


class PreemptWatcher:
    """Polls the file named by AE_PREEMPT_FLAG; calls on_flag(mtime) once when it exists."""

    def __init__(self, path, on_flag, poll_s=1.0):
        self.path, self.on_flag, self.poll_s = path, on_flag, poll_s
        self.fired = False
        self.lock = threading.Lock()
        self.t_last = 0.0
        threading.Thread(target=self._loop, daemon=True, name='preempt-watch').start()

    def check(self, min_interval=0.0):
        """True once the file has been seen. Cheap; also called from safe points (rate-limited)."""
        if self.fired:
            return True
        now = time.time()
        if now - self.t_last < min_interval:
            return False
        self.t_last = now
        try:
            mtime = os.stat(self.path).st_mtime
        except FileNotFoundError:
            return False
        with self.lock:
            if not self.fired:
                self.fired = True
                self.on_flag(mtime)
        return True

    def _loop(self):
        while not self.check():
            time.sleep(self.poll_s)


class Trainer:
    def __init__(self):
        self.t_start = time.time()
        self.stop_requested = False
        self.t_stop_ref = None      # flag file mtime (or SIGTERM receipt time): exit deadline reference
        self.stop_source = None
        self.pid = os.getpid()
        self.done_msg = None
        self.ckpt_lock = threading.Lock()
        signal.signal(signal.SIGTERM, self._on_sigterm)
        flag = os.environ.get('AE_PREEMPT_FLAG')
        self.watcher = PreemptWatcher(flag, self._on_flag) if flag else None

        self.run_dir = FLAGS.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.store = CheckpointStore(os.path.join(self.run_dir, 'ckpt'))
        self.store.test_delay = float(os.environ.get('V1_TEST_CKPT_DELAY', 0))  # e2e test: widen the write
        self.mode = FLAGS.mode
        self.phases = ONLINE_PHASES if self.mode == 'online' else BC_PHASES

    def _on_sigterm(self, signum, frame):
        # Forked eval env workers inherit this handler: they ignore SIGTERM (Slurm signals every process)
        # and keep serving until the main process aborts the eval and closes them.
        if os.getpid() != self.pid:
            return
        if not self.stop_requested:
            self.t_stop_ref, self.stop_source = time.time(), 'sigterm'
            print(f'SIGTERM_RECEIVED t={self.t_stop_ref:.3f} (best effort)', flush=True)
            threading.Thread(target=self._deadline, daemon=True, name='preempt-deadline').start()
        self.stop_requested = True

    def _on_flag(self, mtime):
        """Preemption notice (watcher thread or a safe point)."""
        if not self.stop_requested:
            self.t_stop_ref, self.stop_source = mtime, 'flag'
            print(f'PREEMPT_FLAG_SEEN age_s={time.time() - mtime:.2f} phase={self.phase_name()}', flush=True)
            threading.Thread(target=self._deadline, daemon=True, name='preempt-deadline').start()
        self.stop_requested = True

    def _deadline(self):
        """The next safe point can be far (an episode, a first-time compile, a slow machine). If it has not
        been reached preempt_budget_s after the notice, exit now: the committed checkpoint stays valid and
        the work since it is redone after the restart (as after a hard kill)."""
        time.sleep(max(0.0, self.t_stop_ref + FLAGS.preempt_budget_s - time.time()))
        with self.ckpt_lock:  # never cut a checkpoint write short (harmless, but it would be wasted)
            dt = time.time() - self.t_stop_ref
            print(f'EXIT_PREEMPTED_UNSAVED source={self.stop_source} notice_to_exit_s={dt:.2f} '
                  f'phase={self.phase_name()} committed_gen={self.store.gen}', flush=True)
            try:
                with open(os.path.join(self.run_dir, 'preemptions.jsonl'), 'a') as f:
                    f.write(json.dumps(dict(time=time.time(), source=self.stop_source, unsaved=True,
                                            notice_to_exit_s=round(dt, 3), gen=self.store.gen)) + '\n')
                for proc in getattr(getattr(self, 'eval_envs', None), 'processes', []):
                    proc.terminate()
            finally:
                sys.stdout.flush()
                os._exit(EXIT_PREEMPTED)

    def phase_name(self):
        return getattr(self, 'phase', None)

    # ---- setup ----------------------------------------------------------------------------------
    def check_args(self):
        """Refuse to resume a run directory with different arguments."""
        flag_dict = {k: v for k, v in get_flag_dict().items() if k not in RESTART_FREE_FLAGS}
        path = os.path.join(self.run_dir, 'args.json')
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            new = json.loads(json.dumps(flag_dict, default=str))
            diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new) if old.get(k) != new.get(k)}
            if diff:
                print(f'ARGS_MISMATCH with {path}: {diff}', flush=True)
                sys.exit(2)
        else:
            with open(path + '.tmp', 'w') as f:
                json.dump(flag_dict, f, indent=1, default=str)
            os.replace(path + '.tmp', path)

    def setup(self):
        self.check_args()
        config = resolve_agent_config(FLAGS.method, FLAGS.agent, sys.argv)
        config['horizon_length'] = FLAGS.horizon_length
        check_agent_config(self.mode, config)
        self.config = config
        self.H = FLAGS.horizon_length
        self.batch_size = config['batch_size']

        env, eval_env, ds, _ = make_env_and_datasets(FLAGS.env_name)
        if hasattr(eval_env, 'close'):
            eval_env.close()
        self.env = env
        self.max_ep_len = get_max_episode_length(FLAGS.env_name) if is_robomimic_env(FLAGS.env_name) else 0
        self.demos = process_demos(ds, FLAGS.env_name, self.H, FLAGS.sparse)
        self.action_dim = self.demos['actions'].shape[-1]

        agent = agents[config['agent_name']].create(
            FLAGS.seed, self.demos['observations'][0], self.demos['actions'][0], config)
        self.init_actor_info = None
        if self.mode == 'online':
            with open(FLAGS.init_actor, 'rb') as f:
                blob = pickle.load(f)
            # The BC run may have used any agent: copy this agent's actor modules (same networks in every
            # agent); modules this agent lacks (acbsc's unused one-step flow for acarb) are ignored.
            assert blob['horizon_length'] == self.H, (blob['horizon_length'], self.H)
            params = dict(agent.network.params)
            need = _modules(params, critic=False)
            assert set(need) <= set(blob['actor_params']), (sorted(blob['actor_params']), sorted(params))
            ignored = sorted(set(blob['actor_params']) - set(need))
            for k in need:
                v = blob['actor_params'][k]
                shapes = lambda t: jax.tree_util.tree_map(np.shape, t)  # noqa: E731
                assert shapes(v) == shapes(params[k]), k
                params[k] = jax.tree_util.tree_map(jnp.asarray, v)
            agent = agent.replace(network=agent.network.replace(params=params))
            self.init_actor_info = dict(path=FLAGS.init_actor, agent_name=blob['agent_name'], loaded=need,
                                        ignored=ignored,
                                        **{k: blob[k] for k in ('bc_steps', 'seed', 'source') if k in blob})
        self.agent = split_optimizer(agent)  # fresh: both step counts 0

        # Eval envs before seeding/restoring: AsyncVectorEnv builds a dummy env in this process, and
        # robomimic's make_env reseeds numpy's global RNG. Keep the global state untouched regardless.
        np_state = np.random.get_state()
        self.eval_envs = make_vectorized_eval_envs(FLAGS.env_name, FLAGS.n_eval_envs, seed=FLAGS.eval_seed)
        np.random.set_state(np_state)

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        self.action_key = jax.random.PRNGKey(FLAGS.seed + 7919)

        if self.mode == 'bc':
            self.demo_ds = Dataset.create(**{k: self.demos[k] for k in FIELDS})
        else:
            self.buffers = dict(
                main=SeqReplay(self.demos, FLAGS.buffer_cap, self.H, episodic=False),
                success=SeqReplay(self.demos, FLAGS.buffer_cap, self.H, episodic=True),
            )

        # Probe states and noise: fixed for the whole run (visualization/critic_probe.py).
        idx, self.probe_steps_left = probe_states(self.demos, self.H, FLAGS.probe_states, 0)
        self.probe_obs = self.demos['observations'][idx]
        self.probe_actions = np.stack([self.demos['actions'][i:i + self.H].reshape(-1) for i in idx])
        rs = np.random.RandomState(1)
        self.probe_noises = rs.standard_normal(
            (len(idx), config['actor_num_samples'], self.probe_actions.shape[-1])).astype(np.float32)
        self.probe_rand = rs.uniform(-1, 1, (len(idx), self.probe_actions.shape[-1])).astype(np.float32)

        self.phase = self.phases[0]
        self.ctr = defaultdict(int)
        self.last_eval = 0          # nominal step of the last completed evaluation
        self.env_acc = defaultdict(list)
        self.metrics_offset = 0
        self.wandb_id = None
        self.restore()

        self.metrics = MetricsLog(os.path.join(self.run_dir, 'metrics.jsonl'), self.metrics_offset)
        self.setup_wandb()
        run_config = dict(method=FLAGS.method, agent_name=config['agent_name'], agent=config.to_dict(),
                          init_actor=self.init_actor_info)
        import wandb
        wandb.config.update(dict(v1=run_config), allow_val_change=True)
        if self.metrics_offset == 0:  # first row of the run's log (rewritten if no checkpoint follows it)
            self.metrics.log('config', 0, run_config, wandb_too=False)
        print(f'CONFIG method={FLAGS.method} agent={config["agent_name"]} '
              f'actor_type={config["actor_type"]} actor_num_samples={config["actor_num_samples"]}', flush=True)
        self.t_last_ckpt = time.time()
        self.ob, self.action_queue, self.ep_rows = None, [], []

    def setup_wandb(self):
        import wandb
        id_path = os.path.join(self.run_dir, 'wandb_id')
        if self.wandb_id is None and os.path.exists(id_path):
            with open(id_path) as f:
                self.wandb_id = f.read().strip()
        if self.wandb_id is None:
            import secrets
            self.wandb_id = secrets.token_hex(6)  # W&B run ids: [a-z0-9]
            with open(id_path, 'w') as f:
                f.write(self.wandb_id)
        os.environ['WANDB_RUN_ID'] = self.wandb_id
        os.environ['WANDB_RESUME'] = 'allow'
        setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group,
                    name=FLAGS.wandb_name or os.path.basename(os.path.normpath(self.run_dir)))
        wandb.define_metric('global_step')
        wandb.define_metric('*', step_metric='global_step')

    # ---- checkpoint -----------------------------------------------------------------------------
    def state_dict(self):
        state = dict(
            version=2, mode=self.mode, phase=self.phase, ctr=dict(self.ctr), last_eval=self.last_eval,
            env_acc={k: list(v) for k, v in self.env_acc.items()},
            agent=jax.device_get(flax.serialization.to_state_dict(self.agent)),
            action_key=np.asarray(jax.device_get(self.action_key)),
            np_rng=np.random.get_state(), py_rng=random.getstate(),
            metrics_offset=self.metrics.offset(), wandb_id=self.wandb_id,
        )
        if self.mode == 'online':
            state['buffers'] = {k: b.meta() for k, b in self.buffers.items()}
            state['demo_crc'] = self.demo_crc()
        return state

    def demo_crc(self):
        from utils.ckpt_v1 import rows_crc
        return rows_crc({k: self.demos[k] for k in FIELDS})

    def checkpoint(self, reason):
        print(f'CKPT_BEGIN phase={self.phase} reason={reason}', flush=True)
        segs, keep = {}, {}
        if self.mode == 'online':
            for k, b in self.buffers.items():
                segs[k] = b.new_segment()  # before state_dict(): the saved meta must include it
                keep[k] = b.k_lo
        with self.ckpt_lock:
            state = self.state_dict()
            info = self.store.save(state, segs, keep, meta=dict(phase=self.phase, ctr=dict(self.ctr)))
        self.t_last_ckpt = time.time()
        print(f'CKPT_DONE gen={info["gen"]} reason={reason} state_mb={info["state_bytes"] / 1e6:.1f} '
              f'segment_mb={info["segment_bytes"] / 1e6:.1f} seconds={info["t_total"]:.2f}', flush=True)
        return info

    def restore(self):
        state, man = self.store.load()
        if state is None:
            print('RESUME none (fresh run)', flush=True)
            return
        assert state['mode'] == self.mode and state.get('version') == 2, (state['mode'], state.get('version'))
        self.agent = flax.serialization.from_state_dict(self.agent, state['agent'])
        self.action_key = jnp.asarray(state['action_key'])
        np.random.set_state(state['np_rng'])
        random.setstate(state['py_rng'])
        self.phase = state['phase']
        self.ctr = defaultdict(int, state['ctr'])
        self.last_eval = state['last_eval']
        self.env_acc = defaultdict(list, state['env_acc'])
        self.metrics_offset = state['metrics_offset']
        self.wandb_id = state['wandb_id']
        if self.mode == 'online':
            assert state['demo_crc'] == self.demo_crc(), 'demo data differ from the checkpointed run'
            for k, b in self.buffers.items():
                b.restore(state['buffers'][k], self.store.iter_segments(k))
        print(f'RESUME gen={man["gen"]} phase={self.phase} ctr={dict(self.ctr)}', flush=True)

    def safe_point(self, force=None):
        """Called only where the state is fully defined by state_dict() + replay."""
        if self.watcher is not None:
            self.watcher.check(min_interval=0.5)  # the thread polls too; this bounds the latency at safe points
        due = (time.time() - self.t_last_ckpt) > FLAGS.ckpt_minutes * 60
        if self.stop_requested or force or due:
            self.checkpoint(force or (f'preempt_{self.stop_source}' if self.stop_requested else 'periodic'))
            if self.done_msg:  # the phase is committed as complete only now
                print(self.done_msg, flush=True)
                self.done_msg = None
        if self.stop_requested:
            self.exit_preempted()

    def exit_preempted(self):
        self.close_envs()
        dt = time.time() - self.t_stop_ref
        print(f'EXIT_PREEMPTED source={self.stop_source} notice_to_exit_s={dt:.2f} phase={self.phase} '
              f'ctr={dict(self.ctr)}', flush=True)
        # Not in metrics.jsonl: rows after the last checkpoint are dropped on resume.
        rec = dict(time=time.time(), source=self.stop_source, notice_to_exit_s=round(dt, 3), phase=self.phase,
                   ctr=dict(self.ctr), gen=self.store.gen)
        with open(os.path.join(self.run_dir, 'preemptions.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        try:
            import wandb
            wandb.log({'preempt/notice_to_exit_s': dt, 'global_step': int(self.ctr['env_steps'] or self.ctr['bc_updates'])})
        except Exception:
            pass
        self.close()
        sys.exit(EXIT_PREEMPTED)

    def close_envs(self):
        try:
            self.eval_envs.close()
        except Exception:
            pass

    def close(self):
        self.close_envs()
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # ---- evaluation and probe -------------------------------------------------------------------
    def evaluate(self, rules):
        """{rule: stats} or None if a preemption notice arrived (eval abandoned; no state was changed)."""
        np_state = np.random.get_state()
        out = {}
        try:
            for rule in rules:
                fn = n1_actions if rule == 'n1' else bon_actions
                agent = self.agent

                def act(observations, rng, fn=fn, agent=agent):
                    if self.stop_requested:
                        raise EvalAborted()
                    return fn(agent, observations, rng)

                self._reseed_eval_envs()  # same initial states for every rule and eval point
                np.random.seed(FLAGS.eval_seed)  # evaluate_parallel draws its JAX key from numpy
                stats, _, _ = evaluate_parallel(agent=Policy(act), envs=self.eval_envs,
                                                num_eval_envs=FLAGS.n_eval_envs,
                                                eval_episodes=FLAGS.eval_episodes, action_dim=self.action_dim)
                episodes = FLAGS.n_eval_envs * FLAGS.eval_episodes
                rate = float(stats['success'])
                out[rule] = dict(success_rate=rate, successes=int(round(rate * episodes)), episodes=episodes,
                                 **{k: float(v) for k, v in stats.items() if k.startswith('episode.')})
        except EvalAborted:
            print(f'EVAL_ABORTED ({self.stop_source})', flush=True)
            return None
        finally:
            np.random.set_state(np_state)
        return out

    def _reseed_eval_envs(self):
        import functools
        self.eval_envs.set_attr('v1_reseed', [functools.partial(_seed_numpy, FLAGS.eval_seed + i)
                                              for i in range(FLAGS.n_eval_envs)])
        self.eval_envs.call('v1_reseed')

    def probe(self):
        parts = [probe_batch(self.agent, self.probe_obs[s:s + 256], self.probe_actions[s:s + 256],
                             self.probe_noises[s:s + 256], self.probe_rand[s:s + 256], jax.random.PRNGKey(0))
                 for s in range(0, len(self.probe_obs), 256)]
        q_data, q_cand, q_rand = (np.concatenate([np.asarray(x[k]) for x in parts]) for k in range(3))
        return dict(tie_rate=float((q_cand.max(1) == q_cand.min(1)).mean()),
                    progress_spearman=float(spearmanr(q_data, -self.probe_steps_left).correlation),
                    q_data_mean=float(q_data.mean()),
                    cand_range_mean=float((q_cand.max(1) - q_cand.min(1)).mean()),
                    data_gt_random=float((q_data > q_rand).mean()))

    def run_eval(self, nominal, step, rules, with_probe):
        """Evaluate + probe, log, then checkpoint. On a preemption notice: checkpoint without it and exit."""
        print(f'EVAL_BEGIN nominal={nominal} step={step} phase={self.phase}', flush=True)
        res = self.evaluate(rules)
        if res is None:
            self.safe_point()
        row = {'nominal_step': nominal, 'phase': self.phase}
        for rule, r in res.items():
            row.update({f'{rule}/{k}': v for k, v in r.items()})
        if with_probe:
            row.update({f'probe/{k}': v for k, v in self.probe().items()})
        self.metrics.log('eval', step, row)
        print('EVAL ' + json.dumps({k: v for k, v in row.items() if 'episode.' not in k}), flush=True)
        self.last_eval = nominal
        self.safe_point(force='eval')

    # ---- env interaction ------------------------------------------------------------------------
    def env_step(self, rule):
        if not self.action_queue:
            self.action_key, key = jax.random.split(self.action_key)
            fn = n1_actions if rule == 'n1' else bon_actions
            chunk = np.array(fn(self.agent, self.ob, key)).reshape(-1, self.action_dim)
            self.action_queue = list(chunk)
        action = self.action_queue.pop(0)
        next_ob, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if is_robomimic_env(FLAGS.env_name):
            reward = reward - 1.0
        if FLAGS.sparse:
            reward = (reward != 0.0) * -1.0
        ep_step = len(self.ep_rows)
        row = dict(
            observations=self.ob, actions=action, rewards=reward, terminals=float(done),
            masks=1.0 - terminated, next_observations=next_ob,
            num_chunks_to_timeout=float(np.ceil((self.max_ep_len - ep_step) / self.H)) if self.max_ep_len > 0 else 0.0,
            num_chunks_to_success=-1.0,
        )
        slot = self.buffers['main'].add_transition(row)
        self.ep_rows.append((slot, row))
        self.ctr['env_steps'] += 1
        self.ctr[f'{self.phase}_env_steps'] += 1
        if done:
            self.end_episode(bool(terminated), info)
        else:
            self.ob = next_ob

    def end_episode(self, success, info):
        L = len(self.ep_rows)
        if success:
            slots = np.array([s for s, _ in self.ep_rows])
            ncs = np.ceil((L - np.arange(L)) / self.H).astype(np.float32)
            self.buffers['main'].set_field('num_chunks_to_success', slots, ncs)
            rows = []
            for (_, r), c in zip(self.ep_rows, ncs):
                r = dict(r)
                r['num_chunks_to_success'] = float(c)
                rows.append(r)
            self.buffers['success'].add_episode(rows)
        self.ctr['episodes'] += 1
        self.ctr[f'{self.phase}_episodes'] += 1
        self.ctr[f'{self.phase}_successes' if success else f'{self.phase}_failures'] += 1
        ep = info.get('episode', {})
        summary = dict(phase=self.phase, success=int(success), length=L, ret=float(ep.get('return', np.nan)))
        self.metrics.log('episode', self.ctr['env_steps'], summary, wandb_too=False)
        for k in ('success', 'length', 'ret'):
            self.env_acc[k].append(summary[k])
        self.ob, self.action_queue, self.ep_rows = None, [], []

    def reset_env(self):
        self.ob, _ = self.env.reset()

    def log_env(self, step):
        row = {f'episode_{k}': float(np.mean(v)) for k, v in self.env_acc.items() if v}
        row['episodes_since_log'] = len(self.env_acc.get('success', []))
        for name, b in self.buffers.items():
            row.update({f'{name}/{k}': v for k, v in b.stats().items()})
        for k in ('warmup_successes', 'warmup_failures', 'joint_successes', 'joint_failures'):
            row[k] = self.ctr[k]
        self.metrics.log('env', step, row)
        self.env_acc = defaultdict(list)

    def log_readout(self, phase, batch, step):
        """ARB greedy-vs-expected readout diagnostics on the last main-replay batch (acarb agents only).

        Uses its own RNG (fold_in of the step), so it does not change the training trajectory."""
        if not hasattr(self.agent, 'readout_diagnostics'):
            return
        rng = jax.random.fold_in(jax.random.PRNGKey(FLAGS.seed + 104729), step)
        diag = jax.device_get(self.agent.readout_diagnostics(batch, rng))
        self.metrics.log('readout', self.ctr['env_steps'], dict(phase=phase, updates=step, **diag))

    # ---- phases ---------------------------------------------------------------------------------
    def progress(self, msg):
        print(f'PROGRESS {msg} t={time.time() - self.t_start:.1f}', flush=True)

    def finish_phase(self):
        nxt = self.phases[self.phases.index(self.phase) + 1]
        done = self.phase
        print(f'PHASE_END {done}', flush=True)
        self.metrics.log('phase_end', self.ctr['env_steps'] or self.ctr['bc_updates'],
                         dict(phase=done, **{k: v for k, v in self.ctr.items()}))
        self.phase = nxt
        self.done_msg = f'PHASE_DONE {done}'
        self.safe_point(force='phase_end')

    def run(self):
        while self.phase != 'done':
            print(f'PHASE_BEGIN {self.phase} ctr={dict(self.ctr)}', flush=True)
            getattr(self, f'phase_{self.phase}')()
            self.finish_phase()
        print(f'RUN_COMPLETE wall_s={time.time() - self.t_start:.1f}', flush=True)
        self.close()

    def phase_bc(self):
        n = FLAGS.bc_steps
        while True:
            u = self.ctr['bc_updates']
            if u > self.last_eval and (u % FLAGS.eval_interval == 0 or u == n):
                self.run_eval(u, u, ['n1'], with_probe=False)
            if u >= n:
                break
            self.safe_point()
            batch = self.demo_ds.sample_sequence(self.batch_size, sequence_length=self.H, discount=FLAGS.discount)
            self.agent, info = bc_update(self.agent, batch)
            self.ctr['bc_updates'] += 1
            u += 1
            if u % FLAGS.log_interval == 0 or u == n:
                self.metrics.log('bc', u, jax.device_get(info))
                self.progress(f'bc updates={u}')
        # final_actor.pkl, written atomically; kept permanently.
        blob = dict(actor_params=actor_params(self.agent), agent_name=self.agent.config['agent_name'],
                    env_name=FLAGS.env_name, horizon_length=self.H, bc_steps=n, seed=FLAGS.seed)
        path = os.path.join(self.run_dir, 'final_actor.pkl')
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(blob, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + '.tmp', path)

    def phase_warmup(self):
        while True:
            if self.ob is None:
                if self.ctr['env_steps'] >= FLAGS.warmup_steps:
                    break
                self.safe_point()
                self.reset_env()
            self.env_step('n1')
            if self.ctr['env_steps'] % FLAGS.log_interval == 0:
                self.log_env(self.ctr['env_steps'])
                self.progress(f'warmup env_steps={self.ctr["env_steps"]}')
        self.log_env(self.ctr['env_steps'])

    def phase_pretrain(self):
        main = self.buffers['main']
        while self.ctr['pretrain_updates'] < FLAGS.pretrain_updates:
            self.safe_point()
            batch = main.sample_sequence(self.batch_size, self.H, FLAGS.discount)
            self.agent, info = critic_update(self.agent, batch)
            self.ctr['pretrain_updates'] += 1
            u = self.ctr['pretrain_updates']
            if u % FLAGS.log_interval == 0 or u == FLAGS.pretrain_updates:
                self.metrics.log('pretrain', self.ctr['env_steps'], dict(updates=u, **jax.device_get(info)))
                self.log_readout('pretrain', batch, u)
                self.progress(f'pretrain updates={u}')

    def phase_pretrain_eval(self):
        res = self.evaluate(['n1', 'bon'])
        if res is None:
            self.safe_point()
        row = {}
        for rule, r in res.items():
            row.update({f'{rule}/{k}': v for k, v in r.items()})
        if os.environ.get('V1_TEST_RAISE_IN_PRETRAIN_EVAL'):  # test hook: where job 10888854 failed (probe)
            raise ValueError('V1_TEST_RAISE_IN_PRETRAIN_EVAL')
        row.update({f'probe/{k}': v for k, v in self.probe().items()})
        self.metrics.log('pretrain_eval', self.ctr['env_steps'], row)
        print('EVAL ' + json.dumps({k: v for k, v in row.items() if 'episode.' not in k}), flush=True)
        self.last_eval = self.ctr['env_steps']

    def phase_joint(self):
        interval, end = FLAGS.eval_interval, FLAGS.online_steps
        main, succ = self.buffers['main'], self.buffers['success']
        info = None
        while self.ctr['env_steps'] < end:
            if self.ob is None:
                due = (self.ctr['env_steps'] // interval) * interval if interval > 0 else 0
                if due > self.last_eval and due > 0:
                    self.run_eval(due, self.ctr['env_steps'], ['n1', 'bon'], with_probe=True)
                self.safe_point()
                self.reset_env()
            self.env_step('bon')
            mb = main.sample_sequence(self.batch_size, self.H, FLAGS.discount)
            sb = succ.sample_sequence(self.batch_size, self.H, FLAGS.discount)
            self.agent, info = joint_update(self.agent, mb, sb)
            self.ctr['joint_updates'] += 1
            s = self.ctr['env_steps']
            if s % FLAGS.log_interval == 0:
                self.metrics.log('joint', s, dict(updates=self.ctr['joint_updates'], **jax.device_get(info)))
                self.log_readout('joint', mb, s)
                self.log_env(s)
                self.progress(f'joint env_steps={s}')
        # Reaching online_steps is a safe point (no further env interaction), even mid-episode.
        self.ob, self.action_queue, self.ep_rows = None, [], []
        if self.last_eval < end:
            self.run_eval(end, end, ['n1', 'bon'], with_probe=True)
        self.log_env(self.ctr['env_steps'])


def main(_):
    assert FLAGS.run_dir, '--run_dir is required'
    store = CheckpointStore(os.path.join(FLAGS.run_dir, 'ckpt'))
    man = store.manifest()
    if man is not None and man['meta'].get('phase') == 'done':
        print(f'RUN_ALREADY_COMPLETE gen={man["gen"]} ctr={man["meta"].get("ctr")}', flush=True)
        return
    if FLAGS.mode == 'online':
        assert FLAGS.init_actor, '--init_actor is required for --mode=online'
    trainer = Trainer()
    try:
        trainer.setup()
        trainer.run()
    except SystemExit:
        raise  # preemption (143), refused config / args (2): unchanged
    except BaseException:
        exit_on_exception(trainer)


EXIT_EXCEPTION = 1
EXCEPTION_CLEANUP_S = 20.0


def exit_on_exception(trainer):
    """Unhandled exception: print it, close the eval envs (bounded) and end the process with code 1.

    A normal interpreter exit can hang for a long time here: it joins the eval env worker processes and
    other non-daemon threads (W&B). os._exit skips that; workers still alive after the bounded close are
    terminated."""
    import traceback
    t0 = time.time()
    traceback.print_exc()
    print(f'EXIT_EXCEPTION phase={trainer.phase_name()} ctr={dict(getattr(trainer, "ctr", {}))}', flush=True)

    def cleanup():
        trainer.close_envs()
        try:
            import wandb
            wandb.finish(exit_code=EXIT_EXCEPTION)
        except Exception:
            pass

    t = threading.Thread(target=cleanup, daemon=True, name='exception-cleanup')
    t.start()
    t.join(EXCEPTION_CLEANUP_S)
    for proc in getattr(getattr(trainer, 'eval_envs', None), 'processes', []):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    print(f'EXIT_EXCEPTION_DONE cleanup_s={time.time() - t0:.2f} cleanup_finished={not t.is_alive()}', flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_EXCEPTION)


if __name__ == '__main__':
    app.run(main)
