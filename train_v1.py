"""v1 protocol: resumable BC preparation and online training with two replay buffers.

  --mode=bc      actor-only BC on the demos for --bc_steps updates; n1 eval every --eval_interval and at
                 the end; writes <run_dir>/final_actor.pkl. With --bc_select_threshold=T >= 0 (actor
                 selection): BC stops at the first evaluation whose n1 success rate is > T; else (budget
                 reached) the evaluated actor with the highest rate, the earliest on ties, is selected.
                 The best candidate so far is kept in <run_dir>/best_actor.pkl (and in the checkpointed
                 state); at the end <run_dir>/selected_actor.pkl and selection.json are written.
  --mode=eval_actor  one n1 evaluation of --init_actor (agent, eval envs and seeding as in --mode=bc);
                 writes <run_dir>/eval_actor.json. No training, no checkpoints; W&B only with --project_name.
  --mode=make_probe_candidates  --init_actor=<pkl> --out=<npz>: the actor's actions for the fixed probe noise
                 at the fixed probe states (as the current-actor probe), shared by runs via --probe_candidates.
  --mode=probe_ckpt  --src_run_dir=<run dir> --run_dir=<out dir> --probe_candidates=<npz> plus the source run's
                 --method / --agent.* flags: restore only the agent from the source's committed checkpoint
                 (read-only), run the current-actor and fixed-candidate probes with the run's readout, write
                 <run_dir>/probe_ckpt.json. No env interaction, no training; W&B only with --project_name.
  --mode=final_eval  --src_run_dir=<finished run> --run_dir=<out dir> plus the source run's --method / --agent.*
                 / --env_name / --horizon_length flags (checked against its args.json and config row): the agent
                 of the committed checkpoint (read-only, phase 'done' only), rules n1 and bon on --final_n_seeds
                 env seeds (--final_seed_base + j) x --eval_episodes (default 4 in this mode) episodes, in chunks
                 of --n_eval_envs; each (rule, chunk) is written to <run_dir>/final_eval_parts/ (a restart skips
                 the parts present; a preemption notice exits 143 after the last complete part), then
                 <run_dir>/final_eval.json. W&B only with --project_name.
  --mode=switch_eval  as final_eval (same source checks, seeds, chunks, RNG, parts / resume, preemption exit),
                 but each setting (--switch_order, k in --switch_ks) uses a hybrid rule: bon_first: decisions
                 d < k (policy calls since the episode's reset) use bon, d >= k n1; n1_first: the reverse. Parts
                 <run_dir>/switch_eval_parts/{order}_{k}_{c}.json, then <run_dir>/switch_eval.json. No W&B.
  --mode=value_trace  as final_eval (same source checks, seeds, chunks, RNG, parts / resume, preemption exit) and
                 the same executed actions, recording on the side the critic's reads of the decision's candidates
                 (bon: the agent's own 32 candidates and selection scores, replicated; n1: 32 shadow candidates
                 from fold_in(rng, 1) read with fold_in(rng, 2), the executed chunk read with fold_in(rng, 3)) and
                 --trace_repeats repeated reads (fold_in(rng, 100 + r)). Parts <run_dir>/value_trace_parts/
                 {rule}_{c}.npz + .json; then the episodes are checked against --ref_final_eval, the
                 --probe_candidates panel is read (probe_trace.npz) and value_trace_summary.npz +
                 value_trace.json are written (vd1). No W&B.
  --mode=episode_stages  as final_eval (same source checks and loading): per rule (n1, bon) and env seed
                 final_seed_base + j, j < --stage_n_seeds, ONE episode in a single env (initial state as
                 final_eval's first episode of that seed) with per-step task-stage codes (tool hang: 0 none,
                 1 frame assembled, 2 frame assembled + tool on frame; other tasks: 2 = success). Parts
                 <run_dir>/episode_stages_parts/{rule}_{j}.json, then <run_dir>/episode_stages.json; videos of
                 j < --video_n_seeds in <run_dir>/videos/ (geo8b). No W&B.
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
import hashlib
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
from agents.acarb import geo_config_errors
from envs.env_utils import make_env_and_datasets, make_vectorized_eval_envs
from envs.robomimic_utils import get_max_episode_length, is_robomimic_env
from evaluation import evaluate_parallel
from log_utils import setup_wandb, get_flag_dict
from utils.ckpt_v1 import CheckpointStore
from utils.datasets import Dataset
from utils.replay_v1 import FIELDS, SeqReplay
from visualization.critic_probe import (ensemble_score, make_candidates, probe_batch, probe_batch_scores, probe_fixed,
                                       probe_states, tie_stats, uses_integer_score)
from visualization.eval_checkpoint import Policy

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'online', ['bc', 'online', 'eval_actor', 'make_probe_candidates', 'probe_ckpt',
                                      'final_eval', 'arb_introspect', 'diag_anchors', 'diag_mc', 'diag_score',
                                      'diag_select', 'switch_eval', 'episode_stages', 'value_trace'],
                  'bc: shared BC preparation run; online: one method run; eval_actor: one n1 eval of --init_actor; '
                  'make_probe_candidates: fixed probe candidates of --init_actor to --out; probe_ckpt: probe the '
                  'committed checkpoint of --src_run_dir (read-only); final_eval: n1 + bon evaluation of the '
                  'committed checkpoint of a finished --src_run_dir (read-only); arb_introspect: AR-B critic '
                  'introspection of the committed checkpoint of --src_run_dir on --queries (read-only, geo5b); '
                  'diag_*: geo5a diagnostics of a finished --src_run_dir (read-only), see diag_main; '
                  'switch_eval: final_eval with bon for the first k decisions then n1 (or the reverse, geo8a); '
                  'episode_stages: per-step task stages + videos of single n1 / bon episodes (geo8b); '
                  'value_trace: final_eval recording the critic\'s candidate reads per decision (vd1).')
flags.DEFINE_enum('method', None, list(V1_METHODS), 'v1 method preset (agents.v1_config); required online.')
flags.DEFINE_string('run_dir', None, 'Run directory (checkpoints, metrics log, final_actor.pkl).')
flags.DEFINE_string('init_actor', None, 'online / eval_actor: path to final_actor.pkl or selected_actor.pkl '
                    'from the bc run.')
flags.DEFINE_string('env_name', 'square-mh-low_dim', 'Environment (dataset) name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('horizon_length', 5, 'Action chunk length H.')
flags.DEFINE_bool('sparse', True, 'Sparse reward.')
flags.DEFINE_float('discount', 0.99, 'Discount used by sample_sequence for cumulative rewards.')

flags.DEFINE_integer('bc_steps', 1000000, 'bc: actor BC updates.')
flags.DEFINE_float('bc_select_threshold', -1.0, 'bc: stop at the first n1 evaluation with success rate > this and '
                   'select that actor, else the best evaluated one (earliest on ties). < 0: disabled.')
flags.DEFINE_integer('warmup_steps', 50000, 'online: warm-up env steps (n1 actions, no updates).')
flags.DEFINE_integer('pretrain_updates', 50000, 'online: critic-only updates after warm-up.')
flags.DEFINE_integer('online_steps', 1000000, 'online: total online env steps, warm-up included.')
flags.DEFINE_integer('buffer_cap', 2000000, 'Capacity of each replay buffer (demos included).')

flags.DEFINE_integer('eval_interval', 100000, 'Eval every this many updates (bc) / online env steps (joint).')
flags.DEFINE_integer('n_eval_envs', 32, 'Parallel eval envs.')
flags.DEFINE_integer('eval_episodes', 2, 'Eval episodes per env and rule.')
flags.DEFINE_integer('eval_seed', 1000, 'Eval env i is reseeded with eval_seed + i before every rule.')
flags.DEFINE_integer('probe_states', 1024, 'Fixed demo states for the critic probe.')
flags.DEFINE_string('probe_candidates', None, 'npz from --mode=make_probe_candidates: fixed candidate actions at the '
                    'probe states, scored at every probe (probe_fixed/*, probe_stab/*). None: off.')
flags.DEFINE_enum('replay_rng', 'global', ['global', 'separate'], 'Replay sampling RNG: numpy global state (global) '
                  'or its own np.random.Generator(PCG64(seed + 1_000_003)), checkpointed (separate).')
flags.DEFINE_string('src_run_dir', None, 'probe_ckpt / final_eval: run directory whose committed checkpoint is '
                    'probed / evaluated (read-only).')
flags.DEFINE_integer('final_seed_base', 100000, 'final_eval: env seed j is final_seed_base + j.')
flags.DEFINE_integer('final_n_seeds', 256, 'final_eval: env seeds (a multiple of --n_eval_envs).')
flags.DEFINE_enum('switch_order', 'bon_first', ['bon_first', 'n1_first'], 'switch_eval: bon_first: bon for '
                  'decisions d < k then n1; n1_first: n1 for d < k then bon.')
flags.DEFINE_list('switch_ks', ['0'], 'switch_eval: comma list of switch decisions k (one setting each).')
flags.DEFINE_integer('stage_n_seeds', 16, 'episode_stages: env seeds final_seed_base + j, j < this.')
flags.DEFINE_integer('video_n_seeds', 2, 'episode_stages: write videos for the first this many seeds.')
flags.DEFINE_integer('video_hw', 160, 'episode_stages: video frame height and width.')
flags.DEFINE_integer('video_stride', 4, 'episode_stages: env steps per video frame.')
flags.DEFINE_integer('trace_repeats', 4, 'value_trace: repeated critic reads per decision (keys fold_in(rng, 100 + r)).')
flags.DEFINE_string('ref_final_eval', None, 'value_trace: the source\'s final_eval.json (episodes must reproduce it).')
flags.DEFINE_string('out', None, 'make_probe_candidates: output .npz path.')
flags.DEFINE_string('queries', None, 'arb_introspect: query npz (utils/arb_introspect.py); diag_mc / diag_score: '
                    'query file (anchors.pkl or select_queries.pkl); diag_select: the common-panel anchors.pkl.')
flags.DEFINE_string('candidates', None, 'arb_introspect: optional candidate-set npz (observations, candidates).')
flags.DEFINE_integer('introspect_seed', 0, 'arb_introspect: base PRNG seed of the readout noise.')
flags.DEFINE_integer('introspect_chunk', 256, 'arb_introspect: queries per resumable part.')
flags.DEFINE_list('introspect_components', ['trace', 'forced', 'ablation', 'candidates', 'labelfit'],
                  'arb_introspect: components to run.')
flags.DEFINE_integer('log_interval', 5000, 'Log interval (updates / env steps).')
flags.DEFINE_float('ckpt_minutes', 20.0, 'Periodic checkpoint interval (minutes) at safe points.')
flags.DEFINE_float('preempt_budget_s', 60.0, 'After a preemption notice: if no safe point has been reached '
                   'this many seconds after it, exit without a new checkpoint (resume from the last one).')

flags.DEFINE_integer('diag_seed', 0, 'diag_anchors: actor RNG of the diagnostic replays.')
flags.DEFINE_enum('diag_rule', 'n1', ['n1', 'bo32'], 'diag_anchors: action rule of the diagnostic replays.')
flags.DEFINE_integer('diag_n_seeds', 32, 'diag_anchors: env seeds final_seed_base + j, j < this.')
flags.DEFINE_integer('diag_episodes', 4, 'diag_anchors: episodes per env seed (as final_eval).')
flags.DEFINE_integer('anchors', 32, 'diag_anchors: anchors selected by the stratified rule.')
flags.DEFINE_string('policy_ckpt', None, 'diag_anchors: actor pkl (selected BC actor) instead of --src_run_dir: '
                    'the common-state panel (n1, no critic).')
flags.DEFINE_integer('mc_k', 32, 'diag_mc: continuations per query.')
flags.DEFINE_integer('mc_seed', 0, 'diag_mc: RNG of the continuations.')
flags.DEFINE_integer('mc_t_extra', 757, 'diag_mc (QC contract): steps summed past the step limit (100 * 0.99^757 '
                     '= 0.0497 return units tail bound).')
flags.DEFINE_integer('mc_part_size', 8, 'diag_mc: queries per output part.')
flags.DEFINE_enum('continuation', 'n1', ['n1', 'bo32'], 'diag_mc: continuation policy (bo32: labelled option).')
flags.DEFINE_enum('diag_contract', 'auto', ['auto', 'bsc', 'qc'], 'diag_mc: value contract (auto: the critic\'s).')
flags.DEFINE_integer('sel_seed', 0, 'diag_select: candidate noise / uniform-index / subset RNG.')
flags.DEFINE_integer('mc_candidates', 0, 'diag_select: also this many further candidates (fixed subset incl. the '
                     'chosen one) as queries.')
flags.DEFINE_integer('score_keys', 5, 'diag_score: extra readout keys (variability) for sampled readouts.')

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
# Flags added after runs were started: left out of args.json at their default (older run dirs still resume).
LATER_FLAG_DEFAULTS = {'bc_select_threshold': -1.0, 'probe_candidates': None, 'replay_rng': 'global',
                       'src_run_dir': None, 'out': None, 'final_seed_base': 100000, 'final_n_seeds': 256,
                       'queries': None, 'candidates': None, 'introspect_seed': 0, 'introspect_chunk': 256,
                       'introspect_components': ['trace', 'forced', 'ablation', 'candidates', 'labelfit'],
                       'diag_seed': 0, 'diag_rule': 'n1', 'diag_n_seeds': 32, 'diag_episodes': 4, 'anchors': 32,
                       'policy_ckpt': None, 'mc_k': 32, 'mc_seed': 0, 'mc_t_extra': 757,
                       'mc_part_size': 8, 'continuation': 'n1', 'diag_contract': 'auto', 'sel_seed': 0,
                       'mc_candidates': 0, 'score_keys': 5, 'switch_order': 'bon_first', 'switch_ks': ['0'],
                       'stage_n_seeds': 16, 'video_n_seeds': 2, 'video_hw': 160, 'video_stride': 4}
# Agent config keys added after runs were started: dropped from args.json / the config row at their default.
LATER_AGENT_DEFAULTS = {'arb_readout_temp': 1.0, 'arb_label': 'hard', 'arb_vocab': 'ordinary',
                        'arb_rollin': 'canonical'}
# The same, per agent (the key predates these runs in other agents, where it is kept).
LATER_AGENT_KEY_DEFAULTS = {'acfql': {'critic_target_bon': True}}
REPLAY_SEED_OFFSET = 1_000_003
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
    if config['agent_name'] == 'acarb':
        readout, label = config['arb_readout'], config.get('arb_label', 'hard')
        if readout == 'sample_mean' and config['q_agg'] != 'mean':
            errors.append(f"arb_readout=sample_mean needs q_agg=mean (got {config['q_agg']})")
        if label != 'hard' and readout != 'sample_mean':
            errors.append(f"arb_label={label} needs arb_readout=sample_mean (got {readout})")
        errors.extend(geo_config_errors(config))
    if errors:
        print('CONFIG_REFUSED ' + '; '.join(errors), flush=True)
        sys.exit(2)


def target_family(config):
    """W&B target family of acarb runs: 'ARB-QC-better' (arb_value='qc') or 'ARB-BSC'; None for other agents."""
    if config['agent_name'] != 'acarb':
        return None
    return 'ARB-QC-better' if config['arb_value'] == 'qc' else 'ARB-BSC'


def resolved_qc_args(config):
    """args.json entry of sampled-readout ARB-QC runs (geo3c): the resolved value family and discount, so a resume
    with a different --method / --agent.arb_value / --agent.discount is refused. None otherwise (args.json of
    every other run unchanged)."""
    if config['agent_name'] != 'acarb' or config['arb_value'] != 'qc' or config['arb_readout'] != 'sample_mean':
        return None
    return dict(arb_value=config['arb_value'], discount=config['discount'], target_family=target_family(config))


def strip_later_agent_defaults(agent_dict):
    """Agent config dict without the LATER_AGENT_DEFAULTS (and its agent's LATER_AGENT_KEY_DEFAULTS) keys that are
    at their default."""
    later = dict(LATER_AGENT_DEFAULTS, **LATER_AGENT_KEY_DEFAULTS.get(agent_dict.get('agent_name'), {}))
    return {k: v for k, v in agent_dict.items() if not (k in later and v == later[k])}


def actor_params(agent):
    return {k: jax.device_get(v) for k, v in agent.network.params.items() if 'critic' not in k}


def load_actor(agent, path, H):
    """agent with its actor modules replaced by those of a BC actor blob (final / best / selected_actor.pkl).

    The BC run may have used any agent: copy this agent's actor modules (same networks in every agent);
    modules this agent lacks (acbsc's unused one-step flow for acarb) are ignored. Returns (agent, info)."""
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    assert blob['horizon_length'] == H, (blob['horizon_length'], H)
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
    info = dict(path=path, agent_name=blob['agent_name'], loaded=need, ignored=ignored,
                **{k: blob[k] for k in ('bc_steps', 'seed', 'source', 'selected_update', 'selection_rule')
                   if k in blob})
    return agent, info


def file_sha256(path):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_probe_candidates(path, idx, shape):
    """(candidate actions, info) from a make_probe_candidates npz, checked against the run's probe states."""
    with np.load(path) as z:
        cands, cidx, actions_sha = z['actions'], z['indices'], str(z['actions_sha256'])
    assert np.array_equal(cidx, idx), f"{path}: probe state indices differ from this run's"
    assert cands.shape == tuple(shape), (cands.shape, shape)
    assert hashlib.sha256(np.ascontiguousarray(cands).tobytes()).hexdigest() == actions_sha, f'{path}: corrupt'
    return cands, dict(path=os.path.abspath(path), sha256=file_sha256(path), actions_sha256=actions_sha)


def write_atomic(path, data):
    with open(path + '.tmp', 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + '.tmp', path)


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
        if self.run_dir:  # make_probe_candidates: none
            os.makedirs(self.run_dir, exist_ok=True)
        self.mode = FLAGS.mode
        self.bc_select = None  # actor selection state (bc with --bc_select_threshold >= 0; set in setup)
        self.store = None
        if self.mode in ('bc', 'online'):  # eval_actor / probe tools: no checkpoints
            self.store = CheckpointStore(os.path.join(self.run_dir, 'ckpt'))
            self.store.test_delay = float(os.environ.get('V1_TEST_CKPT_DELAY', 0))  # e2e test: widen the write
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
                  f'phase={self.phase_name()} committed_gen={getattr(self.store, "gen", None)}', flush=True)
            try:
                with open(os.path.join(self.run_dir, 'preemptions.jsonl'), 'a') as f:
                    f.write(json.dumps(dict(time=time.time(), source=self.stop_source, unsaved=True,
                                            notice_to_exit_s=round(dt, 3),
                                            gen=getattr(self.store, 'gen', None))) + '\n')
                for proc in getattr(getattr(self, 'eval_envs', None), 'processes', []):
                    proc.terminate()
            finally:
                sys.stdout.flush()
                os._exit(EXIT_PREEMPTED)

    def phase_name(self):
        return getattr(self, 'phase', None)

    # ---- setup ----------------------------------------------------------------------------------
    def check_args(self, config=None):
        """Refuse to resume a run directory with different arguments (config: the resolved agent config)."""
        flag_dict = {k: v for k, v in get_flag_dict().items() if k not in RESTART_FREE_FLAGS
                     and not (k in LATER_FLAG_DEFAULTS and v == LATER_FLAG_DEFAULTS[k])}
        if isinstance(flag_dict.get('agent'), dict):
            flag_dict['agent'] = strip_later_agent_defaults(flag_dict['agent'])
        resolved = resolved_qc_args(config) if config is not None else None
        if resolved is not None:
            flag_dict['resolved_agent'] = resolved
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
        config = resolve_agent_config(FLAGS.method, FLAGS.agent, sys.argv)
        if self.mode != 'eval_actor':
            self.check_args(config)
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
        if self.mode in ('online', 'eval_actor'):
            agent, self.init_actor_info = load_actor(agent, FLAGS.init_actor, self.H)
        self.agent = split_optimizer(agent)  # fresh: both step counts 0

        # Eval envs before seeding/restoring: AsyncVectorEnv builds a dummy env in this process, and
        # robomimic's make_env reseeds numpy's global RNG. Keep the global state untouched regardless.
        np_state = np.random.get_state()
        self.eval_envs = make_vectorized_eval_envs(FLAGS.env_name, FLAGS.n_eval_envs, seed=FLAGS.eval_seed)
        np.random.set_state(np_state)
        if self.mode == 'eval_actor':
            return

        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        self.action_key = jax.random.PRNGKey(FLAGS.seed + 7919)

        # Replay sampling RNG: None = numpy's global state; separate: one Generator for both buffers.
        self.replay_rng = None
        if FLAGS.replay_rng == 'separate':
            self.replay_rng = np.random.Generator(np.random.PCG64(FLAGS.seed + REPLAY_SEED_OFFSET))
        if self.mode == 'bc':
            self.demo_ds = Dataset.create(**{k: self.demos[k] for k in FIELDS})
        else:
            self.buffers = dict(
                main=SeqReplay(self.demos, FLAGS.buffer_cap, self.H, episodic=False, rng=self.replay_rng),
                success=SeqReplay(self.demos, FLAGS.buffer_cap, self.H, episodic=True, rng=self.replay_rng),
            )

        self.setup_probe(config)

        self.phase = self.phases[0]
        self.ctr = defaultdict(int)
        self.last_eval = 0          # nominal step of the last completed evaluation
        self.env_acc = defaultdict(list)
        self.metrics_offset = 0
        self.wandb_id = None
        # Actor selection (bc): every n1 evaluation [update, successes, episodes, rate]; the best candidate
        # (also its pickled blob, so best_actor.pkl can be rewritten on resume); stopped: rate > threshold.
        self.bc_select = None
        if self.mode == 'bc' and FLAGS.bc_select_threshold >= 0:
            self.bc_select = dict(threshold=float(FLAGS.bc_select_threshold), evals=[], best=None,
                                  best_blob=None, stopped=False)
        self.restore()

        self.metrics = MetricsLog(os.path.join(self.run_dir, 'metrics.jsonl'), self.metrics_offset)
        self.setup_wandb()
        run_config = dict(method=FLAGS.method, agent_name=config['agent_name'],
                          agent=strip_later_agent_defaults(config.to_dict()), init_actor=self.init_actor_info)
        if self.probe_cands is not None:
            run_config['probe_candidates'] = self.probe_cands_info
        if FLAGS.replay_rng != 'global':
            run_config['replay_rng'] = FLAGS.replay_rng
        import wandb
        wandb.config.update(dict(v1=run_config), allow_val_change=True)
        family = target_family(config)
        if family is not None:  # W&B only: the metrics config row is unchanged
            wandb.config.update(dict(target_family=family), allow_val_change=True)
            if wandb.run is not None and not getattr(wandb.run, 'disabled', False):
                wandb.run.tags = tuple(dict.fromkeys((*(wandb.run.tags or ()), family)))
        if self.metrics_offset == 0:  # first row of the run's log (rewritten if no checkpoint follows it)
            self.metrics.log('config', 0, run_config, wandb_too=False)
        print(f'CONFIG method={FLAGS.method} agent={config["agent_name"]} '
              f'actor_type={config["actor_type"]} actor_num_samples={config["actor_num_samples"]}', flush=True)
        self.t_last_ckpt = time.time()
        self.ob, self.action_queue, self.ep_rows = None, [], []

    def setup_probe(self, config):
        """Probe states and noise, fixed for the whole run (visualization/critic_probe.py); with
        --probe_candidates the fixed candidate actions at those states."""
        idx, self.probe_steps_left = probe_states(self.demos, self.H, FLAGS.probe_states, 0)
        self.probe_idx = idx
        self.probe_obs = self.demos['observations'][idx]
        self.probe_actions = np.stack([self.demos['actions'][i:i + self.H].reshape(-1) for i in idx])
        rs = np.random.RandomState(1)
        self.probe_noises = rs.standard_normal(
            (len(idx), config['actor_num_samples'], self.probe_actions.shape[-1])).astype(np.float32)
        self.probe_rand = rs.uniform(-1, 1, (len(idx), self.probe_actions.shape[-1])).astype(np.float32)
        self.probe_cands, self.probe_cands_info = None, None
        if FLAGS.probe_candidates and self.mode != 'make_probe_candidates':
            self.probe_cands, self.probe_cands_info = load_probe_candidates(FLAGS.probe_candidates, idx,
                                                                            self.probe_noises.shape)

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
        if getattr(self, 'replay_rng', None) is not None:  # --replay_rng=separate only (default format unchanged)
            state['replay_rng'] = self.replay_rng.bit_generator.state
        if self.bc_select is not None:
            state['bc_select'] = self.bc_select
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
        if getattr(self, 'replay_rng', None) is not None:
            assert 'replay_rng' in state, 'checkpoint has no replay Generator state'
            self.replay_rng.bit_generator.state = state['replay_rng']
        if self.bc_select is not None:
            self.bc_select = state['bc_select']
            self.restore_best_actor()
        print(f'RESUME gen={man["gen"]} phase={self.phase} ctr={dict(self.ctr)}', flush=True)

    def restore_best_actor(self):
        """best_actor.pkl = the checkpointed best candidate. A process killed after writing a new candidate
        and before the next checkpoint left a newer one on disk: put the committed one back (the resumed
        run redoes that evaluation and writes the newer one again)."""
        best = self.bc_select['best']
        if best is None:
            return  # no committed candidate; the first evaluation overwrites any file left on disk
        path = os.path.join(self.run_dir, 'best_actor.pkl')
        sha = file_sha256(path) if os.path.exists(path) else None
        if sha != best['sha256']:
            write_atomic(path, self.bc_select['best_blob'])
            print(f'BEST_ACTOR_RESTORED update={best["update"]} (file on disk: {sha})', flush=True)

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
        """Current-actor probe ({key: value}, logged under probe/). Ties on the final ensemble score (integer S
        for acarb sample_mean, the aggregated Q otherwise). Uses only the fixed probe key PRNGKey(0)."""
        integer = uses_integer_score(self.agent)
        fn = probe_batch_scores if integer else probe_batch
        parts = [fn(self.agent, self.probe_obs[s:s + 256], self.probe_actions[s:s + 256],
                    self.probe_noises[s:s + 256], self.probe_rand[s:s + 256], jax.random.PRNGKey(0))
                 for s in range(0, len(self.probe_obs), 256)]
        q_data, q_cand, q_rand = (np.concatenate([np.asarray(x[k]) for x in parts]) for k in range(3))
        s_cand = np.concatenate([np.asarray(x[3]) for x in parts]) if integer else q_cand
        tie_all, tie_max = tie_stats(s_cand)
        return dict(tie_rate=tie_all,
                    tie_max=tie_max,
                    progress_spearman=float(spearmanr(q_data, -self.probe_steps_left).correlation),
                    q_data_mean=float(q_data.mean()),
                    cand_range_mean=float((q_cand.max(1) - q_cand.min(1)).mean()),
                    data_gt_random=float((q_data > q_rand).mean()))

    def probe_row(self):
        """probe/* and, with --probe_candidates, probe_fixed/* and probe_stab/*."""
        row = {f'probe/{k}': v for k, v in self.probe().items()}
        if self.probe_cands is not None:
            row.update(probe_fixed(self.agent, self.probe_obs, self.probe_cands))
        return row

    def run_eval(self, nominal, step, rules, with_probe):
        """Evaluate + probe, log, then checkpoint. On a preemption notice: checkpoint without it and exit."""
        print(f'EVAL_BEGIN nominal={nominal} step={step} phase={self.phase}', flush=True)
        res = self.evaluate(rules)
        if res is None:
            self.safe_point()
        if self.bc_select is not None:
            self.bc_select_update(nominal, res['n1'])
        row = {'nominal_step': nominal, 'phase': self.phase}
        for rule, r in res.items():
            row.update({f'{rule}/{k}': v for k, v in r.items()})
        if with_probe:
            row.update(self.probe_row())
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
            if u >= n or (self.bc_select is not None and self.bc_select['stopped']):
                break
            self.safe_point()
            batch = self.demo_ds.sample_sequence(self.batch_size, sequence_length=self.H, discount=FLAGS.discount)
            self.agent, info = bc_update(self.agent, batch)
            self.ctr['bc_updates'] += 1
            u += 1
            if u % FLAGS.log_interval == 0 or u == n:
                self.metrics.log('bc', u, jax.device_get(info))
                self.progress(f'bc updates={u}')
        if self.bc_select is not None:
            self.write_selection()
        # final_actor.pkl (the actor at the last update), written atomically; kept permanently.
        write_atomic(os.path.join(self.run_dir, 'final_actor.pkl'), pickle.dumps(self.actor_blob()))

    def actor_blob(self, **extra):
        return dict(actor_params=actor_params(self.agent), agent_name=self.agent.config['agent_name'],
                    env_name=FLAGS.env_name, horizon_length=self.H, bc_steps=FLAGS.bc_steps, seed=FLAGS.seed,
                    **extra)

    def bc_select_update(self, u, n1):
        """After the n1 evaluation at update u (before its checkpoint): record it; a strictly better rate
        than the best so far (ties keep the earlier one) makes this actor the best candidate; a rate above
        the threshold stops BC here."""
        sel = self.bc_select
        rate, successes, episodes = n1['success_rate'], n1['successes'], n1['episodes']
        rates = os.environ.get('V1_TEST_BC_EVAL_RATES')  # test hook: rates seen here (the eval row is real)
        if rates and len(sel['evals']) < len(rates.split(',')):
            rate = float(rates.split(',')[len(sel['evals'])])
            successes = int(round(rate * episodes))
        sel['evals'].append([u, successes, episodes, rate])
        if sel['best'] is None or rate > sel['best']['rate']:
            data = pickle.dumps(self.actor_blob(bc_updates=u, n1_successes=successes, n1_episodes=episodes,
                                                n1_success_rate=rate))
            write_atomic(os.path.join(self.run_dir, 'best_actor.pkl'), data)
            import hashlib
            sel['best'] = dict(update=u, successes=successes, episodes=episodes, rate=rate,
                               sha256=hashlib.sha256(data).hexdigest())
            sel['best_blob'] = data
            print(f'BEST_ACTOR_WRITTEN update={u} successes={successes}/{episodes} rate={rate}', flush=True)
            time.sleep(float(os.environ.get('V1_TEST_BEST_ACTOR_DELAY', 0)))  # test hook: widen until the ckpt
        if rate > sel['threshold']:
            sel['stopped'] = True
            print(f'BC_SELECT_STOP update={u} rate={rate} > threshold={sel["threshold"]}', flush=True)

    def write_selection(self):
        """selected_actor.pkl (best candidate's blob + selection fields), selection.json, a bc_selection row.
        Everything is derived from the checkpointed selection state (a resume rewrites the same bytes)."""
        sel = self.bc_select
        best = sel['best']
        assert best is not None, 'actor selection: no n1 evaluation was run (bc_steps=0?)'
        path = os.path.join(self.run_dir, 'best_actor.pkl')
        assert file_sha256(path) == best['sha256'], f'{path} differs from the checkpointed best candidate'
        rule = 'first_above_threshold' if sel['stopped'] else 'best_within_budget'
        fields = dict(selection_rule=rule, threshold=sel['threshold'], selected_update=best['update'])
        blob = pickle.loads(sel['best_blob'])
        blob.update(fields)
        write_atomic(os.path.join(self.run_dir, 'selected_actor.pkl'), pickle.dumps(blob))
        info = dict(**fields, n1_successes=best['successes'], n1_episodes=best['episodes'],
                    n1_success_rate=best['rate'], best_actor_sha256=best['sha256'],
                    bc_updates=self.ctr['bc_updates'], bc_steps=FLAGS.bc_steps, eval_interval=FLAGS.eval_interval)
        evals = [dict(update=u, successes=s, episodes=e, success_rate=r) for u, s, e, r in sel['evals']]
        write_atomic(os.path.join(self.run_dir, 'selection.json'),
                     (json.dumps(dict(info, evaluations=evals), indent=1) + '\n').encode())
        self.metrics.log('bc_selection', self.ctr['bc_updates'], dict(info, n_evaluations=len(evals)))
        print('BC_SELECTION ' + json.dumps(info), flush=True)

    def eval_actor(self):
        """--mode=eval_actor: one n1 evaluation of --init_actor, as phase_bc evaluates; eval_actor.json."""
        import socket
        res = self.evaluate(['n1'])
        if res is None:  # preemption notice: nothing written
            self.close_envs()
            sys.exit(EXIT_PREEMPTED)
        r = res['n1']
        out = dict(successes=r['successes'], episodes=r['episodes'], success_rate=r['success_rate'],
                   per_episode_success=None,  # evaluate_parallel returns only the means
                   node=socket.gethostname(), init_actor=self.init_actor_info, env_name=FLAGS.env_name,
                   eval_seed=FLAGS.eval_seed, n_eval_envs=FLAGS.n_eval_envs, eval_episodes=FLAGS.eval_episodes,
                   **{k: v for k, v in r.items() if k.startswith('episode.')})
        write_atomic(os.path.join(self.run_dir, 'eval_actor.json'), (json.dumps(out, indent=1) + '\n').encode())
        print('EVAL_ACTOR ' + json.dumps({k: v for k, v in out.items() if k != 'init_actor'}), flush=True)
        if FLAGS['project_name'].present:
            import wandb
            import secrets
            self.wandb_id = secrets.token_hex(6)  # its own W&B run (not the bc run's wandb_id)
            self.setup_wandb()
            wandb.config.update(dict(v1=dict(mode='eval_actor', init_actor=self.init_actor_info)),
                                allow_val_change=True)
            wandb.log({f'eval_actor/{k}': v for k, v in out.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)})
        self.close()

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
        row.update(self.probe_row())
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


def build_probe_trainer():
    """Trainer with config, demos, agent (actor from --init_actor if given) and probe states only: no eval
    envs, no replay, no checkpoint store, no W&B."""
    tr = build_agent_trainer()
    tr.setup_probe(tr.config)
    return tr


def build_agent_trainer():
    """build_probe_trainer without the probe states."""
    tr = Trainer()
    # make_probe_candidates and the diag_anchors common panel (--policy_ckpt) need only the BC flow actor (same
    # network in every agent): bsc's agent by default.
    actor_only = tr.mode == 'make_probe_candidates' or (tr.mode == 'diag_anchors' and FLAGS.policy_ckpt)
    method = FLAGS.method or ('bsc' if actor_only else None)
    config = resolve_agent_config(method, FLAGS.agent, sys.argv)
    config['horizon_length'] = FLAGS.horizon_length
    check_agent_config(tr.mode, config)
    tr.config, tr.H = config, FLAGS.horizon_length
    env, eval_env, ds, _ = make_env_and_datasets(FLAGS.env_name)
    for e in (env, eval_env):
        if hasattr(e, 'close'):
            e.close()
    tr.demos = process_demos(ds, FLAGS.env_name, tr.H, FLAGS.sparse)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, tr.demos['observations'][0], tr.demos['actions'][0], config)
    tr.init_actor_info = None
    if FLAGS.init_actor:
        agent, tr.init_actor_info = load_actor(agent, FLAGS.init_actor, tr.H)
    tr.agent = split_optimizer(agent)
    return tr


def make_probe_candidates():
    """--mode=make_probe_candidates: the --init_actor's actions for the fixed probe noise (RandomState(1)) at
    the fixed probe states, written to --out (actions, indices, observations, actions_sha256)."""
    assert FLAGS.init_actor and FLAGS.out, '--init_actor and --out are required'
    tr = build_probe_trainer()
    cands = make_candidates(tr.agent, tr.probe_obs, tr.probe_noises).astype(np.float32)
    sha = hashlib.sha256(np.ascontiguousarray(cands).tobytes()).hexdigest()
    import io
    buf = io.BytesIO()
    np.savez(buf, actions=cands, indices=np.asarray(tr.probe_idx), observations=tr.probe_obs.astype(np.float32),
             actions_sha256=np.asarray(sha), init_actor_sha256=np.asarray(file_sha256(FLAGS.init_actor)),
             env_name=np.asarray(FLAGS.env_name), horizon_length=np.asarray(tr.H))
    write_atomic(FLAGS.out, buf.getvalue())
    print(f'PROBE_CANDIDATES out={FLAGS.out} shape={cands.shape} actions_sha256={sha} '
          f'file_sha256={file_sha256(FLAGS.out)}', flush=True)


def read_committed_state(src_run_dir):
    """(state, manifest) of the committed checkpoint of a run dir, read-only: only opens files for reading
    (no CheckpointStore: its load() cleans up uncommitted generations)."""
    root = os.path.join(src_run_dir, 'ckpt')
    with open(os.path.join(root, 'LATEST')) as f:
        name = f.read().strip()
    gdir = os.path.join(root, name)
    with open(os.path.join(gdir, 'manifest.json')) as f:
        man = json.load(f)
    with open(os.path.join(gdir, 'state.pkl'), 'rb') as f:
        data = f.read()
    info = man['files']['state.pkl']
    assert len(data) == info['size'] and hashlib.sha256(data).hexdigest() == info['sha256'], f'{gdir} is corrupt'
    return pickle.loads(data), man


def probe_ckpt():
    """--mode=probe_ckpt: both probes on the agent of --src_run_dir's committed checkpoint -> probe_ckpt.json."""
    assert FLAGS.src_run_dir and FLAGS.probe_candidates, '--src_run_dir and --probe_candidates are required'
    src, out_dir = os.path.realpath(FLAGS.src_run_dir), os.path.realpath(FLAGS.run_dir)
    assert src != out_dir and not out_dir.startswith(src + os.sep), '--run_dir must lie outside --src_run_dir'
    state, man = read_committed_state(src)
    tr = build_probe_trainer()
    tr.agent = flax.serialization.from_state_dict(tr.agent, state['agent'])
    row = tr.probe_row()
    ctr = dict(state['ctr'])
    out = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], ctr=ctr,
               step=int(ctr.get('env_steps', 0) or ctr.get('bc_updates', 0)), method=FLAGS.method,
               agent_name=tr.config['agent_name'], agent=strip_later_agent_defaults(tr.config.to_dict()),
               probe_candidates=tr.probe_cands_info, **row)
    write_atomic(os.path.join(FLAGS.run_dir, 'probe_ckpt.json'),
                 (json.dumps(out, indent=1, default=str) + '\n').encode())
    print('PROBE_CKPT ' + json.dumps({k: v for k, v in out.items() if k != 'agent'}, default=str), flush=True)
    if FLAGS['project_name'].present:
        import wandb
        import secrets
        tr.wandb_id = secrets.token_hex(6)
        tr.setup_wandb()
        wandb.config.update(dict(v1=dict(mode='probe_ckpt', src_run_dir=src, gen=man['gen'])), allow_val_change=True)
        wandb.log({**{k: v for k, v in row.items()}, 'global_step': out['step']})
        wandb.finish()


def wilson_ci(successes, n, z=1.959963984540054):
    """Wilson score 95% interval of a binomial proportion."""
    if n == 0:
        return [None, None]
    p = successes / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, mid - half)), float(min(1.0, mid + half))]


def refuse(msg):
    print(f'FINAL_EVAL_REFUSED {msg}', flush=True)
    sys.exit(2)


def source_config_diff(src, src_args, config):
    """{key: (source, this call)} for env_name / horizon_length / method (args.json), the resolved agent config
    (the source's config row; agent keys added later are equal at their default) and the resolved QC entry."""
    diff = {}
    for k in ('env_name', 'horizon_length', 'method'):
        if k in src_args and src_args[k] != FLAGS[k].value:
            diff[k] = (src_args[k], FLAGS[k].value)
    norm = lambda d: json.loads(json.dumps(strip_later_agent_defaults(d), default=str))  # noqa: E731
    mine = norm(config.to_dict())
    row = None
    path = os.path.join(src, 'metrics.jsonl')
    if os.path.exists(path):
        with open(path) as f:
            first = f.readline()
        row = json.loads(first) if first.strip() else None
    if row is not None and row.get('kind') == 'config':
        theirs = norm(row['agent'])
    else:  # no config row: the --agent flags of args.json (before create() filled in the shapes)
        theirs = norm(src_args.get('agent', {}))
        mine = {k: v for k, v in norm(FLAGS.agent.to_dict()).items() if k not in ('horizon_length', 'action_dim',
                                                                                   'ob_dims')}
        theirs = {k: v for k, v in theirs.items() if k not in ('horizon_length', 'action_dim', 'ob_dims')}
    for k in sorted(set(mine) | set(theirs)):
        if mine.get(k, '<absent>') != theirs.get(k, '<absent>'):
            diff[f'agent.{k}'] = (theirs.get(k, '<absent>'), mine.get(k, '<absent>'))
    resolved = json.loads(json.dumps(resolved_qc_args(config)))
    if src_args.get('resolved_agent') != resolved:
        diff['resolved_agent'] = (src_args.get('resolved_agent'), resolved)
    return diff


def final_eval_chunk_seed(c):
    """numpy seed for chunk c (evaluate_parallel draws its JAX key from numpy): the same for both rules."""
    return int(np.random.SeedSequence([FLAGS.final_seed_base, c]).generate_state(1)[0])


def final_eval(runner=None):
    """--mode=final_eval: n1 and bon on the agent of --src_run_dir's committed checkpoint -> final_eval.json.
    runner: run_switch_eval for --mode=switch_eval (same source checks and loading; no W&B)."""
    assert FLAGS.src_run_dir, '--src_run_dir is required'
    src, out_dir = os.path.realpath(FLAGS.src_run_dir), os.path.realpath(FLAGS.run_dir)
    if src == out_dir or out_dir.startswith(src + os.sep):
        refuse(f'--run_dir {out_dir} lies inside --src_run_dir {src}')
    n_seeds, C = FLAGS.final_n_seeds, FLAGS.n_eval_envs
    E = FLAGS.eval_episodes if FLAGS['eval_episodes'].present else 4
    if n_seeds <= 0 or C <= 0 or n_seeds % C:
        refuse(f'--final_n_seeds={n_seeds} is not a positive multiple of --n_eval_envs={C}')
    state, man = read_committed_state(src)
    if state['phase'] != 'done':
        refuse(f'checkpoint gen={man["gen"]} of {src} has phase={state["phase"]} (need done)')
    args_path = os.path.join(src, 'args.json')
    if not os.path.exists(args_path):
        refuse(f'{args_path} not found')
    with open(args_path) as f:
        src_args = json.load(f)

    tr = build_agent_trainer()
    try:
        diff = source_config_diff(src, src_args, tr.config)
        if diff:
            refuse('config differs from the source run (source, this call): ' + json.dumps(diff, default=str))
        tr.agent = flax.serialization.from_state_dict(tr.agent, state['agent'])
        out = (runner or run_final_eval)(tr, src, state, man, n_seeds, C, E)
    except SystemExit:
        raise
    except BaseException:
        exit_on_exception(tr)
    if runner is None and FLAGS['project_name'].present:
        import wandb
        import secrets
        tr.wandb_id = secrets.token_hex(6)  # its own W&B run
        tr.setup_wandb()
        wandb.config.update(dict(v1=dict(mode='final_eval', src_run_dir=src, gen=man['gen'])), allow_val_change=True)
        log = {}
        for rule in ('n1', 'bon'):
            r = out[rule]
            log.update({f'final1024/{rule}_successes': r['successes'], f'final1024/{rule}_success_rate': r['success_rate'],
                        f'final1024/{rule}_ci_lo': r['wilson95'][0], f'final1024/{rule}_ci_hi': r['wilson95'][1]})
        log['final1024/bon_minus_n1_successes'] = out['paired']['bon_minus_n1_successes']
        log['final1024/bon_minus_n1_rate'] = out['bon']['success_rate'] - out['n1']['success_rate']
        wandb.log(log)
        wandb.finish()


def run_final_eval(tr, src, state, man, n_seeds, C, E):
    import functools
    parts_dir = os.path.join(FLAGS.run_dir, 'final_eval_parts')
    os.makedirs(parts_dir, exist_ok=True)
    tr.eval_envs = None  # built before the first part to evaluate (none if every part is on disk)
    action_dim = tr.config['action_dim']
    n_chunks = n_seeds // C
    ident = dict(src_run_dir=src, gen=man['gen'], final_seed_base=FLAGS.final_seed_base, n_eval_envs=C,
                 eval_episodes=E)
    parts, done = {}, 0

    def exit_preempted():
        tr.close_envs()
        dt = time.time() - tr.t_stop_ref
        print(f'EXIT_PREEMPTED source={tr.stop_source} notice_to_exit_s={dt:.2f} phase=final_eval '
              f'parts_done={done}', flush=True)
        sys.exit(EXIT_PREEMPTED)

    for rule in ('n1', 'bon'):
        fn = n1_actions if rule == 'n1' else bon_actions
        for c in range(n_chunks):
            path = os.path.join(parts_dir, f'{rule}_{c}.json')
            seeds = [FLAGS.final_seed_base + c * C + i for i in range(C)]
            if os.path.exists(path):
                with open(path) as f:
                    part = json.load(f)
                if {k: part.get(k) for k in ident} != ident or part['seeds'] != seeds:
                    tr.close_envs()
                    refuse(f'{path} was written with other settings: '
                           + json.dumps({k: part.get(k) for k in ident}))
                print(f'FINAL_EVAL_PART_SKIP rule={rule} chunk={c}', flush=True)
                parts[rule, c] = part
                done += 1
                continue
            if tr.watcher is not None:
                tr.watcher.check()
            if tr.stop_requested:
                exit_preempted()
            if tr.eval_envs is None:
                np_state = np.random.get_state()  # robomimic's make_env reseeds numpy (as in setup)
                tr.eval_envs = make_vectorized_eval_envs(FLAGS.env_name, C, seed=FLAGS.final_seed_base)
                np.random.set_state(np_state)
            t0 = time.time()

            def act(observations, rng, fn=fn):
                if tr.stop_requested:
                    raise EvalAborted()
                return fn(tr.agent, observations, rng)

            tr.eval_envs.set_attr('v1_reseed', [functools.partial(_seed_numpy, s) for s in seeds])
            tr.eval_envs.call('v1_reseed')
            np.random.seed(final_eval_chunk_seed(c))
            try:
                _, _, _, eps = evaluate_parallel(agent=Policy(act), envs=tr.eval_envs, num_eval_envs=C,
                                                 eval_episodes=E, action_dim=action_dim, return_episodes=True)
            except EvalAborted:
                print(f'EVAL_ABORTED ({tr.stop_source}) rule={rule} chunk={c}', flush=True)
                exit_preempted()
            digest = [[hashlib.sha256(np.ascontiguousarray(o, dtype=np.float64).tobytes()).hexdigest()[:16]
                       for o in env_obs] for env_obs in eps['first_obs']]
            part = dict(ident, rule=rule, chunk=c, seeds=seeds, success=eps['success'].astype(int).tolist(),
                        length=eps['length'].tolist(), first_obs_digest=digest, wall_s=time.time() - t0)
            write_atomic(path, (json.dumps(part) + '\n').encode())
            parts[rule, c] = part
            done += 1
            print(f'FINAL_EVAL_PART rule={rule} chunk={c} successes={int(np.sum(part["success"]))}/{C * E} '
                  f'wall_s={part["wall_s"]:.1f}', flush=True)
    tr.close_envs()

    ctr = dict(state['ctr'])
    out = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], mode=state.get('mode'), ctr=ctr,
               step=int(ctr.get('env_steps', 0) or ctr.get('bc_updates', 0)), env_name=FLAGS.env_name,
               horizon_length=FLAGS.horizon_length, method=FLAGS.method, agent_name=tr.config['agent_name'],
               agent=strip_later_agent_defaults(tr.config.to_dict()), final_seed_base=FLAGS.final_seed_base,
               seeds=[FLAGS.final_seed_base + j for j in range(n_seeds)], n_eval_envs=C, eval_episodes=E,
               wall_s=float(sum(p['wall_s'] for p in parts.values())))
    per_seed, digests = {}, {}
    for rule in ('n1', 'bon'):
        S = np.concatenate([np.asarray(parts[rule, c]['success'], dtype=np.int64) for c in range(n_chunks)])
        L = np.concatenate([np.asarray(parts[rule, c]['length'], dtype=np.int64) for c in range(n_chunks)])
        digests[rule] = sum((parts[rule, c]['first_obs_digest'] for c in range(n_chunks)), [])
        k, n = int(S.sum()), int(S.size)
        per_seed[rule] = S.sum(1)
        out[rule] = dict(successes=k, episodes=n, success_rate=k / n, wilson95=wilson_ci(k, n),
                         per_seed_successes=per_seed[rule].tolist(), per_episode_success=S.tolist(),
                         per_episode_length=L.tolist(),
                         mean_success_length=float(L[S == 1].mean()) if k else None)
    d = per_seed['bon'] - per_seed['n1']
    same = np.array(digests['n1']) == np.array(digests['bon'])
    out['paired'] = dict(bon_minus_n1_successes=int(d.sum()), per_seed_diff_mean=float(d.mean()),
                         per_seed_diff_sd=float(d.std(ddof=1)) if len(d) > 1 else None,
                         same_first_obs_seeds=int(same[:, 0].sum()), same_initial_obs_episodes=int(same.sum()))
    write_atomic(os.path.join(FLAGS.run_dir, 'final_eval.json'),
                 (json.dumps(out, indent=1, default=str) + '\n').encode())
    summary = {k: v for k, v in out.items() if k not in ('agent', 'seeds', 'n1', 'bon', 'ctr')}
    for rule in ('n1', 'bon'):
        summary[rule] = {k: v for k, v in out[rule].items() if not k.startswith('per_')}
    print('FINAL_EVAL ' + json.dumps(summary, default=str), flush=True)
    return out


class _AlignedEnvs:
    """evaluate_parallel's envs, tracking the active envs and the steps since reset, so a policy call can check
    that it was asked for chunks of exactly the active envs (all reset together, H steps per chunk)."""

    def __init__(self, envs, n):
        self.envs, self.n = envs, n

    def reset(self):
        self.obs, info = self.envs.reset()
        self.active, self.t = np.ones(self.n, dtype=bool), 0
        return self.obs, info

    def step(self, actions):
        out = self.envs.step(actions)
        self.obs, self.t = out[0], self.t + 1
        self.active &= ~np.logical_or(out[2], out[3])
        return out

    def check_call(self, observations, d, H):
        a = int(self.active.sum())
        ok = (self.t == d * H and a > 0 and np.array_equal(observations[:a], self.obs[self.active])
              and not np.any(observations[a:]))
        if not ok:
            raise RuntimeError(f'switch_eval alignment broken at decision {d}: t={self.t} active={a}')


def run_switch_eval(tr, src, state, man, n_seeds, C, E):
    """switch_eval settings (order, k) on run_final_eval's seeds / chunks / RNG -> switch_eval.json."""
    import functools
    order = FLAGS.switch_order
    ks = [int(k) for k in FLAGS.switch_ks]
    first, then = (bon_actions, n1_actions) if order == 'bon_first' else (n1_actions, bon_actions)
    parts_dir = os.path.join(FLAGS.run_dir, 'switch_eval_parts')
    os.makedirs(parts_dir, exist_ok=True)
    tr.eval_envs = None
    action_dim, H = tr.config['action_dim'], FLAGS.horizon_length
    n_chunks = n_seeds // C
    ident = dict(src_run_dir=src, gen=man['gen'], final_seed_base=FLAGS.final_seed_base, n_eval_envs=C,
                 eval_episodes=E)
    parts, done = {}, 0

    def exit_preempted():
        tr.close_envs()
        dt = time.time() - tr.t_stop_ref
        print(f'EXIT_PREEMPTED source={tr.stop_source} notice_to_exit_s={dt:.2f} phase=switch_eval '
              f'parts_done={done}', flush=True)
        sys.exit(EXIT_PREEMPTED)

    for k in ks:
        for c in range(n_chunks):
            path = os.path.join(parts_dir, f'{order}_{k}_{c}.json')
            seeds = [FLAGS.final_seed_base + c * C + i for i in range(C)]
            if os.path.exists(path):
                with open(path) as f:
                    part = json.load(f)
                got = {x: part.get(x) for x in (*ident, 'order', 'k')}
                if got != dict(ident, order=order, k=k) or part['seeds'] != seeds:
                    tr.close_envs()
                    refuse(f'{path} was written with other settings: ' + json.dumps(got))
                print(f'SWITCH_EVAL_PART_SKIP order={order} k={k} chunk={c}', flush=True)
                parts[k, c] = part
                done += 1
                continue
            if tr.watcher is not None:
                tr.watcher.check()
            if tr.stop_requested:
                exit_preempted()
            if tr.eval_envs is None:
                np_state = np.random.get_state()  # robomimic's make_env reseeds numpy (as in setup)
                tr.eval_envs = make_vectorized_eval_envs(FLAGS.env_name, C, seed=FLAGS.final_seed_base)
                np.random.set_state(np_state)
            t0 = time.time()
            envs = _AlignedEnvs(tr.eval_envs, C)
            calls = dict(d=0, bon=0, n1=0, rules=[])  # rules: per episode, the rule of each decision (b / n)

            def on_reset(calls=calls):
                calls['d'] = 0
                calls['rules'].append('')

            def act(observations, rng, k=k, envs=envs, calls=calls):
                if tr.stop_requested:
                    raise EvalAborted()
                envs.check_call(observations, calls['d'], H)
                fn = first if calls['d'] < k else then
                rule = 'bon' if fn is bon_actions else 'n1'
                calls[rule] += 1
                calls['rules'][-1] += rule[0]
                calls['d'] += 1
                return fn(tr.agent, observations, rng)

            tr.eval_envs.set_attr('v1_reseed', [functools.partial(_seed_numpy, s) for s in seeds])
            tr.eval_envs.call('v1_reseed')
            np.random.seed(final_eval_chunk_seed(c))
            try:
                _, _, _, eps = evaluate_parallel(agent=Policy(act), envs=envs, num_eval_envs=C, eval_episodes=E,
                                                 action_dim=action_dim, return_episodes=True, on_reset=on_reset)
            except EvalAborted:
                print(f'EVAL_ABORTED ({tr.stop_source}) order={order} k={k} chunk={c}', flush=True)
                exit_preempted()
            assert len(calls['rules']) == E, calls['rules']
            part = dict(ident, order=order, k=k, chunk=c, seeds=seeds, success=eps['success'].astype(int).tolist(),
                        length=eps['length'].tolist(), n_bon_calls=calls['bon'], n_n1_calls=calls['n1'],
                        call_rules=calls['rules'], wall_s=time.time() - t0)
            write_atomic(path, (json.dumps(part) + '\n').encode())
            parts[k, c] = part
            done += 1
            print(f'SWITCH_EVAL_PART order={order} k={k} chunk={c} successes={int(np.sum(part["success"]))}/{C * E} '
                  f'n_bon_calls={calls["bon"]} n_n1_calls={calls["n1"]} wall_s={part["wall_s"]:.1f}', flush=True)
    tr.close_envs()

    ctr = dict(state['ctr'])
    out = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], mode=state.get('mode'), ctr=ctr,
               step=int(ctr.get('env_steps', 0) or ctr.get('bc_updates', 0)), env_name=FLAGS.env_name,
               horizon_length=FLAGS.horizon_length, method=FLAGS.method, agent_name=tr.config['agent_name'],
               agent=strip_later_agent_defaults(tr.config.to_dict()), final_seed_base=FLAGS.final_seed_base,
               seeds=[FLAGS.final_seed_base + j for j in range(n_seeds)], n_eval_envs=C, eval_episodes=E,
               switch_order=order, switch_ks=ks, wall_s=float(sum(p['wall_s'] for p in parts.values())),
               settings=[])
    for k in ks:
        S = np.concatenate([np.asarray(parts[k, c]['success'], dtype=np.int64) for c in range(n_chunks)])
        L = np.concatenate([np.asarray(parts[k, c]['length'], dtype=np.int64) for c in range(n_chunks)])
        s, n = int(S.sum()), int(S.size)
        out['settings'].append(dict(
            order=order, k=k, successes=s, episodes=n, success_rate=s / n, wilson95=wilson_ci(s, n),
            n_bon_calls=sum(parts[k, c]['n_bon_calls'] for c in range(n_chunks)),
            n_n1_calls=sum(parts[k, c]['n_n1_calls'] for c in range(n_chunks)),
            per_seed_successes=S.sum(1).tolist(), per_episode_success=S.tolist(), per_episode_length=L.tolist()))
    write_atomic(os.path.join(FLAGS.run_dir, 'switch_eval.json'),
                 (json.dumps(out, indent=1, default=str) + '\n').encode())
    for r in out['settings']:
        line = {x: v for x, v in r.items() if not x.startswith('per_')}
        line.update(src_run_dir=src, gen=man['gen'])
        print('SWITCH_EVAL ' + json.dumps(line, default=str), flush=True)
    return out


TRACE_AGENTS = ('acarb', 'acbsc', 'acfql')
TRACE_GRIDS = (('q1e3', 1e-3), ('q5e5', 5e-5))
TRACE_SUBSAMPLE = 400


def _trace_candidates(agent, observations, rng):
    """The best-of-n candidates the agent's sample_actions draws from rng (acarb._candidates; acbsc / acfql inline):
    (observations repeated (..., N, obs_dim), candidates (..., N, adim))."""
    c = agent.config
    adim = c['action_dim'] * (c['horizon_length'] if c['action_chunking'] else 1)
    noises = jax.random.normal(rng, (*observations.shape[:-len(c['ob_dims'])], c['actor_num_samples'], adim))
    obs = jnp.repeat(observations[..., None, :], c['actor_num_samples'], axis=-2)
    return obs, jnp.clip(agent.compute_flow_actions(obs, noises), -1, 1)


_TRACE_CANDIDATES = jax.jit(_trace_candidates)
_BATCH_SCORES = jax.jit(ensemble_score)  # (score, value, oor) of a batch read out together, as in sample_actions


def bon_trace_keys(agent, rng):
    """(candidate key, selection-read key) of the agent's sample_actions: acarb splits rng into (rng, readout_rng);
    acbsc / acfql draw the candidates from rng and read them out without a key."""
    if agent.config['agent_name'] == 'acarb':
        k = jax.random.split(rng)
        return k[0], k[1]
    return rng, rng


def trace_u(name, value):
    """Value in u units, float64, unclipped: acfql (Q + 100) / 100; acarb (S / K_den) and acbsc (probability) as is."""
    v = np.asarray(value, dtype=np.float64)
    return (v + 100.0) / 100.0 if name == 'acfql' else v


def trace_q_return(name, value):
    """QC contract return units, float64 (critic_outputs' q_return): acarb 100 u - 100, acfql the raw Q."""
    v = np.asarray(value, dtype=np.float64)
    return 100.0 * v - 100.0 if name == 'acarb' else v


def argmax_changes_after_rounding(u, grid):
    """Per row of u (..., N): does the first-index argmax change when u is rounded to multiples of grid (half up;
    no clipping)?"""
    u = np.asarray(u, dtype=np.float64)
    return np.argmax(np.floor(u / grid + 0.5), axis=-1) != np.argmax(u, axis=-1)


def npz_bytes(arrays):
    """Deterministic .npz (fixed member order and timestamps, deflate): identical arrays give identical bytes."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for k in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.asanyarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f'{k}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, b.getvalue(), compresslevel=9)
    return buf.getvalue()


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def candidate_stats(name, score, u, rep_score, rep_u, exec_u, exec_u_reread, scalar):
    """Compact per-row statistics of N candidates: score / u (rows, N) of the selection read, rep_score / rep_u
    (rows, R, N) of the repeats; exec_u / exec_u_reread (rows,) the executed (or reference) chunk's reads."""
    s = np.sort(u, axis=-1)
    sel = np.argmax(score, axis=-1)
    reads = np.concatenate([u[:, None], rep_u], axis=1)
    out = dict(exec_value=np.asarray(exec_u, np.float32), exec_value_reread=np.asarray(exec_u_reread, np.float32),
               cand_max=s[:, -1].astype(np.float32), cand_second=s[:, -2].astype(np.float32),
               cand_min=s[:, 0].astype(np.float32), cand_mean=u.mean(-1).astype(np.float32),
               n_unique=np.array([len(np.unique(r)) for r in u], dtype=np.int8),
               n_ties_at_max=(score == score.max(-1, keepdims=True)).sum(-1).astype(np.int8),
               readout_sd_mean=reads.std(axis=1).mean(-1).astype(np.float32),
               argmax_agree=(np.argmax(rep_score, axis=-1) == sel[:, None]).sum(-1).astype(np.int8))
    if scalar:
        for tag, grid in TRACE_GRIDS:
            out[f'argmax_change_{tag}'] = argmax_changes_after_rounding(u, grid).astype(np.int8)
    return out


def value_trace_summary(name, scalar, rules, probe, R):
    """value_trace_summary.npz arrays. rules: {rule: dict of concatenated part arrays + 'episode' (global id)};
    probe: the probe_trace arrays."""
    out = {}
    for rule, p in rules.items():
        u, rep_u = trace_u(name, p['sel_value']), trace_u(name, p['rep_value'])
        if rule == 'bon':
            ix = np.arange(len(u))
            exec_u = u[ix, p['sel_index']]
            reread = rep_u[ix, 0, p['sel_index']] if R else np.full(len(u), np.nan)
        else:
            exec_u = trace_u(name, p['exec_value'])
            reread = trace_u(name, p['exec_rep_value'][:, 0]) if R else np.full(len(u), np.nan)
        st = candidate_stats(name, p['sel_score'], u, p['rep_score'], rep_u, exec_u, reread, scalar)
        st['cand_oor_frac'] = p['sel_oor'].mean(-1).astype(np.float32)
        st['episode'] = p['episode'].astype(np.int16)
        st['decision'] = p['decision'].astype(np.int16)
        for k, v in st.items():
            out[f'{rule}/{k}'] = v
        sub = np.sort(np.random.RandomState(0).choice(len(u), size=min(TRACE_SUBSAMPLE, len(u)), replace=False))
        out[f'{rule}/sub_index'] = sub.astype(np.int32)
        out[f'{rule}/sub_cand_value'] = u[sub].astype(np.float32)
        out[f'{rule}/sub_cand_value_repeats'] = rep_u[sub].astype(np.float32)
        if rule == 'n1':
            out[f'{rule}/sub_exec_value_repeats'] = trace_u(name, p['exec_rep_value'][sub]).astype(np.float32)
        else:
            out[f'{rule}/sub_sel_index'] = p['sel_index'][sub].astype(np.int8)
    u, rep_u = trace_u(name, probe['value']), trace_u(name, probe['rep_value'])
    demo_u = trace_u(name, probe['demo_value'])
    reread = trace_u(name, probe['demo_rep_value'][0]) if R else np.full(len(u), np.nan)
    st = candidate_stats(name, probe['score'], u, probe['rep_score'].transpose(1, 0, 2), rep_u.transpose(1, 0, 2),
                         demo_u, reread, scalar)
    st['demo_value'], st['demo_value_reread'] = st.pop('exec_value'), st.pop('exec_value_reread')
    st['cand_oor_frac'] = probe['oor'].mean(-1).astype(np.float32)
    for k, v in st.items():
        out[f'probe/{k}'] = v
    out['probe/indices'] = probe['indices']
    out['u_units'] = np.asarray('(Q + 100) / 100' if name == 'acfql' else 'value')
    return out


def run_value_trace(tr, src, state, man, n_seeds, C, E):
    """value_trace (vd1): run_final_eval's episodes with the critic's candidate reads recorded per decision."""
    import functools
    name, R = tr.config['agent_name'], FLAGS.trace_repeats
    if name not in TRACE_AGENTS:
        refuse(f'value_trace supports agents {TRACE_AGENTS} (got {name})')
    if not FLAGS.ref_final_eval or not FLAGS.probe_candidates:
        refuse('--ref_final_eval and --probe_candidates are required for --mode=value_trace')
    if R < 0:
        refuse(f'--trace_repeats={R} < 0')
    with open(FLAGS.ref_final_eval) as f:
        ref = json.load(f)
    want = dict(src_run_dir=src, gen=man['gen'], final_seed_base=FLAGS.final_seed_base, n_eval_envs=C,
                eval_episodes=E, seeds=[FLAGS.final_seed_base + j for j in range(n_seeds)])
    got = {k: ref.get(k) for k in want}
    if got != want:
        refuse(f'--ref_final_eval {FLAGS.ref_final_eval} is for other settings: '
               + json.dumps({k: (got[k], want[k]) for k in want if got[k] != want[k]}, default=str))
    contract = value_contract(tr.config)
    scalar = not sampled_readout(tr.config)
    agent = tr.agent
    parts_dir = os.path.join(FLAGS.run_dir, 'value_trace_parts')
    os.makedirs(parts_dir, exist_ok=True)
    tr.eval_envs = None
    action_dim, H = tr.config['action_dim'], FLAGS.horizon_length
    n_chunks = n_seeds // C
    ident = dict(src_run_dir=src, gen=man['gen'], final_seed_base=FLAGS.final_seed_base, n_eval_envs=C,
                 eval_episodes=E, trace_repeats=R)
    parts, done = {}, 0

    def exit_preempted():
        tr.close_envs()
        dt = time.time() - tr.t_stop_ref
        print(f'EXIT_PREEMPTED source={tr.stop_source} notice_to_exit_s={dt:.2f} phase=value_trace '
              f'parts_done={done}', flush=True)
        sys.exit(EXIT_PREEMPTED)

    def record(rule, observations, rng, a, m, rec):
        """The critic's reads at this decision for the m active envs (rows 0..m-1); a is the executed action."""
        o, a = jnp.asarray(observations), np.asarray(a)
        if rule == 'bon':
            ck, rk = bon_trace_keys(agent, rng)
            obs_rep, cands = _TRACE_CANDIDATES(agent, o, ck)
            sel = _BATCH_SCORES(agent, obs_rep, cands, rk)
            reps = [_BATCH_SCORES(agent, obs_rep, cands, jax.random.fold_in(rng, 100 + r)) for r in range(R)]
        else:
            _, cands = _TRACE_CANDIDATES(agent, o, jax.random.fold_in(rng, 1))
            sel = _STATE_SCORES(agent, o, cands, jax.random.fold_in(rng, 2))
            reps = [_STATE_SCORES(agent, o, cands, jax.random.fold_in(rng, 100 + r)) for r in range(R)]
            ex = _ROW_SCORES(agent, o, jnp.asarray(a), jax.random.fold_in(rng, 3))
            ex_reps = [_ROW_SCORES(agent, o, jnp.asarray(a), jax.random.fold_in(rng, 100 + r)) for r in range(R)]
            for k, x in zip(('score', 'value', 'oor'), ex):
                rec[f'exec_{k}'].append(np.asarray(x)[:m])
            for i, k in enumerate(('score', 'value', 'oor')):
                rec[f'exec_rep_{k}'].append(np.stack([np.asarray(x[i])[:m] for x in ex_reps], 1) if R else
                                            np.zeros((m, 0), np.asarray(ex[i]).dtype))
        for i, k in enumerate(('score', 'value', 'oor')):
            rec[f'sel_{k}'].append(np.asarray(sel[i])[:m])
            rec[f'rep_{k}'].append(np.stack([np.asarray(x[i])[:m] for x in reps], 1) if R else
                                   np.zeros((m, 0, np.shape(sel[i])[-1]), np.asarray(sel[i]).dtype))
        if rule == 'bon':
            cands = np.asarray(cands)[:m]
            idx = np.argmax(np.asarray(sel[0])[:m], axis=-1)  # first index on ties, as jnp.argmax
            rec['sel_index'].append(idx.astype(np.int8))
            rec['argmax_mismatch'].append(np.any(cands[np.arange(m), idx] != a[:m], axis=-1))

    for rule in ('n1', 'bon'):
        fn = n1_actions if rule == 'n1' else bon_actions
        for c in range(n_chunks):
            path = os.path.join(parts_dir, f'{rule}_{c}.json')
            npz_path = os.path.join(parts_dir, f'{rule}_{c}.npz')
            seeds = [FLAGS.final_seed_base + c * C + i for i in range(C)]
            if os.path.exists(path) and os.path.exists(npz_path):
                with open(path) as f:
                    part = json.load(f)
                if {k: part.get(k) for k in ident} != ident or part['seeds'] != seeds:
                    tr.close_envs()
                    refuse(f'{path} was written with other settings: '
                           + json.dumps({k: part.get(k) for k in ident}))
                print(f'VALUE_TRACE_PART_SKIP rule={rule} chunk={c}', flush=True)
                parts[rule, c] = part
                done += 1
                continue
            if tr.watcher is not None:
                tr.watcher.check()
            if tr.stop_requested:
                exit_preempted()
            if tr.eval_envs is None:
                np_state = np.random.get_state()  # robomimic's make_env reseeds numpy (as in setup)
                tr.eval_envs = make_vectorized_eval_envs(FLAGS.env_name, C, seed=FLAGS.final_seed_base)
                np.random.set_state(np_state)
            t0 = time.time()
            envs = _AlignedEnvs(tr.eval_envs, C)
            calls = dict(d=0, ep=-1)
            rec = defaultdict(list)

            def on_reset(calls=calls):
                calls['d'] = 0
                calls['ep'] += 1

            def act(observations, rng, fn=fn, rule=rule, envs=envs, calls=calls, rec=rec):
                if tr.stop_requested:
                    raise EvalAborted()
                envs.check_call(observations, calls['d'], H)
                a = fn(tr.agent, observations, rng)
                env_ix = np.flatnonzero(envs.active)
                m = len(env_ix)
                record(rule, observations, rng, a, m, rec)
                rec['env'].append(env_ix.astype(np.int16))
                rec['episode'].append(np.full(m, calls['ep'], np.int16))
                rec['decision'].append(np.full(m, calls['d'], np.int16))
                calls['d'] += 1
                return a

            tr.eval_envs.set_attr('v1_reseed', [functools.partial(_seed_numpy, s) for s in seeds])
            tr.eval_envs.call('v1_reseed')
            np.random.seed(final_eval_chunk_seed(c))
            try:
                _, _, _, eps = evaluate_parallel(agent=Policy(act), envs=envs, num_eval_envs=C, eval_episodes=E,
                                                 action_dim=action_dim, return_episodes=True, on_reset=on_reset)
            except EvalAborted:
                print(f'EVAL_ABORTED ({tr.stop_source}) rule={rule} chunk={c}', flush=True)
                exit_preempted()
            assert calls['ep'] == E - 1, calls
            arrays = {k: np.concatenate(v) for k, v in rec.items()}
            if contract == 'qc':
                for k in ('sel', 'exec') if rule == 'n1' else ('sel',):
                    arrays[f'{k}_q_return'] = trace_q_return(name, arrays[f'{k}_value'])
            if name == 'acfql':
                for k in [k for k in arrays if k.endswith('value')]:
                    arrays[k[:-len('value')] + 'u'] = trace_u(name, arrays[k])
            digest = [[hashlib.sha256(np.ascontiguousarray(o, dtype=np.float64).tobytes()).hexdigest()[:16]
                       for o in env_obs] for env_obs in eps['first_obs']]
            reps = [np.abs(arrays['rep_value'] - arrays['sel_value'][:, None]).max(initial=0.0)]
            if rule == 'n1':
                reps.append(np.abs(arrays['exec_rep_value'] - arrays['exec_value'][:, None]).max(initial=0.0))
            part = dict(ident, rule=rule, chunk=c, seeds=seeds, success=eps['success'].astype(int).tolist(),
                        length=eps['length'].tolist(), first_obs_digest=digest, n_decisions=len(arrays['env']),
                        bon_argmax_mismatches=int(arrays['argmax_mismatch'].sum()) if rule == 'bon' else 0,
                        repeat_max_abs_diff=float(max(reps)), wall_s=time.time() - t0)
            write_atomic(npz_path, npz_bytes(arrays))
            write_atomic(path, (json.dumps(part) + '\n').encode())  # written last: marks the part complete
            parts[rule, c] = part
            done += 1
            print(f'VALUE_TRACE_PART rule={rule} chunk={c} successes={int(np.sum(part["success"]))}/{C * E} '
                  f'decisions={part["n_decisions"]} bon_argmax_mismatches={part["bon_argmax_mismatches"]} '
                  f'repeat_max_abs_diff={part["repeat_max_abs_diff"]:.3g} wall_s={part["wall_s"]:.1f}', flush=True)
    tr.close_envs()

    # reproduction of --ref_final_eval (episodes; first-observation digests from its parts if present)
    ref_parts = os.path.join(os.path.dirname(os.path.abspath(FLAGS.ref_final_eval)), 'final_eval_parts')
    n_match, n_eps, n_dig, dig_bad, per_rule = 0, 0, 0, 0, {}
    for rule in ('n1', 'bon'):
        S = np.concatenate([np.asarray(parts[rule, c]['success'], dtype=np.int64) for c in range(n_chunks)])
        L = np.concatenate([np.asarray(parts[rule, c]['length'], dtype=np.int64) for c in range(n_chunks)])
        ok = (S == np.asarray(ref[rule]['per_episode_success'])) & (L == np.asarray(ref[rule]['per_episode_length']))
        n_match, n_eps = n_match + int(ok.sum()), n_eps + int(ok.size)
        for c in range(n_chunks):
            rp = os.path.join(ref_parts, f'{rule}_{c}.json')
            if os.path.exists(rp):
                with open(rp) as f:
                    theirs = np.asarray(json.load(f)['first_obs_digest'])
                mine = np.asarray(parts[rule, c]['first_obs_digest'])
                n_dig, dig_bad = n_dig + mine.size, dig_bad + int((mine != theirs).sum())
        per_rule[rule] = dict(successes=int(S.sum()), episodes=int(S.size), ref_successes=ref[rule]['successes'],
                              n_decisions=sum(parts[rule, c]['n_decisions'] for c in range(n_chunks)))
    mismatches = sum(parts['bon', c]['bon_argmax_mismatches'] for c in range(n_chunks))
    rep_diff = max(p['repeat_max_abs_diff'] for p in parts.values())

    # common panel: fixed demonstration states, fixed BC-actor candidates, the demonstration chunk
    tr.setup_probe(tr.config)
    obs, cands, demo = tr.probe_obs, tr.probe_cands, tr.probe_actions
    keys = [jax.random.PRNGKey(0)] + [jax.random.PRNGKey(100 + r) for r in range(R)]
    reads = [critic_outputs(agent, obs, cands, k, contract, per_state=True) for k in keys]
    dreads = [critic_outputs(agent, obs, demo, k, contract) for k in keys]
    stack = lambda xs, k, like: np.stack([x[k] for x in xs]) if xs else np.zeros((0, *like.shape), like.dtype)  # noqa
    probe = dict(indices=np.asarray(tr.probe_idx), steps_left=np.asarray(tr.probe_steps_left),
                 score=reads[0]['score'], value=reads[0]['value'].astype(np.float32), oor=reads[0]['oor'],
                 rep_score=stack(reads[1:], 'score', reads[0]['score']),
                 rep_value=stack(reads[1:], 'value', reads[0]['value']).astype(np.float32),
                 demo_score=dreads[0]['score'], demo_value=dreads[0]['value'].astype(np.float32),
                 demo_oor=dreads[0]['oor'],
                 demo_rep_value=stack(dreads[1:], 'value', dreads[0]['value']).astype(np.float32))
    for k in ('q_return', 'sigmoid_mean_logit'):
        if k in reads[0]:
            probe[k], probe[f'demo_{k}'] = reads[0][k], dreads[0][k]
    if name == 'acfql':
        probe['u'], probe['demo_u'] = trace_u(name, probe['value']), trace_u(name, probe['demo_value'])
    write_atomic(os.path.join(FLAGS.run_dir, 'probe_trace.npz'), npz_bytes(probe))

    rules = {}
    for rule in ('n1', 'bon'):
        cat = defaultdict(list)
        for c in range(n_chunks):
            z = read_npz(os.path.join(parts_dir, f'{rule}_{c}.npz'))
            z['episode'] = (c * C + z['env'].astype(np.int64)) * E + z['episode']  # final_eval's episode order
            for k, v in z.items():
                cat[k].append(v)
        rules[rule] = {k: np.concatenate(v) for k, v in cat.items()}
    summary_path = os.path.join(FLAGS.run_dir, 'value_trace_summary.npz')
    write_atomic(summary_path, npz_bytes(value_trace_summary(name, scalar, rules, probe, R)))
    summary_bytes = os.path.getsize(summary_path)

    ctr = dict(state['ctr'])
    out = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], ctr=ctr,
               step=int(ctr.get('env_steps', 0) or ctr.get('bc_updates', 0)), env_name=FLAGS.env_name,
               horizon_length=FLAGS.horizon_length, method=FLAGS.method, agent_name=name,
               agent=strip_later_agent_defaults(tr.config.to_dict()), contract=contract, scalar_readout=scalar,
               final_seed_base=FLAGS.final_seed_base, n_eval_envs=C, eval_episodes=E, trace_repeats=R,
               ref_final_eval=os.path.abspath(FLAGS.ref_final_eval), probe_candidates=tr.probe_cands_info,
               n_episodes_match=n_match, n_episodes=n_eps, first_obs_digests_checked=n_dig,
               first_obs_digest_mismatches=dig_bad, bon_argmax_mismatches=mismatches,
               scalar_repeat_max_abs_diff=rep_diff if scalar else None, repeat_max_abs_diff=rep_diff,
               rules=per_rule, probe_states=int(len(obs)), summary_bytes=summary_bytes)
    failures = []
    if n_match != n_eps:
        failures.append(f'{n_eps - n_match}/{n_eps} episodes differ from --ref_final_eval')
    if dig_bad:
        failures.append(f'{dig_bad}/{n_dig} first-observation digests differ from --ref_final_eval\'s parts')
    if mismatches:
        failures.append(f'{mismatches} bon decisions whose replicated argmax candidate is not the executed action')
    if scalar and rep_diff != 0:
        failures.append(f'scalar readout: repeated reads differ from the first (max abs diff {rep_diff})')
    out['ok'], out['failures'] = not failures, failures
    write_atomic(os.path.join(FLAGS.run_dir, 'value_trace.json'),
                 (json.dumps(out, indent=1, default=str) + '\n').encode())
    print('VALUE_TRACE ' + json.dumps({k: v for k, v in out.items() if k not in ('agent', 'ctr', 'probe_candidates')},
                                      default=str), flush=True)
    if failures:
        print('VALUE_TRACE_FAILED ' + '; '.join(failures), flush=True)
        sys.exit(EXIT_TRACE_FAILED)
    return out


EXIT_TRACE_FAILED = 3


def run_episode_stages(tr, src, state, man, *_):
    """--mode=episode_stages (geo8b): one episode per (rule, env seed final_seed_base + j) in a single env, as
    final_eval's actions (one H-step chunk per decision, the env's step limit), recording the task-stage code
    after every env step; videos for j < --video_n_seeds. -> episode_stages.json (resumable per (rule, j) part)."""
    import imageio
    import mujoco
    from PIL import Image, ImageDraw
    from envs.robomimic_utils import make_env
    try:
        import imageio_ffmpeg  # noqa: F401
        ext = 'mp4'
    except ImportError:
        ext = 'gif'
    parts_dir, vid_dir = os.path.join(FLAGS.run_dir, 'episode_stages_parts'), os.path.join(FLAGS.run_dir, 'videos')
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    action_dim, H, n_seeds = tr.config['action_dim'], FLAGS.horizon_length, FLAGS.stage_n_seeds
    hw, stride, fps = FLAGS.video_hw, FLAGS.video_stride, 15
    labels = dict(n1='N1', bon=f'BO{tr.config["actor_num_samples"]}')
    ident = dict(src_run_dir=src, gen=man['gen'], final_seed_base=FLAGS.final_seed_base, video_hw=hw,
                 video_stride=stride)
    env, parts, done = None, {}, 0

    def stage_code(sim):
        """tool hang: robosuite ToolHang._check_frame_assembled / _check_tool_on_frame; else 2 iff success."""
        if hasattr(sim, '_check_frame_assembled'):
            fa = bool(sim._check_frame_assembled())
            return 2 if fa and sim._check_tool_on_frame() else int(fa)
        return 2 * int(bool(sim._check_success()))

    def write_video(path, frames):
        tmp = f'{path[:-len(ext) - 1]}.tmp.{ext}'
        for q in (5, 3, 1):  # keep each file <= 400 KB
            if ext == 'mp4':
                imageio.v2.mimwrite(tmp, frames, format='FFMPEG', fps=fps, codec='libx264', quality=q,
                                    pixelformat='yuv420p', macro_block_size=16)
            else:
                imageio.v2.mimwrite(tmp, frames[::2], duration=2000 / fps, loop=0)
            if os.path.getsize(tmp) <= 400_000:
                break
        os.replace(tmp, path)

    def exit_preempted():
        if env is not None:
            env.close()
        dt = time.time() - tr.t_stop_ref
        print(f'EXIT_PREEMPTED source={tr.stop_source} notice_to_exit_s={dt:.2f} phase=episode_stages '
              f'parts_done={done}', flush=True)
        sys.exit(EXIT_PREEMPTED)

    for rule in ('n1', 'bon'):
        fn = n1_actions if rule == 'n1' else bon_actions
        for j in range(n_seeds):
            seed = FLAGS.final_seed_base + j
            path = os.path.join(parts_dir, f'{rule}_{j}.json')
            if os.path.exists(path):
                with open(path) as f:
                    part = json.load(f)
                if {k: part.get(k) for k in ident} != ident or part['seed'] != seed:
                    refuse(f'{path} was written with other settings: ' + json.dumps({k: part.get(k) for k in ident}))
                print(f'EPISODE_STAGES_PART_SKIP rule={rule} seed={seed}', flush=True)
                parts[rule, j] = part
                done += 1
                continue
            if tr.watcher is not None:
                tr.watcher.check()
            if tr.stop_requested:
                exit_preempted()
            video = j < FLAGS.video_n_seeds
            if env is None:
                np_state = np.random.get_state()  # robomimic's make_env reseeds numpy
                env = make_env(FLAGS.env_name, seed=FLAGS.final_seed_base)
                np.random.set_state(np_state)
            sim = env.env.env  # RobomimicLowdimWrapper -> robomimic EnvRobosuite -> robosuite env
            t0 = time.time()
            np.random.seed(seed)  # as final_eval's _seed_numpy before the seed's first reset
            obs, _ = env.reset()
            digest = hashlib.sha256(np.ascontiguousarray(obs, dtype=np.float64).tobytes()).hexdigest()[:16]
            rng = jax.random.PRNGKey(seed)  # the same actor noise stream for both rules
            codes, frames, success, t, decisions = [], [], 0, 0, 0

            if video:  # robosuite's own offscreen renderer returns corrupted frames once build_agent_trainer has
                # made and closed its envs; mujoco.Renderer (own EGL context, made current per render) does not
                renderer = mujoco.Renderer(sim.sim.model._model, hw, hw)
                scene_opt = mujoco.MjvOption()
                scene_opt.geomgroup[0] = 0  # collision geoms hidden, as robosuite's renderer

            def frame():
                renderer.update_scene(sim.sim.data._data, camera='agentview', scene_option=scene_opt)
                im = Image.fromarray(renderer.render())
                ImageDraw.Draw(im).text((3, 2), f'{labels[rule]} t={t} stage={codes[-1] if codes else 0}',
                                        fill=(255, 255, 0))
                frames.append(np.asarray(im))

            if video:
                frame()
            end = False
            while not end:
                if tr.stop_requested:
                    print(f'EVAL_ABORTED ({tr.stop_source}) rule={rule} seed={seed}', flush=True)
                    exit_preempted()
                rng, key = jax.random.split(rng)
                chunk = np.asarray(fn(tr.agent, obs[None], key)).reshape(-1, action_dim)
                decisions += 1
                for a in chunk[:H]:
                    obs, _, term, trunc, info = env.step(a)
                    t += 1
                    codes.append(stage_code(sim))
                    end = term or trunc
                    if video and (t % stride == 0 or end):
                        frame()
                    if end:
                        success = int(info['success'])
                        break
            c = np.asarray(codes)
            first = {s: (int(np.argmax(c >= s)) + 1 if np.any(c >= s) else None) for s in (1, 2)}
            vpath = None
            if video:
                renderer.close()
                vpath = os.path.join(vid_dir, f'{rule}_seed{seed}.{ext}')
                write_video(vpath, frames)
            part = dict(ident, rule=rule, seed=seed, success=success, length=t, decisions=decisions,
                        first_stage1_step=first[1], first_stage2_step=first[2],
                        steps_at_stage=[int(np.sum(c == s)) for s in range(3)],
                        stage_per_decision=[int(c[min((d + 1) * H, t) - 1]) for d in range(decisions)],
                        first_obs_digest=digest, video=vpath, wall_s=time.time() - t0)
            write_atomic(path, (json.dumps(part) + '\n').encode())
            parts[rule, j] = part
            done += 1
            print(f'EPISODE_STAGES_PART rule={rule} seed={seed} success={success} length={t} '
                  f'stage1={first[1]} stage2={first[2]} wall_s={part["wall_s"]:.1f}', flush=True)
    if env is not None:
        env.close()

    for j in range(min(FLAGS.video_n_seeds, n_seeds)):  # N1 left, BO-n right; the shorter padded with its last frame
        seed = FLAGS.final_seed_base + j
        path = os.path.join(vid_dir, f'side_by_side_seed{seed}.{ext}')
        if os.path.exists(path):
            continue
        a, b = (imageio.v2.mimread(parts[r, j]['video'], memtest=False) for r in ('n1', 'bon'))
        n = max(len(a), len(b))
        a, b = (v + [v[-1]] * (n - len(v)) for v in (a, b))
        write_video(path, [np.concatenate([x[..., :3], y[..., :3]], axis=1) for x, y in zip(a, b)])

    ctr = dict(state['ctr'])
    out = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], mode=state.get('mode'), ctr=ctr,
               env_name=FLAGS.env_name, horizon_length=H, method=FLAGS.method, agent_name=tr.config['agent_name'],
               agent=strip_later_agent_defaults(tr.config.to_dict()), final_seed_base=FLAGS.final_seed_base,
               stage_n_seeds=n_seeds, video_n_seeds=FLAGS.video_n_seeds, video_hw=hw, video_stride=stride,
               stage_codes={0: 'none', 1: 'frame assembled', 2: 'frame assembled + tool on frame (success)'},
               episodes=[parts[r, j] for r in ('n1', 'bon') for j in range(n_seeds)], summary={})
    for rule in ('n1', 'bon'):
        eps = [parts[rule, j] for j in range(n_seeds)]
        s1 = [e['first_stage1_step'] for e in eps if e['first_stage1_step'] is not None]
        out['summary'][rule] = dict(episodes=len(eps), successes=sum(e['success'] for e in eps),
                                    reached_stage1=len(s1),
                                    median_first_stage1_step=float(np.median(s1)) if s1 else None,
                                    mean_length=float(np.mean([e['length'] for e in eps])))
    write_atomic(os.path.join(FLAGS.run_dir, 'episode_stages.json'),
                 (json.dumps(out, indent=1, default=str) + '\n').encode())
    print('EPISODE_STAGES ' + json.dumps(out['summary']), flush=True)
    return out


def arb_introspect():
    """--mode=arb_introspect: utils/arb_introspect.run_mode on the agent of --src_run_dir's committed checkpoint
    (read-only, any phase), the agent built from the query file's shapes (no env, no dataset). Config checked
    against the source as in final_eval. Output: <run_dir>/arb_introspect{.json,.npz} and resumable parts."""
    from utils import arb_introspect as AI
    if not (FLAGS.src_run_dir and FLAGS.queries):
        refuse('--src_run_dir and --queries are required')
    src, out_dir = os.path.realpath(FLAGS.src_run_dir), os.path.realpath(FLAGS.run_dir)
    if src == out_dir or out_dir.startswith(src + os.sep):
        refuse(f'--run_dir {out_dir} lies inside --src_run_dir {src}')
    state, man = read_committed_state(src)
    args_path = os.path.join(src, 'args.json')
    if not os.path.exists(args_path):
        refuse(f'{args_path} not found')
    with open(args_path) as f:
        src_args = json.load(f)
    config = resolve_agent_config(FLAGS.method, FLAGS.agent, sys.argv)
    config['horizon_length'] = FLAGS.horizon_length
    check_agent_config('arb_introspect', config)
    with np.load(FLAGS.queries) as z:
        obs0, act0 = np.asarray(z['observations'][0], np.float32), np.asarray(z['actions'][0], np.float32)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, obs0, np.zeros(act0.size // FLAGS.horizon_length, np.float32), config)
    diff = source_config_diff(src, src_args, config)
    if diff:
        refuse('config differs from the source run (source, this call): ' + json.dumps(diff, default=str))
    agent = flax.serialization.from_state_dict(split_optimizer(agent), state['agent'])
    os.makedirs(out_dir, exist_ok=True)
    ident = dict(src_run_dir=src, gen=man['gen'], phase=state['phase'], method=FLAGS.method,
                 agent=strip_later_agent_defaults(config.to_dict()))
    AI.run_mode(agent, out_dir, ident, FLAGS.queries, FLAGS.candidates, chunk=FLAGS.introspect_chunk,
                seed=FLAGS.introspect_seed, components=tuple(FLAGS.introspect_components))


# ---- geo5a diagnostics (--mode=diag_*) ------------------------------------------------------------------------
# All modes read the source run read-only (as final_eval: committed checkpoint, phase 'done', the call's --method /
# --agent.* / --env_name / --horizon_length checked against it) and write only below --run_dir (refused inside the
# source): manifest.json (inputs, hashes, seeds, counts, code sha per start), atomic parts (a restart skips the
# finished ones; a preemption notice exits 143 after the last finished part), then the mode's outputs.
#   diag_anchors  diagnostic replays (NOT the official final_eval trajectories): --diag_n_seeds env seeds
#                 final_seed_base + j x --diag_episodes episodes (initial states as final_eval), rule --diag_rule
#                 with actor keys from --diag_seed; per decision: seed, episode, step, snapshot, executed chunk,
#                 critic score of the chunk (production readout, key PRNGKey(0)), episode success / length
#                 (parts/replay_<c>.pkl). Then --anchors anchors by diag_v1.select_anchors -> anchors.pkl (a query
#                 file: the anchor states and their executed chunks), anchors.json. --policy_ckpt=<actor pkl>
#                 instead of --src_run_dir: the common-state panel (n1, no critic).
#   diag_mc       per query of --queries: --mc_k continuations: restore -> execute the query chunk -> --continuation
#                 (n1 default) of the source actor; actor key fold_in(fold_in(PRNGKey(mc_seed), q), k), env numpy
#                 seed SeedSequence([mc_seed, q, k]); contract (bsc / qc, diag_v1.ReturnAccumulator) from the critic
#                 -> parts/mc_<p>.pkl, mc.json (per query: every G_k, mean, SE).
#   diag_score    per query: production score / value of the source critic with key PRNGKey(0) and --score_keys
#                 more keys (sampled readouts; scalar critics: one) -> scores.json.
#   diag_select   per state of the panel --queries: actor_num_samples candidates of the source actor (noise key
#                 from SeedSequence([sel_seed, state])), the critic's choice (production score, key PRNGKey(0), first
#                 index on ties), a uniform index (own RNG), optionally --mc_candidates further candidates
#                 -> select.json, select_queries.pkl (input of diag_mc / diag_score).
SCORE_SEED = 20261025
DIAG_QUERY_VERSION = 1


def diag_refuse(msg):
    print(f'DIAG_REFUSED {msg}', flush=True)
    sys.exit(2)


def code_version():
    import subprocess
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True,
                             timeout=30).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return dict(sha=sha or None, dirty=bool(dirty))
    except Exception as e:  # no git: still run
        return dict(sha=None, error=repr(e))


def _check_out_dir(src):
    out_dir = os.path.realpath(FLAGS.run_dir)
    if src == out_dir or out_dir.startswith(src + os.sep):
        diag_refuse(f'--run_dir {out_dir} lies inside the source {src}')


def diag_frozen_agent():
    """(trainer with the source checkpoint's agent, source info); the source is only read."""
    if not FLAGS.src_run_dir:
        diag_refuse('--src_run_dir is required')
    src = os.path.realpath(FLAGS.src_run_dir)
    _check_out_dir(src)
    state, man = read_committed_state(src)
    if state['phase'] != 'done':
        diag_refuse(f'checkpoint gen={man["gen"]} of {src} has phase={state["phase"]} (need done)')
    args_path = os.path.join(src, 'args.json')
    if not os.path.exists(args_path):
        diag_refuse(f'{args_path} not found')
    with open(args_path) as f:
        src_args = json.load(f)
    tr = build_agent_trainer()
    diff = source_config_diff(src, src_args, tr.config)
    if diff:
        diag_refuse('config differs from the source run (source, this call): ' + json.dumps(diff, default=str))
    tr.agent = flax.serialization.from_state_dict(tr.agent, state['agent'])
    info = dict(kind='checkpoint', src_run_dir=src, gen=man['gen'], phase=state['phase'],
                state_sha256=man['files']['state.pkl']['sha256'], args_sha256=file_sha256(args_path),
                method=FLAGS.method, agent_name=tr.config['agent_name'], ctr=dict(state['ctr']))
    return tr, info


def diag_policy_agent():
    """(trainer with the --policy_ckpt actor (bsc agent, critic unused), source info)."""
    path = os.path.realpath(FLAGS.policy_ckpt)
    _check_out_dir(os.path.dirname(path))
    tr = build_agent_trainer()
    tr.agent, info = load_actor(tr.agent, path, tr.H)
    return tr, dict(kind='policy_ckpt', policy_ckpt=path, sha256=file_sha256(path),
                    **{k: v for k, v in info.items() if k in ('agent_name', 'bc_steps', 'seed', 'selected_update',
                                                              'selection_rule')})


def value_contract(config):
    if FLAGS.diag_contract != 'auto':
        return FLAGS.diag_contract
    name = config['agent_name']
    if name == 'acbsc':
        return 'bsc'
    if name == 'acfql':
        return 'qc'
    if name == 'acarb':
        return config['arb_value'] if 'arb_value' in config else 'bsc'
    diag_refuse(f'no value contract for agent {name}; pass --diag_contract')


def diag_manifest(ident, **update):
    """Create or check manifest.json: ident must equal the stored one (else refused); every start appends the code
    version; update adds fields (counts, outputs)."""
    path = os.path.join(FLAGS.run_dir, 'manifest.json')
    norm = json.loads(json.dumps(ident, default=str))
    if os.path.exists(path):
        with open(path) as f:
            man = json.load(f)
        if man['ident'] != norm:
            diff = {k: (man['ident'].get(k), norm.get(k)) for k in set(man['ident']) | set(norm)
                    if man['ident'].get(k) != norm.get(k)}
            diag_refuse(f'{path} was written with other inputs: ' + json.dumps(diff, default=str))
    else:
        man = dict(ident=norm, starts=[])
    if not update:
        man['starts'].append(dict(time=time.time(), code=code_version(), node=os.uname().nodename,
                                  slurm_job=os.environ.get('SLURM_JOB_ID')))
    man.update(json.loads(json.dumps(update, default=str)))
    write_atomic(path, (json.dumps(man, indent=1) + '\n').encode())
    return man


def write_pickle(path, obj):
    write_atomic(path, pickle.dumps(obj, protocol=4))


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_queries(path):
    q = read_pickle(path)
    assert q.get('kind') == 'diag_queries' and q.get('version') == DIAG_QUERY_VERSION, f'{path}: not a query file'
    if q['env_name'] != FLAGS.env_name or q['horizon_length'] != FLAGS.horizon_length:
        diag_refuse(f'{path} is for {q["env_name"]} H={q["horizon_length"]}')
    return q, dict(path=os.path.realpath(path), sha256=file_sha256(path), n_queries=len(q['queries']),
                   n_states=len(q['states']))


def query_file(states, queries, source, **meta):
    return dict(kind='diag_queries', version=DIAG_QUERY_VERSION, env_name=FLAGS.env_name,
                horizon_length=FLAGS.horizon_length, states=states, queries=queries, source=source, **meta)


def _one_row_score(agent, o, a, rng):
    from visualization.critic_probe import ensemble_score
    s, v, oor = ensemble_score(agent, o[None], a[None], rng)
    return s[0], v[0], oor[0]


def _one_state_scores(agent, o, cands, rng):
    from visualization.critic_probe import ensemble_score
    return ensemble_score(agent, jnp.repeat(o[None], cands.shape[0], 0), cands, rng)


_ROW_SCORES = jax.jit(jax.vmap(_one_row_score, in_axes=(None, 0, 0, None)))
_STATE_SCORES = jax.jit(jax.vmap(_one_state_scores, in_axes=(None, 0, 0, None)))


def _batched(fn, agent, obs, actions, rng, batch):
    """fn over fixed-size row batches (padding: no recompiles); rows are independent of the batch layout."""
    obs, actions = np.asarray(obs, np.float32), np.asarray(actions, np.float32)
    outs = []
    for s in range(0, len(obs), batch):
        o, a = obs[s:s + batch], actions[s:s + batch]
        m = len(o)
        if m < batch:
            o = np.concatenate([o, np.zeros((batch - m, *o.shape[1:]), o.dtype)])
            a = np.concatenate([a, np.zeros((batch - m, *a.shape[1:]), a.dtype)])
        outs.append([np.asarray(x)[:m] for x in fn(agent, o, a, rng)])
    return [np.concatenate([o[k] for o in outs]) for k in range(3)]


def critic_outputs(agent, obs, actions, rng, contract, per_state=False):
    """Production ensemble readout of the critic (critic_probe.ensemble_score), per row.

    A sampled readout (acarb sample_mean) draws Gumbel noise of shape (num_qs, *batch, M, L, C) from rng, so the value
    of a chunk depends on its position in the batch. Here it is defined layout-free: each (obs, chunk) row is read out
    alone with rng (the same key for every row); per_state=True: actions (S, N, adim), the N candidates of each state
    read out together with rng, i.e. exactly the layout of the agent's best-of-N for one observation with readout
    key rng. Fields: score (what best-of-n ranks by: integer S for sample_mean, else the q_agg ensemble value), value
    (S / K_den; acbsc: mean of sigmoids (q_agg); acfql: mean Q), q_return (QC contract: acarb 100 value - 100, acfql
    value), oor; acbsc also sigmoid_mean_logit = sigmoid(mean of the 2 logits)."""
    if per_state:
        score, value, oor = _batched(_STATE_SCORES, agent, obs, actions, rng, 8)
    else:
        score, value, oor = _batched(_ROW_SCORES, agent, obs, actions, rng, 256)
    name = agent.config['agent_name']
    out = dict(score=score, value=value.astype(np.float64), oor=oor)
    if contract == 'qc':
        out['q_return'] = 100.0 * out['value'] - 100.0 if name == 'acarb' else out['value']
    if name == 'acbsc':
        o, x = np.asarray(obs, np.float32), np.asarray(actions, np.float32)
        if per_state:
            o, x = np.repeat(o, x.shape[1], 0), x.reshape(-1, x.shape[-1])
        f = jax.jit(lambda a, o, x: a.network.select('critic')(o, x))
        logits = np.concatenate([np.asarray(f(agent, o[s:s + 256], x[s:s + 256])) for s in range(0, len(o), 256)],
                                axis=-1)
        out['sigmoid_mean_logit'] = (1.0 / (1.0 + np.exp(-logits.astype(np.float64).mean(0)))).reshape(score.shape)
    return out


def _row(out, i):
    return {k: (v[i].item() if np.ndim(v[i]) == 0 else np.asarray(v[i]).tolist()) for k, v in out.items()}


def sampled_readout(config):
    return config['agent_name'] == 'acarb' and config['arb_readout'] in ('sample', 'sample_mean')


def diag_stop(tr):
    if tr.watcher is not None:
        tr.watcher.check(min_interval=0.5)
    return tr.stop_requested


def diag_exit_preempted(tr, done):
    tr.close_envs()
    dt = time.time() - tr.t_stop_ref
    print(f'EXIT_PREEMPTED source={tr.stop_source} notice_to_exit_s={dt:.2f} phase={FLAGS.mode} parts_done={done}',
          flush=True)
    sys.exit(EXIT_PREEMPTED)


def _part_done(path, ident):
    if not os.path.exists(path):
        return None
    part = read_pickle(path)
    if json.loads(json.dumps(part['ident'], default=str)) != json.loads(json.dumps(ident, default=str)):
        diag_refuse(f'{path} was written with other inputs')
    return part


def diag_anchors(tr, source):
    from utils import diag_v1
    panel = source['kind'] == 'policy_ckpt'
    rule = 'n1' if panel else FLAGS.diag_rule
    if panel and FLAGS.diag_rule != 'n1':
        diag_refuse('the common panel (--policy_ckpt) uses the n1 rule')
    C, n_seeds, E = FLAGS.n_eval_envs, FLAGS.diag_n_seeds, FLAGS.diag_episodes
    if n_seeds <= 0 or C <= 0 or n_seeds % C:
        diag_refuse(f'--diag_n_seeds={n_seeds} is not a positive multiple of --n_eval_envs={C}')
    contract = None if panel else value_contract(tr.config)
    ident = dict(mode='diag_anchors', source=source, env_name=FLAGS.env_name, horizon_length=tr.H, rule=rule,
                 diag_seed=FLAGS.diag_seed, final_seed_base=FLAGS.final_seed_base, diag_n_seeds=n_seeds,
                 n_eval_envs=C, diag_episodes=E, anchors=FLAGS.anchors, contract=contract,
                 label='diagnostic replays (not the official final_eval trajectories)')
    diag_manifest(ident)
    parts_dir = os.path.join(FLAGS.run_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    policy = diag_v1.row_policy(tr.agent, rule)
    step_dim = tr.config['action_dim']
    base_key = np.asarray(jax.random.PRNGKey(FLAGS.diag_seed))
    tr.eval_envs = None
    parts, done = [], 0
    for c in range(n_seeds // C):
        path = os.path.join(parts_dir, f'replay_{c}.pkl')
        seeds = [FLAGS.final_seed_base + c * C + i for i in range(C)]
        part = _part_done(path, ident)
        if part is not None:
            print(f'DIAG_PART_SKIP replay chunk={c}', flush=True)
            parts.append(part)
            done += 1
            continue
        if diag_stop(tr):
            diag_exit_preempted(tr, done)
        if tr.eval_envs is None:
            tr.eval_envs = diag_v1.make_pool(FLAGS.env_name, C)
        t0 = time.time()
        comp = diag_v1.SnapCompactor()
        try:
            decisions, eps = diag_v1.run_replays(tr.eval_envs, C, seeds, E, policy, base_key, tr.H, step_dim,
                                                 stop=lambda: diag_stop(tr), compactor=comp)
        except diag_v1.Aborted:
            diag_exit_preempted(tr, done)
        if not panel:
            sc = critic_outputs(tr.agent, np.stack([d['obs'] for d in decisions]),
                                np.stack([d['chunk'] for d in decisions]), jax.random.PRNGKey(0), contract)
            for i, d in enumerate(decisions):
                d['critic'] = _row(sc, i)
        part = dict(ident=ident, chunk=c, seeds=seeds, episodes=eps, decisions=decisions, bases=comp.bases,
                    wall_s=time.time() - t0)
        write_pickle(path, part)
        parts.append(part)
        done += 1
        print(f'DIAG_PART replay chunk={c} decisions={len(decisions)} successes={int(eps["success"].sum())}/{C * E} '
              f'wall_s={part["wall_s"]:.1f}', flush=True)
    tr.close_envs()

    # Anchors: the predeclared stratified rule over every decision of every part.
    meta, where = [], []
    for p in parts:
        L, S = p['episodes']['length'], p['episodes']['success']
        for j, d in enumerate(p['decisions']):
            meta.append(dict(seed=d['seed'], episode=d['episode'], step=d['step'], length=int(L[d['env'], d['episode']]),
                             success=int(S[d['env'], d['episode']])))
            where.append((p, j))
    picks = diag_v1.select_anchors(meta, FLAGS.anchors, FLAGS.diag_seed)
    states, queries, rows = [], [], []
    for qi, a in enumerate(picks):
        p, j = where[a['index']]
        d = p['decisions'][j]
        info = dict(seed=d['seed'], episode=d['episode'], decision=d['decision'], step=d['step'], t=d['t'],
                    length=meta[a['index']]['length'], success=meta[a['index']]['success'],
                    **{k: a[k] for k in ('stratum', 'third', 'weight', 'N_h', 'n_h', 'S_h', 'n_hs')},
                    critic=d.get('critic'))
        states.append(diag_v1.expand_snapshot(d['snap'], p['bases']))
        queries.append(dict(qid=qi, state=qi, label='executed', chunk=d['chunk'], meta=info))
        rows.append(info)
    write_pickle(os.path.join(FLAGS.run_dir, 'anchors.pkl'), query_file(states, queries, source, ident=ident))
    S_all = np.concatenate([p['episodes']['success'] for p in parts])
    L_all = np.concatenate([p['episodes']['length'] for p in parts])
    digests = sum((p['episodes']['first_obs_digest'] for p in parts), [])
    summary = dict(ident=ident, decisions=len(meta), episodes=int(S_all.size), successes=int(S_all.sum()),
                   per_episode_success=S_all.tolist(), per_episode_length=L_all.tolist(), first_obs_digest=digests,
                   strata={h: sum(r['stratum'] == h for r in rows) for h in sorted({r['stratum'] for r in rows})},
                   anchors=rows, wall_s=float(sum(p['wall_s'] for p in parts)))
    write_atomic(os.path.join(FLAGS.run_dir, 'anchors.json'), (json.dumps(summary, indent=1, default=str) + '\n').encode())
    diag_manifest(ident, complete=True, counts=dict(decisions=len(meta), episodes=int(S_all.size), anchors=len(rows)),
                  outputs=['anchors.pkl', 'anchors.json'],
                  anchors_sha256=file_sha256(os.path.join(FLAGS.run_dir, 'anchors.pkl')))
    print(f'DIAG_ANCHORS decisions={len(meta)} episodes={S_all.size} successes={int(S_all.sum())} anchors={len(rows)} '
          f'strata={json.dumps(summary["strata"])}', flush=True)


def diag_mc(tr, source):
    from utils import diag_v1
    qf, qinfo = load_queries(FLAGS.queries)
    contract = value_contract(tr.config)
    t_extra = FLAGS.mc_t_extra if contract == 'qc' else 0
    K, P = FLAGS.mc_k, FLAGS.mc_part_size
    ident = dict(mode='diag_mc', source=source, queries=qinfo, env_name=FLAGS.env_name, horizon_length=tr.H,
                 contract=contract, continuation=FLAGS.continuation, mc_k=K, mc_seed=FLAGS.mc_seed, t_extra=t_extra,
                 tail_bound_at_t_extra=diag_v1.qc_tail_bound(t_extra) if contract == 'qc' else None,
                 mc_part_size=P, n_eval_envs=FLAGS.n_eval_envs)
    diag_manifest(ident)
    parts_dir = os.path.join(FLAGS.run_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    policy = diag_v1.row_policy(tr.agent, FLAGS.continuation)
    step_dim = tr.config['action_dim']
    N = FLAGS.n_eval_envs
    qs = qf['queries']
    tr.eval_envs = None
    results, done, walls = {}, 0, []
    for p in range((len(qs) + P - 1) // P):
        path = os.path.join(parts_dir, f'mc_{p}.pkl')
        part = _part_done(path, ident)
        if part is not None:
            print(f'DIAG_PART_SKIP mc part={p}', flush=True)
            results.update(part['results'])
            walls.append(part['wall_s'])
            done += 1
            continue
        if diag_stop(tr):
            diag_exit_preempted(tr, done)
        if tr.eval_envs is None:
            tr.eval_envs = diag_v1.make_pool(FLAGS.env_name, N)
        t0 = time.time()
        jobs = [dict(q=q['qid'], k=k, snap=qf['states'][q['state']], chunk=q['chunk'],
                     np_seed=diag_v1.mc_np_seed(FLAGS.mc_seed, q['qid'], k),
                     key=diag_v1.mc_job_key(FLAGS.mc_seed, q['qid'], k))
                for q in qs[p * P:(p + 1) * P] for k in range(K)]
        try:
            res = diag_v1.run_mc_jobs(tr.eval_envs, N, jobs, policy, contract, t_extra, tr.H, step_dim,
                                      stop=lambda: diag_stop(tr))
        except diag_v1.Aborted:
            diag_exit_preempted(tr, done)
        part = dict(ident=ident, part=p, results=res, wall_s=time.time() - t0)
        write_pickle(path, part)
        results.update(res)
        walls.append(part['wall_s'])
        done += 1
        print(f'DIAG_PART mc part={p} queries={len(jobs) // K} continuations={len(jobs)} '
              f'wall_s={part["wall_s"]:.1f}', flush=True)
    tr.close_envs()
    out = []
    for q in qs:
        runs = [results[q['qid'], k] for k in range(K)]
        row = dict(qid=q['qid'], state=q['state'], label=q['label'], meta=q.get('meta'), K=K,
                   steps=[r['steps'] for r in runs], success=[r['success'] for r in runs],
                   decisions=[r['decisions'] for r in runs],
                   first_actor_digest=[r['first_actor_digest'] for r in runs],
                   final_obs_digest=[r['final_obs_digest'] for r in runs])
        if contract == 'bsc':
            row['G'] = [r['G'] for r in runs]
            row['mean'], row['se'] = diag_v1.mean_se(row['G'])
        else:
            for k in ('G_cut', 'G_ext', 'tail_bound', 'success_before_limit', 'steps_past_limit'):
                row[k] = [r[k] for r in runs]
            row['mean'], row['se'] = diag_v1.mean_se(row['G_ext'])
            row['mean_cut'], row['se_cut'] = diag_v1.mean_se(row['G_cut'])
            row['max_tail_bound'] = max(row['tail_bound'])
        out.append(row)
    res = dict(ident=ident, queries=out, wall_s=float(sum(walls)))
    write_atomic(os.path.join(FLAGS.run_dir, 'mc.json'), (json.dumps(res, indent=1, default=str) + '\n').encode())
    diag_manifest(ident, complete=True, counts=dict(queries=len(qs), continuations=len(qs) * K),
                  outputs=['mc.json'])
    print(f'DIAG_MC queries={len(qs)} K={K} contract={contract} mean_of_means={np.mean([r["mean"] for r in out]):.4f}',
          flush=True)


def diag_score(tr, source):
    qf, qinfo = load_queries(FLAGS.queries)
    contract = value_contract(tr.config)
    R = FLAGS.score_keys if sampled_readout(tr.config) else 0
    keys = [('PRNGKey(0)', jax.random.PRNGKey(0))] + \
        [(f'fold_in(PRNGKey({SCORE_SEED}), {r})', jax.random.fold_in(jax.random.PRNGKey(SCORE_SEED), r)) for r in range(R)]
    ident = dict(mode='diag_score', source=source, queries=qinfo, env_name=FLAGS.env_name, horizon_length=tr.H,
                 contract=contract, keys=[k for k, _ in keys])
    diag_manifest(ident)
    path = os.path.join(FLAGS.run_dir, 'scores.json')
    if os.path.exists(path):
        print('DIAG_SCORE_SKIP (scores.json present)', flush=True)
        return
    qs = qf['queries']
    obs = np.stack([qf['states'][q['state']]['obs'] for q in qs])
    acts = np.stack([q['chunk'] for q in qs])
    outs = [critic_outputs(tr.agent, obs, acts, key, contract) for _, key in keys]
    rows = []
    for i, q in enumerate(qs):
        row = dict(qid=q['qid'], state=q['state'], label=q['label'], **_row(outs[0], i))
        if R:
            row['keys'] = {k: [_row(o, i)[k] for o in outs] for k in outs[0]}
            v = np.array(row['keys']['value'], np.float64)
            row['value_sd_over_keys'] = float(v.std(ddof=1))
        rows.append(row)
    write_atomic(path, (json.dumps(dict(ident=ident, queries=rows), indent=1, default=str) + '\n').encode())
    diag_manifest(ident, complete=True, counts=dict(queries=len(qs), keys=len(keys)), outputs=['scores.json'])
    print(f'DIAG_SCORE queries={len(qs)} keys={len(keys)}', flush=True)


def diag_select(tr, source):
    from utils import diag_v1
    from visualization.critic_probe import make_candidates
    qf, qinfo = load_queries(FLAGS.queries)
    contract = value_contract(tr.config)
    N = tr.config['actor_num_samples']
    n_sub = FLAGS.mc_candidates
    if not 0 <= n_sub <= N:
        diag_refuse(f'--mc_candidates={n_sub} not in [0, {N}]')
    ident = dict(mode='diag_select', source=source, panel=qinfo, env_name=FLAGS.env_name, horizon_length=tr.H,
                 contract=contract, sel_seed=FLAGS.sel_seed, candidates=N, mc_candidates=n_sub,
                 readout_key='PRNGKey(0)')
    diag_manifest(ident)
    if os.path.exists(os.path.join(FLAGS.run_dir, 'select.json')):
        print('DIAG_SELECT_SKIP (select.json present)', flush=True)
        return
    adim = tr.config['action_dim'] * tr.H
    S = len(qf['states'])
    obs = np.stack([s['obs'] for s in qf['states']]).astype(np.float32)
    noise = np.stack([np.asarray(jax.random.normal(diag_v1.selection_noise_key(FLAGS.sel_seed, i), (N, adim)))
                      for i in range(S)]).astype(np.float32)
    cands = np.clip(make_candidates(tr.agent, obs, noise), -1, 1).astype(np.float32)  # as agent._candidates
    sc = critic_outputs(tr.agent, obs, cands, jax.random.PRNGKey(0), contract, per_state=True)  # each (S, N)
    states, queries, rows = [], [], []
    for i in range(S):
        chosen = int(np.argmax(sc['score'][i]))  # first index on ties, as best-of-n
        uni = diag_v1.uniform_index(FLAGS.sel_seed, i, N)
        sub = diag_v1.candidate_subset(FLAGS.sel_seed, i, N, n_sub, chosen) if n_sub else []
        states.append(qf['states'][i])
        for label, j in [('chosen', chosen), ('uniform', uni)] + [('cand', j) for j in sub]:
            queries.append(dict(qid=len(queries), state=i, label=label, chunk=cands[i, j],
                                meta=dict(panel_state=i, candidate=j, panel_meta=_panel_meta(qf, i),
                                          critic={k: sc[k][i, j].item() for k in sc})))
        top = sc['score'][i] == sc['score'][i].max()
        rows.append(dict(state=i, chosen=chosen, uniform=uni, subset=sub, ties_at_max=int(top.sum()),
                         **{k: sc[k][i].tolist() for k in sc}))
    write_pickle(os.path.join(FLAGS.run_dir, 'select_queries.pkl'), query_file(states, queries, source, ident=ident))
    write_atomic(os.path.join(FLAGS.run_dir, 'select.json'),
                 (json.dumps(dict(ident=ident, states=rows), indent=1, default=str) + '\n').encode())
    diag_manifest(ident, complete=True, counts=dict(states=S, queries=len(queries)),
                  outputs=['select.json', 'select_queries.pkl'],
                  queries_sha256=file_sha256(os.path.join(FLAGS.run_dir, 'select_queries.pkl')))
    print(f'DIAG_SELECT states={S} queries={len(queries)} chosen_eq_uniform={sum(r["chosen"] == r["uniform"] for r in rows)}',
          flush=True)


def _panel_meta(qf, state):
    for q in qf['queries']:
        if q['state'] == state:
            return {k: v for k, v in (q.get('meta') or {}).items() if k != 'critic'}
    return None


def diag_main():
    if FLAGS.mode == 'diag_anchors' and FLAGS.policy_ckpt:
        if FLAGS.src_run_dir:
            diag_refuse('--policy_ckpt and --src_run_dir are exclusive')
        tr, source = diag_policy_agent()
    else:
        tr, source = diag_frozen_agent()
    if FLAGS.mode != 'diag_anchors' and not FLAGS.queries:
        diag_refuse('--queries is required')
    try:
        dict(diag_anchors=diag_anchors, diag_mc=diag_mc, diag_score=diag_score, diag_select=diag_select)[FLAGS.mode](
            tr, source)
    except SystemExit:
        raise
    except BaseException:
        exit_on_exception(tr)
    tr.close_envs()


def main(_):
    assert FLAGS.run_dir or FLAGS.mode == 'make_probe_candidates', '--run_dir is required'
    if FLAGS.mode == 'arb_introspect':
        arb_introspect()
        return
    if FLAGS.mode == 'make_probe_candidates':
        make_probe_candidates()
        return
    if FLAGS.mode == 'probe_ckpt':
        probe_ckpt()
        return
    if FLAGS.mode == 'final_eval':
        final_eval()
        return
    if FLAGS.mode == 'switch_eval':
        final_eval(runner=run_switch_eval)
        return
    if FLAGS.mode == 'episode_stages':
        final_eval(runner=run_episode_stages)
        return
    if FLAGS.mode == 'value_trace':
        final_eval(runner=run_value_trace)
        return
    if FLAGS.mode.startswith('diag_'):
        diag_main()
        return
    if FLAGS.mode == 'eval_actor':
        assert FLAGS.init_actor, '--init_actor is required for --mode=eval_actor'
        trainer = Trainer()
        try:
            trainer.setup()
            trainer.eval_actor()
        except SystemExit:
            raise
        except BaseException:
            exit_on_exception(trainer)
        return
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
