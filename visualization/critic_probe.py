"""Probe saved critics: does Q rank actions and states, or is it flat?

For each params_<step>.pkl in --run_dir, on a fixed set of dataset states (same states for every
checkpoint), compute:
  q_data            Q of the dataset's own action chunk
  progress_spearman Spearman(q_data, -steps left in the demo): does Q rank states by progress?
  cand_*            Q over the best-of-n candidates from the BC flow policy (the actor's choice set)
  cand_rel_spread   mean within-state candidate std / across-state std of q_data (scale-free)
  data_gt_random    fraction of states with Q(dataset chunk) > Q(uniform random chunk)
Q uses the same aggregation over the ensemble as best-of-n (config q_agg). Writes one JSON line per
checkpoint to --out.

Usage (from the project root):
  python visualization/critic_probe.py --run_dir <save_dir leaf with flags.json> --out probe.jsonl
"""
import argparse
import glob
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scipy.stats import spearmanr

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent_with_file


def q_values(agent, observations, actions, rng):
    """Ensemble-aggregated Q, as best-of-n uses it.

    Agents with critic_value (accat, achlg: expected value over the bin support; acarb: with rng) use it;
    acbsc's critic outputs logits (sigmoid), acfql's the Q itself."""
    name = agent.config['agent_name']
    if hasattr(agent, 'critic_value'):
        if 'rng' in inspect.signature(agent.critic_value).parameters:
            qs = agent.critic_value(observations, actions, rng=rng)
        else:
            qs = agent.critic_value(observations, actions)
    elif name == 'acbsc':
        qs = jax.nn.sigmoid(agent.network.select('critic')(observations, actions))
    elif name == 'acfql':
        qs = agent.network.select('critic')(observations, actions)
    else:
        raise ValueError(f'unsupported agent {name}')
    return qs.min(axis=0) if agent.config['q_agg'] == 'min' else qs.mean(axis=0)


@jax.jit
def probe_batch(agent, obs, data_actions, cand_noises, rand_actions, rng):
    n = agent.config['actor_num_samples']
    q_data = q_values(agent, obs, data_actions, rng)
    obs_rep = jnp.repeat(obs[:, None, :], n, axis=1)
    cand = agent.compute_flow_actions(obs_rep, cand_noises)
    q_cand = q_values(agent, obs_rep, cand, rng)
    q_rand = q_values(agent, obs, rand_actions, rng)
    return q_data, q_cand, q_rand


def uses_integer_score(agent):
    return agent.config['agent_name'] == 'acarb' and agent.config['arb_readout'] == 'sample_mean'


def ensemble_score(agent, observations, actions, rng):
    """(score, value, oor_frac) of the final ensemble readout, shape (...).

    score is what best-of-n ranks by: the integer S for acarb's sample_mean readout (value S / K_den),
    else the aggregated q_values (score == value). oor_frac: decoded codes outside [0, B^L - 1] (sample_mean
    only, else 0). Ties are exact equality of score."""
    if uses_integer_score(agent):
        s, oor = agent.critic_score(observations, actions, rng=rng)
        return s, s.astype(jnp.float32) / agent.readout_den(), oor
    q = q_values(agent, observations, actions, rng)
    return q, q, jnp.zeros_like(q)


@jax.jit
def probe_batch_scores(agent, obs, data_actions, cand_noises, rand_actions, rng):
    """probe_batch with the readout's final ensemble score of the candidates: (q_data, q_cand, q_rand, s_cand)."""
    n = agent.config['actor_num_samples']
    q_data = ensemble_score(agent, obs, data_actions, rng)[1]
    obs_rep = jnp.repeat(obs[:, None, :], n, axis=1)
    cand = agent.compute_flow_actions(obs_rep, cand_noises)
    s_cand, q_cand, _ = ensemble_score(agent, obs_rep, cand, rng)
    q_rand = ensemble_score(agent, obs, rand_actions, rng)[1]
    return q_data, q_cand, q_rand, s_cand


@jax.jit
def fixed_scores(agent, obs, cands, rng):
    """ensemble_score of fixed candidates cands (n, N, adim) at states obs (n, obs_dim)."""
    obs_rep = jnp.repeat(obs[:, None, :], cands.shape[1], axis=1)
    return ensemble_score(agent, obs_rep, cands, rng)


def tie_stats(s):
    """(tie_all, tie_max) of scores (n, N): all N equal / >= 2 candidates share the maximum."""
    s = np.asarray(s)
    top = s.max(1, keepdims=True)
    return float((s.max(1) == s.min(1)).mean()), float(((s == top).sum(1) >= 2).mean())


STAB_SEED, STAB_REPEATS, STAB_STRIDE = 20261022, 5, 8


def probe_fixed(agent, obs, cands, batch=256):
    """Probe on fixed candidates (shared across runs): probe_fixed/* and probe_stab/* metrics.

    Main pass with key PRNGKey(0) (as the current-actor probe). Stability: states 0, 8, 16, ... scored
    STAB_REPEATS times with keys fold_in(PRNGKey(STAB_SEED), rep), rep = 0..4; per-candidate variance of
    the ensemble value across repeats (averaged), and the fraction of (state, repeat >= 1) whose first-index
    argmax of the score equals repeat 0's. Uses only these probe keys (no training RNG)."""
    parts = [fixed_scores(agent, obs[s:s + batch], cands[s:s + batch], jax.random.PRNGKey(0))
             for s in range(0, len(obs), batch)]
    score, value, oor = (np.concatenate([np.asarray(p[k]) for p in parts]) for k in range(3))
    tie_all, tie_max = tie_stats(score)
    out = {'probe_fixed/tie_all': tie_all, 'probe_fixed/tie_max': tie_max,
           'probe_fixed/score_mean': float(value.mean()),
           'probe_fixed/cand_range_mean': float((value.max(1) - value.min(1)).mean()),
           'probe_fixed/oor_frac': float(oor.mean())}
    sub = np.arange(0, len(obs), STAB_STRIDE)
    reps = [fixed_scores(agent, obs[sub], cands[sub], jax.random.fold_in(jax.random.PRNGKey(STAB_SEED), r))
            for r in range(STAB_REPEATS)]
    values = np.stack([np.asarray(r[1]) for r in reps])           # (reps, n_sub, N)
    argmax = np.stack([np.asarray(r[0]).argmax(-1) for r in reps])  # (reps, n_sub), first index on ties
    out['probe_stab/score_var_mean'] = float(values.astype(np.float64).var(0).mean())
    out['probe_stab/same_argmax_frac'] = float((argmax[1:] == argmax[0]).mean())
    return out


def make_candidates(agent, obs, noises, batch=256):
    """The agent's BC flow actions for fixed noises (n, N, adim) at states obs (n, obs_dim), as probe_batch."""
    f = jax.jit(lambda a, o, z: a.compute_flow_actions(jnp.repeat(o[:, None, :], z.shape[1], axis=1), z))
    return np.concatenate([np.asarray(f(agent, obs[s:s + batch], noises[s:s + batch]))
                           for s in range(0, len(obs), batch)])


def probe_states(ds, horizon, num_states, seed):
    """Fixed chunk start indices that stay inside one trajectory, and steps left in that demo."""
    terminals = np.asarray(ds['terminals']) > 0
    ends = np.flatnonzero(terminals)
    end_of = ends[np.searchsorted(ends, np.arange(len(terminals)))]  # terminal index of each step's demo
    starts = np.arange(len(terminals) - horizon + 1)
    starts = starts[end_of[starts] >= starts + horizon - 1]  # chunk does not cross the demo end
    rs = np.random.RandomState(seed)
    idx = np.sort(rs.choice(starts, size=num_states, replace=False))
    steps_left = end_of[idx] - idx + 1
    return idx, steps_left


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--num_states', type=int, default=1024)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    with open(os.path.join(args.run_dir, 'flags.json')) as f:
        flags = json.load(f)
    config = ml_collections.ConfigDict(flags['agent'])
    config['horizon_length'] = flags['horizon_length']
    H = flags['horizon_length']

    _, _, ds, _ = make_env_and_datasets(flags['env_name'])
    obs_all, act_all = np.asarray(ds['observations']), np.asarray(ds['actions'])
    idx, steps_left = probe_states(ds, H, args.num_states, args.seed)
    obs = obs_all[idx]
    data_actions = np.stack([act_all[i:i + H].reshape(-1) for i in idx])
    adim = data_actions.shape[-1]

    rs = np.random.RandomState(args.seed + 1)  # same noise and random chunks for every checkpoint
    cand_noises = rs.standard_normal((len(idx), config['actor_num_samples'], adim)).astype(np.float32)
    rand_actions = rs.uniform(-1, 1, (len(idx), adim)).astype(np.float32)

    agent = agents[config['agent_name']].create(args.seed, obs_all[0], act_all[0], config)
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, 'params_*.pkl')),
                   key=lambda s: int(re.search(r'params_(\d+)\.pkl', s).group(1)))
    assert ckpts, f'no checkpoints in {args.run_dir}'

    with open(args.out, 'w') as out:
        for path in ckpts:
            step = int(re.search(r'params_(\d+)\.pkl', path).group(1))
            agent = restore_agent_with_file(agent, path)
            parts = [probe_batch(agent, obs[s:s + args.batch], data_actions[s:s + args.batch],
                                 cand_noises[s:s + args.batch], rand_actions[s:s + args.batch],
                                 jax.random.PRNGKey(step))
                     for s in range(0, len(idx), args.batch)]
            q_data, q_cand, q_rand = (np.concatenate([np.asarray(x[k]) for x in parts]) for k in range(3))

            across = float(q_data.std())
            cand_std = q_cand.std(axis=1)
            row = dict(
                step=step,
                agent=config['agent_name'],
                env=flags['env_name'],
                num_states=len(idx),
                q_data_mean=float(q_data.mean()),
                q_data_p05=float(np.percentile(q_data, 5)),
                q_data_p50=float(np.percentile(q_data, 50)),
                q_data_p95=float(np.percentile(q_data, 95)),
                q_data_frac_gt_0p9=float((q_data > 0.9).mean()),
                q_data_across_state_std=across,
                progress_spearman=float(spearmanr(q_data, -steps_left).correlation),
                cand_std_mean=float(cand_std.mean()),
                cand_range_mean=float((q_cand.max(1) - q_cand.min(1)).mean()),
                cand_rel_spread=float(cand_std.mean() / max(across, 1e-12)),
                bon_gain_rel=float((q_cand.max(1) - q_cand.mean(1)).mean() / max(across, 1e-12)),
                q_rand_mean=float(q_rand.mean()),
                data_gt_random=float((q_data > q_rand).mean()),
            )
            out.write(json.dumps(row) + '\n')
            out.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
