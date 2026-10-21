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
    """Ensemble-aggregated Q, as best-of-n uses it."""
    name = agent.config['agent_name']
    if name == 'acbsc':
        qs = jax.nn.sigmoid(agent.network.select('critic')(observations, actions))
    elif name == 'acfql':
        qs = agent.network.select('critic')(observations, actions)
    elif name == 'accat':
        qs = agent.critic_value(observations, actions)
    elif name == 'acarb':
        qs = agent.critic_value(observations, actions, rng=rng)
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
