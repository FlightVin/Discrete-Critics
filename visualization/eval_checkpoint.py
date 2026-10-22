"""Evaluate a saved agent with different best-of-n scoring rules.

Rules:
  native  the agent's own sample_actions
  bon     best-of-n reimplemented here, scored like the agent (sigmoid prob for acbsc, Q for acfql)
  logit   best-of-n scored by the mean critic logit (acbsc only; avoids the float32 sigmoid tie)
  n1      one actor sample, no critic (best-of-n with a single candidate)

Before evaluating, the script checks that `bon` reproduces `native` exactly for the same RNG key.
Usage (from the project root):
  python visualization/eval_checkpoint.py --run_dir <dir with flags.json> --step <N> \
      --rules native,n1,logit --eval_episodes 4 --n_eval_envs 32 --out result.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents import agents
from envs.env_utils import make_env_and_datasets, make_vectorized_eval_envs
from evaluation import evaluate_parallel
from utils.flax_utils import restore_agent_with_file


def make_bon(agent, n, score):
    """Best-of-n as in agents/acbsc.py and agents/acfql.py, with a chosen score; n=1 means no critic."""
    adim = agent.config['action_dim'] * agent.config['horizon_length']

    @jax.jit
    def sample(observations, rng):
        noises = jax.random.normal(rng, (*observations.shape[:-1], n, adim))
        obs = jnp.repeat(observations[..., None, :], n, axis=-2)
        cands = jnp.clip(agent.compute_flow_actions(obs, noises), -1, 1)
        if n == 1:
            return cands[..., 0, :]
        out = agent.network.select('critic')(obs, cands)           # (num_qs, ..., n)
        if score == 'prob':
            out = jax.nn.sigmoid(out)
        q = out.mean(axis=0) if agent.config['q_agg'] == 'mean' else out.min(axis=0)
        idx = jnp.argmax(q, axis=-1)
        return jnp.take_along_axis(cands, idx[..., None, None], axis=-2)[..., 0, :]
    return sample


class Policy:
    def __init__(self, fn):
        self.fn = fn

    def sample_actions(self, observations, rng=None):
        return self.fn(observations, rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--rules', default='native,n1')
    p.add_argument('--eval_episodes', type=int, default=4)
    p.add_argument('--n_eval_envs', type=int, default=32)
    p.add_argument('--eval_seed', type=int, default=1000)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    flags = json.load(open(os.path.join(args.run_dir, 'flags.json')))
    config = ml_collections.ConfigDict(flags['agent'])
    config['horizon_length'] = flags['horizon_length']
    name = config['agent_name']
    env, _, ds, _ = make_env_and_datasets(flags['env_name'])
    obs0, act0 = np.asarray(ds['observations'][0]), np.asarray(ds['actions'][0])
    agent = agents[name].create(flags['seed'], obs0, act0, config)
    agent = restore_agent_with_file(agent, os.path.join(args.run_dir, f"params_{args.step}.pkl"))
    score = 'prob' if name == 'acbsc' else 'q'
    n = config['actor_num_samples']

    # Exactness check: reimplemented best-of-n == agent.sample_actions for the same key.
    obs = np.asarray(ds['observations'][:64])
    key = jax.random.PRNGKey(123)
    a_native = np.asarray(agent.sample_actions(observations=obs, rng=key))
    a_bon = np.asarray(make_bon(agent, n, score)(obs, key))
    exact = bool(np.array_equal(a_native, a_bon))
    print(json.dumps(dict(check='bon_equals_native', exact=exact, max_abs=float(np.abs(a_native - a_bon).max()))), flush=True)
    assert exact, 'reimplemented best-of-n differs from the agent; stop'

    action_dim = act0.shape[-1]
    with open(args.out, 'a') as out:
        for rule in args.rules.split(','):
            if rule == 'native':
                pol = agent
            elif rule == 'bon':
                pol = Policy(make_bon(agent, n, score))
            elif rule == 'logit':
                assert name == 'acbsc', 'logit rule is for acbsc only'
                pol = Policy(make_bon(agent, n, 'logit'))
            elif rule == 'n1':
                pol = Policy(make_bon(agent, 1, score))
            else:
                raise ValueError(rule)
            np.random.seed(0)  # same eval RNG key for every rule (evaluate_parallel draws it from np.random)
            envs = make_vectorized_eval_envs(flags['env_name'], args.n_eval_envs, seed=args.eval_seed)
            stats, _, _ = evaluate_parallel(agent=pol, envs=envs, num_eval_envs=args.n_eval_envs,
                                            eval_episodes=args.eval_episodes, action_dim=action_dim)
            envs.close()
            row = dict(agent=name, env=flags['env_name'], step=args.step, rule=rule,
                       episodes=args.n_eval_envs * args.eval_episodes, success=float(stats['success']))
            out.write(json.dumps(row) + '\n'); out.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
