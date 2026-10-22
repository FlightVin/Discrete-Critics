"""Extract the actor modules of a main.py agent checkpoint (params_<N>.pkl) into train_v1's final_actor.pkl format.

  python tests/v1_actor_from_params.py SRC_params.pkl OUT_final_actor.pkl [--horizon_length 5]

The source is only read. agent_name / env_name are taken from the run's flags.json next to it when present.
"""
import argparse
import json
import os
import pickle

import numpy as np


def extract(src, out, horizon_length=None):
    with open(src, 'rb') as f:
        blob = pickle.load(f)
    params = blob['agent']['network']['params']
    actor = {k: _to_numpy(v) for k, v in params.items() if 'critic' not in k}
    flags_path = os.path.join(os.path.dirname(src), 'flags.json')
    run_flags = json.load(open(flags_path)) if os.path.exists(flags_path) else {}
    agent_cfg = run_flags.get('agent', {})
    H = horizon_length or run_flags.get('horizon_length')
    assert H, 'horizon_length unknown: pass --horizon_length'
    res = dict(actor_params=actor, agent_name=agent_cfg.get('agent_name', 'unknown'),
               env_name=run_flags.get('env_name'), horizon_length=int(H), bc_steps=None,
               seed=run_flags.get('seed'), source=os.path.abspath(src))
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(res, f)
    os.replace(tmp, out)
    return res


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    return np.asarray(tree)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--horizon_length', type=int, default=None)
    a = p.parse_args()
    r = extract(a.src, a.out, a.horizon_length)
    print(f'wrote {a.out}: agent_name={r["agent_name"]} env={r["env_name"]} H={r["horizon_length"]} '
          f'modules={sorted(r["actor_params"])}')
