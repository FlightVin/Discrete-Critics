"""
Inspect the dones field of a robomimic dataset to diagnose length-1 trajectories.

Usage:
    python runscripts/inspect_dones.py [--env_name=square-mh-low_dim]
"""

import os, sys, argparse
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robomimic_utils import _check_dataset_exists

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="square-mh-low_dim")
parser.add_argument("--n_demos", type=int, default=10)
args = parser.parse_args()

dataset_path = _check_dataset_exists(args.env_name)
f = h5py.File(dataset_path, 'r')
demos = sorted(list(f['data'].keys()), key=lambda x: int(x[5:]))

print(f"Total demos: {len(demos)}")
print(f"\n--- First {args.n_demos} demos ---")
for ep in demos[:args.n_demos]:
    dones = np.array(f[f'data/{ep}/dones'])
    rewards = np.array(f[f'data/{ep}/rewards'])
    n_done = int(dones.sum())
    print(f"{ep}: len={len(dones):4d}, sum(dones)={n_done:3d}, "
          f"dones[:3]={dones[:3].tolist()}, dones[-3:]={dones[-3:].tolist()}, "
          f"sum(rewards)={rewards.sum():.1f}")

print("\n--- Global stats across ALL demos ---")
all_dones = np.concatenate([np.array(f[f'data/{ep}/dones']) for ep in demos])
terminal_locs = np.nonzero(all_dones > 0)[0]
traj_endpoints = np.concatenate([[0], terminal_locs + 1])
if traj_endpoints[-1] < len(all_dones):
    traj_endpoints = np.concatenate([traj_endpoints, [len(all_dones)]])
traj_lengths = np.diff(traj_endpoints)

print(f"Total timesteps: {len(all_dones)}")
print(f"Total terminals (dones=1): {(all_dones > 0).sum()}")
print(f"Expected terminals (1 per demo): {len(demos)}")
print(f"Extra terminals: {(all_dones > 0).sum() - len(demos)}")
print(f"\nTrajectory count: {len(traj_lengths)}")
print(f"Length == 1:  {(traj_lengths == 1).sum()}")
print(f"Length <= 5:  {(traj_lengths <= 5).sum()}")
print(f"Min: {traj_lengths.min()}  Max: {traj_lengths.max()}  Mean: {traj_lengths.mean():.1f}")

# Show where consecutive dones occur
consec = np.where(np.diff(terminal_locs) == 1)[0]
if len(consec) > 0:
    print(f"\nConsecutive terminal locs (cause of length-1): {len(consec)} occurrences")
    print(f"First few: {terminal_locs[consec[:5]]} -> {terminal_locs[consec[:5]+1]}")
else:
    print("\nNo consecutive terminals found.")

f.close()
