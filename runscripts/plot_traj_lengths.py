"""
Plot trajectory length distribution for a robomimic dataset.

Usage:
    python runscripts/plot_traj_lengths.py [--env_name=square-mh-low_dim]
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robomimic_utils import get_dataset, make_env, get_max_episode_length

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="square-mh-low_dim")
parser.add_argument("--out", default="runscripts/plots/traj_lengths.png")
args = parser.parse_args()

print(f"Loading {args.env_name} ...")
env = make_env(args.env_name, seed=0)
ds  = get_dataset(env, args.env_name)

terminals      = np.array(ds['terminals'])
terminal_locs  = np.nonzero(terminals > 0)[0]
traj_endpoints = np.concatenate([[0], terminal_locs + 1])
# Include any trailing steps after the last terminal as a final trajectory
if traj_endpoints[-1] < ds.size:
    traj_endpoints = np.concatenate([traj_endpoints, [ds.size]])
traj_lengths   = np.diff(traj_endpoints)
max_ep_len     = get_max_episode_length(args.env_name)

n_over = (traj_lengths > max_ep_len).sum()
print(f"{len(traj_lengths)} trajectories, max_ep_len={max_ep_len}")
print(f"Longer than max_ep_len: {n_over} ({100*n_over/len(traj_lengths):.1f}%)")
print(f"Min: {traj_lengths.min()}  Max: {traj_lengths.max()}  Mean: {traj_lengths.mean():.1f}")
print(f"Length == 1:  {(traj_lengths == 1).sum()}")
print(f"Length <= 5:  {(traj_lengths <= 5).sum()}")
print(f"Length <= 17: {(traj_lengths <= 17).sum()}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(traj_lengths, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
ax.axvline(max_ep_len, color='red', linestyle='--', linewidth=1.2, label=f'max_ep_len={max_ep_len}')
ax.axvline(traj_lengths.mean(), color='orange', linestyle='--', linewidth=1.2, label=f'mean={traj_lengths.mean():.0f}')
ax.axvline(traj_lengths.min(), color='green', linestyle=':', linewidth=1.2, label=f'min={traj_lengths.min()}')
ax.axvline(traj_lengths.max(), color='purple', linestyle=':', linewidth=1.2, label=f'max={traj_lengths.max()}')
ax.set_title(f'Trajectory length distribution — {args.env_name}')
ax.set_xlabel('steps')
ax.set_ylabel('# trajectories')
ax.legend()
ax.text(0.98, 0.95, f'n={len(traj_lengths)}  >max_ep_len: {n_over} ({100*n_over/len(traj_lengths):.1f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f"Saved → {args.out}")
