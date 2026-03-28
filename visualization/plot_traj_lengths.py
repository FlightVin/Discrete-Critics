"""
Plot trajectory length distribution for a robomimic dataset.
Shows both the dones-based lengths (from get_dataset) and the true HDF5 episode lengths side-by-side.

Usage:
    python runscripts/plot_traj_lengths.py [--env_name=square-mh-low_dim]
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robomimic_utils import _check_dataset_exists, get_max_episode_length

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="square-mh-low_dim")
parser.add_argument("--out", default="runscripts/plots/traj_lengths.png")
args = parser.parse_args()

dataset_path = _check_dataset_exists(args.env_name)
max_ep_len   = get_max_episode_length(args.env_name)

# --- True episode lengths directly from HDF5 ---
f = h5py.File(dataset_path, 'r')
demos = sorted(list(f['data'].keys()), key=lambda x: int(x[5:]))
true_lengths = np.array([f[f'data/{ep}/actions'].shape[0] for ep in demos])
f.close()

# --- Fixed dones (as get_dataset now produces: terminal only at last step per episode) ---
# terminal_locs are the last indices of each episode: cumsum(lengths) - 1
fixed_terminal_locs = np.cumsum(true_lengths) - 1
traj_endpoints = np.concatenate([[0], fixed_terminal_locs + 1])
dones_lengths = np.diff(traj_endpoints)

def print_stats(name, lengths):
    n_over = (lengths > max_ep_len).sum()
    print(f"\n--- {name} ---")
    print(f"{len(lengths)} trajectories, max_ep_len={max_ep_len}")
    print(f"Min: {lengths.min()}  Max: {lengths.max()}  Mean: {lengths.mean():.1f}")
    print(f"Length == 1:  {(lengths == 1).sum()}")
    print(f"Length <= 5:  {(lengths <= 5).sum()}")
    print(f"Longer than max_ep_len: {n_over} ({100*n_over/len(lengths):.1f}%)")

print_stats("True HDF5 episode lengths", true_lengths)
print_stats("Fixed dones lengths (get_dataset after fix)", dones_lengths)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, lengths, title in [
    (axes[0], true_lengths,  f"True HDF5 episode lengths — {args.env_name}"),
    (axes[1], dones_lengths, f"Fixed dones lengths (get_dataset after fix) — {args.env_name}"),
]:
    n_over = (lengths > max_ep_len).sum()
    ax.hist(lengths, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(max_ep_len, color='red',    linestyle='--', linewidth=1.2, label=f'max_ep_len={max_ep_len}')
    ax.axvline(lengths.mean(), color='orange', linestyle='--', linewidth=1.2, label=f'mean={lengths.mean():.0f}')
    ax.axvline(lengths.min(), color='green',  linestyle=':',  linewidth=1.2, label=f'min={lengths.min()}')
    ax.axvline(lengths.max(), color='purple', linestyle=':',  linewidth=1.2, label=f'max={lengths.max()}')
    ax.set_title(title)
    ax.set_xlabel('steps')
    ax.set_ylabel('# trajectories')
    ax.legend()
    ax.text(0.98, 0.95, f'n={len(lengths)}  >max_ep_len: {n_over} ({100*n_over/len(lengths):.1f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f"\nSaved → {args.out}")
