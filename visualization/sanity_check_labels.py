"""
Sanity checks for num_chunks_to_timeout and num_chunks_to_success after dataset processing.

Checks:
  1. num_chunks_to_timeout >= 0 everywhere
  2. num_chunks_to_success <= num_chunks_to_timeout for all success transitions
  3. num_chunks_to_timeout is non-increasing within each trajectory
  4. num_chunks_to_success is non-increasing within each success trajectory, ending at 1
  5. final_outcome is uniform within each trajectory

Usage:
    python visualization/sanity_check_labels.py [--env_name=square-mh-low_dim] [--horizon_length=5]
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robomimic_utils import get_dataset, make_env, get_max_episode_length, is_robomimic_env
from utils.datasets import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="square-mh-low_dim")
parser.add_argument("--horizon_length", type=int, default=5)
args = parser.parse_args()

H = args.horizon_length
max_ep_len = get_max_episode_length(args.env_name)

print(f"Loading {args.env_name} ...")
env = make_env(args.env_name, seed=0)
ds = get_dataset(env, args.env_name)
ds = Dataset.create(**ds)

# Apply reward shift and compute labels (mirrors main.py process_train_dataset)
ds_dict = {k: np.array(v) for k, v in ds.items()}
ds_dict["rewards"] = ds_dict["rewards"] - 1.0  # robomimic -1 shift

terminal_locs = np.nonzero(ds_dict['terminals'] > 0)[0]
traj_endpoints = np.concatenate([[0], terminal_locs + 1])
traj_lengths = np.diff(traj_endpoints)
success_flags = (ds_dict['rewards'][terminal_locs] > -1e-6).astype(np.float32)

final_outcomes = np.zeros(ds.size, dtype=np.float32)
final_outcomes[:traj_endpoints[-1]] = np.repeat(success_flags, traj_lengths)

num_chunks_to_timeout = np.zeros(ds.size, dtype=np.float32)
num_chunks_to_success = np.full(ds.size, -1.0, dtype=np.float32)
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    traj_len = end_excl - start
    local_steps = np.arange(traj_len)
    num_chunks_to_timeout[start:end_excl] = np.ceil((max_ep_len - local_steps) / H)
    if success_flags[i] == 1.0:
        num_chunks_to_success[start:end_excl] = np.ceil((traj_len - local_steps) / H)

num_chunks_to_timeout = np.maximum(num_chunks_to_timeout, 0)
success_mask = num_chunks_to_success >= 0
num_chunks_to_success[success_mask] = np.minimum(
    num_chunks_to_success[success_mask], num_chunks_to_timeout[success_mask]
)

n_traj = len(traj_lengths)
n_success = int(success_flags.sum())
print(f"\nDataset: {ds.size} transitions, {n_traj} trajectories, {n_success} successful")
print(f"horizon_length={H}, max_ep_len={max_ep_len}")

failures = []

# --- Check 1: num_chunks_to_timeout >= 0 ---
n_neg = (num_chunks_to_timeout < 0).sum()
if n_neg > 0:
    failures.append(f"[FAIL] Check 1: num_chunks_to_timeout < 0 in {n_neg} transitions")
else:
    print("[PASS] Check 1: num_chunks_to_timeout >= 0 everywhere")

# --- Check 2: num_chunks_to_success <= num_chunks_to_timeout for success transitions ---
bad = success_mask & (num_chunks_to_success > num_chunks_to_timeout)
if bad.any():
    failures.append(f"[FAIL] Check 2: num_chunks_to_success > num_chunks_to_timeout in {bad.sum()} transitions")
else:
    print("[PASS] Check 2: num_chunks_to_success <= num_chunks_to_timeout for all success transitions")

# --- Check 3: num_chunks_to_timeout non-increasing within each trajectory ---
n_bad_traj = 0
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    vals = num_chunks_to_timeout[start:end_excl]
    diffs = np.diff(vals)
    if (diffs > 0).any():
        n_bad_traj += 1
if n_bad_traj > 0:
    failures.append(f"[FAIL] Check 3: num_chunks_to_timeout increases within {n_bad_traj} trajectories")
else:
    print("[PASS] Check 3: num_chunks_to_timeout non-increasing within every trajectory")

# --- Check 4: num_chunks_to_success non-increasing within success trajectories, ends at 1 ---
n_bad_mono, n_bad_end = 0, 0
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    if success_flags[i] != 1.0:
        continue
    vals = num_chunks_to_success[start:end_excl]
    if (np.diff(vals) > 0).any():
        n_bad_mono += 1
    expected_end = min(1.0, num_chunks_to_timeout[end_excl - 1])
    if vals[-1] != expected_end:
        n_bad_end += 1
if n_bad_mono > 0:
    failures.append(f"[FAIL] Check 4a: num_chunks_to_success increases within {n_bad_mono} success trajectories")
else:
    print("[PASS] Check 4a: num_chunks_to_success non-increasing within every success trajectory")
if n_bad_end > 0:
    failures.append(f"[FAIL] Check 4b: num_chunks_to_success does not end at 1 in {n_bad_end} success trajectories")
else:
    print("[PASS] Check 4b: num_chunks_to_success ends at 1 for every success trajectory")

# --- Check 5: final_outcome uniform within each trajectory ---
n_bad = 0
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    if not np.all(final_outcomes[start:end_excl] == success_flags[i]):
        n_bad += 1
if n_bad > 0:
    failures.append(f"[FAIL] Check 5: final_outcome not uniform in {n_bad} trajectories")
else:
    print("[PASS] Check 5: final_outcome uniform within every trajectory")

# --- Summary ---
print(f"\n{'='*50}")
if failures:
    print(f"FAILED ({len(failures)} checks):")
    for f in failures:
        print(f"  {f}")
else:
    print("ALL CHECKS PASSED")

print(f"\nLabel ranges:")
print(f"  num_chunks_to_timeout : min={num_chunks_to_timeout.min():.1f}  max={num_chunks_to_timeout.max():.1f}")
print(f"  num_chunks_to_success : min={num_chunks_to_success[success_mask].min():.1f}  max={num_chunks_to_success[success_mask].max():.1f}  (-1 for failed)")
print(f"  final_outcome         : {final_outcomes.mean()*100:.1f}% success transitions")
