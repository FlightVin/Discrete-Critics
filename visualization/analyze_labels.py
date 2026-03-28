"""
Visualise and sanity-check the new per-transition labels added to the offline dataset:
  - final_outcome        (1.0 success / 0.0 timeout, per trajectory)
  - num_chunks_to_timeout  (chunks left before episode times out)
  - num_chunks_to_success  (chunks left until task success; -1 for failed trajectories)

Checks:
  1. final_outcome distribution across all transitions
  2. num_chunks_to_timeout decreases by 1/H each step within every trajectory
  3. num_chunks_to_success decreases by 1/H each step within every success trajectory,
     and is -1 for every step of failed trajectories
  4. Per-trajectory traces for a random sample of episodes

Usage:
    cd /home/mananaga/BSC_branches/manan_qc/Binary-Success-Critic
    python runscripts/analyze_labels.py [--env_name=square-mh-low_dim] [--horizon=5]
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.robomimic_utils import get_dataset, make_env, is_robomimic_env, get_max_episode_length
from utils.datasets import Dataset

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--env_name",  default="square-mh-low_dim")
parser.add_argument("--horizon",   type=int, default=5)
parser.add_argument("--n_trajs",   type=int, default=6,   help="trajectories to plot in trace panel")
parser.add_argument("--out",       default="runscripts/plots/analyze_labels.png")
args = parser.parse_args()

H        = args.horizon
ENV_NAME = args.env_name

# ── Load + process dataset ────────────────────────────────────────────────────

print(f"Loading {ENV_NAME} ...")
env = make_env(ENV_NAME, seed=0)
ds  = get_dataset(env, ENV_NAME)
ds  = Dataset.create(**ds)

# robomimic reward shift
rewards = np.array(ds['rewards']) - 1.0

ds_dict = {k: np.array(v) for k, v in ds.items()}
ds_dict['rewards'] = rewards
ds = Dataset.create(**ds_dict)

# ── Compute labels (mirrors process_train_dataset in main.py) ─────────────────

ds_dict        = {k: np.array(v) for k, v in ds.items()}
terminal_locs  = np.nonzero(ds_dict['terminals'] > 0)[0]
traj_endpoints = np.concatenate([[0], terminal_locs + 1])
traj_lengths   = np.diff(traj_endpoints)
success_flags  = (ds_dict['rewards'][terminal_locs] > -1e-6).astype(np.float32)

N = ds.size
final_outcomes        = np.zeros(N, dtype=np.float32)
num_chunks_to_timeout = np.zeros(N, dtype=np.float32)
num_chunks_to_success = np.full(N, -1.0, dtype=np.float32)

max_ep_len = get_max_episode_length(ENV_NAME)

for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    traj_len    = end_excl - start
    local_steps = np.arange(traj_len)

    final_outcomes[start:end_excl]        = success_flags[i]
    num_chunks_to_timeout[start:end_excl] = np.ceil((max_ep_len - local_steps) / H)
    if success_flags[i] == 1.0:
        num_chunks_to_success[start:end_excl] = np.ceil((traj_len - local_steps) / H)

n_trajs   = len(traj_lengths)
n_success = int(success_flags.sum())
n_timeout = n_trajs - n_success

print(f"\nDataset: {N} transitions, {n_trajs} trajectories")
print(f"  Success: {n_success} ({100*n_success/n_trajs:.1f}%)")
print(f"  Timeout: {n_timeout} ({100*n_timeout/n_trajs:.1f}%)")
print(f"  Max ep len: {max_ep_len}  |  Horizon H: {H}")
print(f"  Max num_chunks_to_timeout: {num_chunks_to_timeout.max():.0f}  (expect {max_ep_len/H:.0f})")
print(f"  Max num_chunks_to_success (success trajs): {num_chunks_to_success[num_chunks_to_success >= 0].max():.0f}")

# ── Sanity checks ─────────────────────────────────────────────────────────────

errors = 0

# 1. num_chunks_to_timeout: non-increasing (diffs in {-1, 0}), first value = ceil(max_ep_len / H),
#    total number of -1 diffs = first_value - last_value (monotone step-function due to ceil).
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    vals = num_chunks_to_timeout[start:end_excl]
    diffs = np.diff(vals)
    bad_diffs = diffs[~np.isin(diffs, [-1., 0.])]
    if len(bad_diffs) > 0:
        print(f"  [FAIL] traj {i}: num_chunks_to_timeout has unexpected diffs: {np.unique(bad_diffs)}")
        errors += 1
    expected_first = float(np.ceil(max_ep_len / H))
    if vals[0] != expected_first:
        print(f"  [FAIL] traj {i}: num_chunks_to_timeout[0]={vals[0]}, expected {expected_first}")
        errors += 1
    # last value: ceil((max_ep_len - (traj_len-1)) / H)
    traj_len = end_excl - start
    expected_last = float(np.ceil((max_ep_len - (traj_len - 1)) / H))
    if vals[-1] != expected_last:
        print(f"  [FAIL] traj {i}: num_chunks_to_timeout[-1]={vals[-1]}, expected {expected_last}")
        errors += 1

# 2. num_chunks_to_success: success trajs are non-increasing step-function (diffs in {-1, 0}),
#    last value = 1; failure trajs are all -1.
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    vals = num_chunks_to_success[start:end_excl]
    if success_flags[i] == 1.0:
        diffs = np.diff(vals)
        bad_diffs = diffs[~np.isin(diffs, [-1., 0.])]
        if len(bad_diffs) > 0:
            print(f"  [FAIL] success traj {i}: num_chunks_to_success has unexpected diffs: {np.unique(bad_diffs)}")
            errors += 1
        if vals[-1] != 1.0:
            print(f"  [FAIL] success traj {i}: last step num_chunks_to_success={vals[-1]}, expected 1")
            errors += 1
        traj_len = end_excl - start
        expected_first = float(np.ceil(traj_len / H))
        if vals[0] != expected_first:
            print(f"  [FAIL] success traj {i}: num_chunks_to_success[0]={vals[0]}, expected {expected_first}")
            errors += 1
    else:
        if not np.all(vals == -1.0):
            print(f"  [FAIL] timeout traj {i}: num_chunks_to_success has non -1 values")
            errors += 1

# 3. final_outcome matches success_flags
for i, (start, end_excl) in enumerate(zip(traj_endpoints[:-1], traj_endpoints[1:])):
    if not np.all(final_outcomes[start:end_excl] == success_flags[i]):
        print(f"  [FAIL] traj {i}: final_outcome not uniform within trajectory")
        errors += 1

if errors == 0:
    print("\n  [PASS] All sanity checks passed.")
else:
    print(f"\n  [{errors} FAILURES]")

# ── Plots ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(args.out), exist_ok=True)

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"Label Analysis — {ENV_NAME}  (H={H})", fontsize=13, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1: final_outcome distribution ---
ax = fig.add_subplot(gs[0, 0])
counts = [n_success, n_timeout]
ax.bar(['success\n(1.0)', 'timeout\n(0.0)'], counts, color=['steelblue', 'salmon'])
for bar, c in zip(ax.patches, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{c}\n({100*c/n_trajs:.1f}%)', ha='center', va='bottom', fontsize=9)
ax.set_title('final_outcome (trajectory counts)')
ax.set_ylabel('# trajectories')

# --- Panel 2: trajectory length distribution ---
ax = fig.add_subplot(gs[0, 1])
ax.hist(traj_lengths, bins=40, color='steelblue', edgecolor='white', linewidth=0.3)
ax.axvline(traj_lengths.mean(), color='orange', linestyle='--', label=f'mean={traj_lengths.mean():.0f}')
ax.set_title('Trajectory length distribution')
ax.set_xlabel('steps')
ax.set_ylabel('# trajectories')
ax.legend(fontsize=8)

# --- Panel 3: num_chunks_to_timeout histogram ---
ax = fig.add_subplot(gs[0, 2])
ax.hist(num_chunks_to_timeout, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
ax.set_title('num_chunks_to_timeout (all transitions)')
ax.set_xlabel('chunks')
ax.set_ylabel('# transitions')

# --- Panel 4: num_chunks_to_success histogram (success trajs only) ---
ax = fig.add_subplot(gs[1, 0])
success_vals = num_chunks_to_success[num_chunks_to_success >= 0]
ax.hist(success_vals, bins=50, color='steelblue', edgecolor='white', linewidth=0.3)
ax.set_title('num_chunks_to_success\n(success trajectories only)')
ax.set_xlabel('chunks')
ax.set_ylabel('# transitions')

# --- Panel 5: num_chunks_to_success vs num_chunks_to_timeout scatter (success trajs) ---
ax = fig.add_subplot(gs[1, 1])
mask = num_chunks_to_success >= 0
ax.scatter(num_chunks_to_timeout[mask], num_chunks_to_success[mask],
           alpha=0.02, s=1, color='steelblue', rasterized=True)
lim = max(num_chunks_to_timeout.max(), success_vals.max())
ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='y=x')
ax.set_title('timeout vs success chunks\n(success trajs)')
ax.set_xlabel('num_chunks_to_timeout')
ax.set_ylabel('num_chunks_to_success')
ax.legend(fontsize=8)

# --- Panel 6: num_chunks_to_success == -1 indicator ---
ax = fig.add_subplot(gs[1, 2])
frac_neg1 = (num_chunks_to_success == -1).mean()
ax.bar(['-1\n(timeout traj)', '≥1\n(success traj)'],
       [frac_neg1, 1 - frac_neg1], color=['salmon', 'steelblue'])
ax.set_title('num_chunks_to_success\n(-1 fraction = timeout transitions)')
ax.set_ylabel('fraction of all transitions')
ax.set_ylim(0, 1)

# --- Panels 7-9: per-trajectory traces for random sample of trajectories ---
rng = np.random.default_rng(0)
sample_idxs = rng.choice(n_trajs, size=min(args.n_trajs, n_trajs), replace=False)

ax_timeout = fig.add_subplot(gs[2, 0])
ax_success = fig.add_subplot(gs[2, 1])
ax_outcome = fig.add_subplot(gs[2, 2])

for idx in sample_idxs:
    start, end_excl = traj_endpoints[idx], traj_endpoints[idx + 1]
    steps = np.arange(end_excl - start)
    color = 'steelblue' if success_flags[idx] == 1.0 else 'salmon'
    label = 'success' if success_flags[idx] == 1.0 else 'timeout'

    ax_timeout.plot(steps, num_chunks_to_timeout[start:end_excl], color=color, alpha=0.7, linewidth=1)
    ax_success.plot(steps, num_chunks_to_success[start:end_excl], color=color, alpha=0.7, linewidth=1)
    ax_outcome.plot(steps, final_outcomes[start:end_excl], color=color, alpha=0.7, linewidth=1)

# legend proxy
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='steelblue', label='success'),
    Line2D([0], [0], color='salmon',    label='timeout'),
]
for ax, title, ylabel in [
    (ax_timeout, 'num_chunks_to_timeout traces', 'chunks'),
    (ax_success, 'num_chunks_to_success traces\n(-1 = timeout traj)', 'chunks'),
    (ax_outcome, 'final_outcome traces', 'value'),
]:
    ax.set_title(title)
    ax.set_xlabel('step within trajectory')
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles, fontsize=8)

plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f"\nSaved figure → {args.out}")
