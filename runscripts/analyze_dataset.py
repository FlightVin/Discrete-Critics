"""
Dataset analysis script for robomimic offline data.

Validates:
  1. Episode structure — which episodes are success vs timeout, and their lengths
  2. Reward distribution — confirms {0, 1} raw / {-1, 0} after shift
  3. terminals vs bsc_terminals — shows that `dones` only marks success boundaries,
     while `bsc_terminals` marks ALL episode ends
  4. BSC TD signal simulation — what fraction of valid sequences would be
     success / timeout / non-terminal under each terminal scheme, and the
     resulting fixed-point Q value
  5. Sequence sampling preview — spot-check that bsc_is_terminal and bsc_is_success
     fire correctly after the fix

Usage:
    cd /home/mananaga/BSC_branches/manan_qc/Binary-Success-Critic
    python runscripts/analyze_dataset.py
"""

import os
import sys
import numpy as np
import h5py

# ── Config ────────────────────────────────────────────────────────────────────

ENV_NAME     = "square-mh-low_dim"
HORIZON      = 5          # horizon_length used in BSC
DISCOUNT     = 0.99       # discount used in sample_sequence
REWARD_SHIFT = -1.0       # robomimic shift applied in process_train_dataset
N_SAMPLE     = 50_000     # sequences to sample for BSC signal simulation
BATCH_SIZE   = 256

# ── Dataset path (mirrors robomimic_utils._check_dataset_exists) ─────────────

def get_dataset_path(env_name):
    task, dataset_type, hdf5_type = env_name.split("-")
    file_name = "low_dim_sparse_v15.hdf5" if dataset_type == "mg" else "low_dim_v15.hdf5"
    return os.path.join("/data/user_data/mananaga/robomimic", task, dataset_type, file_name)


# ── Load raw HDF5 ─────────────────────────────────────────────────────────────

def load_raw(env_name):
    path = get_dataset_path(env_name)
    print(f"Loading: {path}")
    f = h5py.File(path, "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x[5:]))

    episodes = []
    for ep in demos:
        r     = np.array(f[f"data/{ep}/rewards"])
        dones = np.array(f[f"data/{ep}/dones"])
        episodes.append({"name": ep, "rewards": r, "dones": dones, "length": len(r)})

    f.close()
    return episodes


# ── Episode-level analysis ────────────────────────────────────────────────────

def analyze_episodes(episodes):
    print("\n" + "="*70)
    print("  EPISODE-LEVEL ANALYSIS")
    print("="*70)

    n_total   = len(episodes)
    n_success = sum(1 for e in episodes if e["dones"][-1] == 1)
    n_timeout = n_total - n_success

    lengths_all     = [e["length"] for e in episodes]
    lengths_success = [e["length"] for e in episodes if e["dones"][-1] == 1]
    lengths_timeout = [e["length"] for e in episodes if e["dones"][-1] == 0]

    print(f"  Total episodes : {n_total}")
    print(f"  Success (dones[-1]=1) : {n_success}  ({100*n_success/n_total:.1f}%)")
    print(f"  Timeout (dones[-1]=0) : {n_timeout}  ({100*n_timeout/n_total:.1f}%)")
    print()

    def _stats(arr, label):
        if not arr:
            print(f"  {label}: (none)")
            return
        print(f"  {label} length — mean={np.mean(arr):.1f}  min={np.min(arr)}  max={np.max(arr)}  "
              f"median={np.median(arr):.0f}")

    _stats(lengths_all,     "All     ")
    _stats(lengths_success, "Success ")
    _stats(lengths_timeout, "Timeout ")

    # Verify reward structure
    print()
    all_rewards = np.concatenate([e["rewards"] for e in episodes])
    unique_r = np.unique(all_rewards)
    print(f"  Unique raw rewards in dataset: {unique_r}")

    # Check that dones only fires at reward=1 steps (or not)
    all_dones = np.concatenate([e["dones"] for e in episodes])
    done_at_reward1 = ((all_dones == 1) & (all_rewards > 0)).sum()
    done_at_reward0 = ((all_dones == 1) & (all_rewards == 0)).sum()
    print(f"  dones=1 at reward>0 (success) : {done_at_reward1}")
    print(f"  dones=1 at reward=0 (timeout) : {done_at_reward0}")

    # Confirm: for timeout episodes (dones[-1]=0), are ALL dones 0?
    any_done_in_timeout = sum(1 for e in episodes if e["dones"][-1] == 0 and e["dones"].sum() > 0)
    print(f"  Timeout episodes with any dones=1 inside : {any_done_in_timeout}")

    return n_success, n_timeout


# ── Build flat dataset (mimics get_dataset + process_train_dataset) ───────────

def build_flat_dataset(episodes):
    """Returns flat arrays: rewards (shifted+sparse), terminals, bsc_terminals."""
    rewards_list    = []
    terminals_list  = []
    bsc_term_list   = []

    for e in episodes:
        r     = e["rewards"].copy().astype(np.float32)
        dones = e["dones"].copy().astype(np.float32)

        # robomimic shift
        r = r - 1.0
        # sparse: map non-zero to -1
        r = (r != 0.0).astype(np.float32) * -1.0

        # terminals = dones (only success marked)
        # bsc_terminals = dones with last step forced to 1
        ep_boundaries = dones.copy()
        ep_boundaries[-1] = 1.0

        rewards_list.append(r)
        terminals_list.append(dones)
        bsc_term_list.append(ep_boundaries)

    rewards      = np.concatenate(rewards_list)
    terminals    = np.concatenate(terminals_list)
    bsc_terminals = np.concatenate(bsc_term_list)
    return rewards, terminals, bsc_terminals


# ── Simulate sample_sequence + BSC TD signal ─────────────────────────────────

def simulate_bsc_signal(rewards, terminals_arr, label, n_sample=N_SAMPLE, H=HORIZON, discount=DISCOUNT):
    """
    Sample random H-step sequences and compute what fraction would be:
      success, timeout, non-terminal (in the VALID loss).
    Also computes the theoretical Q fixed point.
    """
    N = len(rewards)
    discount_powers = discount ** np.arange(H)

    n_success = n_timeout = n_nonterminal = n_invalid = 0

    rng = np.random.default_rng(0)
    idxs = rng.integers(0, N - H + 1, size=n_sample)

    for idx in idxs:
        chunk_r   = rewards[idx : idx + H]
        chunk_t   = terminals_arr[idx : idx + H]

        # Cumulative discounted reward
        cum_r = np.zeros(H)
        cum_r[0] = chunk_r[0]
        for i in range(1, H):
            cum_r[i] = cum_r[i-1] + chunk_r[i] * discount_powers[i]

        # Propagated terminals
        prop_t = np.zeros(H)
        prop_t[0] = chunk_t[0]
        for i in range(1, H):
            prop_t[i] = max(prop_t[i-1], chunk_t[i])

        # valid[i] = 1 - prop_t[i-1]
        valid_last = 1.0 - (prop_t[H-2] if H > 1 else 0.0)

        if valid_last < 0.5:
            n_invalid += 1
            continue

        last_contrib = cum_r[H-1] - cum_r[H-2] if H > 1 else cum_r[0]
        bsc_is_terminal = prop_t[H-1] > 0.5
        bsc_is_success  = last_contrib > -1e-6

        if bsc_is_terminal and bsc_is_success:
            n_success += 1
        elif bsc_is_terminal and not bsc_is_success:
            n_timeout += 1
        else:
            n_nonterminal += 1

    n_valid = n_success + n_timeout + n_nonterminal
    if n_valid == 0:
        print(f"  [{label}] No valid sequences sampled!")
        return

    s_frac  = n_success    / n_valid
    t_frac  = n_timeout    / n_valid
    nt_frac = n_nonterminal / n_valid
    invalid_frac = n_invalid / n_sample

    # Fixed point: Q* = s_frac / (s_frac + t_frac)  (assuming discount=1 for simplicity)
    denom = s_frac + t_frac
    q_star = s_frac / denom if denom > 0 else float('nan')

    print(f"\n  [{label}]  (sampled {n_sample:,} sequences, H={H})")
    print(f"    invalid (cross-boundary, masked out) : {100*invalid_frac:.1f}%")
    print(f"    valid breakdown:")
    print(f"      success    (target=1.0) : {100*s_frac:.2f}%")
    print(f"      timeout    (target=0.0) : {100*t_frac:.2f}%")
    print(f"      non-terminal (bootstrap) : {100*nt_frac:.2f}%")
    print(f"    → theoretical Q fixed point (discount=1): {q_star:.4f}")


# ── Spot-check: print first few terminal sequences ────────────────────────────

def spot_check_terminals(rewards, terminals_arr, bsc_terminals_arr, H=HORIZON, discount=DISCOUNT, n_show=5):
    print("\n" + "="*70)
    print("  SPOT CHECK — sequences ending at a bsc_terminal boundary")
    print("="*70)

    N = len(rewards)
    discount_powers = discount ** np.arange(H)

    # Find all positions where bsc_terminals=1 that could be at position H-1
    boundary_idxs = np.where(bsc_terminals_arr == 1)[0]
    start_idxs = boundary_idxs - (H - 1)
    start_idxs = start_idxs[(start_idxs >= 0) & (start_idxs + H <= N)]

    shown = 0
    for idx in start_idxs[:200]:
        chunk_r   = rewards[idx : idx + H]
        chunk_t   = terminals_arr[idx : idx + H]
        chunk_bt  = bsc_terminals_arr[idx : idx + H]

        cum_r = np.zeros(H)
        cum_r[0] = chunk_r[0]
        for i in range(1, H):
            cum_r[i] = cum_r[i-1] + chunk_r[i] * discount_powers[i]

        prop_bt = np.zeros(H)
        prop_bt[0] = chunk_bt[0]
        for i in range(1, H):
            prop_bt[i] = max(prop_bt[i-1], chunk_bt[i])

        valid_last = 1.0 - (prop_bt[H-2] if H > 1 else 0.0)
        if valid_last < 0.5:
            continue

        last_contrib = cum_r[H-1] - cum_r[H-2] if H > 1 else cum_r[0]
        bsc_is_terminal = prop_bt[H-1] > 0.5
        bsc_is_success  = last_contrib > -1e-6
        kind = "success" if bsc_is_success else "timeout"

        # Check what OLD terminals would say
        prop_t = np.zeros(H)
        prop_t[0] = chunk_t[0]
        for i in range(1, H):
            prop_t[i] = max(prop_t[i-1], chunk_t[i])
        old_terminal = prop_t[H-1] > 0.5
        old_valid    = 1.0 - (prop_t[H-2] if H > 1 else 0.0)

        print(f"  idx={idx:6d}  rewards={np.round(chunk_r,3)}  cum_r[-1]={cum_r[-1]:.3f}")
        print(f"    dones_in_chunk={chunk_t.tolist()}  bsc_t_in_chunk={chunk_bt.tolist()}")
        print(f"    last_contrib={last_contrib:.4f}  kind={kind}")
        print(f"    OLD: terminal={old_terminal}, valid={old_valid>0.5}  "
              f"→ would appear in loss: {old_terminal and old_valid > 0.5}")
        print(f"    NEW: terminal={bsc_is_terminal}, valid={valid_last>0.5}  "
              f"→ appears in loss: {bsc_is_terminal and valid_last > 0.5}")
        print()

        shown += 1
        if shown >= n_show:
            break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    episodes = load_raw(ENV_NAME)

    n_success, n_timeout = analyze_episodes(episodes)

    rewards, terminals, bsc_terminals = build_flat_dataset(episodes)

    print("\n" + "="*70)
    print("  FLAT DATASET SUMMARY")
    print("="*70)
    print(f"  Total transitions  : {len(rewards):,}")
    print(f"  terminals=1        : {int(terminals.sum()):,}  "
          f"({100*terminals.mean():.2f}%  — success boundaries only)")
    print(f"  bsc_terminals=1    : {int(bsc_terminals.sum()):,}  "
          f"({100*bsc_terminals.mean():.2f}%  — all episode boundaries)")
    print(f"  Extra boundaries added by bsc_terminals: "
          f"{int((bsc_terminals - terminals).clip(0).sum()):,}  (timeout episodes)")

    print("\n" + "="*70)
    print("  BSC TD SIGNAL SIMULATION")
    print("="*70)
    simulate_bsc_signal(rewards, terminals,     "OLD terminals    (dones only)")
    simulate_bsc_signal(rewards, bsc_terminals, "NEW bsc_terminals (all boundaries)")

    spot_check_terminals(rewards, terminals, bsc_terminals)

    print("\nDone.")


if __name__ == "__main__":
    main()
