"""
BSC vs QC wandb analysis script.

Fetches training metrics from wandb for BSC and QC runs and prints:
  1. A per-metric table with one column per run, broken into offline/online phases
  2. Time-series plots (one subplot per metric, all runs overlaid)

Metric namespacing (from main.py):
  offline_agent/*   — metrics logged during the offline training phase
  online_agent/*    — metrics logged during the online training phase
  eval/*            — evaluation metrics (logged at eval_interval throughout)

BSC-specific critic metrics:  critic/q, critic/target_q, critic/success_frac, critic/timeout_frac
QC/FQL critic metrics:        critic/q_mean, critic/q_max, critic/q_min
Shared actor metrics:         actor/actor_loss, actor/bc_flow_loss, actor/distill_loss
BSC-only actor metric:        actor/q_loss

Usage: python runscripts/wandb_analysis.py
"""

import datetime
import os
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configure runs here ──────────────────────────────────────────────────────

ENTITY  = 'RL-experiments'
PROJECT = 'BSC'

RUNS = [
    # (wandb run name,  short label for tables/plots)
    ('QC',  'QC'),
    ('BSC', 'BSC'),
]

# Phase boundaries in log_step (continuous across offline + online).
# Default: offline_steps=1_000_000, online_steps=1_000_000.
PHASES = [
    ('Offline   0–1M',   0,         1_000_000),
    ('Online  1M–2M',    1_000_000, 2_000_000),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
EMA        = 0.3    # EMA smoothing for plots (0 = none)
SAMPLES    = 5000   # max history points requested from wandb per metric

# ─── Metrics ──────────────────────────────────────────────────────────────────
# Each entry: (wandb key, display name, format string)

METRICS = [
    # ── Evaluation ────────────────────────────────────────────────────────────
    ('eval/success',                              'Eval success rate',                   '.3f'),

    # ── Offline: critic ───────────────────────────────────────────────────────
    ('offline_agent/critic/critic_loss',          'Critic loss (offline)',               '.5f'),
    ('offline_agent/critic/q',                    'Q = P(success) (offline) [BSC]',      '.4f'),
    ('offline_agent/critic/target_q',             'Target Q (offline) [BSC]',            '.4f'),
    ('offline_agent/critic/success_frac',         'Success frac (offline) [BSC]',        '.3f'),
    ('offline_agent/critic/timeout_frac',         'Timeout frac (offline) [BSC]',        '.3f'),
    ('offline_agent/critic/q_mean',               'Q mean (offline) [QC]',               '.4f'),
    ('offline_agent/critic/q_max',                'Q max (offline) [QC]',                '.4f'),
    ('offline_agent/critic/q_min',                'Q min (offline) [QC]',                '.4f'),

    # ── Offline: actor ────────────────────────────────────────────────────────
    ('offline_agent/actor/actor_loss',            'Actor loss (offline)',                '.5f'),
    ('offline_agent/actor/bc_flow_loss',          'BC flow loss (offline)',              '.5f'),
    ('offline_agent/actor/distill_loss',          'Distill loss (offline)',              '.5f'),
    ('offline_agent/actor/q_loss',                'Q loss (offline) [BSC]',              '.5f'),

    # ── Online: critic ────────────────────────────────────────────────────────
    ('online_agent/critic/critic_loss',           'Critic loss (online)',                '.5f'),
    ('online_agent/critic/q',                     'Q = P(success) (online) [BSC]',       '.4f'),
    ('online_agent/critic/target_q',              'Target Q (online) [BSC]',             '.4f'),
    ('online_agent/critic/success_frac',          'Success frac (online) [BSC]',         '.3f'),
    ('online_agent/critic/timeout_frac',          'Timeout frac (online) [BSC]',         '.3f'),
    ('online_agent/critic/q_mean',                'Q mean (online) [QC]',                '.4f'),
    ('online_agent/critic/q_max',                 'Q max (online) [QC]',                 '.4f'),
    ('online_agent/critic/q_min',                 'Q min (online) [QC]',                 '.4f'),

    # ── Online: actor ─────────────────────────────────────────────────────────
    ('online_agent/actor/actor_loss',             'Actor loss (online)',                 '.5f'),
    ('online_agent/actor/bc_flow_loss',           'BC flow loss (online)',               '.5f'),
    ('online_agent/actor/distill_loss',           'Distill loss (online)',               '.5f'),
    ('online_agent/actor/q_loss',                 'Q loss (online) [BSC]',               '.5f'),
]

# Which metrics to include in time-series plots
PLOT_METRICS = [
    'eval/success',
    # offline critic
    'offline_agent/critic/critic_loss',
    'offline_agent/critic/q',
    'offline_agent/critic/q_mean',
    'offline_agent/critic/success_frac',
    # offline actor
    'offline_agent/actor/actor_loss',
    'offline_agent/actor/bc_flow_loss',
    'offline_agent/actor/distill_loss',
    'offline_agent/actor/q_loss',
    # online critic
    'online_agent/critic/critic_loss',
    'online_agent/critic/q',
    'online_agent/critic/q_mean',
    # online actor
    'online_agent/actor/actor_loss',
    'online_agent/actor/q_loss',
]

PLOT_TITLES = {
    'eval/success':                           'Eval Success Rate  [primary metric]',
    'offline_agent/critic/critic_loss':       'Critic Loss (offline)',
    'offline_agent/critic/q':                 'Q=P(success) (offline) [BSC]',
    'offline_agent/critic/q_mean':            'Q Mean (offline) [QC/FQL]',
    'offline_agent/critic/success_frac':      'Success Fraction (offline) [BSC]',
    'offline_agent/actor/actor_loss':         'Actor Loss (offline)',
    'offline_agent/actor/bc_flow_loss':       'BC Flow Loss (offline)',
    'offline_agent/actor/distill_loss':       'Distill Loss (offline)',
    'offline_agent/actor/q_loss':             'Q Loss (offline) [BSC]',
    'online_agent/critic/critic_loss':        'Critic Loss (online)',
    'online_agent/critic/q':                  'Q=P(success) (online) [BSC]',
    'online_agent/critic/q_mean':             'Q Mean (online) [QC/FQL]',
    'online_agent/actor/actor_loss':          'Actor Loss (online)',
    'online_agent/actor/q_loss':              'Q Loss (online) [BSC]',
}

# ─── Wandb helpers ────────────────────────────────────────────────────────────

def fetch_run(entity, project, run_name):
    api = wandb.Api()
    all_runs = list(api.runs(f"{entity}/{project}"))
    matched = [r for r in all_runs if r.name == run_name]
    if not matched:
        print(f"  [WARN] Run '{run_name}' not found in {entity}/{project}.")
        return None
    if len(matched) > 1:
        print(f"  [WARN] {len(matched)} runs match '{run_name}', using most recent.")
    return matched[0]


def fetch_one_metric(run, metric_key):
    """Fetch (steps, values) for a single metric. Returns (None, None) if missing."""
    try:
        history = run.history(keys=['_step', metric_key], samples=SAMPLES)
    except Exception as e:
        print(f"    [ERROR] {metric_key}: {e}")
        return None, None
    if history.empty or metric_key not in history.columns:
        return None, None
    steps  = history['_step'].values.astype(float)
    values = history[metric_key].values.astype(float)
    mask   = ~np.isnan(values) & ~np.isnan(steps)
    if mask.sum() == 0:
        return None, None
    idx = np.argsort(steps[mask])
    return steps[mask][idx], values[mask][idx]


def fetch_all_metrics(run, metrics):
    """Returns dict: metric_key -> (steps, values)."""
    result = {}
    for key, *_ in metrics:
        result[key] = fetch_one_metric(run, key)
    return result

# ─── Statistics ───────────────────────────────────────────────────────────────

def phase_mean(steps, values, lo, hi):
    if steps is None:
        return float('nan')
    mask = (steps >= lo) & (steps < hi)
    v = values[mask]
    return float(np.mean(v)) if len(v) > 0 else float('nan')


def phase_npts(steps, lo, hi):
    if steps is None:
        return 0
    return int(((steps >= lo) & (steps < hi)).sum())

# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS     = ['#2A52BE', '#B30000', '#50C878', '#925BFF', '#F9A825']
LINESTYLES = ['-', '--', '-.', ':', '-']


def apply_ema(values, alpha):
    if alpha == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1 - alpha) * values[i] + alpha * out[i - 1]
    return out


def make_plots(run_data_list, output_path):
    """run_data_list: list of (label, data_dict) where data_dict: key -> (steps, values)"""
    ncols = 3
    nrows = (len(PLOT_METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 4))
    axes = axes.flatten()

    # Draw vertical line at offline/online boundary
    boundary = PHASES[0][2]  # end of offline phase

    for ax_idx, key in enumerate(PLOT_METRICS):
        ax = axes[ax_idx]
        title = PLOT_TITLES.get(key, key.split('/')[-1])

        for run_idx, (label, data) in enumerate(run_data_list):
            color = COLORS[run_idx % len(COLORS)]
            ls    = LINESTYLES[run_idx % len(LINESTYLES)]
            steps, values = data.get(key, (None, None))
            if steps is not None:
                ax.plot(steps, apply_ema(values, EMA),
                        label=label, color=color, linestyle=ls, linewidth=1.8, alpha=0.9)

        # Mark offline→online boundary
        ax.axvline(x=boundary, color='gray', linestyle=':', linewidth=1.0, alpha=0.6, label='offline→online')

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Steps', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        handles, labels_leg = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc='best')

    for ax_idx in range(len(PLOT_METRICS), len(axes)):
        axes[ax_idx].set_visible(False)

    run_labels = ', '.join(label for label, _ in run_data_list)
    plt.suptitle(f'BSC vs QC Diagnostics — {run_labels}', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()

# ─── Table printing ───────────────────────────────────────────────────────────

def print_stats_table(run_data_list):
    """Print one table per phase: rows=metrics, columns=runs."""
    n_runs = len(run_data_list)
    labels = [label for label, _ in run_data_list]
    col_w  = 16
    name_w = 46

    for phase_name, lo, hi in PHASES:
        print(f"\n{'='*100}")
        print(f"  PHASE: {phase_name}")
        print(f"{'='*100}")

        header = f"{'Metric':<{name_w}}"
        for label in labels:
            header += f"{label:>{col_w}}"
        print(header)
        print('-' * (name_w + col_w * n_runs))

        prev_group = None
        for key, display_name, fmt in METRICS:
            # Group separator by top-level prefix
            group = key.split('/')[0]
            if group != prev_group and prev_group is not None:
                print()
            prev_group = group

            row = f"{display_name:<{name_w}}"
            for _, data in run_data_list:
                steps, values = data.get(key, (None, None))
                mean = phase_mean(steps, values, lo, hi)
                if np.isnan(mean):
                    cell = 'n/a'
                else:
                    cell = f"{mean:{fmt}}"
                row += f"{cell:>{col_w}}"
            print(row)

        print()
        # Data-point count uses eval/success as the reference metric
        count_row = f"{'  [eval data points]':<{name_w}}"
        for _, data in run_data_list:
            steps, _ = data.get('eval/success', (None, None))
            n = phase_npts(steps, lo, hi) if steps is not None else 0
            count_row += f"{'(%d pts)' % n:>{col_w}}"
        print(count_row)


def print_coverage(run_data_list):
    """Show which metrics were found for each run."""
    labels = [label for label, _ in run_data_list]
    col_w  = 18
    name_w = 55

    print(f"\n{'='*100}")
    print('  METRIC COVERAGE')
    print(f"{'='*100}")
    header = f"{'Metric key':<{name_w}}"
    for label in labels:
        header += f"{label:>{col_w}}"
    print(header)
    print('-' * (name_w + col_w * len(labels)))

    prev_group = None
    for key, display_name, _ in METRICS:
        group = key.split('/')[0]
        if group != prev_group and prev_group is not None:
            print()
        prev_group = group

        row = f"{key:<{name_w}}"
        for _, data in run_data_list:
            steps, _ = data.get(key, (None, None))
            cell = 'x' if steps is None else f"ok ({len(steps)}pts)"
            row += f"{cell:>{col_w}}"
        print(row)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 80)
    print(f"Fetching {len(RUNS)} runs from {ENTITY}/{PROJECT}")
    print('=' * 80)

    run_data_list = []
    for run_name, label in RUNS:
        print(f"\n  Fetching: {run_name}  ({label})")
        run = fetch_run(ENTITY, PROJECT, run_name)
        if run is None:
            print(f"    Skipping — run not found.")
            continue
        print(f"    Found run id={run.id}, state={run.state}, step={run.summary.get('_step', '?')}")
        data = fetch_all_metrics(run, METRICS)
        found = sum(1 for s, _ in data.values() if s is not None)
        print(f"    Fetched {found}/{len(METRICS)} metrics.")
        run_data_list.append((label, data))

    if not run_data_list:
        print("No runs found. Exiting.")
        return

    print_coverage(run_data_list)
    print_stats_table(run_data_list)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%m%d_%H%M%S')
    make_plots(run_data_list, os.path.join(OUTPUT_DIR, f"bsc_qc_analysis_{ts}.png"))

    print('\nDone.')


if __name__ == '__main__':
    main()
