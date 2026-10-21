"""Plot eval/success for the {Square,Toolhang}-{QC,BSC,ARB,CAT} wandb runs.

Usage: python visualization/plot_eval_success.py [--out PATH]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

ENTITY = 'RL-experiments'
PROJECT = 'BSC'
TASKS = ['Square', 'Toolhang']
METHODS = ['QC', 'BSC', 'ARB', 'CAT']
LABELS = {'QC': 'QC', 'BSC': 'Binary-Success Critic', 'ARB': 'Autoregressive Binning',
          'CAT': 'Categorical Critic (Two-Hot)'}
OFFLINE_STEPS = 1_000_000


def fetch_eval(api, name):
    """Return sorted (steps, success) for the run with this display name, or None."""
    runs = list(api.runs(f'{ENTITY}/{PROJECT}', filters={'display_name': name}))
    if not runs:
        print(f'[skip] no run named {name}')
        return None
    if len(runs) > 1:
        print(f'[warn] {len(runs)} runs named {name}, using the most recent')
    run = max(runs, key=lambda r: r.created_at)
    # eval is logged twice at some steps (offline end / online start); keep one per step
    points = {}
    for row in run.history(keys=['eval/success'], samples=100_000, pandas=False):
        points[int(row['_step'])] = row['eval/success']
    steps = sorted(points)
    print(f'{name} ({run.id}, {run.state}): {len(steps)} evals, last step {steps[-1] if steps else None}')
    return steps, [points[s] for s in steps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'plots', 'eval_success.png'))
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(12, 4.5))

    for ax, task in zip(axes, TASKS):
        for method in METHODS:
            data = fetch_eval(api, f'{task}-{method}')
            if data is None:
                continue
            steps, success = data
            ax.plot([s / 1e6 for s in steps], success, label=LABELS[method], lw=2)

        ax.axvline(OFFLINE_STEPS / 1e6, color='gray', ls='--')
        ax.set_title(task)
        ax.set_xlabel('Environment steps (M)')
        ax.set_ylim(0, 1.0)
        ax.legend()
    axes[0].set_ylabel('Eval success rate')

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
