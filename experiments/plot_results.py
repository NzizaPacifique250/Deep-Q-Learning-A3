"""
plot_results.py — plot reward-trend comparisons across experiment CSVs.

Each train.py run writes experiments/logs/<run-name>.csv with columns:
    episode, timestep, reward, length, rolling_mean_reward_20

This script overlays the rolling mean reward curves for a set of runs so you
can visually back up your "which change helped / hurt" claims in the README
and presentation, instead of relying on eyeballing raw numbers.

Usage:
    python experiments/plot_results.py --runs edwin_exp01 edwin_exp02 edwin_exp08 edwin_exp09 --out experiments/logs/eps_decay_comparison.png
    python experiments/plot_results.py --runs edwin_exp01 edwin_exp02 edwin_exp03 edwin_exp04 edwin_exp05 edwin_exp06 edwin_exp07 edwin_exp08 edwin_exp09 edwin_exp10 --out experiments/logs/all_runs_comparison.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_csv(path):
    timesteps, rolling = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rolling_col = [c for c in reader.fieldnames if c.startswith("rolling_mean_reward")][0]
        for row in reader:
            timesteps.append(int(row["timestep"]))
            rolling.append(float(row[rolling_col]))
    return timesteps, rolling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="experiments/logs")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="Run names to overlay, e.g. edwin_exp01 edwin_exp02")
    parser.add_argument("--out", type=str, default="experiments/logs/comparison.png")
    args = parser.parse_args()

    plt.figure(figsize=(9, 5))
    for run in args.runs:
        path = os.path.join(args.log_dir, f"{run}.csv")
        if not os.path.exists(path):
            print(f"skipping {run}: {path} not found")
            continue
        ts, rolling = load_csv(path)
        plt.plot(ts, rolling, label=run)

    plt.xlabel("Timestep")
    plt.ylabel("Rolling mean episode reward (window=20)")
    plt.title("DQN on ALE/Breakout-v5 — hyperparameter comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
