"""
summarize_results.py — print a compact final-performance summary for each
experiment CSV, so the "Noted Behavior" column in edwin_results.md can be
filled in from real numbers instead of guesswork.

Usage:
    python experiments/summarize_results.py
    python experiments/summarize_results.py --runs edwin_exp01 edwin_exp02
"""

import argparse
import csv
import glob
import os


def summarize(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rolling_col = [c for c in reader.fieldnames if c.startswith("rolling_mean_reward")][0]
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    final_rolling = float(rows[-1][rolling_col])
    max_rolling = max(float(r[rolling_col]) for r in rows)
    max_rolling_episode = max(rows, key=lambda r: float(r[rolling_col]))["episode"]
    final_ep_len = float(rows[-1]["length"])
    max_ep_len = max(float(r["length"]) for r in rows)
    n_episodes = len(rows)
    final_timestep = rows[-1]["timestep"]

    return {
        "n_episodes": n_episodes,
        "final_timestep": final_timestep,
        "final_rolling_reward": round(final_rolling, 2),
        "max_rolling_reward": round(max_rolling, 2),
        "max_rolling_at_episode": max_rolling_episode,
        "final_episode_length": round(final_ep_len, 1),
        "max_episode_length": round(max_ep_len, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="experiments/logs")
    parser.add_argument("--runs", type=str, nargs="*", default=None, help="Specific run names; default = all edwin_exp* CSVs found")
    args = parser.parse_args()

    if args.runs:
        paths = [os.path.join(args.log_dir, f"{r}.csv") for r in args.runs]
    else:
        paths = sorted(glob.glob(os.path.join(args.log_dir, "edwin_exp*.csv")))

    print(f"{'run':<14} {'episodes':>9} {'final_ts':>9} {'final_roll_r':>13} {'max_roll_r':>11} {'@episode':>9} {'final_len':>10} {'max_len':>8}")
    for path in paths:
        run_name = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            print(f"{run_name:<14} MISSING ({path})")
            continue
        s = summarize(path)
        if s is None:
            print(f"{run_name:<14} EMPTY CSV")
            continue
        print(f"{run_name:<14} {s['n_episodes']:>9} {s['final_timestep']:>9} {s['final_rolling_reward']:>13} "
              f"{s['max_rolling_reward']:>11} {s['max_rolling_at_episode']:>9} {s['final_episode_length']:>10} {s['max_episode_length']:>8}")


if __name__ == "__main__":
    main()
