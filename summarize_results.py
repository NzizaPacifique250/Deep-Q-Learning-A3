"""
summarize_results.py — turn experiments/results.csv into a Markdown table.

After running experiments, this prints a table you can paste into the README /
your experiments/<member>_results.md file. It fills the "Noted Behavior" column
with the best mean eval reward; you still add a short qualitative note per row.

    python summarize_results.py
    python summarize_results.py --member alice     # filter to one member
"""

import argparse
import csv
import os

RESULTS_CSV = "./experiments/results.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--member", type=str, default=None, help="Filter rows whose run_name starts with this")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(RESULTS_CSV):
        print(f"No results yet at {RESULTS_CSV}. Run train.py / run_experiments.py first.")
        return

    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    if args.member:
        rows = [r for r in rows if r["run_name"].startswith(args.member)]

    if not rows:
        print("No matching rows.")
        return

    print("| Run | Policy | lr | gamma | batch | eps_start | eps_end | eps_decay | Best mean reward | Noted behavior |")
    print("|-----|--------|----|-------|-------|-----------|---------|-----------|------------------|----------------|")
    for r in rows:
        print(f"| {r['run_name']} | {r['policy']} | {r['lr']} | {r['gamma']} | "
              f"{r['batch_size']} | {r['eps_start']} | {r['eps_end']} | {r['eps_decay_frac']} | "
              f"{r['best_mean_reward']} | _(add note)_ |")

    # Highlight the best run overall
    def score(r):
        try:
            return float(r["best_mean_reward"])
        except ValueError:
            return float("-inf")

    best = max(rows, key=score)
    print(f"\nBest run so far: {best['run_name']} "
          f"(best mean reward {best['best_mean_reward']}) -> model: {best['best_model']}")


if __name__ == "__main__":
    main()
