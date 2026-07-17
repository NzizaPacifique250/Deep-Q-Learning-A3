"""
run_experiments.py — run one member's 10 hyperparameter experiments in sequence.

Each experiment is a distinct combination of (lr, gamma, batch_size, eps_*),
trained by calling train.py as a subprocess so every run gets its own logs,
monitor CSVs, best model, and a summary row in experiments/results.csv.

Edit MEMBER and the EXPERIMENTS list below for your own sweep, then:

    python run_experiments.py                 # run all 10
    python run_experiments.py --only 1 2 3    # run just experiments 1,2,3
    python run_experiments.py --timesteps 100000   # override length for a faster pass

The 10 combos below are a sensible starting sweep for Pong: they vary ONE knob
at a time from a baseline so the "noted behavior" is attributable. Change the
values to explore your own hypotheses — the assignment wants YOUR reasoning.
"""

import argparse
import subprocess
import sys

MEMBER = "nzizapacifique250"

# Baseline is exp 1; each later row changes one thing (mostly) vs the baseline.
EXPERIMENTS = [
    # id,  lr,     gamma, batch, eps_start, eps_end, eps_decay_frac
    (1,  1e-4,  0.99,  32,   1.0,  0.05, 0.10),   # baseline
    (2,  5e-4,  0.99,  32,   1.0,  0.05, 0.10),   # higher lr
    (3,  5e-5,  0.99,  32,   1.0,  0.05, 0.10),   # lower lr
    (4,  1e-4,  0.95,  32,   1.0,  0.05, 0.10),   # shorter horizon (low gamma)
    (5,  1e-4,  0.999, 32,   1.0,  0.05, 0.10),   # longer horizon (high gamma)
    (6,  1e-4,  0.99,  64,   1.0,  0.05, 0.10),   # larger batch
    (7,  1e-4,  0.99,  128,  1.0,  0.05, 0.10),   # largest batch
    (8,  1e-4,  0.99,  32,   1.0,  0.01, 0.10),   # exploit more (low final eps)
    (9,  1e-4,  0.99,  32,   1.0,  0.10, 0.30),   # explore longer (slower decay)
    (10, 5e-4,  0.99,  64,   1.0,  0.02, 0.20),   # combined "best guess" config
]


def parse_args():
    p = argparse.ArgumentParser(description="Run a member's 10 DQN hyperparameter experiments.")
    p.add_argument("--member", type=str, default=MEMBER, help="Member name used in run-name prefix")
    p.add_argument("--timesteps", type=int, default=200_000, help="Timesteps per experiment")
    p.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    p.add_argument("--only", type=int, nargs="*", default=None, help="Only run these experiment ids")
    p.add_argument("--buffer-size", type=int, default=100_000)
    return p.parse_args()


def main():
    args = parse_args()
    to_run = [e for e in EXPERIMENTS if args.only is None or e[0] in args.only]
    print(f"Running {len(to_run)} experiment(s) for member '{args.member}' "
          f"at {args.timesteps} timesteps each ({args.policy}).\n")

    for (eid, lr, gamma, batch, es, ee, ed) in to_run:
        run_name = f"{args.member}_exp{eid:02d}"
        cmd = [
            sys.executable, "train.py",
            "--lr", str(lr),
            "--gamma", str(gamma),
            "--batch-size", str(batch),
            "--eps-start", str(es),
            "--eps-end", str(ee),
            "--eps-decay-frac", str(ed),
            "--timesteps", str(args.timesteps),
            "--policy", args.policy,
            "--buffer-size", str(args.buffer_size),
            "--run-name", run_name,
        ]
        print("=" * 70)
        print(f"[{eid}/10] {run_name}: lr={lr} gamma={gamma} batch={batch} "
              f"eps={es}->{ee} decay={ed}")
        print("=" * 70)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!! Experiment {eid} failed (exit {result.returncode}). Continuing.\n")

    print("\nAll requested experiments finished. See experiments/results.csv for the table.")
    print("Then run:  python summarize_results.py   to build a Markdown table for the README.")


if __name__ == "__main__":
    main()
