"""
run_experiments.py — run one member's 10 hyperparameter experiments in sequence.

Each experiment is a distinct combination of (lr, gamma, batch_size, eps_*),
trained by calling train.py as a subprocess so every run gets its own logs,
monitor CSVs, best model, and a summary row in experiments/results.csv.
"""

import argparse
import subprocess
import sys

MEMBER = "Yinka_Ajao"

# Aligned with Edwin's sweep layout: modifying ONE variable at a time from baseline
EXPERIMENTS = [
    # id,  lr,     gamma, batch, eps_start, eps_end, eps_decay_frac
    (1,  1e-4,  0.99,  32,   1.0,  0.05, 0.10),   # Baseline config
    (2,  1e-3,  0.99,  32,   1.0,  0.05, 0.10),   # Higher learning rate
    (3,  1e-5,  0.99,  32,   1.0,  0.05, 0.10),   # Lower learning rate
    (4,  1e-4,  0.90,  32,   1.0,  0.05, 0.10),   # Shorter horizon (low gamma)
    (5,  1e-4,  0.995, 32,   1.0,  0.05, 0.10),   # Longer horizon (high gamma)
    (6,  1e-4,  0.99,  16,   1.0,  0.05, 0.10),   # Smaller batch size
    (7,  1e-4,  0.99,  128,  1.0,  0.05, 0.10),   # Larger batch size
    (8,  1e-4,  0.99,  32,   1.0,  0.05, 0.02),   # Rapid epsilon decay
    (9,  1e-4,  0.99,  32,   1.0,  0.10, 0.30),   # Prolonged exploration
    (10, 5e-4,  0.99,  64,   1.0,  0.02, 0.15),   # Combined optimal guess config
]

def parse_args():
    p = argparse.ArgumentParser(description="Run a member's 10 DQN hyperparameter experiments.")
    p.add_argument("--member", type=str, default=MEMBER, help="Member name used in run-name prefix")
    p.add_argument("--timesteps", type=int, default=150_000, help="Timesteps per experiment")
    p.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    p.add_argument("--only", type=int, nargs="*", default=None, help="Only run these experiment ids")
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
            "--run-name", run_name,
        ]
        print("=" * 70)
        print(f"[{eid}/10] {run_name}: lr={lr} gamma={gamma} batch={batch} "
              f"eps={es}->{ee} decay={ed}")
        print("=" * 70)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"!! Experiment {eid} failed (exit {result.returncode}). Continuing.\n")

    print("\nAll requested experiments finished.")

if __name__ == "__main__":
    main()
