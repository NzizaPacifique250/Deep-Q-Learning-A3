#!/usr/bin/env bash
# Runs Edwin's 10 experiments one at a time, in order — each command waits
# for the previous one to finish before starting (never parallel, since
# they'd contend for the same CPU/RAM and skew each other's timing).
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# You can Ctrl+C at any point and resume later by commenting out the runs
# that already finished (their models/CSVs will already exist).

set -e  # stop immediately if any run errors out, instead of silently continuing

python train.py --run-name edwin_exp01 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp02 --lr 1e-3  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp03 --lr 1e-5  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp04 --lr 1e-4  --gamma 0.90  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp05 --lr 1e-4  --gamma 0.995 --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp06 --lr 1e-4  --gamma 0.99  --batch-size 16  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp07 --lr 1e-4  --gamma 0.99  --batch-size 128 --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp08 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.02 --timesteps 150000 --device auto
python train.py --run-name edwin_exp09 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.10 --eps-decay-frac 0.30 --timesteps 150000 --device auto
python train.py --run-name edwin_exp10 --lr 5e-4  --gamma 0.99  --batch-size 64  --eps-start 1.0 --eps-end 0.02 --eps-decay-frac 0.15 --timesteps 150000 --device auto

echo "All 10 experiments finished. Models are in models/, logs are in experiments/logs/."
