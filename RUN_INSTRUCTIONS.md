# Run Instructions

## One-time setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sanity-check the environments register and torch sees your accelerator:
```bash
python3 -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); [print(gym.make(e).action_space) for e in ['ALE/Breakout-v5','ALE/SpaceInvaders-v5','ALE/Pong-v5']]"
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"
```

Do a tiny smoke-test run before committing to a full one (~2 minutes):
```bash
python train.py --env-id ALE/Breakout-v5 --run-name smoketest --timesteps 5000 --device auto
python play.py --env-id ALE/Breakout-v5 --model models/dqn_model_smoketest.zip --mode record --episodes 1 --video-length 500
```

## Edwin — `ALE/Breakout-v5` (10 experiments, 150k steps each)

```bash
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp01 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp02 --lr 1e-3  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp03 --lr 1e-5  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp04 --lr 1e-4  --gamma 0.90  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp05 --lr 1e-4  --gamma 0.995 --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp06 --lr 1e-4  --gamma 0.99  --batch-size 16  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp07 --lr 1e-4  --gamma 0.99  --batch-size 128 --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp08 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.02 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp09 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.10 --eps-decay-frac 0.30 --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --run-name edwin_exp10 --lr 5e-4  --gamma 0.99  --batch-size 64  --eps-start 1.0 --eps-end 0.02 --eps-decay-frac 0.15 --timesteps 150000 --device auto

# MLP vs CNN comparison
python train.py --env-id ALE/Breakout-v5 --member edwin --policy MlpPolicy --run-name edwin_mlp_comparison --timesteps 150000 --device auto
python train.py --env-id ALE/Breakout-v5 --member edwin --policy CnnPolicy --run-name edwin_cnn_comparison --timesteps 150000 --device auto

# Promote the winning model (exp10) and record the submission video
cp models/dqn_model_edwin_exp10.zip models/dqn_model_edwin_breakout.zip
python play.py --env-id ALE/Breakout-v5 --model models/dqn_model_edwin_breakout.zip --mode record --episodes 20
```

## David — `ALE/SpaceInvaders-v5` (10 experiments, 150k steps each, GPU)

```bash
# Run all 10 in sequence (edit run_experiments.py's EXPERIMENTS list if you need to change configs)
python run_experiments.py --member Yinka_Ajao --timesteps 150000   # if using the branch-preserved run_experiments.py

# Or individually with the unified train.py:
python train.py --env-id ALE/SpaceInvaders-v5 --member david --run-name david_exp09 --lr 1e-4 --gamma 0.99 --batch-size 32 --eps-start 1.0 --eps-end 0.10 --eps-decay-frac 0.30 --timesteps 150000 --device cuda
python train.py --env-id ALE/SpaceInvaders-v5 --member david --run-name david_exp10 --lr 5e-4 --gamma 0.99 --batch-size 64 --eps-start 1.0 --eps-end 0.02 --eps-decay-frac 0.15 --timesteps 150000 --device cuda

# Resolve the exp09-vs-exp10 discrepancy (see experiments/results.md), then:
cp models/dqn_model_david_exp10.zip models/dqn_model_david_spaceinvaders.zip   # if re-promoting to exp10
python play.py --env-id ALE/SpaceInvaders-v5 --model models/dqn_model_david_spaceinvaders.zip --mode record --episodes 3
```

## Nziza — `ALE/Pong-v5` (10 experiments, 200k steps each)

```bash
python train.py --env-id ALE/Pong-v5 --member nziza --run-name nziza_exp01 --lr 1e-4 --gamma 0.99 --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 200000 --device auto
python train.py --env-id ALE/Pong-v5 --member nziza --run-name nziza_exp07 --lr 1e-4 --gamma 0.99 --batch-size 128 --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 200000 --device auto
# (full 10-run list in experiments/results.md)

# Still outstanding — record the gameplay video:
python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode record --episodes 3

# Still outstanding — MLP vs CNN comparison:
python train.py --env-id ALE/Pong-v5 --member nziza --policy MlpPolicy --run-name nziza_mlp_comparison --timesteps 200000 --device auto
```

## Plotting / summarizing results

```bash
python experiments/plot_results.py --runs edwin_exp01 edwin_exp02 edwin_exp10 --out experiments/logs/comparison.png
python experiments/summarize_results.py --runs edwin_exp01 edwin_exp02 edwin_exp10
```

## Troubleshooting

- **`ROM not found` / license error**: `pip install "gymnasium[atari,accept-rom-license]" ale-py` again.
- **Memory pressure**: lower `--buffer-size` (e.g. `25000`) or close other apps.
- **MPS/CUDA errors mid-training**: fall back to `--device cpu` for the affected run.
- **`play.py --mode record` produces no video**: confirm `moviepy` installed (`pip show moviepy`).
