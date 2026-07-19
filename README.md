# DQN Atari Agent — David (Yinka) Ajao's Individual Work (ALE/SpaceInvaders-v5)

Part of the group's [Deep-Q-Learning-A3](../README.md) submission. Training and evaluating a
DQN agent (Stable Baselines3 + Gymnasium) on **`ALE/SpaceInvaders-v5`** — see the top-level
README for why each member used a different Atari environment.

## Setup

```bash
pip install -r ../requirements.txt
```

This repo trains with `device="cuda"` by default (see `train.py`) — it was run on a GPU
(e.g. Google Colab). If running on a machine without CUDA, change `device="cuda"` to
`device="auto"` in `train.py` first.

## Files

- `train.py` — trains the DQN agent (`CnnPolicy` or `MlpPolicy`), saves `models/dqn_model_<run-name>.zip`
- `play.py` — loads a trained model and plays greedily (`deterministic=True`). `--mode record`
  (default) saves an `.mp4` via Gymnasium's `RecordVideo`; `--mode fast` runs headless and just
  prints scores.
- `run_experiments.py` — runs all 10 hyperparameter experiments in sequence by shelling out to `train.py`
- `experiments/david_results.md` — the 10-experiment table with real final/max reward pulled from TensorBoard logs, plus the MLP vs CNN comparison
- `models/dqn_model.zip` — David's promoted champion model (see "Open item" in `experiments/david_results.md` regarding exp09 vs exp10)
- `video/champion-video-episode-0.mp4` — gameplay recording of the champion model

## Usage

Run all 10 experiments:
```bash
python run_experiments.py --member Yinka_Ajao --timesteps 150000
```

Run the MLP vs CNN comparison:
```bash
python run_experiments.py --member Yinka_Ajao_MLP --policy MlpPolicy --only 1 2
```

Watch the trained agent play (records a video by default):
```bash
python play.py --model models/dqn_model.zip --mode record --episodes 3
```

## Policy Architecture: MLP vs CNN

See `experiments/david_results.md` for the numbers. Summary: on SpaceInvaders, `CnnPolicy`
and `MlpPolicy` landed in the same performance ballpark at this training budget — a smaller
gap than Edwin found on Breakout, where CNN clearly won. Worth raising in Q&A as a genuine,
environment-dependent nuance rather than assuming CNN always dominates by a wide margin.

## Hyperparameter Tuning Results

Full table with real per-run numbers (extracted from TensorBoard logs) and reasoning:
[`experiments/david_results.md`](experiments/david_results.md).

## Final Chosen Configuration

See the "Best configuration" and "Open item for David" sections in
[`experiments/david_results.md`](experiments/david_results.md) — the promoted model
(exp09) is not the numerically top-scoring run (exp10 scored higher); be ready to explain
why, or consider re-promoting exp10 before presenting.

## Gameplay Demo

[`video/champion-video-episode-0.mp4`](video/champion-video-episode-0.mp4) — 27 seconds,
817 frames, the champion model (exp09) playing SpaceInvaders greedily.

## Individual Contribution — David (Yinka) Ajao

- Adapted the shared `train.py`/`play.py` skeleton to `ALE/SpaceInvaders-v5`, with GPU
  (`device="cuda"`) training.
- Ran 10 hyperparameter experiments plus 2 MLP-vs-CNN comparison runs — see
  `experiments/david_results.md`.
- Selected and promoted a champion model, recorded the gameplay video.
