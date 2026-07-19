# DQN Atari Agent — Edwin's Individual Work (ALE/Breakout-v5)

Part of the group's [Deep-Q-Learning-A3](../README.md) submission. Training and evaluating a
DQN agent (Stable Baselines3 + Gymnasium) on **`ALE/Breakout-v5`** — see the top-level README
for why each member used a different Atari environment.

> Settings fixed across all 10 of Edwin's runs (only changed for this environment's setup,
> not per-experiment): `ENV_ID=ALE/Breakout-v5`, `N_ENVS=4`, `FRAME_STACK=4`,
> `POLICY=CnnPolicy` (primary), `timesteps=150,000` per run, `buffer_size=50,000` with
> `optimize_memory_usage=True` (keeps replay-buffer RAM manageable on 16GB laptops at
> 84x84x4 uint8 frames). Only `lr`, `gamma`, `batch_size`, and the epsilon schedule vary
> between the 10 experiments.

## Setup

```bash
pip install -r requirements.txt
```

Apple Silicon (M-series) note: PyTorch will try `mps` automatically via `--device auto`.
If you hit an "operator not implemented for MPS" error, rerun with `--device cpu`.

## Files

- `train.py` — trains the DQN agent, saves `models/dqn_model_<run-name>.zip` + a per-episode
  reward/length CSV under `experiments/logs/`
- `play.py` — loads a trained model and plays greedily (`deterministic=True`); `--mode record`
  saves an `.mp4` (the required gameplay deliverable), `--mode human` opens a live window
- `experiments/edwin_results.md` — Edwin's 10 hyperparameter experiments (config, hypothesis,
  observed behavior)
- `experiments/plot_results.py` — overlays reward-trend CSVs into a comparison chart
- `experiments/logs/` — per-run CSV logs + generated comparison plots
- `models/` — saved models
- `RUN_INSTRUCTIONS_EDWIN.md` — step-by-step commands to reproduce Edwin's 10 runs locally
- `PRESENTATION_NOTES_EDWIN.md` — talking points + Q&A prep for Edwin's 2-minute segment

## Usage

Train:
```bash
python train.py --lr 1e-4 --gamma 0.99 --batch-size 32 --run-name edwin_exp01 --device auto
```

Play with the trained model (record the submission video):
```bash
python play.py --model models/dqn_model.zip --mode record --episodes 3
```

Play with a live window instead:
```bash
python play.py --model models/dqn_model.zip --mode human --episodes 3
```

## Policy Architecture: MLP vs CNN

One shared run (not one of the 10 individual experiments) compares `MlpPolicy` against
`CnnPolicy` on the same Breakout setup, to have concrete evidence for Q&A:

```bash
python train.py --policy MlpPolicy --run-name edwin_mlp_comparison --timesteps 150000 --device auto
python train.py --policy CnnPolicy --run-name edwin_cnn_comparison --timesteps 150000 --device auto
python experiments/plot_results.py --runs edwin_mlp_comparison edwin_cnn_comparison --out experiments/logs/mlp_vs_cnn.png
```

**Result:** `CnnPolicy` clearly outperformed `MlpPolicy` on Breakout at the same 150k-step
budget — final rolling-mean reward 7.55 (peak 11.9) vs. MLP's 5.65 (peak 7.65). The gap shows
up in episode count too: MLP needed 3,225 episodes to use its step budget vs. CNN's 2,092,
meaning its average life was noticeably shorter (less effective ball tracking). This matches
the theory: convolutional layers exploit the spatial structure of the 84×84×4 pixel stack
(paddle/ball/brick shapes and relative position) directly, while `MlpPolicy` flattens the
frame into an unordered vector and has to learn every pixel-position relationship from
scratch, which is harder to do well in a modest training budget. Worth noting for Q&A: the
gap is real but not catastrophic — MLP wasn't hopeless, it just consistently trailed CNN on
every metric. See `experiments/logs/mlp_vs_cnn.png` for the overlaid reward curves.

## Hyperparameter Tuning Results (merged from all members)

Edwin's full 10-experiment table, with a pre-run hypothesis and the actual observed behavior
for every row, lives in [`experiments/edwin_results.md`](experiments/edwin_results.md). (This
is combined with David's and Nziza's tables in the top-level repo README — see
[`../README.md`](../README.md).)

## Final Chosen Configuration

- **Run:** exp10
- **Config:** `lr=5e-4, gamma=0.99, batch_size=64, eps_start=1.0, eps_end=0.02, eps_decay_frac=0.15`
- **Reasoning:** exp10 combined three individually-promising directions — a moderate LR bump
  (validated safe by exp02), a larger batch size for smoother gradients (validated by exp07),
  and a lower epsilon floor for decisive exploitation (validated by exp08 beating exp09). Final
  rolling-mean reward 16.85, peak 20.9 — roughly double the baseline, and reached in the fewest
  episodes of any run (most sample-efficient). Promoted to `models/dqn_model.zip`.

## Gameplay Demo

[`videos/edwin_breakout_gameplay.mp4`](videos/edwin_breakout_gameplay.mp4) — 20 episodes,
~27 seconds, recorded with `python play.py --model models/dqn_model.zip --mode record --episodes 20`
using the promoted exp10 model. Fresh greedy evaluation over 10 episodes averaged reward 3.3
(range 2-6, steps 24-55 per life) — lower than the 16.85 training-time rolling mean, because
that mean was pulled up by rare long "rally" episodes (training's `max_episode_length` was
1193 vs. a typical ~35-55). Most individual lives are short; occasionally the agent finds a
sustained rhythm and racks up a much longer rally. This variance is expected at a 150k-step
budget — published DQN Breakout benchmarks train for millions of steps to get consistently
long rallies.

## Discussion

The clearest lesson from this sweep: hyperparameters that looked "risky" on paper (10x higher
LR, 4x larger batch) turned out to help, not hurt, at this training budget — both beat the
baseline individually, and combining them (exp10) compounded the gains rather than cancelling
out. The one hypothesis that inverted was epsilon scheduling: a slower decay with a higher
floor (exp09) performed *worse* than a fast decay (exp08), likely because sustained random
actions (10% floor for the entire run) kept injecting noise that prevented the policy from
stabilizing, rather than helping it explore into a better optimum. Full per-run reasoning is
in `experiments/edwin_results.md`; reward-curve comparisons are in
`experiments/logs/all_runs_comparison.png` and `experiments/logs/mlp_vs_cnn.png`.

## Individual Contribution — Edwin Bayingana

- Adapted the shared `train.py`/`play.py` skeleton to `ALE/Breakout-v5` (memory-safe replay
  buffer settings, MPS device support, per-episode CSV logging, video-recording playback mode).
- Designed and ran 10 hyperparameter experiments (OFAT sweep + 2 combined configs) — see
  `experiments/edwin_results.md`.
- Selected best-performing configuration and prepared the 2-minute presentation segment — see
  `PRESENTATION_NOTES_EDWIN.md`.
