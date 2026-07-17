# DQN Atari Agent — Group Project

Training and evaluating a **DQN** agent (Stable Baselines3 + Gymnasium) on **`ALE/Pong-v5`**.

Pong is chosen for its clear, dense-ish reward signal (+1 when the agent scores, −1 when it
concedes; episode score ranges from −21 to +21) and relatively fast learning, which makes
hyperparameter effects visible within a modest training budget.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only wheel
pip install -r requirements.txt
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Trains a DQN agent; logs reward + episode length; saves best & final models |
| `play.py` | Loads a trained model and plays greedily (`deterministic=True`), rendering gameplay |
| `run_experiments.py` | Runs one member's 10 hyperparameter experiments in sequence |
| `summarize_results.py` | Turns `experiments/results.csv` into a Markdown table |
| `experiments/` | Per-member experiment logs, monitor CSVs, and `results.csv` |
| `models/` | Saved models (submit your best as `models/dqn_model.zip`) |

## Usage

**Single training run:**
```bash
python train.py --lr 1e-4 --gamma 0.99 --batch-size 32 --run-name alice_exp01
```
This writes:
- `models/dqn_model_<run>.zip` (final) and `models/<run>/best_model.zip` (best during eval)
- reward/episode-length CSVs under `experiments/monitor/<run>/`
- a summary row in `experiments/results.csv`
- TensorBoard logs under `tb_logs/` — view with `tensorboard --logdir tb_logs`

**Run all 10 experiments** (edit `MEMBER` and the `EXPERIMENTS` list first):
```bash
python run_experiments.py --member alice --timesteps 200000
python summarize_results.py --member alice     # prints a Markdown table
```

**MLP vs CNN comparison** (same hyperparameters, different policy):
```bash
python train.py --policy CnnPolicy --run-name cnn_baseline
python train.py --policy MlpPolicy --run-name mlp_baseline
```

**Watch the trained agent play** (greedy / GreedyQPolicy-equivalent):
```bash
cp models/<best-run>/best_model.zip models/dqn_model.zip   # promote your best model
python play.py --model models/dqn_model.zip --episodes 3
```

## Policy Architecture: MLP vs CNN

Atari observations are raw **84×84×4 stacked grayscale frames** (image data).

- **CnnPolicy** feeds these frames through convolutional layers (the classic DQN Nature
  architecture). Convolutions share weights across the image and preserve spatial structure,
  so the network can learn features like "the ball is here, the paddle is there" efficiently.
- **MlpPolicy** first *flattens* the frames into a long vector, discarding spatial locality.
  It must learn every pixel-position relationship independently, needing far more parameters
  and data to reach the same understanding.

**Conclusion:** `CnnPolicy` preserves the spatial structure of Atari frames and is therefore
used for all tuning experiments. `MlpPolicy` remains available as an optional architecture
baseline.

## Hyperparameter Tuning Results (merged from all members)

| Member | Run | lr | gamma | batch | eps_start | eps_end | eps_decay | Best mean reward | Noted behavior |
|--------|-----|----|-------|-------|-----------|---------|-----------|------------------|----------------|
| Nzizapacifique250 | exp01 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Baseline stayed at the minimum evaluation score. |
| Nzizapacifique250 | exp02 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | Higher learning rate did not improve evaluation. |
| Nzizapacifique250 | exp03 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | Lower learning rate also stayed at -21.00. |
| Nzizapacifique250 | exp04 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | Shorter reward horizon produced no measurable gain. |
| Nzizapacifique250 | exp05 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.10 | Longer reward horizon produced no measurable gain. |
| Nzizapacifique250 | exp06 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | First measurable improvement; peaked at -20.60. |
| Nzizapacifique250 | exp07 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | Best run; improved after 125k steps and peaked at -20.20. |
| Nzizapacifique250 | exp08 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | Faster exploitation did not improve evaluation. |
| Nzizapacifique250 | exp09 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | Longer exploration did not improve within 200k steps. |
| Nzizapacifique250 | exp10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.20 | Combined configuration stayed at -21.00. |

_(10 Nzizapacifique250 rows completed; other members can append their results.)_

## Final Chosen Configuration

- **Member:** Nzizapacifique250
- **Run:** `nzizapacifique250_exp07`
- **Config:** `lr=1e-4, gamma=0.99, batch_size=128, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.10`
- **Best mean reward:** `-20.20`
- **Reasoning:** This was the strongest greedy evaluation result in the sweep. It began
  improving after 125,000 steps and outperformed the next-best batch-64 run (`-20.60`). Its
  best checkpoint is promoted to `models/dqn_model.zip`.

## Gameplay Evaluation

Run the promoted model headlessly with:

```bash
python play.py --model models/dqn_model.zip --episodes 3 --no-render
```

## Discussion

- **What helped:** Increasing the batch size was the only tested change that produced a
  measurable greedy-evaluation improvement. Batch 64 reached `-20.60`, while batch 128 reached
  `-20.20`.
- **What did not help within this budget:** Changing learning rate, gamma, final epsilon, or
  exploration decay left the best mean evaluation at `-21.00` in these single-seed runs.
- **Interpretation:** The winning model is still close to the minimum Pong score, so 200,000
  timesteps were enough to compare early learning behavior but not enough to produce a strong
  player. More timesteps and multiple seeds are needed before making a robust general claim.
