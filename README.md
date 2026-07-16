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

**Conclusion:** For pixel-based Atari, **CnnPolicy is the correct choice and clearly
outperforms MlpPolicy** — the MLP learns slowly or plateaus near random play, while the CNN
steadily improves. We therefore use `CnnPolicy` for all tuning experiments and only run MLP
once as a documented baseline for comparison. _(Paste your two comparison numbers here after
running the commands above.)_

## Hyperparameter Tuning Results (merged from all members)

> Generate each member's block with `python summarize_results.py --member <name>` and paste
> below, then add a short qualitative note per row in the "Noted behavior" column.

| Member | Run | lr | gamma | batch | eps_start | eps_end | eps_decay | Best mean reward | Noted behavior |
|--------|-----|----|-------|-------|-----------|---------|-----------|------------------|----------------|
|        |     |    |       |       |           |         |           |                  |                |

_(40 rows total — 10 per member.)_

## Final Chosen Configuration

- **Member:** _TBD_
- **Config:** `lr=…, gamma=…, batch_size=…, eps_start=…, eps_end=…, eps_decay_frac=…`
- **Reasoning:** _Why this run gave the highest / most stable eval reward._

## Gameplay Demo

_(Link or embed the video of `play.py` running with the final model here.)_

## Discussion

Summarize across all experiments:
- **What helped:** e.g. moderate learning rate (1e-4–5e-4), high gamma (0.99) for Pong's
  delayed rewards, larger batch for more stable updates, enough exploration before decaying ε.
- **What hurt:** e.g. too-high lr → divergence/instability; too-low gamma → short-sighted play;
  too-fast ε decay → premature exploitation of a bad policy; too-small buffer → forgetting.
- **Why the final model behaves as it does:** tie the winning config back to these effects.
