# Deep Q-Learning — Group Formative 3

DQN agents trained with Stable Baselines3 + Gymnasium/ALE, evaluated with greedy
(`deterministic=True`) playback. `train.py` and `play.py` are shared, single implementations
used by all three members — `--env-id` selects which Atari environment to train or play.

## Environments

**Each member trained on a different Atari game.** The assignment's presentation flow assumes
one shared environment, and this group's own pre-planning doc flagged that risk before anyone
started — but by the time it was caught, all three members had already completed full
10-experiment sweeps on different games, and redoing them wasn't realistic before the
deadline. Disclosed here for transparency going into Q&A:

| Member | Environment | Best config | Result |
|--------|-------------|--------------|--------|
| Edwin Bayingana | `ALE/Breakout-v5` | `lr=5e-4, gamma=0.99, batch=64, eps_end=0.02, decay=0.15` | reward 16.85 (peak 20.9) |
| David (Yinka) Ajao | `ALE/SpaceInvaders-v5` | `lr=1e-4, gamma=0.99, batch=32, eps_end=0.10, decay=0.30`* | reward 219.95* |
| Nziza Aime Pacifique | `ALE/Pong-v5` | `lr=1e-4, gamma=0.99, batch=128` | eval reward -20.20 |

*David's promoted model is exp09 (reward 219.95). His own logs show exp10 scoring higher
(286.55, config `lr=5e-4, gamma=0.99, batch=64, eps_end=0.02, decay=0.15`) — this discrepancy
is unresolved; see `experiments/results.md` for detail and be ready to address it in Q&A.

If asked in Q&A "why three different games": the honest answer is a coordination gap on
environment selection, caught after training was already complete. What each member
individually delivered — 10 documented hyperparameter experiments, a shared `train.py`/
`play.py`, and a gameplay video — meets the assignment's requirements per-member even though
cross-member environment consistency does not. Each member's individual, unmerged work is
also preserved on its own branch (`edwin`, `david`, `nziza`) for reference.

## Setup

```bash
pip install -r requirements.txt
```

Apple Silicon (M-series): `--device auto` picks up `mps` automatically; fall back to
`--device cpu` if you hit an "operator not implemented for MPS" error. Colab/GPU boxes: pass
`--device cuda` explicitly.

## Files

```
Deep-Q-Learning-A3/
├── train.py                      shared training script (--env-id required)
├── play.py                       shared playback/recording script (--env-id required)
├── requirements.txt
├── experiments/
│   ├── results.md                 merged 30-run hyperparameter table + cross-member analysis
│   ├── plot_results.py            overlay reward-trend CSVs into a comparison chart
│   ├── summarize_results.py       print final/peak reward per run from a CSV log
│   └── logs/                      per-run CSVs, TensorBoard-derived numbers, comparison plots
├── models/
│   ├── dqn_model_edwin_breakout.zip
│   ├── dqn_model_david_spaceinvaders.zip
│   └── dqn_model_nziza_pong.zip
└── videos/
    ├── edwin_breakout_gameplay.mp4
    ├── david_spaceinvaders_gameplay.mp4
    └── nziza_pong_gameplay.mp4
```

## Usage

Train:
```bash
python train.py --env-id ALE/Breakout-v5 --lr 5e-4 --gamma 0.99 --batch-size 64 --eps-end 0.02 --eps-decay-frac 0.15 --run-name edwin_exp10 --device auto
```

Watch a trained agent play and record a gameplay video:
```bash
python play.py --env-id ALE/Breakout-v5 --model models/dqn_model_edwin_breakout.zip --mode record --episodes 3
python play.py --env-id ALE/SpaceInvaders-v5 --model models/dqn_model_david_spaceinvaders.zip --mode record --episodes 3
python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode record --episodes 3
```

Every hyperparameter value used for all 30 runs (10 per member) is documented in
[`experiments/results.md`](experiments/results.md), which is also what to use to reproduce any
individual run via `train.py`.

## Policy Architecture: MLP vs CNN

Summary table and full discussion in [`experiments/results.md`](experiments/results.md#policy-architecture-mlp-vs-cnn).
Short version: CNN clearly beat MLP on Breakout (the textbook result), the two were
comparable on SpaceInvaders (a genuine, environment-dependent nuance worth raising in Q&A),
and the comparison was not run for Pong.

## Hyperparameter Tuning Results

Full merged table (30 runs, all three environments, each with a noted-behavior explanation)
plus cross-member insights: [`experiments/results.md`](experiments/results.md).

## Discussion

The clearest cross-cutting lesson: hyperparameter effects don't generalize across
environments by default. Batch size 128 was one of the best changes on Breakout and the worst
on SpaceInvaders. Learning rate was more forgiving than expected on both Breakout and
SpaceInvaders (10x higher helped, not hurt) but irrelevant on Pong, where nothing moved the
needle — pointing at training budget, not hyperparameters, as Pong's real bottleneck. Combined
"best guess" configurations won for two of three members, suggesting individually-validated
improvements compounded rather than conflicted when stacked together. Full reasoning per
experiment is in `experiments/results.md`.

## Individual Contribution

- **Edwin Bayingana:** built and debugged the shared `train.py`/`play.py` (memory-safe replay
  buffer, MPS device support, per-episode CSV logging, video-recording playback with the
  reset-per-episode bug fixed), ran all 10 Breakout experiments + the MLP/CNN comparison,
  merged all three members' individual work into this unified repo structure.
- **David (Yinka) Ajao:** adapted the skeleton to SpaceInvaders with GPU training, ran 10
  hyperparameter experiments plus 2 MLP-vs-CNN comparison runs, recorded the gameplay video.
- **Nziza Aime Pacifique:** authored the original shared skeleton this group's `train.py`/
  `play.py` were built from (`EvalCallback` best-model tracking, `results.csv` auto-logging),
  ran 10 hyperparameter experiments on Pong, documented an honest negative result rather than
  overstating the outcome, recorded the gameplay video.

## Known gaps

Being upfront about what this repo does not yet include, rather than leaving it to be
discovered:

- **David's promoted model (exp09) isn't his best-scoring run (exp10).** Explain this in the
  presentation, or re-promote exp10 before submitting — see `experiments/results.md`.
- **Nziza's MLP vs CNN comparison was never run** — the assignment's explicit
  "compare MLPPolicy and CNNPolicy" requirement only has real data for two of three members.
- **Coach booking sheet** (copy the assignment's Google Sheet, fill it, save as PDF, add to
  the repo) is not in this repo.

## Branches

Each member's original, individual (pre-merge) work is preserved on its own branch for
reference: `edwin`, `david`, `nziza`. `main` is the unified, submission-ready version.
