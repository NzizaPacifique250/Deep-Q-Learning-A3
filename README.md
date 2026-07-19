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
| David (Yinka) Ajao | `ALE/SpaceInvaders-v5` | `lr=5e-4, gamma=0.99, batch=64, eps_end=0.02, decay=0.15`* | reward 286.55* |
| Nziza Aime Pacifique | `ALE/Pong-v5` | `lr=1e-4, gamma=0.99, batch=128` | eval reward -20.20 |

*David's numerically-best run is exp10, but the model currently promoted in this repo is
exp09 (reward 219.95) — see `experiments/results.md` for the open item to resolve.

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
├── RUN_INSTRUCTIONS.md           setup + exact commands to reproduce every run
├── PRESENTATION_NOTES.md         Q&A prep, one section per member
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
    └── nziza_pong_gameplay.mp4        (outstanding — see Submission Checklist)
```

## Usage

Train:
```bash
python train.py --env-id ALE/Breakout-v5 --lr 5e-4 --gamma 0.99 --batch-size 64 --eps-end 0.02 --eps-decay-frac 0.15 --run-name edwin_exp10 --device auto
```

Watch a trained agent play and record the submission video:
```bash
python play.py --env-id ALE/Breakout-v5 --model models/dqn_model_edwin_breakout.zip --mode record --episodes 3
python play.py --env-id ALE/SpaceInvaders-v5 --model models/dqn_model_david_spaceinvaders.zip --mode record --episodes 3
python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode record --episodes 3
```

Full per-member commands (all 30 experiments, exact hyperparameters) are in
[`RUN_INSTRUCTIONS.md`](RUN_INSTRUCTIONS.md).

## Policy Architecture: MLP vs CNN

Summary table and full discussion in [`experiments/results.md`](experiments/results.md#policy-architecture-mlp-vs-cnn).
Short version: CNN clearly beat MLP on Breakout (the textbook result), the two were
comparable on SpaceInvaders (a genuine, environment-dependent nuance worth raising in Q&A),
and the comparison was never run for Pong.

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
  overstating the outcome.

## Submission Checklist

- [x] `train.py` and `play.py` — single shared scripts, `--env-id` selects the environment
- [x] Merged hyperparameter table, 30 documented experiments across 3 environments
- [x] Each member's best model in `models/`
- [x] Edwin's and David's gameplay videos in `videos/`
- [ ] **Nziza's gameplay video** — outstanding, she's sending it separately. Run
      `python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode record --episodes 3`
      and add the `.mp4` to `videos/` once available.
- [ ] **David: resolve the exp09-vs-exp10 champion discrepancy** (see `experiments/results.md`).
- [ ] **Nziza's MLP vs CNN comparison** — not yet executed.
- [ ] **Coach booking sheet** — copy the assignment's Google Sheet, fill it, save as PDF, add to this repo. Every group must book a Week 6 slot.
- [ ] Rehearse the 10-minute presentation using `PRESENTATION_NOTES.md`.
- [ ] Zip the repo (Attempt 1) or confirm the pushed repo URL (Attempt 2).

## Branches

Each member's original, individual (pre-merge) work is preserved on its own branch for
reference: `edwin`, `david`, `nziza`. `main` is the unified, submission-ready version.
