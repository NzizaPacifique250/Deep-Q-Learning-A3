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

| Member               | Environment            | Best config                                                 | Result                   |
| -------------------- | ---------------------- | ----------------------------------------------------------- | ------------------------ |
| Edwin Bayingana      | `ALE/Breakout-v5`      | `lr=5e-4, gamma=0.99, batch=64, eps_end=0.02, decay=0.15`   | reward 16.85 (peak 20.9) |
| David (Yinka) Ajao   | `ALE/SpaceInvaders-v5` | `lr=1e-4, gamma=0.99, batch=32, eps_end=0.10, decay=0.30`\* | reward 219.95\*          |
| Nziza Aime Pacifique | `ALE/Pong-v5`          | `lr=1e-4, gamma=0.99, batch=128`                            | eval reward -20.20       |

\*David's promoted model is exp09 (reward 219.95). His own logs show exp10 scoring higher
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

| Member | Environment | CNN config | CNN result | MLP config | MLP result | Verdict |
|--------|-------------|-------------|------------|-------------|------------|---------|
| Edwin | Breakout | baseline | reward 7.55 (peak 11.9) | same hyperparams | reward 5.65 (peak 7.65) | CNN clearly wins — matches theory (convolutions exploit spatial structure in pixel frames). |
| David | SpaceInvaders | baseline / exp02 | reward 222.30 / 230.15 | same hyperparams | reward 208.80 / 236.05 | Comparable — MLP even edges ahead on the higher-LR config. |
| Nziza | Pong | best config, -20.20 | -20.20 | not run | — | Outstanding — never executed. |

CNN clearly beat MLP on Breakout (the textbook result), the two were comparable on
SpaceInvaders (a genuine, environment-dependent nuance worth raising in Q&A), and the
comparison was not run for Pong. Full reasoning: [`experiments/results.md`](experiments/results.md#policy-architecture-mlp-vs-cnn).

## Hyperparameter Tuning Results

Each member ran a one-factor-at-a-time (OFAT) sweep: a baseline, then one hyperparameter
changed at a time, then a combined "best guess" config applying whatever individually helped.

| Member | Env | # | lr | gamma | batch | eps_start | eps_end | eps_decay | Final metric | Noted behavior |
|--------|-----|---|----|-------|-------|-----------|---------|-----------|---------------|-----------------|
| Edwin | Breakout | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 7.55 | Baseline reference point. |
| Edwin | Breakout | 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 10.25 | 10x higher LR **beat** baseline instead of destabilizing. |
| Edwin | Breakout | 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 4.45 | Worst run — LR too low to learn meaningfully in 150k steps. |
| Edwin | Breakout | 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | reward 9.4 | Lower gamma — mild improvement. |
| Edwin | Breakout | 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | reward 10.4 | Higher gamma — modest improvement. |
| Edwin | Breakout | 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | reward 9.1 | Smaller batch — roughly neutral. |
| Edwin | Breakout | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | reward 10.25 | Larger batch **beat** baseline. |
| Edwin | Breakout | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | reward 10.8 | Fast epsilon decay beat baseline. |
| Edwin | Breakout | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | reward 7.45 | Slow decay + higher floor — one of the **worst** runs. |
| Edwin | Breakout | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | reward 16.85 | **Best run overall** — combined config compounded individual gains. |
| David | SpaceInvaders | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 222.30 | Baseline reference point. |
| David | SpaceInvaders | 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 230.15 | Higher LR slightly beat baseline. |
| David | SpaceInvaders | 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 281.75 | **2nd-best run** — opposite of Edwin's Breakout LR finding. |
| David | SpaceInvaders | 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | reward 261.70 | Lower gamma clearly helped. |
| David | SpaceInvaders | 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | reward 213.20 | Higher gamma — worst gamma setting tested. |
| David | SpaceInvaders | 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | reward 240.90 | Smaller batch beat baseline. |
| David | SpaceInvaders | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | reward 206.80 | **Worst CNN run** — opposite of Edwin's Breakout batch finding. |
| David | SpaceInvaders | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | reward 248.25 | Fast epsilon decay beat baseline. |
| David | SpaceInvaders | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | reward 219.95 | Roughly baseline-level. **Promoted champion model.** |
| David | SpaceInvaders | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | reward 286.55 | **Numerically best run** — same combined-config pattern as Edwin. |
| Nziza | Pong | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval -21.00 | Baseline stayed at the minimum score. |
| Nziza | Pong | 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval -21.00 | Higher LR did not improve evaluation. |
| Nziza | Pong | 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval -21.00 | Lower LR also stayed at the minimum. |
| Nziza | Pong | 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | eval -21.00 | Shorter horizon — no measurable gain. |
| Nziza | Pong | 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.10 | eval -21.00 | Longer horizon — no measurable gain. |
| Nziza | Pong | 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | eval -20.60 | First measurable improvement, after 150k steps. |
| Nziza | Pong | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | eval -20.20 | **Best run** — same "larger batch helps" direction as Edwin. |
| Nziza | Pong | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | eval -21.00 | Faster exploitation did not improve evaluation. |
| Nziza | Pong | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | eval -21.00 | Longer exploration did not improve within budget. |
| Nziza | Pong | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.20 | eval -21.00 | Combined best-guess config did **not** improve, unlike Edwin's and David's. |

*(Edwin's/David's "final metric" is the training-time rolling-mean reward from the last 20
episodes; Nziza's is the stricter `EvalCallback` greedy-evaluation best mean reward — which is
also why her numbers read as more consistently poor: her agent genuinely never learned to win
at Pong within 200k steps, rather than the metric being unusually harsh.)*

### Cross-member insights

- **Batch size had opposite effects on Breakout vs. SpaceInvaders.** 128 was one of the best changes for Edwin, the single worst for David.
- **Learning rate was more forgiving than expected on Breakout and SpaceInvaders** (10x higher helped, not hurt on both) but irrelevant on Pong, where nothing moved the needle.
- **Combined "best guess" configs won for 2 of 3 members** (Edwin, David by raw score) — individually-tested improvements were additive rather than conflicting.

Full per-experiment reasoning and detail: [`experiments/results.md`](experiments/results.md).

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

## Branches

Each member's original, individual (pre-merge) work is preserved on its own branch for
reference: `edwin`, `david`, `nziza`. `main` is the unified, submission-ready version.
