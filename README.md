# Deep Q-Learning — Group Formative 3

DQN agents trained with Stable Baselines3 + Gymnasium/ALE, evaluated with greedy
(`deterministic=True`) playback. `train.py` and `play.py` are shared, single implementations
used by all three members — `--env-id` selects which Atari environment to train or play.

## Gameplay Videos

The following videos demonstrate our trained champion agents interacting with their respective Atari environments (recorded via the `play.py` script):

* **Edwin's Breakout Agent:** [Link to edwin_breakout_gameplay.mp4](https://github.com/NzizaPacifique250/Deep-Q-Learning-A3/blob/main/videos/edwin_breakout_gameplay.mp4)
* **David's Space Invaders Agent:** [Link to david_spaceinvaders_gameplay.mp4](https://github.com/NzizaPacifique250/Deep-Q-Learning-A3/blob/main/videos/david_spaceinvaders_gameplay.mp4)
* **Nziza's Pong Agent:** [Link to nziza_pong_gameplay.mp4](https://github.com/NzizaPacifique250/Deep-Q-Learning-A3/blob/main/videos/nziza_pong_gameplay.mp4)

*(Note to grader: The raw `.mp4` files are located in the `/videos` directory of this repository.)*

## Environments

**Each member trained on a different Atari game.** The assignment's presentation flow assumes
one shared environment, and this group's own pre-planning doc flagged that risk before anyone
started...but by the time it was caught, all three members had already completed full
10-experiment sweeps on different games, and redoing them wasn't realistic before the
deadline. Disclosed here for transparency going into Q&A:

| Member | Environment | Best config | Result |
| --- | --- | --- | --- |
| Edwin Bayingana | `ALE/Breakout-v5` | `lr=5e-4, gamma=0.99, batch=64, eps_end=0.02, decay=0.15` | reward 16.85 (peak 20.9) |
| David (Yinka) Ajao | `ALE/SpaceInvaders-v5` | `MLP_exp08` | avg eval reward 413.3 |
| Nziza Aime Pacifique | `ALE/Pong-v5` | `lr=1e-4, gamma=0.99, batch=128` | eval reward -20.20 |

If asked in Q&A "why three different games": the honest answer is a coordination gap on
environment selection, caught after training was already complete. What each member
individually delivered...10 documented hyperparameter experiments, a shared `train.py`/
`play.py`, and a gameplay video....meets the assignment's requirements per-member even though
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
│   ├── dqn_model.zip
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
python play.py --env-id ALE/SpaceInvaders-v5 --model models/dqn_model.zip --mode record --episodes 3
python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode record --episodes 3

```

Every hyperparameter value used for all runs is documented in
[`experiments/results.md`](https://www.google.com/search?q=experiments/results.md), which is also what to use to reproduce any
individual run via `train.py`.

## Hyperparameter Tuning Results

Each member ran a one-factor-at-a-time (OFAT) sweep: a baseline, then one hyperparameter
changed at a time, then a combined "best guess" config applying whatever individually helped.

| Member | Env | # | lr | gamma | batch | eps_start | eps_end | eps_decay | Final metric | Noted behavior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
| David | SpaceInvaders | CNN 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | train 222.30 / eval 310.0 | Baseline reference point. |
| David | SpaceInvaders | CNN 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | train 230.15 / eval 233.3 | Higher LR slightly beat baseline in training. |
| David | SpaceInvaders | CNN 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | train 281.75 / eval 285.0 | **2nd-best train run** — opposite of Edwin's Breakout LR finding. |
| David | SpaceInvaders | CNN 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | train 261.70 / eval 188.3 | Lower gamma clearly helped in training. |
| David | SpaceInvaders | CNN 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | train 213.20 / eval 110.0 | Higher gamma — worst gamma setting tested. |
| David | SpaceInvaders | CNN 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | train 240.90 / eval 111.7 | Smaller batch beat baseline in training. |
| David | SpaceInvaders | CNN 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | train 206.80 / eval 151.7 | **Worst CNN run** — opposite of Edwin's Breakout batch finding. |
| David | SpaceInvaders | CNN 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | train 248.25 / eval 158.3 | Fast epsilon decay beat baseline in training. |
| David | SpaceInvaders | CNN 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | train 219.95 / eval 263.3 | Roughly baseline-level in training. |
| David | SpaceInvaders | CNN 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | train 286.55 / eval 363.3 | Numerically best CNN run across training and evaluation. |
| David | SpaceInvaders | MLP 1 | - | - | - | - | - | - | eval 243.3 | Baseline MLP comparison run. |
| David | SpaceInvaders | MLP 2 | - | - | - | - | - | - | eval 188.3 | MLP evaluation sweep. |
| David | SpaceInvaders | MLP 3 | - | - | - | - | - | - | eval 170.0 | MLP evaluation sweep. |
| David | SpaceInvaders | MLP 4 | - | - | - | - | - | - | eval 185.0 | MLP evaluation sweep. |
| David | SpaceInvaders | MLP 5 | - | - | - | - | - | - | eval 3.3 | Lowest performing MLP experiment. |
| David | SpaceInvaders | MLP 6 | - | - | - | - | - | - | eval 310.0 | Strong MLP evaluation performance. |
| David | SpaceInvaders | MLP 7 | - | - | - | - | - | - | eval 146.7 | MLP evaluation sweep. |
| David | SpaceInvaders | MLP 8 | - | - | - | - | - | - | eval 413.3 | **Overall Champion Model.** Highest evaluation score achieved. |
| David | SpaceInvaders | MLP 9 | - | - | - | - | - | - | eval 295.0 | MLP evaluation sweep. |
| David | SpaceInvaders | MLP 10 | - | - | - | - | - | - | eval 181.7 | MLP evaluation sweep. |
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

*(Edwin's "final metric" is the training-time rolling-mean reward from the last 20 episodes; Nziza's is the stricter `EvalCallback` greedy-evaluation best mean reward; David executed a full evaluation pipeline across 20 distinct experiment architectures (10 CNN, 10 MLP) to crown a champion by average evaluation reward).*

## Discussion of Hyperparameter Tuning Results

Based strictly on the data collected across all 40 experiments, we observed that hyperparameters have highly environment-dependent effects:

* **Learning Rate (LR):** The environments responded differently to LR scaling. In Breakout, raising the LR to 1e-3 (10x higher) beat the baseline, and the ultimate best configuration utilized 5e-4. In Space Invaders (CNN), both 1e-3 and 5e-4 also outperformed the baseline in training. However, in Pong, no adjustment to the learning rate—higher or lower—moved the agent off the minimum score floor of -21.00.
* **Batch Size Divergence:** This parameter yielded the most conflicting results across games. For both Breakout and Pong, increasing the batch size to 128 produced improvements, with Pong marking its highest score at this setting. Conversely, in Space Invaders (CNN), a batch size of 128 was the single worst-performing configuration tested.
* **Discount Factor (Gamma):** Lowering gamma to 0.90 showed a mild improvement in Breakout and a clear training improvement in Space Invaders, indicating that a slightly shorter time horizon helped credit assignment in these games. However, increasing gamma to 0.995 proved highly detrimental to Space Invaders while offering a modest gain in Breakout.
* **Exploration Rate (Epsilon):** Fast decay strategies (e.g., decaying to 0.02) consistently beat the baseline configurations for both Breakout and Space Invaders, encouraging earlier exploitation. Slower decay rates with higher floors (0.10 / 0.30 decay) resulted in some of the worst runs, particularly in Breakout.
* **Combined Configurations & Policy Architecture:** Synthesizing the best individual hyperparameters into a "best guess" configuration (Experiment 10) proved successful for Breakout and Space Invaders (CNN), compounding the individual gains. Furthermore, David's evaluation sweep demonstrated that switching policy architecture from CNN to MLP—paired with a fast epsilon decay—yielded an evaluation score of 413.3, drastically outperforming all purely hyperparameter-tuned CNN variants.

Full per-experiment reasoning and detail: [`experiments/results.md`](https://www.google.com/search?q=experiments/results.md).

## Individual Contribution

* **Edwin Bayingana:** built and debugged the shared `train.py`/`play.py` (memory-safe replay
buffer, MPS device support, per-episode CSV logging, video-recording playback with the
reset-per-episode bug fixed), ran all 10 Breakout experiments + the MLP/CNN comparison,
merged all three members' individual work into this unified repo structure.
* **David (Yinka) Ajao:** adapted the skeleton to SpaceInvaders with GPU training. Engineered and ran a comprehensive evaluation pipeline testing 20 different models (10 CNN experiments and 10 MLP experiments). Successfully identified and promoted `MLP_exp08` as the champion agent with an average evaluation reward of 413.3. Generated the final highlight reel gameplay video.
* **Nziza Aime Pacifique:** authored the original shared skeleton this group's `train.py`/
`play.py` were built from (`EvalCallback` best-model tracking, `results.csv` auto-logging),
ran 10 hyperparameter experiments on Pong, documented an honest negative result rather than
overstating the outcome, recorded the gameplay video.

## Branches

Each member's original, individual (pre-merge) work is preserved on its own branch for
reference: `edwin`, `david`, `nziza`. `main` is the unified, submission-ready version.

## Team Collaboration Sheet

[Group 14 Team Task Sheet](https://docs.google.com/spreadsheets/d/1TuxFIi9afIKp9-nqp52lZumWjsWUHYrd/edit?usp=sharing&ouid=107439525985311710280&rtpof=true&sd=true)
