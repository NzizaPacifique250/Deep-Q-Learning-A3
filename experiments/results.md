# Hyperparameter Tuning Results — All Members

Each member ran a one-factor-at-a-time (OFAT) sweep: a baseline configuration, then one
hyperparameter changed at a time (learning rate, gamma, batch size, epsilon schedule), then a
combined "best guess" config applying whatever individually helped. This isolates *why* a
change mattered instead of just reporting a final score — see each member's "Noted behavior"
column below.

Per-episode reward/length CSVs and comparison plots backing these numbers are in
[`experiments/logs/`](logs/). Raw TensorBoard event files for David's runs, and the raw
`results.csv` used to generate Nziza's numbers, are in `logs/` as reference/backup.

## Full merged table (30 runs across 3 environments)

| Member | Environment | # | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Final metric | Noted behavior |
|--------|-------------|---|----|-------|------------|-----------|---------|-----------------|---------------|-----------------|
| Edwin | Breakout | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 7.55 | Baseline reference point — steady but slow climb. |
| Edwin | Breakout | 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 10.25 | 10x higher LR **beat** baseline instead of destabilizing — didn't diverge within this budget. |
| Edwin | Breakout | 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 4.45 | Worst run — LR too low to meaningfully update Q-values in 150k steps. |
| Edwin | Breakout | 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | reward 9.4 | Lower gamma (myopic) — mild improvement, not clearly capped as hypothesized. |
| Edwin | Breakout | 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | reward 10.4 | Higher gamma — modest improvement over baseline. |
| Edwin | Breakout | 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | reward 9.1 | Smaller batch — roughly neutral to slightly positive. |
| Edwin | Breakout | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | reward 10.25 | Larger batch **beat** baseline — smoother gradients mattered more than fewer updates here. |
| Edwin | Breakout | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | reward 10.8 | Fast epsilon decay — beat baseline; early exploitation didn't cause premature lock-in. |
| Edwin | Breakout | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | reward 7.45 | Slow decay + higher floor — one of the **worst** runs; sustained randomness prevented the policy from stabilizing. |
| Edwin | Breakout | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | reward 16.85 | **Best run overall** — combining the individually-good directions compounded rather than cancelled. |
| David | SpaceInvaders | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 222.30 | Baseline reference point. |
| David | SpaceInvaders | 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 230.15 | Higher LR slightly beat baseline — same direction as Edwin's Breakout finding. |
| David | SpaceInvaders | 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | reward 281.75 | **2nd-best run** — low LR suited SpaceInvaders' larger reward scale, opposite of Edwin's Breakout result. |
| David | SpaceInvaders | 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | reward 261.70 | Lower gamma clearly helped — bigger effect than on Breakout. |
| David | SpaceInvaders | 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | reward 213.20 | Higher gamma — worst gamma setting tested. |
| David | SpaceInvaders | 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | reward 240.90 | Smaller batch beat baseline, longest average episode of any single-factor run. |
| David | SpaceInvaders | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | reward 206.80 | **Worst CNN run** — larger batch hurt here, the opposite of Edwin's Breakout result. |
| David | SpaceInvaders | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | reward 248.25 | Fast epsilon decay beat baseline — same direction as Edwin's finding. |
| David | SpaceInvaders | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | reward 219.95 | Roughly baseline-level. **Promoted as champion model** — see open item below. |
| David | SpaceInvaders | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | reward 286.55 | **Numerically best run** — same combined-config pattern winning as for Edwin. |
| Nziza | Pong | 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval reward -21.00 | Baseline stayed at the minimum evaluation score. |
| Nziza | Pong | 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval reward -21.00 | Higher LR did not improve evaluation. |
| Nziza | Pong | 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | eval reward -21.00 | Lower LR also stayed at the minimum. |
| Nziza | Pong | 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | eval reward -21.00 | Shorter horizon — no measurable gain. |
| Nziza | Pong | 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.10 | eval reward -21.00 | Longer horizon — no measurable gain. |
| Nziza | Pong | 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | eval reward -20.60 | First measurable improvement, appearing after 150k steps. |
| Nziza | Pong | 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | eval reward -20.20 | **Best run** — improvement began after 125k steps. Same "larger batch helps" direction as Edwin's Breakout result. |
| Nziza | Pong | 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | eval reward -21.00 | Faster exploitation did not improve evaluation. |
| Nziza | Pong | 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | eval reward -21.00 | Longer exploration did not improve within budget. |
| Nziza | Pong | 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.20 | eval reward -21.00 | Combined best-guess config did **not** improve — unlike Edwin's and David's combined runs, which were each member's best. |

*(Note on metrics: Edwin's and David's "final metric" is the training-time rolling-mean reward
from the last 20 episodes of the CNN sweep, extracted from per-episode CSV logs / TensorBoard.
Nziza's is the `EvalCallback` best mean reward from periodic greedy evaluation — a stricter,
more reliable metric, which is also why her numbers read as more consistently poor: her agent
genuinely never learned to win at Pong within 200k steps, rather than the metric being unusually
harsh.)*

## Policy Architecture: MLP vs CNN

| Member | Environment | CNN config | CNN result | MLP config | MLP result | Verdict |
|--------|-------------|-------------|------------|-------------|------------|---------|
| Edwin | Breakout | baseline | reward 7.55 (peak 11.9) | same hyperparams | reward 5.65 (peak 7.65) | CNN clearly wins — matches theory (convolutions exploit spatial structure in pixel frames). |
| David | SpaceInvaders | baseline / exp02 | reward 222.30 / 230.15 | same hyperparams | reward 208.80 / 236.05 | Comparable — MLP even edges ahead on the higher-LR config. Theoretical CNN advantage is real but this environment's regular, repetitive visual field may be easier for an MLP to approximate than Breakout's fast ball-tracking. |
| Nziza | Pong | best config, -20.20 | -20.20 | not run | — | **Outstanding** — Nziza's MLP comparison was never executed. |

## Best configuration per member

- **Edwin (Breakout):** exp10 — `lr=5e-4, gamma=0.99, batch_size=64, eps_end=0.02, eps_decay_frac=0.15`. Final reward 16.85 (peak 20.9), most sample-efficient run. Promoted to `models/dqn_model_edwin_breakout.zip`.
- **David (SpaceInvaders):** numerically best is exp10 (reward 286.55), but the **promoted model is exp09** (reward 219.95). This is an open item — David should confirm the reasoning (e.g. video/qualitative preference) or re-promote exp10 before presenting.
- **Nziza (Pong):** exp07 — `lr=1e-4, gamma=0.99, batch_size=128`. Best mean evaluation reward -20.20 — still close to Pong's -21 minimum; documented honestly as a negative result rather than overstated.

## Cross-member insights

- **Batch size had opposite effects on Breakout vs. SpaceInvaders.** 128 was one of the best changes for Edwin, the single worst for David. No hyperparameter effect generalizes across environments by default.
- **Larger batch size helped in 2 of 3 environments** (Edwin's Breakout, Nziza's best Pong run) — a mild pattern worth mentioning, but SpaceInvaders' opposite result shows it isn't universal.
- **Learning rate was more forgiving than expected almost everywhere** — a 10x higher LR helped, not hurt, on both Breakout and SpaceInvaders. It only failed to move anything on Pong, where *nothing* moved the needle — suggesting Pong's difficulty at 200k steps is about training budget, not hyperparameters.
- **Combined "best guess" configs won for 2 of 3 members** (Edwin, David by raw score) — suggesting individually-tested improvements were additive. Nziza's combined config did *not* win, consistent with her overall finding that no hyperparameter change escaped the -21 floor.
