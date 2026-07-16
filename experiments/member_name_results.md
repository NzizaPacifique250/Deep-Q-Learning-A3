# Hyperparameter Experiments — <Member Name>

Environment: `ALE/Pong-v5`
Fixed training length: `200,000` timesteps (keep the same across all 10 runs unless testing timesteps itself).
Policy: `CnnPolicy` (MLP used only for the architecture-comparison baseline).

Each row changes mostly ONE knob vs the baseline (row 1) so the effect is attributable.
Fill "Best mean reward" from `experiments/results.csv` (or `summarize_results.py`) and add a
short qualitative note. On Pong, reward ranges from −21 (loses every point) to +21 (perfect).

| # | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Best mean reward | Noted Behavior |
|---|------|-------|------------|-----------|---------|-----------------|------------------|-----------------|
| 1 | 1e-4 | 0.99  | 32  | 1.0 | 0.05 | 0.10 |  | Baseline |
| 2 | 5e-4 | 0.99  | 32  | 1.0 | 0.05 | 0.10 |  | Higher lr — faster but watch for instability |
| 3 | 5e-5 | 0.99  | 32  | 1.0 | 0.05 | 0.10 |  | Lower lr — slower, smoother |
| 4 | 1e-4 | 0.95  | 32  | 1.0 | 0.05 | 0.10 |  | Short horizon — more short-sighted |
| 5 | 1e-4 | 0.999 | 32  | 1.0 | 0.05 | 0.10 |  | Long horizon — values delayed reward |
| 6 | 1e-4 | 0.99  | 64  | 1.0 | 0.05 | 0.10 |  | Larger batch — steadier updates |
| 7 | 1e-4 | 0.99  | 128 | 1.0 | 0.05 | 0.10 |  | Largest batch — slower per step |
| 8 | 1e-4 | 0.99  | 32  | 1.0 | 0.01 | 0.10 |  | Exploit more (low final ε) |
| 9 | 1e-4 | 0.99  | 32  | 1.0 | 0.10 | 0.30 |  | Explore longer (slow decay) |
| 10| 5e-4 | 0.99  | 64  | 1.0 | 0.02 | 0.20 |  | Combined best-guess config |

## Best configuration

- Run #:
- Config:
- Why it performed best:

## Key takeaways (for presentation, max 2 min)

- What helped:
- What hurt:
- Final config + reasoning:
