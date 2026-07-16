# Hyperparameter Experiments — Nzizapacifique250

- Environment: `ALE/Pong-v5`
- Policy: `CnnPolicy`
- Training length: `200,000` timesteps per run
- Seed: `0`

| # | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Best mean reward | Noted behavior |
|---|----|-------|------------|-----------|---------|----------------|------------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Baseline stayed at the minimum evaluation score. |
| 2 | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Higher learning rate did not improve evaluation. |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Lower learning rate did not improve evaluation. |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Shorter reward horizon produced no measurable gain. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 0.10 | -21.00 | Longer reward horizon produced no measurable gain. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | -20.60 | First measurable improvement, appearing after 150k steps. |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | -20.20 | Best run; improvement began after 125k steps. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.10 | -21.00 | Faster exploitation did not improve evaluation. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | -21.00 | Longer exploration did not improve within the budget. |
| 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.20 | -21.00 | Combined best-guess configuration did not improve. |

## Best configuration

- Run: `nzizapacifique250_exp07`
- Configuration: `lr=1e-4`, `gamma=0.99`, `batch_size=128`, `eps_start=1.0`,
  `eps_end=0.05`, `eps_decay_frac=0.10`
- Best mean evaluation reward: `-20.20`
- Promoted model: `models/dqn_model.zip`

The batch-128 run achieved the strongest greedy evaluation, ahead of the batch-64 run at
`-20.60`. All other tested configurations remained at `-21.00`. Because every run used one
seed and the winning score is still close to Pong's minimum, the result should be treated as an
early-learning comparison rather than evidence of a strong final policy. Longer training and
multiple seeds would provide a more reliable conclusion.
