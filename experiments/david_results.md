# Hyperparameter Experiments — David (Yinka) Ajao

- Environment: `ALE/SpaceInvaders-v5`
- Policy: `CnnPolicy` (primary), plus 2 `MlpPolicy` comparison runs
- Training length: `150,000` timesteps per run
- Model naming in this repo uses "Yinka_Ajao" as the run-name prefix (David's own naming from `run_experiments.py`)

> **Note on this table:** these runs were trained and logged to TensorBoard, but not written
> up in a results file. The final/max reward and episode-length numbers below were pulled
> directly from each run's `tb_logs/` event file (`rollout/ep_rew_mean`, `rollout/ep_len_mean`),
> so they're real, not estimated — but the "Noted behavior" column is a reconstruction from
> those numbers rather than notes taken live during training. David, sanity-check these against
> your own recollection before presenting.

| # | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Final ep_rew_mean | Max ep_rew_mean | Final ep_len_mean | Noted behavior |
|---|----|-------|------------|-----------|---------|-----------------|--------------------|------------------|---------------------|----------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 222.30 | 279.20 | 626.07 | Baseline reference point — solid mid-pack performance. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 230.15 | 255.30 | 640.89 | Higher LR (10x) slightly beat baseline, not the destabilization one might expect. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 281.75 | 290.25 | 713.52 | Lower LR (10x) was the **2nd-best run overall** — the small, conservative updates suited SpaceInvaders' larger unclipped reward scale well within this budget. |
| 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 261.70 | 305.35 | 625.78 | Lower gamma (more myopic) clearly helped here — opposite of what a "long-horizon credit assignment" story would predict. |
| 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | 213.20 | 257.40 | 637.59 | Higher gamma slightly hurt — the worst gamma setting tested. |
| 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 240.90 | 272.80 | 702.89 | Smaller batch beat baseline, with the longest average episode length of any single-factor run. |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | 206.80 | 299.95 | 564.00 | Larger batch was the **worst CNN run** — fewer effective updates per timestep budget cost more than the smoother gradients gained. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | 248.25 | 291.80 | 684.06 | Fast epsilon decay beat baseline — committing to exploitation early paid off. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | 219.95 | 275.80 | 654.47 | Roughly baseline-level by this metric. **This is David's chosen champion model** — see the note below on why that doesn't match the numerically top run. |
| 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | 286.55 | 304.85 | 651.54 | **Best run overall** — combining a moderate LR bump with a larger batch and a lower epsilon floor produced the top score, same pattern as the other two members' combined-config experiments. |

## MLP vs CNN comparison

`run_experiments.py` was re-run with `--policy MlpPolicy --only 1` and `--only 2` (baseline and
higher-LR configs) to produce a direct comparison against the matching CNN runs:

| Config | CnnPolicy final ep_rew_mean | MlpPolicy final ep_rew_mean |
|--------|------------------------------|-------------------------------|
| Baseline (exp01) | 222.30 | 208.80 |
| Higher LR (exp02) | 230.15 | 236.05 |

Unlike Edwin's Breakout comparison (where CNN clearly beat MLP), on SpaceInvaders the two
architectures landed in the same ballpark, with MLP even edging ahead on the higher-LR config.
Worth raising in Q&A: this doesn't mean CNN's theoretical advantage (exploiting spatial
structure in pixel frames) isn't real — SpaceInvaders' larger, more repetitive visual field
(rows of aliens moving in lockstep) may simply be easier for an MLP to approximate within
150k steps than Breakout's fast, precise ball-tracking. A third `MlpPolicy` run (`--only 3`)
was started but didn't complete/log any data.

## Best configuration

- **Numerically best run:** exp10 (`lr=5e-4, gamma=0.99, batch_size=64, eps_end=0.02, eps_decay_frac=0.15`) — final reward 286.55, the top score in the sweep.
- **David's promoted champion model:** exp09 (`lr=1e-4, gamma=0.99, batch_size=32, eps_end=0.10, eps_decay_frac=0.30`) — this is what `models/dqn_model.zip` and the gameplay video actually contain.

**Open item for David:** the promoted model (exp09) is not the numerically top run by
`ep_rew_mean` — exp10 scored noticeably higher (286.55 vs 219.95). If exp09 was chosen for a
specific reason (e.g. it looked more stable in the recorded video, or a different evaluation
method was used), say so explicitly in the presentation — otherwise it's worth either
re-promoting exp10's model to `models/dqn_model.zip` and re-recording the gameplay video, or
being ready to explain the discrepancy if a coach asks "why didn't you pick your best-scoring
run?"

## Key takeaways (for presentation, max 2 min)

- What helped: a lower learning rate (1e-5), a lower gamma (0.90), a smaller batch (16), and
  fast epsilon decay all individually beat the baseline. Combining a moderate LR increase with
  a larger batch and low epsilon floor (exp10) produced the best result overall.
- What hurt: a larger batch size (128) was the single worst change tested — the opposite
  direction from Edwin's Breakout sweep, where batch=128 was one of the *best* changes. A
  higher gamma (0.995) also underperformed.
- Final config + reasoning: *(David to confirm exp09 vs exp10 — see "Open item" above before presenting).*
