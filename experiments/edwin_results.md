# Hyperparameter Experiments — Edwin Bayingana

Environment: `ALE/Breakout-v5`
Policy: `CnnPolicy` (see `README.md` for the one shared MLP-vs-CNN comparison run)
Fixed settings across all 10 runs (do not change these — only the columns below vary):
`N_ENVS=4`, `FRAME_STACK=4`, `timesteps=150,000`, `buffer_size=50,000`, `seed=0`

## Design rationale

The 10 runs are not random — they're a one-factor-at-a-time (OFAT) sweep around a
baseline (run 1), so each experiment isolates the effect of a single hyperparameter,
followed by two combined configs that apply what the individual runs suggested works.
This makes the "which change helped / which hurt" story in the presentation traceable
to a specific, explainable cause rather than a black box.

| # | lr | gamma | batch_size | eps_start | eps_end | eps_decay_frac | Hypothesis (before running) | Noted Behavior (after running) |
|---|----|-------|------------|-----------|---------|-----------------|------------------------------|----------------------------------|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | Baseline — SB3's standard Atari DQN defaults. Expect steady but slow reward climb. | Baseline reference point: final rolling reward 7.55, peak 11.9 (episode 1607), max episode length 806. Steady but unremarkable climb — matches the hypothesis, nothing pathological. |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | LR 10x higher than baseline. Expect noisy/unstable Q-values, possibly reward collapse or oscillation rather than faster learning. | Hypothesis **contradicted**: outperformed baseline (final 10.25, peak 16.0, max ep length 879) instead of collapsing. At 150k steps the higher LR seems to accelerate useful learning faster than it destabilizes Q-values — divergence risk is real but wasn't triggered in this budget. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | LR 10x lower than baseline. Expect very slow, flat learning curve — likely won't clear bricks reliably within 150k steps. | Confirmed: worst run overall (final 4.45, peak only 6.45, shortest episodes at 596 max). Updates were too small to meaningfully shift Q-values in the available steps. |
| 4 | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | Lower discount factor (more myopic). Expect the agent to chase immediate paddle hits but undervalue setting up multi-brick combos, capping reward lower. | Mild effect only: final 9.4, peak 11.5 — slightly above baseline, not clearly capped. Breakout's reward signal may be immediate enough (brick breaks right after contact) that a shorter horizon isn't very costly. |
| 5 | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | Higher discount factor (more far-sighted). Expect similar or slightly better long-run reward than baseline, but possibly slower/noisier early learning since credit assignment spans further. | Slightly better than baseline and than exp04 (final 10.4, peak 12.75) — modest evidence that valuing future reward more helps, but the effect size is small compared to the LR/combined changes. |
| 6 | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | Smaller batch size. Expect noisier gradient estimates per update but more updates/sec — could go either way on wall-clock efficiency. | Comparable to baseline, marginally better (final 9.1, peak 11.35). More frequent noisier updates roughly offset each other here — no strong signal either way. |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | Larger batch size. Expect smoother, more stable gradient updates but fewer effective updates for the same timestep budget — slower apparent progress. | Hypothesis **contradicted**: second-best individual run (final 10.25, peak 16.1). Smoother gradients apparently mattered more than update count in this budget — echoes exp02's surprise that "riskier-looking" hyperparameters didn't hurt here. |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | Fast epsilon decay (greedy almost immediately). Expect premature convergence to a mediocre policy — agent stops exploring before finding good brick-clearing strategies. | Performed well (final 10.8, peak 12.4) — better than baseline, and clearly better than the slow-decay exp09 below. Committing to exploitation early didn't cause premature lock-in here. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | Slow epsilon decay + higher exploration floor. Expect slower early reward growth (more random actions) but potentially a better final policy from broader exploration. | Hypothesis **contradicted** — this was one of the *worst* runs (final 7.45, peak 11.1), worse than fast-decay exp08. A floor of 0.10 keeps 10% random actions active for the entire 150k steps, which likely kept injecting noise that prevented the policy from ever fully stabilizing, rather than helping it explore into a better optimum. |
| 10 | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | Combined "best guess": moderate LR bump, larger batch for stability, slightly extended decay with a lower floor. Expect this to beat the baseline if the individual effects above are additive. | **Best run by a wide margin**: final rolling reward 16.85, peak 20.9 — roughly double the baseline and clearly ahead of every other run. Also the most sample-efficient (reached its peak by episode 1378, needed only 1436 episodes total vs ~2000-3200 for other runs) and survived longest (max episode length 1193). Combining a moderately higher LR with a larger batch (more stable gradients per exp07) and a lower epsilon floor (per exp08) compounded rather than cancelled out — the individual "risky" changes (exp02, exp07) that helped on their own helped together too. |

## Exact commands (run from the repo root, on the M1 Pro)

```bash
python train.py --run-name edwin_exp01 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp02 --lr 1e-3  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp03 --lr 1e-5  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp04 --lr 1e-4  --gamma 0.90  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp05 --lr 1e-4  --gamma 0.995 --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp06 --lr 1e-4  --gamma 0.99  --batch-size 16  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp07 --lr 1e-4  --gamma 0.99  --batch-size 128 --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.10 --timesteps 150000 --device auto
python train.py --run-name edwin_exp08 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.05 --eps-decay-frac 0.02 --timesteps 150000 --device auto
python train.py --run-name edwin_exp09 --lr 1e-4  --gamma 0.99  --batch-size 32  --eps-start 1.0 --eps-end 0.10 --eps-decay-frac 0.30 --timesteps 150000 --device auto
python train.py --run-name edwin_exp10 --lr 5e-4  --gamma 0.99  --batch-size 64  --eps-start 1.0 --eps-end 0.02 --eps-decay-frac 0.15 --timesteps 150000 --device auto
```

Each run saves:
- `models/dqn_model_edwin_expNN.zip` — the trained model
- `experiments/logs/edwin_expNN.csv` — per-episode reward/length + rolling mean (for plotting reward trends)
- TensorBoard logs under `tb_logs/edwin_expNN/`

## How to fill "Noted Behavior" after each run

Open `experiments/logs/edwin_expNN.csv` (or run `tensorboard --logdir tb_logs`) and note,
in 1-2 sentences per row:
- Did reward trend up, stay flat, or collapse/oscillate?
- Was the rolling mean reward at the end of training higher or lower than the baseline (run 1)?
- Did episode length grow (agent surviving longer / clearing more bricks)?
- Did the observed behavior match the hypothesis? If not, why might that be?

## Best configuration

- Run #: **10**
- Config: `lr=5e-4, gamma=0.99, batch_size=64, eps_start=1.0, eps_end=0.02, eps_decay_frac=0.15`
- Why it performed best: it combined three individually-promising directions — a moderate
  learning-rate increase (validated as beneficial, not destabilizing, by exp02), a larger
  batch size for smoother gradient updates (validated by exp07), and a lower epsilon floor
  for more decisive exploitation once training progressed (consistent with exp08 beating
  exp09). Rather than these effects cancelling out, they compounded: exp10 reached roughly
  double the peak reward of the baseline, in fewer episodes than any other run, indicating
  both faster and more stable learning.

## Key takeaways (for presentation, max 2 min)

- What helped: a moderately higher learning rate (1e-3 territory, not just the 1e-4
  default), a larger batch size (64-128) for smoother updates, and a lower epsilon floor
  (0.02) once exploration had run its course. Combining these (exp10) outperformed every
  individual change.
- What hurt: a learning rate 10x too low (exp03) starved the network of meaningful updates
  within the 150k-step budget. Counter-intuitively, a *slower* epsilon decay with a higher
  exploration floor (exp09) also hurt — sustained random actions prevented the policy from
  stabilizing, rather than helping it explore into a better optimum.
- Final config + reasoning: exp10's combined configuration
  (`lr=5e-4, batch_size=64, eps_end=0.02, eps_decay_frac=0.15`) — the individually-tested
  "riskier" changes turned out to be net positives at this training budget, and stacking
  them compounded rather than conflicted.
