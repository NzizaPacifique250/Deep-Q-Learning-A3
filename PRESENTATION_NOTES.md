# Presentation Notes — Q&A Prep (all members)

The rubric weights "Understanding of DQN & RL Concepts" at 10/30 — the single largest line
item, evaluated through the presentation and Q&A, not the code. Each section below is one
member's 2-minute segment script + Q&A prep. Open with the group-level framing (see README's
"Environments" section) before individual segments so the coach isn't confused by three games.

---

## Edwin Bayingana — `ALE/Breakout-v5`

### 2-minute segment script

**[~20s] What I tested:** "I ran 10 experiments on Breakout, a one-factor-at-a-time sweep —
baseline, then one hyperparameter changed at a time (learning rate, gamma, batch size,
epsilon schedule), then a combined config applying what worked."

**[~50s] What helped / what hurt:**
- "A 10x higher learning rate (1e-3) actually *beat* the baseline (10.25 vs 7.55 reward)
  instead of destabilizing — it didn't diverge within this training budget."
- "The opposite of my hypothesis: a *slower* epsilon decay with a higher exploration floor
  (0.10) performed *worse* (7.45) than a fast decay (10.8) — sustained random actions kept
  injecting noise that prevented the policy from stabilizing, rather than helping it explore
  into a better optimum."
- "Combining the individually-good changes (moderate LR bump, larger batch, low epsilon
  floor) nearly doubled the baseline — reward 16.85, peak 20.9."

**[~30s] Final configuration and why:** "My best run was experiment 10:
lr=5e-4, batch=64, eps_end=0.02, decay=0.15. It combined three individually-validated
improvements, and they compounded rather than conflicted — reached its peak in the fewest
episodes of any run, meaning it was also the most sample-efficient."

**[~20s] Hand off:** "That's my part — over to [next member] for their environment."

### Q&A prep

**Exploration vs. exploitation:** epsilon-greedy — probability epsilon takes a random action,
otherwise the highest-Q action. Epsilon decays from `eps_start` to `eps_end` over
`eps_decay_frac` of training. Fast decay (exp08) beat slow decay + high floor (exp09) here —
sustained randomness cost more than the extra exploration gained within this budget.

**Reward structure in Breakout:** +1 per brick broken (SB3's `AtariWrapper` clips to
{-1,0,+1}). Reward is delayed relative to the paddle-positioning action that causes it —
a credit-assignment problem gamma addresses.

**Gamma:** low (0.90) → myopic, mild effect; high (0.995) → modest improvement. Effect size
was small compared to LR/batch/epsilon changes on this environment.

**Batch size:** 128 (larger) *beat* baseline (10.25 vs 7.55) — smoother gradients mattered
more than fewer updates in this budget. Contrast with David's SpaceInvaders result, where
batch=128 was his *worst* run — hyperparameter effects don't generalize across environments.

**Learning rate:** too low (1e-5) was the worst run overall (4.45) — too small to meaningfully
update Q-values in 150k steps. Too high (1e-3) was fine here, though theory says it risks
overshoot/divergence with a longer budget.

**CnnPolicy vs MlpPolicy:** CNN convolutions exploit spatial structure in pixel frames
(paddle/ball/brick shapes, position-independent). MLP flattens to an unordered vector.
Measured: CNN reward 7.55 (peak 11.9) vs MLP 5.65 (peak 7.65) — CNN clearly wins, and MLP
needed more episodes (3225 vs 2092) to use the same step budget, meaning shorter average
lives (weaker ball tracking).

**Why does the final model behave the way it does?** Fresh 10-episode greedy eval averaged
reward 3.3 (range 2-6, steps 24-55) — lower than the 16.85 training-time rolling mean, because
that mean was pulled up by rare long "rally" episodes (training's max episode length was
1193). Most lives are short; occasionally the agent finds a sustained rhythm. This variance is
expected at 150k steps — published DQN Breakout benchmarks train for millions of steps.

---

## David (Yinka) Ajao — `ALE/SpaceInvaders-v5`

### 2-minute segment script

**[~20s] What I tested:** "10 experiments on SpaceInvaders, same one-factor-at-a-time
structure as the rest of the group — baseline, then LR, gamma, batch size, and epsilon
changes individually, then a combined config."

**[~50s] What helped / what hurt:**
- "A lower learning rate (1e-5) was my 2nd-best run (281.75 reward) — the opposite direction
  from Edwin's Breakout finding, where low LR was the *worst* run. SpaceInvaders' larger,
  unclipped reward scale seems to suit smaller, more conservative updates."
- "A larger batch size (128) was my *worst* run (206.80) — again, the opposite of Edwin's
  Breakout result, where batch=128 was one of the best changes. Same hyperparameter, opposite
  effect, different environment."
- "Combining the promising directions (moderate LR, larger batch, low epsilon floor) produced
  my numerically best run (286.55) — same pattern as Edwin's combined config winning."

**[~30s] Final configuration and why:** *(Resolve before presenting — see "Open item" in
`experiments/results.md`.)* "My promoted model is exp09, but exp10 scored higher in my logs
(286.55 vs 219.95). [Explain why exp09 was chosen, or state that exp10 has been re-promoted.]"

**[~20s] Hand off:** "Over to [next member]."

### Q&A prep

**CnnPolicy vs MlpPolicy:** unlike Edwin's clear Breakout result, CNN and MLP performed
comparably on SpaceInvaders (baseline: 222.30 CNN vs 208.80 MLP; higher-LR config: 230.15 CNN
vs 236.05 MLP — MLP even edged ahead). Be ready to explain: this doesn't disprove CNN's
theoretical advantage — SpaceInvaders' regular, repetitive visual field (rows of aliens
moving in lockstep) may simply be easier for an MLP to approximate within 150k steps than
Breakout's fast, precise ball-tracking.

**Cross-environment contrast (batch size and LR):** be ready to discuss *why* the same
hyperparameter change helped on one environment and hurt on another — the honest answer is
that hyperparameter effects interact with each game's reward density, episode length, and
visual complexity; there's no universal "good" batch size or LR independent of the problem.

**The exp09/exp10 discrepancy:** if asked "why didn't you promote your best-scoring run,"
have a real answer ready (visual/qualitative preference, an evaluation methodology
difference, or acknowledge it should be fixed).

---

## Nziza Aime Pacifique — `ALE/Pong-v5`

### 2-minute segment script

**[~20s] What I tested:** "10 experiments on Pong — baseline, then LR, gamma, batch size, and
epsilon changes, evaluated with a proper greedy `EvalCallback` every 25k steps rather than
just the training rollout mean."

**[~50s] What helped / what hurt:**
- "Only one change produced any measurable improvement: increasing batch size. Batch 64
  reached -20.60, batch 128 reached -20.20 — both still close to Pong's -21 minimum score."
- "Every other change — learning rate up or down, gamma up or down, faster or slower epsilon
  decay, even the combined 'best guess' config — stayed at exactly -21.00, the minimum
  possible score. My agent never learned to consistently return the ball within 200k steps."
- "This is a genuine negative result, not a bug: Pong needs sustained rallies to score, and
  200k steps wasn't enough training time for any tested configuration to discover that."

**[~30s] Final configuration and why:** "My best (least-bad) run was experiment 7:
lr=1e-4, gamma=0.99, batch=128, at -20.20. I'm presenting this honestly as a training-budget
limitation rather than claiming a strong player — more timesteps and multiple seeds would be
needed to draw a reliable conclusion."

**[~20s] Hand off:** "Over to [next member] / back to the group for the gameplay clips."

### Q&A prep

**Why -21.00 for almost every run:** -21 is Pong's minimum possible score (the agent loses
every point). A flat -21.00 across most configurations means the agent's paddle-control
policy never improved enough to win even a single point reliably — this is a stronger,
stricter signal than a low-but-nonzero score, since it shows the *evaluation* policy
(deterministic, no exploration noise) was still essentially non-functional.

**Why batch size was the one variable that helped:** larger batches produce smoother, less
noisy gradient estimates per update. In a sparse/delayed-reward game like Pong, noisy
gradients from a small batch may be more likely to push the policy in an unhelpful direction
before enough evidence has accumulated — though with only two data points (64, 128) this is a
tentative pattern, not a proven mechanism.

**Reward structure in Pong:** +1 when the agent scores, -1 when it concedes, episode score
ranges -21 to +21. Reward is sparse relative to the many paddle-positioning actions leading up
to each point — a harder credit-assignment problem than Breakout's more frequent brick-break
rewards, which may partly explain why 200k steps wasn't enough here but was more productive
for Edwin's and David's environments.

**Be ready to own the honest framing:** if a coach asks "so your agent doesn't actually play
Pong well," the correct answer is yes, that's accurate, and explain *why* (training budget,
not a broken implementation) rather than overstating the result. The train/play pipeline
itself works correctly — `EvalCallback` and the results logging prove that — it's specifically
the *learning* that didn't converge to a strong policy in this budget.

---

## Group-level trade-off questions (any member should be ready to answer)

- **Why 150k-200k timesteps instead of more?** Time/compute budget across 3 members × 10
  experiments each before the deadline. Published DQN Atari benchmarks use millions of steps;
  this group's numbers should be read as early-training comparisons, not final performance.
- **Why didn't you use the same environment?** See README's "Environments" section — a
  coordination gap caught after training was already complete.
- **What would you do differently with more time?** Agree on environment before any training
  starts (as the group's own pre-planning doc recommended); run Nziza's MLP comparison and
  video; resolve David's champion-model discrepancy; consider longer training runs given
  Pong's apparent need for more than 200k steps.
