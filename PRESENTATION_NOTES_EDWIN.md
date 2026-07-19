# Presentation Notes — Edwin (2-minute segment + Q&A prep)

The rubric weights "Understanding of DQN & RL Concepts" at 10/30 — the single largest
line item, evaluated through the presentation and Q&A, not the code. These notes are
built to cover that directly. Fill in the bracketed placeholders with your actual
numbers once the 10 runs are done — the reasoning structure stays the same.

## 2-minute segment script skeleton

**[~20s] What I tested:**
"I ran 10 experiments on Breakout, structured as a one-factor-at-a-time sweep: starting
from a baseline config, I changed one hyperparameter at a time — learning rate, gamma,
batch size, then epsilon schedule — before combining what worked into a final config."

**[~50s] What helped / what hurt (pick your 3 most interesting findings once you have
results — this template shows the shape of a strong answer):**
- "[Hyperparameter X] at [value] caused [specific observed effect — e.g. reward
  plateaued around Y, or oscillated]. This makes sense because [mechanism — e.g. a
  learning rate that's too high causes large Q-value updates that overshoot and
  destabilize the target]."
- "[Hyperparameter Y] improved things because [mechanism tied to exploration/exploitation
  or credit assignment]."
- Contrast at least one "helped" and one "hurt" result — graders are explicitly listening
  for this.

**[~30s] Final configuration and why:**
"My best run was experiment #[N] with [config]. I believe it performed best because
[tie back to a specific mechanism, not just 'the numbers were higher']."

**[~20s] Hand off / gameplay clip cue:** "That's my part — [next member] will cover
theirs, and we'll close with the gameplay clip."

Rehearse with a timer. 2 minutes goes fast with 10 experiments to summarize — the
script above is deliberately terse; expand only the 3 most interesting results, don't
narrate all 10 rows.

## Q&A prep — core concepts you must be able to explain live

**Exploration vs. exploitation:**
- Epsilon-greedy: with probability epsilon, take a random action (explore); otherwise
  take the action with the highest predicted Q-value (exploit).
- Epsilon decays from `eps_start` to `eps_end` over `eps_decay_frac` of training — early
  on the agent should explore broadly since its Q-value estimates are unreliable; later
  it should exploit its (by then more accurate) estimates.
- Trade-off: decay too fast (see experiment #8, `eps_decay_frac=0.02`) → the agent locks
  into a policy before it's discovered good strategies (e.g. tunneling through the brick
  wall in Breakout). Decay too slow / floor too high (experiment #9) → wastes training
  budget on random actions, slower apparent progress, though potentially a better final
  policy if given enough steps.

**Reward structure in Breakout:**
- +1 per brick broken (values differ by row in the true ALE reward, but SB3's
  `AtariWrapper` clips rewards to {-1, 0, +1} by default for training stability).
- Reward is frequent but not immediate relative to the *action* that causes it — moving
  the paddle to the right position is only rewarded several steps later when the ball
  connects, which is a credit-assignment problem gamma helps solve.

**Gamma (discount factor):**
- Determines how much future reward is worth relative to immediate reward.
- Low gamma (experiment #4, `gamma=0.90`) → agent is myopic, may fail to set up
  multi-brick combos or track the ball's longer-term trajectory.
- High gamma (experiment #5, `gamma=0.995`) → agent values long-term reward almost as
  much as immediate reward, which suits Breakout's delayed-reward structure but can make
  early training noisier since value estimates propagate across longer horizons.

**Batch size:**
- Number of transitions sampled from the replay buffer per gradient update.
- Small batch (#6) → noisier gradient estimates, but more frequent/cheaper updates.
- Large batch (#7) → smoother, more stable gradient estimates, but each update is more
  expensive and — for a fixed timestep budget — you get fewer total updates.

**Learning rate:**
- Step size for the Q-network's gradient updates.
- Too high (#2, `1e-3`) → risk of overshooting and destabilizing the Bellman target,
  visible as reward spikes/collapses rather than steady improvement.
- Too low (#3, `1e-5`) → safe but slow; may not converge to good behavior within the
  150k-step budget at all.

**Policy architecture — CnnPolicy vs. MlpPolicy:**
- CNNs apply convolutional filters that detect spatial patterns (edges, the ball,
  paddle position, brick layout) regardless of where they appear on screen — this
  matches the structure of pixel-frame observations directly.
- MLPs flatten the image into a flat vector and treat every pixel as an independent,
  unordered input — no notion of "this pixel is spatially next to that one" — so they
  generally need far more capacity/data to learn anything useful from raw pixels, and in
  our shared comparison run [fill in: underperformed / failed to learn within budget].
- Why this matters for Breakout specifically: the relevant signal (ball trajectory,
  paddle alignment, brick-wall gaps) is inherently spatial and benefits from
  translation-invariant feature detection — exactly what convolution provides.

**Why does the final model behave the way it does?**
- Be ready to connect the final chosen config's *specific* values back to Breakout's
  reward structure and the exploration/exploitation and credit-assignment points above,
  not just restate the numbers. E.g.: "the extended epsilon decay in run #10 gave it
  enough exploration time to discover that angling the ball to break through one side of
  the wall opens a fast-scoring channel behind the bricks — a strategy that requires
  discovering a specific action sequence, so it needs sustained exploration to find."

**Trade-offs to be ready to discuss:**
- Compute/time budget vs. hyperparameter search breadth (why 150k steps, why these 10
  configs and not a full grid search).
- Stability (low lr, larger batch) vs. speed of learning (higher lr, smaller batch).
- Exploration duration vs. sample efficiency (slow epsilon decay finds better policies
  but "wastes" more of the fixed step budget on random actions).
