# Run Instructions — Edwin (MacBook M1 Pro, 16GB RAM)

This repo was validated in a cloud sandbox (no GPU): `ALE/Breakout-v5` registers correctly
under Gymnasium/ALE with the expected 4-action space, and `train.py`/`play.py` compile
cleanly. Full training was intentionally *not* run there (CPU-only, would take many hours
for 10 runs) — run the real experiments locally, where an M1 Pro with `mps` acceleration
will be meaningfully faster.

## 1. One-time setup

```bash
cd "starter-repo"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sanity-check the environment registers and torch sees MPS:
```bash
python3 -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); e = gym.make('ALE/Breakout-v5'); print(e.action_space, e.observation_space)"
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
Expected: `Discrete(4)` action space, `Box(0, 255, (210, 160, 3), uint8)` observation space.

## 2. Do a tiny smoke-test run first (don't skip this)

Before committing to a 150k-step run, confirm the full train → save → load → play loop
works end-to-end in under 2 minutes:

```bash
python train.py --run-name smoketest --timesteps 5000 --device auto
python play.py --model models/dqn_model_smoketest.zip --mode record --episodes 1 --video-length 500
```
Check that `videos/dqn_breakout_play-*.mp4` was created and plays. If you hit an MPS
"operator not implemented" error, rerun both commands with `--device cpu` — Breakout's
CNN is small enough that CPU is still workable, just slower.

## 3. Run the 10 real experiments

Copy-paste each line from the "Exact commands" section of
`experiments/edwin_results.md` one at a time (or save them as a `.sh` file and run
sequentially — don't run them in parallel, DQN + 4 vec envs will contend hard for your
4-8 CPU cores/RAM and skew wall-clock behavior between runs).

Rough runtime expectation on M1 Pro: ~25–45 minutes per 150k-step run depending on
`--device` and background load — budget roughly half a day total for all 10, and start
this well before the deadline.

After each run finishes, immediately jot the observed behavior into the corresponding
row of `experiments/edwin_results.md` while it's fresh — don't batch this at the end.

## 4. Pick your best run and generate deliverables

```bash
# compare all 10 reward curves
python experiments/plot_results.py --runs edwin_exp01 edwin_exp02 edwin_exp03 edwin_exp04 edwin_exp05 edwin_exp06 edwin_exp07 edwin_exp08 edwin_exp09 edwin_exp10 --out experiments/logs/all_runs_comparison.png

# promote your best run to the canonical model play.py defaults to
cp models/dqn_model_edwin_expNN.zip models/dqn_model.zip

# record the final gameplay video for the README/submission
python play.py --model models/dqn_model.zip --mode record --episodes 3
```

The `.mp4` lands in `videos/`. Upload it (or a short screen-recording of it playing) and
link/embed it in the README's "Gameplay Demo" section — the assignment requires this
video be visible from the README, not just present somewhere in the repo.

## 5. Fill in the "Noted Behavior" column honestly

Grade-wise, this matters more than raw scores: the rubric rewards **explanations of why**
a configuration behaved the way it did, not just "it worked" / "it didn't." Reference the
comparison plots and the CSVs in `experiments/logs/` when writing these.

## 6. Troubleshooting

- **`ROM not found` / license error**: `pip install "gymnasium[atari,accept-rom-license]" ale-py` again — occasionally the ROM license flag needs a clean reinstall.
- **Memory pressure / swapping**: lower `--buffer-size` further (e.g. `25000`) or close other apps; the 84x84x4 uint8 frame stack is the dominant memory cost.
- **MPS errors mid-training**: fall back to `--device cpu` for the affected run; note in your table if a run used a different device (shouldn't affect learning behavior, only speed).
- **`play.py --mode record` produces no video**: confirm `moviepy` installed correctly (`pip show moviepy`); it's required by SB3's `VecVideoRecorder`.
