"""
train.py — DQN training for the Atari group project (Stable Baselines3 + Gymnasium).

Each member copies the CONFIG defaults or passes CLI flags for their own 10
hyperparameter experiments. Every run logs:
  - reward trends + episode length  -> TensorBoard AND a per-run CSV (Monitor)
  - a periodic greedy evaluation     -> eval reward over training
  - the best model seen during eval  -> models/<run-name>/best_model.zip
  - the final model                  -> models/dqn_model_<run-name>.zip
plus a one-line summary appended to experiments/results.csv so the whole group's
runs collect into a single table.

Usage:
    python train.py
    python train.py --lr 5e-4 --gamma 0.98 --batch-size 64 --timesteps 300000 --run-name alice_exp02
    python train.py --policy MlpPolicy --run-name mlp_vs_cnn   # architecture comparison
"""

import argparse
import csv
import os
from datetime import datetime

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register Atari (ALE/*) environments with Gymnasium (required by ale-py >= 0.10)
gym.register_envs(ale_py)

# --------------------------------------------------------------------------
# Group-agreed constants — keep these the SAME across every member's runs
# so results are comparable. Change ENV_ID once as a group, not per-member.
# --------------------------------------------------------------------------
ENV_ID = "ALE/Pong-v5"          # e.g. swap for ALE/Breakout-v5, ALE/SpaceInvaders-v5
N_ENVS = 4                      # parallel envs for faster data collection
FRAME_STACK = 4                 # standard Atari frame stacking
POLICY = "CnnPolicy"            # "CnnPolicy" (default for Atari) or "MlpPolicy" for comparison
LOG_DIR = "./tb_logs"           # TensorBoard logs
MODEL_DIR = "./models"          # saved models
MONITOR_DIR = "./experiments/monitor"   # per-run reward/episode-length CSVs
RESULTS_CSV = "./experiments/results.csv"   # one row per run, appended


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on an Atari environment.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon (exploration_initial_eps)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon (exploration_final_eps)")
    parser.add_argument("--eps-decay-frac", type=float, default=0.1, help="Fraction of training over which epsilon decays (exploration_fraction)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size (lower if you hit memory limits, e.g. 50000)")
    parser.add_argument("--learning-starts", type=int, default=10_000, help="Steps of random play before learning begins")
    parser.add_argument("--target-update", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--train-freq", type=int, default=4, help="Environment steps between gradient updates")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Steps between greedy evaluations (per env)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--policy", type=str, default=POLICY, choices=["CnnPolicy", "MlpPolicy"], help="Policy network type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--run-name", type=str, default="run", help="Name used for logs/model file, e.g. 'alice_exp03'")
    return parser.parse_args()


def make_env(seed, monitor_dir=None):
    """Vectorized, frame-stacked Atari env. make_atari_env wraps each sub-env in
    a Monitor, so episode reward + length are recorded to CSV when monitor_dir is set."""
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=seed, monitor_dir=monitor_dir)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def append_results_row(args, best_mean, final_path, best_path):
    """Append a one-line summary of this run to experiments/results.csv."""
    header = [
        "timestamp", "run_name", "policy", "env", "timesteps",
        "lr", "gamma", "batch_size", "eps_start", "eps_end", "eps_decay_frac",
        "best_mean_reward", "final_model", "best_model",
    ]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"), args.run_name, args.policy, ENV_ID, args.timesteps,
        args.lr, args.gamma, args.batch_size, args.eps_start, args.eps_end, args.eps_decay_frac,
        f"{best_mean:.2f}" if best_mean is not None else "n/a", final_path, best_path,
    ]
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def main():
    args = parse_args()
    for d in (MODEL_DIR, LOG_DIR, MONITOR_DIR):
        os.makedirs(d, exist_ok=True)
    run_monitor_dir = os.path.join(MONITOR_DIR, args.run_name)
    best_model_dir = os.path.join(MODEL_DIR, args.run_name)
    os.makedirs(best_model_dir, exist_ok=True)

    # Training env (logs episode reward + length to per-run monitor CSVs)
    env = make_env(args.seed, monitor_dir=run_monitor_dir)
    # Separate eval env (different seed) for unbiased greedy evaluation
    eval_env = make_env(args.seed + 100)

    model = DQN(
        policy=args.policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_decay_frac,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update,
        seed=args.seed,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # EvalCallback runs a greedy (deterministic) evaluation every eval_freq steps,
    # logs mean reward to TensorBoard, and saves the best model automatically.
    # eval_freq is per-env, so divide by N_ENVS to get the intended global cadence.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=max(args.eval_freq // N_ENVS, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"Training config: {vars(args)}")
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, f"dqn_model_{args.run_name}.zip")
    model.save(final_path)
    best_path = os.path.join(best_model_dir, "best_model.zip")
    best_mean = getattr(eval_callback, "best_mean_reward", None)

    append_results_row(args, best_mean, final_path, best_path)
    print(f"\nSaved final model to {final_path}")
    if os.path.exists(best_path):
        print(f"Best eval model at   {best_path} (best mean reward: {best_mean:.2f})")
    print(f"Per-run reward/length logs: {run_monitor_dir}/  |  summary row -> {RESULTS_CSV}")
    print("\nTip: once you pick your best run, copy its model to models/dqn_model.zip for play.py:")
    print(f"    cp {best_path} models/dqn_model.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
