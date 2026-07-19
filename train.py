"""
train.py — DQN training script for the group's Deep Q-Learning assignment.

Shared by all three members. Each member trained on a different Atari
environment (see README.md for why), so --env-id is a required flag rather
than a hardcoded constant — everything else (the network, preprocessing,
logging, and the hyperparameters under test) is identical code across
Edwin's, David's, and Nziza's runs.

Usage:
    python train.py --env-id ALE/Breakout-v5 --run-name edwin_exp01
    python train.py --env-id ALE/SpaceInvaders-v5 --member david --run-name david_exp01
    python train.py --env-id ALE/Pong-v5 --lr 5e-4 --gamma 0.98 --batch-size 64 --run-name nziza_exp02
"""

import argparse
import csv
import os
from datetime import datetime

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register Atari (ALE/*) environments with Gymnasium (required by ale-py >= 0.10)
gym.register_envs(ale_py)

# --------------------------------------------------------------------------
# Fixed scaffolding — identical for every run regardless of member or
# environment, so results only differ because of the hyperparameters under
# test, not because of inconsistent preprocessing.
# --------------------------------------------------------------------------
N_ENVS = 4                      # parallel envs for faster data collection
FRAME_STACK = 4                 # standard Atari frame stacking (lets the agent see motion)
DEFAULT_POLICY = "CnnPolicy"    # "CnnPolicy" (default for Atari) or "MlpPolicy" for comparison
LOG_DIR = "./tb_logs"
MODEL_DIR = "./models"
METRICS_DIR = "./experiments/logs"
RESULTS_CSV = "./experiments/results.csv"


class EpisodeCSVLogger(BaseCallback):
    """Writes one row per completed episode (across all vec envs) with the
    reward and length, plus a rolling mean, so reward-trend/episode-length
    plots can be built without needing TensorBoard installed."""

    def __init__(self, csv_path, rolling_window=20, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rolling_window = rolling_window
        self._rewards = []
        self._episode_count = 0

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "timestep", "reward", "length", f"rolling_mean_reward_{self.rolling_window}"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self._episode_count += 1
                self._rewards.append(ep_info["r"])
                rolling_mean = sum(self._rewards[-self.rolling_window:]) / min(len(self._rewards), self.rolling_window)
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([self._episode_count, self.num_timesteps, ep_info["r"], ep_info["l"], round(rolling_mean, 3)])
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on an Atari environment.")
    parser.add_argument("--env-id", type=str, required=True, help="Atari environment id, e.g. ALE/Breakout-v5")
    parser.add_argument("--member", type=str, default="", help="Member name, used only for the results.csv summary row")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon (exploration_initial_eps)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon (exploration_final_eps)")
    parser.add_argument("--eps-decay-frac", type=float, default=0.1, help="Fraction of training over which epsilon decays (exploration_fraction)")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size (kept modest to fit 16GB RAM machines)")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, choices=["CnnPolicy", "MlpPolicy"], help="Policy network type")
    parser.add_argument("--run-name", type=str, default="run", help="Name used for logs/model file, e.g. 'edwin_exp03'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                         help="'auto' picks the best available device (mps on Apple Silicon, cuda on Colab/GPU boxes)")
    return parser.parse_args()


def make_env(env_id, seed=0):
    """Build a vectorized, frame-stacked Atari environment. Identical for
    every member/environment — only env_id changes."""
    env = make_atari_env(env_id, n_envs=N_ENVS, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def append_results_row(args, csv_path):
    """Append a one-line summary of this run to experiments/results.csv, so
    the whole group's runs collect into a single shared table."""
    header = [
        "timestamp", "member", "run_name", "policy", "env_id", "timesteps",
        "lr", "gamma", "batch_size", "eps_start", "eps_end", "eps_decay_frac", "model_path",
    ]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"), args.member, args.run_name, args.policy, args.env_id, args.timesteps,
        args.lr, args.gamma, args.batch_size, args.eps_start, args.eps_end, args.eps_decay_frac, csv_path,
    ]
    write_header = not os.path.exists(RESULTS_CSV)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    env = make_env(args.env_id, seed=args.seed)

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
        optimize_memory_usage=True,          # halves replay buffer RAM (needed at 84x84x4 uint8 frames)
        replay_buffer_kwargs=dict(handle_timeout_termination=False),  # required when optimize_memory_usage=True
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=LOG_DIR,
    )

    csv_path = os.path.join(METRICS_DIR, f"{args.run_name}.csv")
    callback = EpisodeCSVLogger(csv_path)

    print(f"Training config: {vars(args)}")
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        callback=callback,
        progress_bar=True,
    )

    save_path = os.path.join(MODEL_DIR, f"dqn_model_{args.run_name}.zip")
    model.save(save_path)
    append_results_row(args, csv_path)
    print(f"Saved model to {save_path}")
    print(f"Per-episode reward/length log saved to {csv_path}")
    print(f"Summary row appended to {RESULTS_CSV}")

    # Once you've decided this is your best run, copy it to a clearly-named
    # canonical file for play.py, e.g.:
    #   cp models/dqn_model_edwin_exp10.zip models/dqn_model_edwin_breakout.zip

    env.close()


if __name__ == "__main__":
    main()
