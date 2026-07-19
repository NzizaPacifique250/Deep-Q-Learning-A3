"""
train.py — DQN training script for the Atari group project.

Environment: ALE/Breakout-v5 (group-agreed constant — do not change per-run).

Everyone forks/copies this file for their own 10 hyperparameter experiments.
Only the CLI args below (lr, gamma, batch-size, epsilon settings) should
change between an individual member's runs — the constants in the
"Group-agreed constants" block are locked so every member's results are
directly comparable.

Usage:
    python train.py --run-name edwin_exp01
    python train.py --lr 5e-4 --gamma 0.98 --batch-size 64 --run-name edwin_exp02
"""

import argparse
import csv
import os

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # registers the ALE/* env ids with Gymnasium

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# --------------------------------------------------------------------------
# Group-agreed constants — keep these the SAME across every member's runs
# so results are comparable. Only change as a group, never per-member.
# --------------------------------------------------------------------------
ENV_ID = "ALE/Breakout-v5"      # group-selected Atari environment
N_ENVS = 4                      # parallel envs for faster data collection
FRAME_STACK = 4                 # standard Atari frame stacking (lets the agent see motion)
POLICY = "CnnPolicy"            # "CnnPolicy" (default for Atari) or "MlpPolicy" for the one shared comparison run
LOG_DIR = "./tb_logs"
MODEL_DIR = "./models"
METRICS_DIR = "./experiments/logs"


class EpisodeCSVLogger(BaseCallback):
    """Writes one row per completed episode (across all vec envs) with the
    reward and length, plus a rolling mean, so reward-trend/episode-length
    plots can be built for the README without needing TensorBoard installed.
    """

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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon (exploration_initial_eps)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon (exploration_final_eps)")
    parser.add_argument("--eps-decay-frac", type=float, default=0.1, help="Fraction of training over which epsilon decays (exploration_fraction)")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Total training timesteps")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer size (kept modest to fit 16GB RAM machines)")
    parser.add_argument("--policy", type=str, default=POLICY, choices=["CnnPolicy", "MlpPolicy"], help="Policy network type")
    parser.add_argument("--run-name", type=str, default="run", help="Name used for logs/model file, e.g. 'edwin_exp03'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                         help="'auto' picks MPS on Apple Silicon if available; use 'cpu' if you hit MPS op-support errors")
    return parser.parse_args()


def make_env(seed=0):
    """Build a vectorized, frame-stacked Atari environment."""
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=seed)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    env = make_env(seed=args.seed)

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
    print(f"Saved model to {save_path}")
    print(f"Per-episode reward/length log saved to {csv_path}")

    # Once you've decided this run is your best, copy/save it as the
    # canonical file play.py defaults to — do this manually so you don't
    # overwrite a better earlier run by accident:
    #   cp models/dqn_model_<run-name>.zip models/dqn_model.zip

    env.close()


if __name__ == "__main__":
    main()
