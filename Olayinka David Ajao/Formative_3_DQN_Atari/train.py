"""
train.py — DQN training skeleton for the Atari group project.
"""

import argparse
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register ALE environments
gym.register_envs(ale_py)

# --------------------------------------------------------------------------
# Group-agreed constants
# --------------------------------------------------------------------------
ENV_ID = "ALE/SpaceInvaders-v5" # Updated to your assigned environment
N_ENVS = 4
FRAME_STACK = 4
POLICY = "CnnPolicy"
LOG_DIR = "./tb_logs"
MODEL_DIR = "./models"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on an Atari environment.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon (exploration_initial_eps)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon (exploration_final_eps)")
    parser.add_argument("--eps-decay-frac", type=float, default=0.1, help="Fraction of training over which epsilon decays (exploration_fraction)")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--policy", type=str, default=POLICY, choices=["CnnPolicy", "MlpPolicy"], help="Policy network type")
    parser.add_argument("--run-name", type=str, default="run", help="Name used for logs/model file")
    return parser.parse_args()


def make_env():
    """Build a vectorized, frame-stacked Atari environment."""
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=0)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env = make_env()

    model = DQN(
        policy=args.policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_decay_frac,
        buffer_size=100_000,
        learning_starts=10_000,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda" # Force GPU usage in Colab
    )

    print(f"Training config: {vars(args)}")
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    save_path = os.path.join(MODEL_DIR, f"dqn_model_{args.run_name}.zip")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    env.close()

if __name__ == "__main__":
    main()
