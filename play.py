"""
play.py — load a trained DQN model and watch it play, greedily.

Stable Baselines3 has no separate "GreedyQPolicy" class — the equivalent is
model.predict(obs, deterministic=True), which always picks the highest-Q action
instead of sampling exploration actions. That is what this script does, matching
the assignment's "use GreedyQPolicy for evaluation" requirement.

Usage:
    python play.py                                   # uses models/dqn_model.zip
    python play.py --model models/alice_exp10/best_model.zip --episodes 5
    python play.py --no-render                       # headless (e.g. over SSH); prints scores only
"""

import argparse
import os
import time

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Register Atari (ALE/*) environments with Gymnasium (required by ale-py >= 0.10)
gym.register_envs(ale_py)

# Must match the settings used in train.py for the loaded model
ENV_ID = "ALE/Pong-v5"
FRAME_STACK = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Play an Atari environment with a trained DQN model.")
    parser.add_argument("--model", type=str, default="models/dqn_model.zip", help="Path to the saved model .zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--sleep", type=float, default=0.01, help="Delay between frames (seconds) to slow rendering")
    parser.add_argument("--no-render", action="store_true", help="Run headless (no window) — useful over SSH")
    return parser.parse_args()


def make_play_env(render):
    """Single-env, frame-stacked version of the training environment."""
    env_kwargs = {"render_mode": "human"} if render else None
    env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "Train first (python train.py) then copy your best model to models/dqn_model.zip,\n"
            "or point --model at a best_model.zip under models/<run-name>/."
        )

    render = not args.no_render
    env = make_play_env(render)
    model = DQN.load(args.model, env=env)
    print(f"Loaded model: {args.model}")

    rewards = []
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # greedy action selection
            obs, reward, dones, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            if render:
                env.render()
                if args.sleep:
                    time.sleep(args.sleep)
            done = bool(dones[0])

        rewards.append(total_reward)
        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    print(f"\nAverage reward over {len(rewards)} episodes: {sum(rewards)/len(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    main()
