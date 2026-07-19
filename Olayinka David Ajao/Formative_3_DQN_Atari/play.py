"""
play.py — load a trained DQN model and watch it play, greedily.
FAST EVALUATION MODE (Video Disabled)
"""

import argparse
import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN

# Register environments
gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
FRAME_STACK = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Play an Atari environment with a trained DQN model.")
    parser.add_argument("--model", type=str, default="models/dqn_model_exp01.zip", help="Path to the saved model .zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")

    # Recreate the exact environment wrapping with frameskip=1
    env = gym.make(ENV_ID, render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)

    model = DQN.load(args.model)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    env.close()

if __name__ == "__main__":
    main()
