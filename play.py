"""
play.py — load a trained DQN model and watch it play, greedily.

Stable Baselines3 has no separate "GreedyQPolicy" class — the equivalent is
model.predict(obs, deterministic=True), which always picks the highest-Q
action instead of sampling exploration actions. That's what this script does.

Two modes:
  --mode record   (default) Renders off-screen and saves an .mp4 via
                   Gymnasium's RecordVideo wrapper — use this for the
                   submission deliverable ("video showing the agent playing").
  --mode fast      Original no-render mode: just prints episode scores,
                   useful for quickly checking a model without waiting on
                   video encoding.

Usage:
    python play.py --model models/dqn_model.zip --mode record --episodes 3
    python play.py --model models/dqn_model.zip --mode fast --episodes 10
"""

import argparse
import os

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN

# Register environments
gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"
FRAME_STACK = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Play an Atari environment with a trained DQN model.")
    parser.add_argument("--model", type=str, default="models/dqn_model.zip", help="Path to the saved model .zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--mode", type=str, default="record", choices=["record", "fast"], help="record = save .mp4, fast = headless, scores only")
    parser.add_argument("--video-folder", type=str, default="video", help="Where to save the .mp4 (record mode)")
    return parser.parse_args()


def make_env(mode, video_folder):
    """Recreate the exact environment wrapping used in train.py (frameskip=1,
    AtariPreprocessing + FrameStackObservation) so the loaded model sees the
    observation shape it was trained on."""
    render_mode = "rgb_array" if mode == "record" else None
    env = gym.make(ENV_ID, render_mode=render_mode, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, FRAME_STACK)
    if mode == "record":
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix="dqn_spaceinvaders_play",
                                        episode_trigger=lambda ep: True)
    return env


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")
    env = make_env(args.mode, args.video_folder)
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
    if args.mode == "record":
        print(f"Video(s) saved to {args.video_folder}/")


if __name__ == "__main__":
    main()
