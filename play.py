"""
play.py — load a trained DQN model and watch it play, greedily.

Stable Baselines3 doesn't have a separate "GreedyQPolicy" class — the
equivalent is calling model.predict(obs, deterministic=True), which always
picks the action with the highest Q-value instead of sampling exploration
actions. That's what this script does.

Two modes:
  --mode human   Opens a live GUI window (env.render()). Good for watching
                  locally, but produces no file you can attach to the README.
  --mode record   Renders off-screen and saves an .mp4 via SB3's
                  VecVideoRecorder. This is the mode to use for the
                  submission deliverable ("video showing the agent playing").

Usage:
    python play.py --model models/dqn_model.zip --mode human
    python play.py --model models/dqn_model.zip --mode record --episodes 3
"""

import argparse
import os
import time

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # registers the ALE/* env ids with Gymnasium

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

# Must match the settings used in train.py for the loaded model
ENV_ID = "ALE/Breakout-v5"
FRAME_STACK = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Play an Atari environment with a trained DQN model.")
    parser.add_argument("--model", type=str, default="models/dqn_model.zip", help="Path to the saved model .zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--sleep", type=float, default=0.01, help="Delay between frames (seconds), human mode only")
    parser.add_argument("--mode", type=str, default="record", choices=["human", "record"], help="human = live GUI window, record = save mp4")
    parser.add_argument("--video-folder", type=str, default="videos", help="Where to save the .mp4 (record mode)")
    parser.add_argument("--video-length", type=int, default=3000, help="Max frames to capture (record mode)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Inference device")
    return parser.parse_args()


def make_play_env(mode):
    """Single-env version of the training environment, in the render mode requested."""
    render_mode = "human" if mode == "human" else "rgb_array"
    env = make_atari_env(ENV_ID, n_envs=1, seed=0, env_kwargs={"render_mode": render_mode})
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    args = parse_args()

    env = make_play_env(args.mode)

    if args.mode == "record":
        os.makedirs(args.video_folder, exist_ok=True)
        env = VecVideoRecorder(
            env,
            args.video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix="dqn_breakout_play",
        )

    model = DQN.load(args.model, env=env, device=args.device)

    # Reset exactly ONCE before the loop, not per-episode. Breakout's Atari
    # preprocessing (EpisodicLifeEnv) reports "done=True" after every life
    # lost, not just full game over, and the underlying VecEnv already
    # auto-resets internally when that happens. Calling env.reset() again
    # ourselves on every loop iteration would be harmless in human mode, but
    # VecVideoRecorder closes and restarts a brand-new video file on every
    # explicit reset() call — so it silently produced a ~1-second clip per
    # life instead of one continuous recording. Resetting once avoids that.
    obs = env.reset()

    for ep in range(1, args.episodes + 1):
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)  # greedy action selection
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            if args.mode == "human":
                env.render()
                time.sleep(args.sleep)
            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)

        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    env.close()
    if args.mode == "record":
        print(f"Video(s) saved to {args.video_folder}/ — attach one to the README as the gameplay deliverable.")


if __name__ == "__main__":
    main()
