"""
play.py — load a trained DQN model and watch it play, greedily.

Shared by all three members. Stable Baselines3 has no separate "GreedyQPolicy"
class — the equivalent is model.predict(obs, deterministic=True), which
always picks the highest-Q action instead of sampling exploration actions.
That's what this script does.

Four modes:
  --mode record   (default) Renders off-screen and saves an .mp4 via SB3's
                   VecVideoRecorder — this is the mode used for every
                   submission gameplay video in this repo.
  --mode human     Opens a live GUI window instead of saving a file.
  --mode fast      Headless, no render/recording — just prints scores.
  --mode keyboard  Opens a window that you control with the keyboard. A
                   trained model is not needed in this mode.

Usage:
    python play.py --env-id ALE/Breakout-v5 --model models/dqn_model_edwin_breakout.zip --mode record --episodes 3
    python play.py --env-id ALE/SpaceInvaders-v5 --model models/dqn_model_david_spaceinvaders.zip --mode human
    python play.py --env-id ALE/Pong-v5 --model models/dqn_model_nziza_pong.zip --mode fast --episodes 10
    python play.py --env-id ALE/Pong-v5 --mode keyboard
    python play.py --env-id ALE/Breakout-v5 --mode keyboard
    python play.py --env-id ALE/SpaceInvaders-v5 --mode keyboard
"""

import argparse
import os
import time

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

gym.register_envs(ale_py)

FRAME_STACK = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a DQN model or play an Atari game with the keyboard.")
    parser.add_argument("--env-id", type=str, required=True, help="Atari environment id, e.g. ALE/Breakout-v5 — must match the model's training env")
    parser.add_argument("--model", type=str, help="Path to the saved model .zip (not needed in keyboard mode)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--sleep", type=float, default=0.01, help="Delay between frames (seconds), human mode only")
    parser.add_argument("--mode", type=str, default="record", choices=["record", "human", "fast", "keyboard"],
                         help="record = save .mp4 (default), human = watch the trained model, fast = scores only, keyboard = play yourself")
    parser.add_argument("--video-folder", type=str, default="videos", help="Where to save the .mp4 (record mode)")
    parser.add_argument("--video-length", type=int, default=3000, help="Max frames to capture (record mode)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def keyboard_controls(env_id, action_ids, pygame):
    """Return the correct key map and help text for a supported game."""
    game = env_id.rsplit("/", maxsplit=1)[-1].split("-", maxsplit=1)[0].lower()
    common_actions = {"NOOP", "FIRE", "RIGHT", "LEFT"}
    missing_actions = common_actions.difference(action_ids)
    if missing_actions:
        missing = ", ".join(sorted(missing_actions))
        raise SystemExit(f"Keyboard controls are not configured for {env_id}; missing actions: {missing}")

    fire = action_ids["FIRE"]

    if game == "pong":
        up = action_ids["RIGHT"]
        down = action_ids["LEFT"]
        up_fire = action_ids.get("RIGHTFIRE", up)
        down_fire = action_ids.get("LEFTFIRE", down)
        return {
            "w": up,
            pygame.K_UP: up,
            "s": down,
            pygame.K_DOWN: down,
            " ": fire,
            ("w", " "): up_fire,
            (pygame.K_UP, pygame.K_SPACE): up_fire,
            ("s", " "): down_fire,
            (pygame.K_DOWN, pygame.K_SPACE): down_fire,
        }, [
            "Move up:   W or Up Arrow",
            "Move down: S or Down Arrow",
            "Serve:     Space",
        ]

    if game in {"breakout", "spaceinvaders"}:
        left = action_ids["LEFT"]
        right = action_ids["RIGHT"]
        left_fire = action_ids.get("LEFTFIRE", left)
        right_fire = action_ids.get("RIGHTFIRE", right)
        fire_label = "Launch ball" if game == "breakout" else "Fire"
        return {
            "a": left,
            pygame.K_LEFT: left,
            "d": right,
            pygame.K_RIGHT: right,
            " ": fire,
            ("a", " "): left_fire,
            (pygame.K_LEFT, pygame.K_SPACE): left_fire,
            ("d", " "): right_fire,
            (pygame.K_RIGHT, pygame.K_SPACE): right_fire,
        }, [
            "Move left:  A or Left Arrow",
            "Move right: D or Right Arrow",
            f"{fire_label}:".ljust(12) + " Space",
        ]

    raise SystemExit(
        f"Keyboard mode supports ALE/Pong-v5, ALE/Breakout-v5, and "
        f"ALE/SpaceInvaders-v5; received {env_id}"
    )


def play_with_keyboard(env_id):
    """Open an Atari game and let a person control it with the keyboard."""
    try:
        import pygame
        from gymnasium.utils.play import play as gym_play
    except ImportError as exc:
        raise SystemExit("Keyboard mode requires pygame. Run: pip install -r requirements.txt") from exc

    env = gym.make(env_id, render_mode="rgb_array")
    meanings = env.unwrapped.get_action_meanings()
    action_ids = {name: index for index, name in enumerate(meanings)}

    try:
        keys_to_action, instructions = keyboard_controls(env_id, action_ids, pygame)
    except SystemExit:
        env.close()
        raise

    score = 0.0

    def print_episode_score(_obs, _next_obs, _action, reward, terminated, truncated, _info):
        nonlocal score
        score += float(reward)
        if terminated or truncated:
            print(f"Game over — score: {score:.1f}")
            score = 0.0

    print("Keyboard controls:")
    for instruction in instructions:
        print(f"  {instruction}")
    print("  Quit:      Escape or close the window")

    try:
        gym_play(
            env,
            keys_to_action=keys_to_action,
            noop=action_ids["NOOP"],
            callback=print_episode_score,
            zoom=3,
        )
    finally:
        env.close()


def make_play_env(env_id, mode):
    """Single-env, frame-stacked version of the training environment."""
    if mode == "human":
        env_kwargs = {"render_mode": "human"}
    elif mode == "record":
        env_kwargs = {"render_mode": "rgb_array"}
    else:
        env_kwargs = None
    env = make_atari_env(env_id, n_envs=1, seed=0, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    args = parse_args()

    if args.mode == "keyboard":
        play_with_keyboard(args.env_id)
        return

    if not args.model:
        raise SystemExit("--model is required unless --mode keyboard is used")

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    env = make_play_env(args.env_id, args.mode)

    if args.mode == "record":
        os.makedirs(args.video_folder, exist_ok=True)
        env = VecVideoRecorder(
            env,
            args.video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix=f"dqn_{args.env_id.split('/')[-1].split('-')[0].lower()}_play",
        )

    model = DQN.load(args.model, env=env, device=args.device)
    print(f"Loaded model: {args.model} (env: {args.env_id})")

    # Reset exactly ONCE before the loop, not per-episode. The underlying
    # VecEnv auto-resets internally when an episode ends, and
    # VecVideoRecorder closes/restarts a new video file on every explicit
    # reset() call — resetting per-episode would silently truncate the
    # recording to just the first episode/life.
    obs = env.reset()

    rewards = []
    for ep in range(1, args.episodes + 1):
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)  # greedy action selection
            obs, reward, dones, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
            if args.mode == "human":
                env.render()
                if args.sleep:
                    time.sleep(args.sleep)
            done = bool(dones[0])

        rewards.append(total_reward)
        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    print(f"\nAverage reward over {len(rewards)} episodes: {sum(rewards)/len(rewards):.1f}")
    env.close()
    if args.mode == "record":
        print(f"Video(s) saved to {args.video_folder}/")


if __name__ == "__main__":
    main()
