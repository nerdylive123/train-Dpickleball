#!/usr/bin/env python
"""
Minimal PPO training example using SharedObsUnityGymWrapper and CustomCNN.

Requirements (install in your env):
    pip install stable-baselines3 torch gymnasium opencv-python numpy

Usage:
    python examples/train_simple_sb3.py --executable "E:/DpickleBallEnv/PickleBallFinal/Pickleball_Build_Training/dp.exe" --timesteps 10000

Notes:
- The script wraps the Unity environment with SharedObsUnityGymWrapper, which
  preprocesses images, stacks frames, and shapes rewards.
- CustomCNN is used as the SB3 feature extractor for CnnPolicy.
"""
import argparse
import os
from datetime import datetime
from typing import cast

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

from stable_baselines3 import PPO

from lib import SharedObsUnityGymWrapper, CustomCNN


def make_unity_env(executable_path: str) -> UnityEnvironment:
    # Optional custom channels used by the dPickleBall build
    string_channel = StringSideChannel()
    data_channel = CustomDataChannel()
    # Initialize serve/scores to safe defaults
    data_channel.send_data(serve=212, p1=0, p2=0)
    return UnityEnvironment(executable_path, side_channels=[string_channel, data_channel])


def main():
    parser = argparse.ArgumentParser(description="Train PPO on dPickleBall with SB3")
    parser.add_argument("--executable", required=False,
                        default="E:/DpickleBallEnv/PickleBallFinal/Pickleball_Build_Training/dp.exe", help=("Path to "
                                                                                                            "Unity executable (training build)"))
    parser.add_argument("--timesteps", type=int, default=10000, help="Total PPO timesteps")
    parser.add_argument("--frame_stack", type=int, default=8, help="Frames to stack in observations")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale observations")
    parser.add_argument("--features_dim", type=int, default=256, help="CustomCNN features dimension")
    parser.add_argument("--out", default="runs", help="Output directory for checkpoints/logs")
    args = parser.parse_args()

    run_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    unity_env = make_unity_env(args.executable)

    env = SharedObsUnityGymWrapper(
        unity_env, frame_stack=args.frame_stack, img_size=(168, 84), grayscale=args.grayscale
    )

    # PPO with custom CNN extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )

    # Reasonable small defaults for a smoke test; tune for real training
    model = PPO(
        policy="CnnPolicy",
        env=cast("Env", env),  # quiet type checkers; SB3 accepts Gym/Gymnasium envs
        policy_kwargs=policy_kwargs,
        n_steps=512,
        batch_size=128,
        learning_rate=2.5e-4,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
    )

    try:
        model.learn(total_timesteps=args.timesteps)
        model.save(os.path.join(run_dir, "ppo_dpickleball"))
    finally:
        # Ensure Unity is closed cleanly
        env.close()


if __name__ == "__main__":
    main()
