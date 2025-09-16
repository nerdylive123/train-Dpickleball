import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import traceback


# Import Gym and Stable-Baselines3 components
from gym import Env, spaces
from gym.utils import seeding
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import the mlagents-envs components
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from mylib import SharedObsUnityGymWrapper


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# -----------------------------
# Main Training Logic
# -----------------------------
if __name__ == "__main__":
    ENV_PATH = r"C:\Users\David\DPICKLEBALL COMPETITIONS\PickleBallFinal\Pickleball_Build_Training\dp.exe"
    MODEL_SAVE_PATH = "ppo_pickleball_agent"

    env = None
    try:
        # 1. Create the side channels
        string_channel = StringSideChannel()
        channel = CustomDataChannel()

        # 2. Send the initialization command
        channel.send_data(serve=212, p1=0, p2=0)

        # 3. Initialize the Unity Environment with the side channels
        unity_env = UnityEnvironment(
            ENV_PATH,
            worker_id=1,
            no_graphics=False,
            side_channels=[string_channel, channel]
        )

        env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True)

        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
        )

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=1024,
            batch_size=32,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=3e-4,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Starting training on device: {model.device}")
        model.learn(total_timesteps=1_000_000)

        model.save(MODEL_SAVE_PATH)
        print(f"Training complete. Model saved to '{MODEL_SAVE_PATH}.zip'")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
    finally:
        if env:
            print("Closing environment.")
            env.close()