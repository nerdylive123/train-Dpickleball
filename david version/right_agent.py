import torch
from stable_baselines3 import PPO
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from mlagents_envs.environment import UnityEnvironment

from shared_env import create_env
from mylib import SharedObsUnityGymWrapper
from custom_cnn import CustomCNN


def train_right_agent():
    ENV_PATH = r"C:\Users\David\DPICKLEBALL COMPETITIONS\PickleBallFinal\Pickleball_Build_Training\dp.exe"
    MODEL_SAVE_PATH = "right_agent_model"

    # 1. Create the side channels
    string_channel = StringSideChannel()
    channel = CustomDataChannel()
    channel.send_data(serve=212, p1=0, p2=0)

    # 2. Initialize the Unity Environment with the side channels
    env = create_env()


    # 4. Use the refactored custom CNN features extractor
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
    try:
        model.learn(total_timesteps=1_000_000)
        model.save(MODEL_SAVE_PATH)
        print("Right agent training complete.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    train_right_agent()