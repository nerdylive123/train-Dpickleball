import torch
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from mlagents_envs.environment import UnityEnvironment
from mylib import SharedObsUnityGymWrapper, CustomCNN
from stable_baselines3 import PPO

def train_right_agent():
    ENV_PATH = r"C:\Users\David\DPICKLEBALL COMPETITIONS\PickleBallFinal\Pickleball_Build_Training\dp.exe"
    MODEL_SAVE_PATH = "right_agent_model"

    # 1. Create the side channels
    string_channel = StringSideChannel()
    channel = CustomDataChannel()
    channel.send_data(serve=212, p1=0, p2=0)

    # 2. Initialize the Unity Environment with the side channels
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
        n_steps=1024, #modifies from 1024
        batch_size=32, #modified from 64
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Starting training on device: {model.device}")
    model.learn(total_timesteps=100_000)
    model.save(MODEL_SAVE_PATH)
    print("Right agent training complete.")

if __name__ == "__main__":
    train_right_agent()
