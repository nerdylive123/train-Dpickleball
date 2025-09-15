from stable_baselines3 import PPO
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
import numpy as np
import os
import random
import gymnasium as gym
from gymnasium import spaces
import traceback

class PickleballGymWrapper(gym.Env):
    def __init__(self, env_path, worker_id=None, no_graphics=True):
        super().__init__()
        
        # Setup Unity environment
        self.string_channel = StringSideChannel()
        self.channel = CustomDataChannel()
        self.channel.send_data(serve=212, p1=0, p2=0)
        self.worker_id = worker_id if worker_id is not None else random.randint(5000, 6000)
        
        self.unity_env = UnityEnvironment(
            env_path,
            side_channels=[self.string_channel, self.channel],
            worker_id=self.worker_id,
            no_graphics=no_graphics
        )
        self.env = UnityParallelEnv(self.unity_env)
        
        # Define observation space for image input
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(84, 168, 3), dtype=np.uint8)
        })
        
        # Define action space for discrete actions
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        if self.env.agents:
            agent_obs = next(iter(obs.values()))
            if isinstance(agent_obs, dict):
                visual_obs = np.array(agent_obs.get('visual_obs', np.zeros((84, 168, 3))))
                return {'image': visual_obs.astype(np.uint8)}, {}
            return {'image': np.array(agent_obs).astype(np.uint8)}, {}
        return {'image': np.zeros((84, 168, 3), dtype=np.uint8)}, {}
    
    def step(self, action):
        if not self.env.agents:
            return (
                {'image': np.zeros((84, 168, 3), dtype=np.uint8)},
                0.0,
                True,
                False,
                {}
            )
        
        discrete_action = np.array(action, dtype=np.int32)
        actions = {self.env.agents[0]: discrete_action}
        obs, rewards, dones, infos = self.env.step(actions)
        
        agent_obs = obs[self.env.agents[0]]
        if isinstance(agent_obs, dict):
            visual_obs = np.array(agent_obs.get('visual_obs', np.zeros((84, 168, 3))))
            processed_obs = {'image': visual_obs.astype(np.uint8)}
        else:
            processed_obs = {'image': np.array(agent_obs).astype(np.uint8)}
        
        return (
            processed_obs,
            float(rewards[self.env.agents[0]]),
            bool(dones[self.env.agents[0]]),
            False,
            infos
        )
    
    def close(self):
        if hasattr(self, 'env'):
            try:
                self.env.close()
            except:
                pass
        if hasattr(self, 'unity_env'):
            try:
                self.unity_env.close()
            except:
                pass

def main():
    env = None
    try:
        os.makedirs("models", exist_ok=True)
        # GANTI PATH SENDIRI
        env = PickleballGymWrapper(
            r"C:\Users\Vanessa\Downloads\dpickleball\Pickleball_Build_Training\dp.exe"
        )
        
        # Increase training parameters since basic training works
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            batch_size=64,    # Increased from 32
            n_steps=256,      # Increased from 128
            learning_rate=3e-4,
            ent_coef=0.01,
            n_epochs=5,       # Increased from 3
            device='cpu'
        )
        
        print("Starting training...")
        model.learn(total_timesteps=2000)  # Increased from 500
        model.save("models/pickeball_ppo")
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Error during environment cleanup: {e}")

if __name__ == "__main__":
    main()
