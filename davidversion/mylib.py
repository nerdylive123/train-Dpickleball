import os

import numpy as np
import cv2
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv


class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=4, img_size=(240, 120), grayscale=True, left_agent=None):
        self.env = UnityParallelEnv(unity_env)

        self._save_debug_frames = True
        self._debug_frame_count = 0
        self._debug_dir = "debug_frames"
        os.makedirs(self._debug_dir, exist_ok=True)
        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]  # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0]  # agent at opposite (left)
        self.agent_obs = self.env.possible_agents[0]  # obs is only available in agent 0, always 0
        self.left_agent = left_agent  # optional predefined controller for left agent
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Final obs shape after manual preprocessing
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])  # (stack, H, W)
        else:
            obs_shape = (frame_stack * c, img_size[1], img_size[0])  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.env.action_spaces[self.agent]

        print(f"✓ Wrapper initialized:")
        print(f"  - Observation space: {self.observation_space.shape}")
        print(f"  - Action space: {self.action_space}")

    def _save_frame(self, frame, tag):
        # frame is (1,H,W) after grayscale preprocess; convert to uint8 PNG
        img = (frame* 255).astype(np.uint8)[0]
        path = os.path.join(self._debug_dir, f"{tag}_{self._debug_frame_count:04d}.png")
        cv2.imwrite(path, img)

    def _preprocess(self, obs):
        """
        Preprocess observation to match training format.

        CRITICAL FIX: Unity ML-Agents outputs images already normalized to [0, 1].
        We should NOT divide by 255 again!
        """
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)  # (H, W, C)

        # The court should be most of the frame
        # Try a MINIMAL crop - just remove obvious UI areas
        h, w = obs.shape[:2]

        # Remove only top UI bar (score) and bottom text
        # Start with conservative values
        crop_top = int(h * 0.27)  # Remove top ~30%
        crop_bottom = int(h * 0.95)  # Keep to 95%
        crop_left = int(w * 0.09)  # Remove left ~11%
        crop_right = int(w * 0.91)  # Keep to 89%

        obs = obs[crop_top:crop_bottom, crop_left:crop_right]
        if self._debug_frame_count % 10000 == 0:
            cv2.imwrite(f"{self._debug_dir}/after_crop_{self._debug_frame_count:04d}.png",
                        (obs * 255).astype(np.uint8))
        # Then resize
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        if self._debug_frame_count % 10000 == 0:
            cv2.imwrite(f"{self._debug_dir}/after_crop_R_{self._debug_frame_count:04d}.png",
                        (obs * 255).astype(np.uint8))

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # ✅ FIXED: Don't divide by 255 - Unity already outputs [0, 1]
        obs = obs.astype(np.float32)

        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()

        # DEBUG: Inspect the full observation dict on first reset
        if not hasattr(self, "_debugged_obs_space"):
            self._debugged_obs_space = True
            print("=== Unity obs dict summary on reset ===")
            for agent_id, od in obs_dict.items():
                try:
                    if isinstance(od, dict):
                        for k, v in od.items():
                            arr = np.array(v)
                            print(
                                f"  agent {agent_id} key '{k}': shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
                    else:
                        arr = np.array(od)
                        print(
                            f"  agent {agent_id}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
                except Exception as e:
                    print(f"  agent {agent_id}: error inspecting obs: {e}")

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        if self._save_debug_frames and self._debug_frame_count < 100:
            self._save_frame(obs, "reset")

        # Debug first observation
        print(f"Reset: obs range = [{obs.min():.3f}, {obs.max():.3f}]")

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        if self.left_agent is not None and hasattr(self.left_agent, "reset"):
            self.left_agent.reset()

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def step(self, action):
        # Compute left agent action if provided
        actions = {}
        if self.left_agent is not None:
            current_single_frame = self.frames[-1] if len(self.frames) > 0 else None
            left_action = self.left_agent.act(current_single_frame)
            actions[self.agent_other] = left_action

        # Right agent action from the learning policy
        actions[self.agent] = action

        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        if self._debug_frame_count % 1000 == 0:
            # Check if ball exists in observation dict
            print(f"Step {self._debug_frame_count}: obs_dict keys = {obs_dict.keys()}")
            print(f"  Agent obs shape: {obs_dict[self.agent_obs]['observation'][0].shape}")
            print(
                f"  Agent obs range: [{obs_dict[self.agent_obs]['observation'][0].min():.3f}, {obs_dict[self.agent_obs]['observation'][0].max():.3f}]")

        self._debug_frame_count += 1
        # Debug rewards
        if (rewards[self.agent] != 0 or rewards[self.agent_other] != 0):
            print(f"Rewards: agent={rewards[self.agent]:.2f}, opponent={rewards[self.agent_other]:.2f}")

        # ✅ IMPROVED REWARD STRUCTURE
        # Reward shaping: +1 for scoring, -1 for opponent scoring
        # Remove the 0.001 constant reward - it encourages doing nothing
        reward =  rewards[self.agent] - rewards[self.agent_other]

        if terminations[self.agent] or terminations[self.agent_other]:
            print("Episode terminated. Resetting left agent if applicable.")
            if self.left_agent is not None and hasattr(self.left_agent, "reset"):
                self.left_agent.reset()

        # Small bonus for each step without losing
        # or for keeping rally going
        # if not terminations[self.agent]:
        #     reward -= 0.001

        # Optional: small penalty per step to encourage ending points
        if abs(reward) >= 1.0:
            print(f"Step reward: {reward:.3f}")
            if self.left_agent is not None and hasattr(self.left_agent, "reset"):
                self.left_agent.reset()
        else:
            reward -= 0.005

        return stacked_obs, reward, terminations[self.agent], False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for processing stacked grayscale frames.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))