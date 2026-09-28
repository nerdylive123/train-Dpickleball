import numpy as np
from collections import deque
import cv2
import os
import torch
import time

from mlagents.torch_utils import nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium import Env, spaces
import gymnasium as gym


class TeamY:
    def __init__(self, frame_stack=4, model_path=None, vec_normalize_path="vecnormalize.pkl"):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.img_size = (240, 120)  # (width, height) for cv2.resize
        self.state = None  # For LSTM state
        self.vec_normalize = None
        self.episode_start = True
        self.prev_reward = None
        self.step_count = 0

        # Resolve paths relative to this file
        base_dir = os.path.dirname(__file__)
        if model_path is None:
            model_path = os.path.join(base_dir, "right_agent_policy_weights.pth")
        if not os.path.isabs(vec_normalize_path):
            vec_normalize_path = os.path.join(base_dir, vec_normalize_path)

        # Load model
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, using fallback no-op action")
            self.model = None
            self.policy_net = None
        else:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = device

                # Option 1: Load from .pth (policy weights only)
                if model_path.endswith('.pth'):
                    print(f"Loading policy weights from {model_path}")
                    self.policy_net = self._create_policy_network()
                    self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
                    self.policy_net.eval()
                    self.model = None  # No full model
                    print(f"✓ Policy network loaded successfully from {model_path}")

                # Option 2: Load from .zip (full SB3 model)
                else:
                    print(f"Loading full model from {model_path}")
                    self.model = RecurrentPPO.load(model_path, device=device)
                    self.model.n_envs = 1
                    self.policy_net = self.model.policy
                    print(f"✓ Full model loaded successfully from {model_path}")

                print(f"  - Device: {device}")

            except Exception as e:
                print(f"Error loading model: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
                self.policy_net = None

    def _create_policy_network(self):
        """
        Recreate the policy network architecture.
        This MUST match your training configuration exactly.
        """
        from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch.nn as nn

        # Define CustomCNN (must match training code exactly)
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
                    sample_output = self.cnn(sample_input)
                    cnn_output_dim = sample_output.shape[1]

                self.linear = nn.Sequential(
                    nn.Linear(cnn_output_dim, features_dim),
                    nn.ReLU()
                )

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self.linear(self.cnn(observations))

        # Define observation and action spaces (must match training)
        observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, 120, 240),
            dtype=np.float32
        )
        action_space = gym.spaces.MultiDiscrete([3, 3, 3])

        # CHANGED: Match the architecture from your checkpoint (64 instead of 256)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),
            # net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Changed from [256, 256]
            lstm_hidden_size=256,
            n_lstm_layers=1,
        )

        policy = RecurrentActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 0.0003,
            **policy_kwargs
        )

        return policy.to(self.device)
    def _preprocess_observation(self, observation):
        """Preprocess observation to match training format."""
        obs = observation.transpose(1, 2, 0)  # → (H, W, C)

        # Apply cropping
        h, w = obs.shape[:2]
        crop_top = int(h * 0.27)
        crop_bottom = int(h * 0.95)
        crop_left = int(w * 0.09)
        crop_right = int(w * 0.91)
        obs = obs[crop_top:crop_bottom, crop_left:crop_right]

        # Resize
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)

        return obs

    def _detect_episode_reset(self, reward):
        """Detect if episode should reset."""
        if isinstance(reward, (list, np.ndarray)):
            reward_val = reward[0] if len(reward) > 0 else 0
        else:
            reward_val = reward

        return abs(reward_val) >= 1.0

    def policy(self, observation, reward):
        """Policy function called at each timestep."""
        self.step_count += 1

        # Detect episode reset
        if self._detect_episode_reset(reward):
            print(f"🔄 Episode reset at step {self.step_count}")
            self.reset()

        self.prev_reward = reward

        # Episode start flag for LSTM
        episode_start = np.array([self.episode_start], dtype=np.float32)
        if self.episode_start:
            self.episode_start = False

        # Preprocess observation
        processed_obs = self._preprocess_observation(observation)

        # Frame stacking
        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(processed_obs)
        else:
            self.frames.append(processed_obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)

        # Model prediction
        if self.policy_net is not None:
            try:
                obs_batch = stacked_obs[None, ...].astype(np.float32)
                obs_tensor = torch.from_numpy(obs_batch).to(self.device)
                episode_start_tensor = torch.from_numpy(episode_start).to(self.device)

                with torch.no_grad():
                    # Forward pass through policy network
                    if self.state is None:
                        from sb3_contrib.common.recurrent.type_aliases import RNNStates

                        # Initialize LSTM state for both actor and critic
                        num_layers = 1
                        hidden_size = 256
                        batch_size = 1

                        h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)
                        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=self.device)

                        # Create RNNStates object with separate states for actor (pi) and critic (vf)
                        self.state = RNNStates(
                            pi=(h_0.clone(), c_0.clone()),
                            vf=(h_0.clone(), c_0.clone())
                        )

                    # Get action from policy
                    actions, _, _, self.state = self.policy_net.forward(
                        obs_tensor,
                        self.state,
                        episode_start_tensor,
                        deterministic=True
                    )

                    action = actions.cpu().numpy()[0]

                if self.step_count % 1000 == 0:
                    print(f"Step {self.step_count}: Action={action}")

                return action

            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                import traceback
                traceback.print_exc()
                self.policy_net = None

        # Fallback: no-op action
        print(f"⚠️ Using fallback action at step {self.step_count}")
        return np.array([1, 1, 0], dtype=np.int64)

    def reset(self):
        """Reset agent state for new episode."""
        print(f"🔄 Resetting agent (step {self.step_count})")
        self.frames.clear()
        self.state = None
        self.episode_start = True


