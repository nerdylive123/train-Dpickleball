import numpy as np
import cv2
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from gym.spaces import Tuple as GymTuple
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True, left_agent=None):
        self.env = UnityParallelEnv(unity_env)

        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]       # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0] # agent at opposite (left)
        self.agent_obs = self.env.possible_agents[0]   # obs is only available in agent 0, always 0
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None
        self.left_agent = left_agent  # Store the left agent instance

        # Add reward tracking for episode detection
        self._last_reward = 0.0
        self._total_score_left = 0
        self._total_score_right = 0

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs]
        if isinstance(base_obs, GymTuple):
            base_obs = base_obs.spaces[0]
        elif hasattr(base_obs, 'spaces') and isinstance(base_obs.spaces, (tuple, list)):
            base_obs = base_obs.spaces[0]

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

    def _preprocess(self, obs):
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        # Resize and grayscale
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return obs

    def reset(self, *, seed=None, options=None):
        if self.left_agent is not None and hasattr(self.left_agent, 'reset'):
            self.left_agent.reset()
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        # Reset score tracking
        self._total_score_left = 0
        self._total_score_right = 0
        print("🆕 NEW GAME STARTED! Score reset to 0-0")

        obs_dict = self.env.reset()
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def step(self, action):
        # Get left agent action - both agents run in the same environment
        left_action = None
        if self.left_agent is not None:
            # The left agent doesn't need observations, it just returns a fixed action
            left_action = self.left_agent.act(None)  # Pass None since left_agent doesn't use obs
            # Convert to numpy array if it's a list
            if isinstance(left_action, list):
                left_action = np.array(left_action, dtype=np.int32)
        else:
            # Fallback: use random action if no left agent
            left_action = self.env.action_spaces[self.agent_other].sample()

        # Execute actions for both agents simultaneously
        actions = {self.agent: action, self.agent_other: left_action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        # Store last observation for debugging
        self._last_obs = obs_dict

        # Process observation for the right agent (agent being trained)
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)

        # Update scores when points are scored
        if rewards[self.agent] > 0:
            self._total_score_right += 1
            print(f"🏓 Right agent scored! Current Score: Right={self._total_score_right}, Left={self._total_score_left}")
        elif rewards[self.agent_other] > 0:
            self._total_score_left += 1
            print(f"🏓 Left agent scored! Current Score: Right={self._total_score_right}, Left={self._total_score_left}")

        # Check if someone has reached 21 points (game end condition)
        game_ended_by_score = (self._total_score_left >= 21 or self._total_score_right >= 21)

        # Also check Unity's termination signal
        game_ended_by_termination = terminations[self.agent] or terminations[self.agent_other]

        # Game ends if either condition is met
        game_ended = game_ended_by_score or game_ended_by_termination

        if game_ended:
            if game_ended_by_score:
                winner = "Right" if self._total_score_right >= 21 else "Left"
                print(f"\n🏆 GAME OVER! {winner} agent WINS!")
                print(f"🎯 Final Score: Right={self._total_score_right}, Left={self._total_score_left}")
                print(f"🎮 Game ended because {winner} reached 21 points")
            else:
                print(f"\n🎯 GAME ENDED by Unity termination signal!")
                print(f"🎮 Final Score: Right={self._total_score_right}, Left={self._total_score_left}")

            print("=" * 50)

        # Calculate reward difference (right agent - left agent)
        reward_diff = rewards[self.agent] - rewards[self.agent_other]

        # Store the reward for episode tracking
        self._last_reward = reward_diff
        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print(f"🎯 Rewards this step - Right: {rewards[self.agent]:.3f}, Left: {rewards[self.agent_other]:.3f}")
        # Return reward difference (what the training agent receives)
        return stacked_obs, rewards[self.agent], game_ended, False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layers to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, features_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class FastCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN optimized for low latency.
    - Fewer channels and layers
    - Aggressive strides to downsample early
    - Optional channels-last and mixed precision for CUDA
    """
    def __init__(self, observation_space, features_dim=256, use_amp=True, use_channels_last=True):
        super().__init__(observation_space, features_dim)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_channels_last = use_channels_last
        n_input_channels = observation_space.shape[0]

        # Compact CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Determine flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            if self.use_channels_last:
                sample = sample.to(memory_format=torch.channels_last)
            out = self.cnn(sample)
            flat_dim = out.shape[1]

        # Smaller head
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        # Optionally set modules to channels-last for better GPU throughput
        if self.use_channels_last:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
                    pass  # layers operate fine with channels-last inputs

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        if self.use_channels_last:
            x = x.to(memory_format=torch.channels_last)
        if self.use_amp:
            # autocast speeds up convs on GPU
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = self.cnn(x)
                feats = self.linear(feats)
            # Project back to fp32 for downstream stability
            return feats.float()
        else:
            feats = self.cnn(x)
            feats = self.linear(feats)
            return feats
