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
            print(f"🔄 Reward difference (Right - Left): {reward_diff:.3f}")

        return stacked_obs, reward_diff, game_ended, False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class PickleballCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=384, use_amp=True):
        super().__init__(observation_space, features_dim)
        self.use_amp = use_amp and torch.cuda.is_available()
        n_input_channels = observation_space.shape[0]

        # More gradual downsampling to preserve spatial details
        self.cnn = nn.Sequential(
            # First layer: moderate downsampling to preserve details
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),

            # Second layer: extract low-level features
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Third layer: maintain spatial resolution for detail
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Fourth layer: slight downsampling
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Fifth layer: high-level features
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Final pooling: smaller spatial size to reduce parameters
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Determine flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            out = self.cnn(sample)
            flat_dim = out.shape[1]

        # Much smaller and more efficient head
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = self.cnn(x)
                feats = self.linear(feats)
            return feats.float()
        else:
            feats = self.cnn(x)
            return self.linear(feats)


class ImprovedPickleballCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=384, use_amp=True,
                 use_temporal_compression=True, use_batchnorm=False):  # Disable BN by default for rollout
        super().__init__(observation_space, features_dim)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_temporal_compression = use_temporal_compression
        self.use_batchnorm = use_batchnorm
        n_input_channels = observation_space.shape[0]

        # Disable temporal compression for small channel counts to reduce overhead
        if use_temporal_compression and n_input_channels >= 32:
            compressed_channels = max(16, n_input_channels // 4)
            self.temporal_compressor = nn.Conv2d(
                n_input_channels, compressed_channels,
                kernel_size=1, stride=1, padding=0, bias=True  # Use bias since no BN
            )
            # Remove temporal BN as suggested - it's slow with small batches
            n_input_channels = compressed_channels
        else:
            self.temporal_compressor = None

        layers = []

        layers.extend([
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2, bias=True),  # Always use bias
            nn.ReLU(inplace=True)  # Remove BN, just use ReLU
        ])

        layers.extend([
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])

        layers.extend([
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ])

        # Remove the heavy 96->128->96 layers as suggested
        # This eliminates 2 conv layers and their associated overhead

        # Use smaller pooling to reduce linear input size by 4x
        layers.extend([
            nn.AdaptiveAvgPool2d((2, 2)),  # Changed from (4,4) to (2,2)
            nn.Flatten()
        ])

        self.cnn = nn.Sequential(*layers)

        # Calculate flattened dimension
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            if self.temporal_compressor:
                sample = self.temporal_compressor(sample)
            out = self.cnn(sample)
            flat_dim = out.shape[1]

        # Smaller MLP as suggested: 512->256
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 256),  # Reduced from 512
            nn.ReLU(inplace=True),
            # Remove dropout for inference speed
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations
        use_amp_now = self.use_amp and x.is_cuda and x.shape[0] >= 4

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp_now):
            if self.temporal_compressor:
                x = self.temporal_compressor(x)

            # Main CNN processing
            feats = self.cnn(x)
            feats = self.linear(feats)

        # Always ensure output is float32 for LSTM compatibility
        if feats.dtype != torch.float32:
            feats = feats.float()

        return feats
