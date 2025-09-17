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
from gym.spaces import Tuple as GymTuple

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True, left_agent=None):
        self.env = UnityParallelEnv(unity_env)
        self.agent = self.env.possible_agents[1]       # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0] # agent at opposite (left)
        self.agent_obs = self.env.possible_agents[0]   # obs is only available in agent 0, always 0
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None
        self.left_agent = left_agent  # Store the left agent instance

        # Observation space
        base_obs = self.env.observation_spaces.get(self.agent_obs, None)
        if base_obs is None:
            raise ValueError(f"Observation space for agent '{self.agent_obs}' not found. Available: {self.env.observation_spaces}")
        # If it's a Tuple, use the first element (image)
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
        if self._transpose:
            obs = obs.transpose(1, 2, 0)
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=0)
        else:
            obs = obs.transpose(2, 0, 1)
        obs = obs.astype(np.float32) / 255.0
        return obs

    def reset(self, *, seed=None, options=None):
        if self.left_agent is not None and hasattr(self.left_agent, 'reset'):
            self.left_agent.reset()
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
        obs_dict = self.env.reset()
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), {}

    def step(self, action):
        # Get left agent's observation
        obs_dict = self.env.observe() if hasattr(self.env, 'observe') else None
        if obs_dict is None:
            # If no observe method, step with dummy action to get obs_dict
            dummy_actions = {self.agent: action, self.agent_other: self.env.action_spaces[self.agent_other].sample()}
            obs_dict, _, _, _ = self.env.step(dummy_actions)
        obs_left = obs_dict[self.agent_other]['observation'][0]
        left_action = self.left_agent.act(obs_left) if self.left_agent is not None else self.env.action_spaces[self.agent_other].sample()
        actions = {self.agent: action, self.agent_other: left_action}

        # Store the previous state info to determine if a ball hit occurred
        prev_obs = obs_dict[self.agent_obs]['observation'][0] if obs_dict else None

        obs_dict, rewards, terminations, infos = self.env.step(actions)
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)

        # Check if this is a reward from active play (ball hit) or a 5-second stationary rule
        # We'll use the infos dictionary to determine this, and if it doesn't contain
        # that information, we'll use a heuristic based on the observation changes

        reward = 0
        raw_reward = rewards[self.agent]

        # Check if infos contains information about the reward source
        if 'reward_source' in infos[self.agent]:
            if infos[self.agent]['reward_source'] == 'active_hit':
                reward = raw_reward
        else:
            # Heuristic: If the agent's position is close to the ball's previous position
            # and the ball's trajectory changed significantly, consider it an active hit
            # We assume active play rewards when both agents have non-zero rewards
            if raw_reward > 0 and 'ball_contact' in infos[self.agent] and infos[self.agent]['ball_contact']:
                reward = raw_reward
            # Otherwise, we assume it's a stationary ball rule reward and don't count it

        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print("Rewards - Raw: ", rewards[self.agent], rewards[self.agent_other], "Adjusted:", reward)

        return stacked_obs, reward, terminations[self.agent], False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
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
