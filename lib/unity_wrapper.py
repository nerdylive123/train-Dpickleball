"""
This module contains the SharedObsUnityGymWrapper, a Gymnasium wrapper for a
Unity environment with shared observations.
"""
import numpy as np
import cv2
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class SharedObsUnityGymWrapper(Env):
    """
    A Gymnasium wrapper for a Unity environment with two agents, where only one agent
    provides observations. The observations are shared and preprocessed before
    being passed to the learning agent.
    """
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=True):
        """
        Initializes the SharedObsUnityGymWrapper.

        Args:
            unity_env: The Unity environment to wrap.
            frame_stack (int): The number of frames to stack for the observation.
            img_size (tuple): The size to resize the observation images to (W, H).
            grayscale (bool): Whether to convert the observation images to grayscale.
        """
        self.env = UnityParallelEnv(unity_env)

        # Agent configuration
        # agent 1 (right) is the agent to be controlled
        self.agent_to_control = self.env.possible_agents[1]
        # agent 0 (left) is the opponent
        self.opponent_agent = self.env.possible_agents[0]
        # Observations are only available from agent 0
        self.observation_agent = self.env.possible_agents[0]

        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Define observation space
        base_obs_space = self.env.observation_spaces[self.observation_agent][0]
        num_channels, height, width = base_obs_space.shape
        self._transpose_obs = (num_channels == 3)

        # Calculate the shape of the preprocessed observations
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])  # (stack, H, W)
        else:
            obs_shape = (frame_stack * num_channels, img_size[1], img_size[0])  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        self.action_space = self.env.action_spaces[self.agent_to_control]

    def _preprocess_observation(self, obs):
        """
        Preprocesses the raw observation from the environment. This includes
        transposing, resizing, grayscaling, and normalizing the image.
        """
        # Transpose from (C, H, W) to (H, W, C) if necessary
        if self._transpose_obs:
            obs = obs.transpose(1, 2, 0)

        # Resize and apply grayscale
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        return obs

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()

        # The observation is taken from a specific agent
        initial_obs = self._preprocess_observation(obs_dict[self.observation_agent]['observation'][0])

        # Initialize the frame stack with the first observation
        for _ in range(self.frame_stack):
            self.frames.append(initial_obs)

        return np.concatenate(list(self.frames), axis=0), {}

    def step(self, action):
        """
        Takes a step in the environment with the given action.
        """
        actions = {self.agent_to_control: action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        # Preprocess and stack the new observation
        new_obs = self._preprocess_observation(obs_dict[self.observation_agent]['observation'][0])
        self.frames.append(new_obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)

        # The reward is designed to encourage winning over the opponent
        reward = rewards[self.agent_to_control] - rewards[self.opponent_agent]

        if (rewards[self.agent_to_control] + rewards[self.opponent_agent]) > 0:
            print ("Rewards: ", rewards[self.agent_to_control], rewards[self.opponent_agent])

        return stacked_obs, reward, terminations[self.agent_to_control], False, infos[self.agent_to_control]

    def render(self, mode='human'):
        """
        Renders the environment.
        """
        return self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()
