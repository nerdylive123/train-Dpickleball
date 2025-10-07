# left_agent.py
import numpy as np
import os
import torch
from sb3_contrib import RecurrentPPO


class LeftAgent:
    def __init__(self):
        self.time_elapsed = 0
        self.__step = 0

    def pattern_1(self):
        # Cycle between left and right movements each 5 time steps
        if (self.__step // 12) % 2 == 0:
            action = np.array([0, 2, 2])  # Move left
            self.__step += 2
        else:
            action = np.array([0, 1, 0])
            self.__step += 1
        return action

    def pattern_simple_right(self):
        self.__step += 1
        return [0, 1, np.random.randint(0,3, dtype=np.int32)]  # just keep moving right until the middle of the field

    def pattern_simple2(self):
        # move left [0,2,0] for 10 steps then next should be stop [0,0,0] for 60 steps then
        # move right [0,1,0] for 10 steps then stop [0,0,0] for 50 steps
        cycle_length = 100

        step_in_cycle = self.__step % cycle_length
        self.__step += 1

        if step_in_cycle < 10:
            action = np.array([0, 2, 0])
        elif step_in_cycle < 70:
            action = np.array([0, 0, 0])
        elif step_in_cycle < 80:
            action = np.array([0, 1, 0])
        else:
            action = np.array([0, 0, 0])
        return action

    def pattern_still(self):
        self.__step += 1
        return np.array([0, 0, 0])

    def act(self, observation):
        return self.pattern_simple_right()

    def reset(self):
        self.time_elapsed = 0
        self.__step = 0


class ModelLeftAgent:
    def __init__(self, model_path: str, device: str | None = None, deterministic: bool = True):
        self.model_path = model_path if os.path.exists(model_path) else f"{model_path}.zip"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model for left agent not found at {self.model_path}")
        full_agent = RecurrentPPO.load(self.model_path, device="cpu")
        self.model = full_agent.policy
        self.state = None
        self.deterministic = deterministic
        # Expected input channels from the trained policy's observation space
        self.expected_stack = getattr(self.model.observation_space, 'shape', (1,))[0]
        print(f"LeftAgent loaded model from {self.model_path}")

    def act(self, observation):
        if observation is None:
            raise ValueError("ModelLeftAgent requires observation input")
        obs = observation
        # observation may be single-frame (1,H,W); pad to expected stack if needed
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            c, h, w = obs.shape
            if c == 1 and self.expected_stack > 1:
                obs = np.repeat(obs, self.expected_stack, axis=0)
        # add batch dimension
        obs = np.expand_dims(obs, axis=0)
        action, self.state = self.model.predict(obs, state=self.state, episode_start=None, deterministic=self.deterministic)
        print(f"LeftAgent action: {action[0]}")
        return action[0]

    def reset(self):
        self.state = None