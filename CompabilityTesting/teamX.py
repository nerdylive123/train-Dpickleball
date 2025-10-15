import numpy as np
from collections import deque
import cv2
import os
import torch
from sb3_contrib import RecurrentPPO

# Build a Python class for your solution, do preprocessing (image processing, frame stacking, etc) here.
# During competition, only the policy function is called at each time step, providing the observation and reward for that time step only.
# Your agent is expected to return actions to be executed.
class TeamX:
    def __init__(self, frame_stack=4, model_path=None):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.img_size = (168, 84)  # Target size for preprocessing
        self.state = None  # For LSTM state

        # Load your checkpoint for policy network
        if model_path is None:
            # Default model path - adjust this to your trained model
            model_path = r"left_agent_600000_steps.zip"

        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, using fallback square movement")
            self.model = None
        else:
            try:
                # Try loading the model without specifying policy_kwargs
                # This will use the saved policy configuration
                self.model = RecurrentPPO.load(model_path, device="cuda").policy
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}, using fallback square movement")
                self.model = None

        # Fallback square movement parameters
        self.square_directions = [
            np.array([1, 0, 0], dtype=np.int32),   # right
            np.array([0, 1, 0], dtype=np.int32),   # up
            np.array([2, 0, 0], dtype=np.int32),   # left
            np.array([0, 2, 0], dtype=np.int32)    # down
        ]
        self.steps_per_side = 30
        self.step = 0

    def _preprocess_observation(self, observation):
        """Preprocess observation to match training format"""
        # observation comes in as (C, H, W), typically (3, H, W)
        # Convert to (H, W, C) for OpenCV processing
        obs = (observation * 255).astype(np.uint8)
        obs = obs.transpose(1, 2, 0)

        # Resize to match training size
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)

        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0

        return obs

    # Your policy takes only visual representation as input,
    # and reward is 1 when you score, -1 when your opponent scores
    # Your policy function returns actions
    def policy(self, observation, reward):
        # Implement your solution here

        # Image processing
        processed_obs = self._preprocess_observation(observation)
        self.frames.append(processed_obs)

        # Ensure we have enough frames for stacking
        while len(self.frames) < self.frame_stack:
            self.frames.append(processed_obs)  # Repeat the current frame

        # Create stacked observation
        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # Use your policy network here
        if self.model is not None:
            try:
                # Add batch dimension for model prediction
                obs_batch = np.expand_dims(stacked_obs, axis=0)  # (1, stack, H, W)

                # Get action from trained model
                action, self.state = self.model.predict(obs_batch, state=self.state, deterministic=True)
                return action[0]  # Remove batch dimension

            except Exception as e:
                print(f"Error during model prediction: {e}, falling back to square movement")
                self.model = None  # Disable model for future calls

        # Fallback: square motion
        side = (self.step // self.steps_per_side) % 4
        action = self.square_directions[side]
        self.step += 1

        return action

    def reset(self):
        """Reset internal state"""
        self.frames.clear()
        self.state = None
        self.step = 0
