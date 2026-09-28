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
        
        # LSTM state management (improved from ModelLeftAgent)
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.deterministic = True

        # Load your checkpoint for policy network
        if model_path is None:
            # Default model path - adjust this to your trained model
            model_path = r"left_agent_600000_steps.zip"

        # Check for .zip extension (improved from ModelLeftAgent)
        model_path = model_path if os.path.exists(model_path) else f"{model_path}.zip"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, using fallback square movement")
            self.model = None
            self.expected_shape = None
        else:
            try:
                # Load full agent then extract policy (improved from ModelLeftAgent)
                full_agent = RecurrentPPO.load(model_path, device="cuda")
                self.model = full_agent.policy
                self.model.set_training_mode(False)
                
                # Get expected observation shape from policy
                obs_space = self.model.observation_space
                self.expected_shape = obs_space.shape
                
                print(f"Successfully loaded model from {model_path}")
                print(f"Expected observation shape: {self.expected_shape}")
            except Exception as e:
                print(f"Error loading model: {e}, using fallback square movement")
                self.model = None
                self.expected_shape = None

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
                # Validate and fix observation shape if needed (improved from ModelLeftAgent)
                if self.expected_shape is not None and stacked_obs.shape != self.expected_shape:
                    print(f"[TeamX] Warning: obs shape {stacked_obs.shape} != expected {self.expected_shape}")
                    
                    # Handle different cases
                    if len(stacked_obs.shape) == 3:
                        c, h, w = stacked_obs.shape
                        expected_c, expected_h, expected_w = self.expected_shape
                        
                        # If frame stack mismatch
                        if c != expected_c:
                            if c > expected_c:
                                # Take most recent frames
                                stacked_obs = stacked_obs[-expected_c:, :, :]
                            else:
                                # Pad with last frame
                                padding = np.repeat(stacked_obs[-1:, :, :], expected_c - c, axis=0)
                                stacked_obs = np.concatenate([stacked_obs, padding], axis=0)
                        
                        # If spatial dimensions mismatch (shouldn't happen, but defensive)
                        if h != expected_h or w != expected_w:
                            print(f"[TeamX] Spatial dimension mismatch - this shouldn't happen!")
                
                # Add batch dimension for model prediction
                obs_batch = np.expand_dims(stacked_obs, axis=0)  # (1, stack, H, W)

                # Get action from trained model with improved LSTM state management
                with np.errstate(all='ignore'):
                    action, self.lstm_states = self.model.predict(
                        obs_batch,
                        state=self.lstm_states,
                        episode_start=self.episode_start,
                        deterministic=self.deterministic
                    )
                
                self.episode_start[0] = False
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
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.step = 0
