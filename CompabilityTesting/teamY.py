import numpy as np
from collections import deque
import cv2
import os
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class TeamY:
    def __init__(self, frame_stack=4, model_path=None, vec_normalize_path="vecnormalize.pkl"):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.img_size = (240, 120)  # (width, height) for cv2.resize - MUST match training resolution
        self.state = None  # For LSTM state
        self.vec_normalize = None
        self.episode_start = True
        self.prev_reward = None
        self.step_count = 0

        # Resolve paths relative to this file
        base_dir = os.path.dirname(__file__)
        if model_path is None:
            model_path = os.path.join(base_dir, "right_agent_150000_steps.zip")
        if not os.path.isabs(vec_normalize_path):
            vec_normalize_path = os.path.join(base_dir, vec_normalize_path)

        # Load VecNormalize statistics
        # if os.path.exists(vec_normalize_path):
        #     try:
        #         dummy_venv = self._build_dummy_vecenv()
        #         self.vec_normalize = VecNormalize.load(vec_normalize_path, venv=dummy_venv)
        #         self.vec_normalize.training = False
        #         self.vec_normalize.norm_reward = False
        #         print(f"Successfully loaded VecNormalize stats from {vec_normalize_path}")
        #     except Exception as e:
        #         print(f"Warning: Failed to load VecNormalize stats: {e}")
        #         self.vec_normalize = None
        # else:
        #     print(f"Warning: VecNormalize stats not found at {vec_normalize_path}")

        # Load model
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, using fallback no-op action")
            self.model = None
        else:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = RecurrentPPO.load(model_path, device=device)
                print(f"✓ Model loaded successfully from {model_path}")
                print(f"  - Device: {device}")
                print(f"  - Action space: {self.model.action_space}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None

    def _build_dummy_vecenv(self):
        """Create a minimal DummyVecEnv for VecNormalize loading."""
        try:
            import gym
        except Exception:
            import gymnasium as gym

        obs_shape = (self.frame_stack, self.img_size[1], self.img_size[0])

        class _ObsOnlyEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
                )
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                try:
                    super().reset(seed=seed)
                except Exception:
                    pass
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                return obs, 0.0, False, False, {}

        return DummyVecEnv([lambda: _ObsOnlyEnv()])

    def _save_frame(self, frame, tag):
        # frame is (1,H,W) after grayscale preprocess; convert to uint8 PNG
        img = (frame* 255).astype(np.uint8)[0]
        path = f"{tag}.png"
        cv2.imwrite(path, img)

    def _preprocess_observation(self, observation):
        """
        Preprocess observation to match training format.

        CRITICAL: Unity environment outputs images in [0, 1] range already!
        We should NOT divide by 255 again.
        """

        # self._save_frame(observation, "raw")
        # observation: (C, H, W) with values in [0, 1]
        obs = observation.transpose(1, 2, 0)  # → (H, W, C)

        # CRITICAL: Apply same cropping as training to remove UI elements
        h, w = obs.shape[:2]
        crop_top = int(h * 0.27)     # Remove top ~27%
        crop_bottom = int(h * 0.95)  # Keep to 95%
        crop_left = int(w * 0.09)    # Remove left ~9%
        crop_right = int(w * 0.91)   # Keep to 91%
        obs = obs[crop_top:crop_bottom, crop_left:crop_right]

        # Resize to target size
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)

        # Already normalized! Don't divide by 255
        obs = obs.astype(np.float32)

        return obs

    def _detect_episode_reset(self, reward):
        """
        Detect if episode should reset.
        Handles both scalar rewards and array rewards.
        """
        # Handle array rewards (e.g., [1.0, 0.0])
        if isinstance(reward, (list, np.ndarray)):
            # Check if any element indicates a point scored
            reward_val = reward[0] if len(reward) > 0 else 0
        else:
            reward_val = reward

        # Reset if a point was scored (|reward| >= 1)
        if abs(reward_val) >= 1.0:
            return True

        return False

    def policy(self, observation, reward):
        """
        Policy function called at each timestep.

        Args:
            observation: (C, H, W) image, typically (3, 84, 168) in [0, 1] range
            reward: Scalar or array reward

        Returns:
            action: Action array (for MultiDiscrete: [x_move, y_move, hit])
        """
        self.step_count += 1

        # Debug first few steps
        if self.step_count <= 3:
            print(f"\n=== Step {self.step_count} ===")
            print(f"Observation: shape={observation.shape}, range=[{observation.min():.3f}, {observation.max():.3f}]")
            print(f"Reward: {reward}")

        # Detect episode reset
        if self._detect_episode_reset(reward):
            print(f"🔄 Episode reset at step {self.step_count} (reward={reward})")
            self.reset()

        self.prev_reward = reward

        # Episode start flag for LSTM
        episode_start = np.array([self.episode_start], dtype=bool)
        if self.episode_start:
            self.episode_start = False

        # Preprocess observation
        processed_obs = self._preprocess_observation(observation)

        if self.step_count <= 3:
            print(
                f"Processed: shape={processed_obs.shape}, range=[{processed_obs.min():.3f}, {processed_obs.max():.3f}]")

        # Frame stacking
        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(processed_obs)
        else:
            self.frames.append(processed_obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # Model prediction
        if self.model is not None:
            try:
                obs_batch = stacked_obs[None, ...].astype(np.float32)  # (1, stack, H, W)

                # Debug: Print shapes before prediction
                if self.step_count <= 3:
                    print(f"[DEBUG] obs_batch shape: {obs_batch.shape}, dtype: {obs_batch.dtype}")
                    print(f"[DEBUG] obs_batch range: [{obs_batch.min():.3f}, {obs_batch.max():.3f}]")
                    print(f"[DEBUG] episode_start: {episode_start}, type: {type(episode_start)}")
                    print(f"[DEBUG] state: {type(self.state)}")
                    print(f"[DEBUG] Model observation space: {self.model.observation_space}")
                    print(f"[DEBUG] Model action space: {self.model.action_space}")

                # Get action
                # print(f"[DEBUG] About to call model.predict()...")
                action, self.state = self.model.predict(
                    obs_batch,
                    state=self.state,
                    episode_start=episode_start,
                    deterministic=True
                )
                # print(f"[DEBUG] model.predict() succeeded! Action: {action}")

                # Debug periodic output
                if self.step_count <= 10 or self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: Action={action[0]}, Episode_start={episode_start[0]}")

                return action[0]

            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                print(f"❌ Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print(f"❌ Disabling model for future calls")
                self.model = None

        # Fallback: no-op action (no movement, no hit)
        # For MultiDiscrete([3, 3, 3]): [1, 1, 0] = stay in place, no hit
        print(f"⚠️ Using fallback action at step {self.step_count}")
        return np.array([1, 1, 0], dtype=np.int64)

    def reset(self):
        """Reset agent state for new episode."""
        print(f"🔄 Resetting agent (step {self.step_count})")
        self.frames.clear()
        self.state = None  # Clear LSTM hidden state
        self.episode_start = True
