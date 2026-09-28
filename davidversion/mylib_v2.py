"""
Refactored Unity Gym Wrapper with Modular Reward System
Clean, maintainable version with all fixes applied.
"""

import os
import numpy as np
import cv2
from collections import deque
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import modular reward system
from .reward_system import RewardCalculator
from .reward_system.detection_utils import BallDetector, PaddleDetector, DetectionDebugger


class SharedObsUnityGymWrapper(Env):
    """
    Gymnasium wrapper for Unity ML-Agents pickleball environment.

    Features:
    - Frame stacking for temporal information
    - Modular reward system with robust contact detection
    - Velocity-aware positioning rewards
    - OOB detection and handling
    - Optional left agent (opponent) support
    """

    def __init__(self, unity_env, frame_stack=4, img_size=(240, 120), grayscale=True, left_agent=None):
        self.env = UnityParallelEnv(unity_env)

        # Agent configuration
        self.agent = self.env.possible_agents[1]  # Right agent (we control)
        self.agent_other = self.env.possible_agents[0]  # Left agent (opponent)
        self.agent_obs = self.env.possible_agents[0]  # Observations come from agent 0
        self.left_agent = left_agent  # Optional fixed opponent

        # Preprocessing configuration
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Determine if we need to transpose observations
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Define observation space
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])
        else:
            obs_shape = (frame_stack * c, img_size[1], img_size[0])

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

        # Define action space
        orig_action_space = self.env.action_spaces[self.agent]
        self.action_space = self._to_gymnasium_space(orig_action_space)

        print("="*70)
        print("SharedObsUnityGymWrapper initialized with MODULAR REWARD SYSTEM")
        print("="*70)
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Action space: {self.action_space}")
        print(f"Frame stack: {frame_stack}")
        print(f"Grayscale: {grayscale}")
        print("="*70)

        # ============================================================
        # MODULAR REWARD SYSTEM
        # ============================================================
        self.reward_calc = RewardCalculator()
        self.ball_detector = BallDetector()
        self.paddle_detector = PaddleDetector()

        # Debug configuration
        self._debug_dir = "debug_frames"
        os.makedirs(self._debug_dir, exist_ok=True)
        self.debugger = DetectionDebugger(self._debug_dir)

        # Debug flags
        self._log_reward_components = False  # Log reward breakdown
        self._save_detection_debug = False  # Save detection overlays
        self._detection_debug_interval = 100  # Save every N frames

        # Tracking state
        self._episode_step_count = 0
        self._total_step_count = 0
        self._last_raw_obs = None
        self._last_rgb01 = None

        # Paddle position estimation (fallback when detection fails)
        self._estimated_paddle_x = 0.75
        self._estimated_paddle_y = 0.50

    def _to_gymnasium_space(self, space):
        """Convert Unity ML-Agents space to Gymnasium space."""
        nvec = getattr(space, "nvec", None)
        if nvec is not None:
            return spaces.MultiDiscrete(np.asarray(nvec, dtype=np.int64))
        if hasattr(space, "n") and not hasattr(space, "nvec"):
            return spaces.Discrete(int(space.n))
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        shape = getattr(space, "shape", None)
        if low is not None and high is not None and shape is not None:
            low_arr = np.array(low, dtype=np.float32)
            high_arr = np.array(high, dtype=np.float32)
            return spaces.Box(low=low_arr, high=high_arr, shape=shape, dtype=np.float32)
        if space.__class__.__name__ == "MultiBinary":
            n = getattr(space, "n", None)
            shape = getattr(space, "shape", None)
            return spaces.MultiBinary(n if n is not None else shape)
        raise TypeError(f"Unsupported space type: {type(space)}")

    def _preprocess(self, obs):
        """
        Preprocess observation for the policy network.

        Steps:
        1. Transpose if needed (C,H,W) -> (H,W,C)
        2. Crop UI elements
        3. Resize to target size
        4. Detect ball and paddle
        5. Convert to grayscale if needed

        Note: Unity ML-Agents outputs images already normalized to [0, 1]
        """
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        h, w = obs.shape[:2]

        # Crop UI elements (score, text)
        crop_top = int(h * 0.27)
        crop_bottom = int(h * 0.95)
        crop_left = int(w * 0.09)
        crop_right = int(w * 0.91)

        obs = obs[crop_top:crop_bottom, crop_left:crop_right]

        # Resize to target size
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # Store RGB for detection
        rgb01 = obs.copy()
        self._last_rgb01 = rgb01

        # Detect ball and paddle positions
        ball_pos = self.ball_detector.detect(rgb01)
        paddle_pos = self.paddle_detector.detect(rgb01)

        # Update reward system with ball position
        self.reward_calc.update_ball_position(ball_pos)

        # Debug visualization
        if self._save_detection_debug and self._total_step_count % self._detection_debug_interval == 0:
            self.debugger.save_detection_overlay(rgb01, ball_pos, paddle_pos)

        # Convert to grayscale if needed
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # Unity already outputs [0, 1], so just ensure float32
        obs = obs.astype(np.float32)

        return obs, ball_pos, paddle_pos

    def reset(self, *, seed=None, options=None):
        """Reset environment for new episode."""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        # Reset counters
        self._episode_step_count = 0

        print(f"\n{'='*70}")
        print(f"EPISODE RESET - Total steps: {self._total_step_count}")
        print(f"{'='*70}\n")

        # Reset Unity environment
        obs_dict = self.env.reset()
        self._last_raw_obs = obs_dict[self.agent_obs]['observation'][0]

        # Preprocess observation
        obs, _, _ = self._preprocess(self._last_raw_obs)

        # Fill frame stack
        for _ in range(self.frame_stack):
            self.frames.append(obs)

        # Reset reward calculator
        self.reward_calc.reset()

        # Reset left agent if applicable
        if self.left_agent is not None and hasattr(self.left_agent, "reset"):
            self.left_agent.reset()

        stacked_obs = np.concatenate(list(self.frames), axis=0)
        return stacked_obs, {}

    def step(self, action):
        """Execute one step in the environment."""
        # Prepare actions for both agents
        actions = {}

        # Left agent (opponent) action
        if self.left_agent is not None:
            left_action = self.left_agent.act(self._last_raw_obs)
            actions[self.agent_other] = left_action

        # Right agent (our) action
        actions[self.agent] = action

        # Step environment
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        # Store raw observation for left agent
        self._last_raw_obs = obs_dict[self.agent_obs]['observation'][0]

        # Preprocess observation
        obs, ball_pos, paddle_pos = self._preprocess(self._last_raw_obs)
        self.frames.append(obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)

        # Get paddle position (use detection or fallback to estimate)
        if paddle_pos is None:
            # Use estimated position
            paddle_pos = (self._estimated_paddle_x, self._estimated_paddle_y)
        else:
            # Update estimate with detected position
            self._estimated_paddle_x, self._estimated_paddle_y = paddle_pos

        # Update estimated position based on action (for fallback)
        self._update_paddle_estimate(action)

        # Extract game event (win/loss)
        event = rewards[self.agent] - rewards[self.agent_other]
        done = terminations[self.agent] or terminations[self.agent_other]

        # Increment counters
        self._episode_step_count += 1
        self._total_step_count += 1
        self.debugger.increment_frame()

        # ============================================================
        # CALCULATE REWARD USING MODULAR SYSTEM
        # ============================================================
        reward, components = self.reward_calc.calculate_reward(
            paddle_pos=paddle_pos,
            event_reward=event,
            current_frame=self._total_step_count,
            action=action,
            log_components=self._log_reward_components
        )

        # Print step info
        print(f"Step {self._episode_step_count:4d} | Total: {self._total_step_count:6d} | "
              f"Event: {event:+.1f} | Reward: {reward:+.3f} | Done: {done}")

        # Log reward components if enabled and present
        if self._log_reward_components and len(components) > 0:
            components_str = ", ".join([f"{k}={v:+.4f}" for k, v in components.items()])
            print(f"  └─ Components: {components_str}")

        # Print debug info on important events
        if abs(event) > 0:
            if event > 0:
                print(f"\n{'='*70}")
                print(f"🎯 WE SCORED! Episode step {self._episode_step_count}")
                print(f"   Rally duration: {self._episode_step_count} steps ({self._episode_step_count/60:.1f}s)")
                print(f"   Reward: {reward:+.3f}")
                debug_info = self.reward_calc.get_debug_info()
                print(f"   Debug: {debug_info}")
                print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print(f"❌ OPPONENT SCORED! Episode step {self._episode_step_count}")
                print(f"   Rally duration: {self._episode_step_count} steps ({self._episode_step_count/60:.1f}s)")
                print(f"   Reward: {reward:+.3f}")
                debug_info = self.reward_calc.get_debug_info()
                print(f"   Debug: {debug_info}")
                print(f"{'='*70}\n")

        # Reset left agent on episode end
        if done:
            if self.left_agent is not None and hasattr(self.left_agent, "reset"):
                self.left_agent.reset()

        return stacked_obs, reward, terminations[self.agent], False, infos[self.agent]

    def _update_paddle_estimate(self, action):
        """Update estimated paddle position based on action (fallback for detection failures)."""
        try:
            horiz_action = int(action[1]) if hasattr(action, "__len__") and len(action) > 1 else 0
            vert_action = int(action[0]) if hasattr(action, "__len__") and len(action) > 0 else 0

            move_speed = 0.03

            # Horizontal movement
            if horiz_action == 1:  # Right
                self._estimated_paddle_x = min(0.95, self._estimated_paddle_x + move_speed)
            elif horiz_action == 2:  # Left
                self._estimated_paddle_x = max(0.50, self._estimated_paddle_x - move_speed)

            # Vertical movement
            if vert_action == 1:  # Up
                self._estimated_paddle_y = max(0.05, self._estimated_paddle_y - move_speed)
            elif vert_action == 2:  # Down
                self._estimated_paddle_y = min(0.95, self._estimated_paddle_y + move_speed)
        except Exception:
            pass  # Keep current estimate if action parsing fails

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for processing stacked grayscale frames.
    Architecture optimized for pickleball observation space.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # Frame stack size

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output dimension
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Linear layer to features
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

