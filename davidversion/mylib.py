# mylib.py
import os

import numpy as np
import cv2
from collections import deque
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv


class SharedObsUnityGymWrapper(Env):
    def _to_gymnasium_space(self, space):
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

    def __init__(self, unity_env, frame_stack=4, img_size=(240, 120), grayscale=True, left_agent=None):
        self.env = UnityParallelEnv(unity_env)

        self._save_debug_frames = True
        self._debug_frame_count = 0
        self._debug_dir = "debug_frames"
        os.makedirs(self._debug_dir, exist_ok=True)
        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]  # agent to be controlled (right)
        self.agent_other = self.env.possible_agents[0]  # agent at opposite (left)
        self.agent_obs = self.env.possible_agents[0]  # obs is only available in agent 0, always 0
        self.left_agent = left_agent  # optional predefined controller for left agent
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
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
        orig_action_space = self.env.action_spaces[self.agent]
        self.action_space = self._to_gymnasium_space(orig_action_space)

        print("Wrapper initialized:")
        print(f"  - Observation space: {self.observation_space.shape}")
        print(f"  - Action space: {self.action_space}")

        # Simple reward configuration (Option A)
        self._use_simple_rewards = True
        self._step_penalty = 0.001
        self._win_bonus = 6.0
        self._loss_penalty = 6.0
        self._log_reward_components = False

        # Ball-based dense shaping to improve recovery behavior (right-side agent)
        # These small terms give the agent directional feedback before the point ends.
        self._use_ball_shaping = True
        self._k_side = 0.002     # penalty per step while ball remains on our (right) half
        self._k_away = 0.015     # extra penalty if we move left while ball is behind us near right wall
        self._k_toward = 0.005   # small reward if we move right while ball is behind us near right wall
        self._frames_on_right = 0
        self._last_ball = None

        # Detection debug config: save every few hundred steps until limit
        self._dbg_detect = False
        self._dbg_detect_max_steps = 1000
        self._dbg_detect_every = 0   # disabled; use shaping-triggered saves instead
        self._use_paddle_detection = True
        self._last_paddle = None

        # Shaping-triggered debug overlay saving
        self._dbg_shaping = True
        self._last_rgb01 = None

    def _save_frame(self, frame, tag):
        # frame is (1,H,W) after grayscale preprocess; convert to uint8 PNG
        img = (frame* 255).astype(np.uint8)[0]
        path = os.path.join(self._debug_dir, f"{tag}_{self._debug_frame_count:04d}.png")
        cv2.imwrite(path, img)

    # --- Detection debugging utilities ---
    def _should_dbg_detect(self):
        s = self._debug_frame_count
        return self._dbg_detect and (s < self._dbg_detect_max_steps) and (self._dbg_detect_every > 0) and (s % self._dbg_detect_every == 0)

    def _save_dbg_rgb01(self, name, rgb01):
        # rgb01 float [0,1] HxWx3
        path = os.path.join(self._debug_dir, f"{name}_{self._debug_frame_count:06d}.png")
        cv2.imwrite(path, (np.clip(rgb01, 0, 1) * 255).astype(np.uint8))

    def _save_dbg_mask(self, name, mask):
        # mask uint8 HxW
        path = os.path.join(self._debug_dir, f"{name}_{self._debug_frame_count:06d}.png")
        cv2.imwrite(path, mask)

    def _save_shaping_overlay(self, tags, horiz_action, deltas, total_delta):
        """
        Save an overlay image when shaping terms are applied.
        Includes ball/paddle markers and text annotations.
        Filename encodes which terms fired, e.g., shape_k_side+k_away_XXXXXX.png
        """
        try:
            base = getattr(self, "_last_rgb01", None)
            if not isinstance(base, np.ndarray):
                return
            overlay = (np.clip(base, 0.0, 1.0) * 255).astype(np.uint8)

            # Draw markers for ball and right paddle if available
            if getattr(self, "_last_ball", None) is not None:
                xb, yb = self._last_ball
                cv2.circle(overlay, (int(xb * overlay.shape[1]), int(yb * overlay.shape[0])), 5, (0, 0, 255), -1)
            if getattr(self, "_last_paddle", None) is not None:
                xp, yp = self._last_paddle
                cv2.circle(overlay, (int(xp * overlay.shape[1]), int(yp * overlay.shape[0])), 5, (255, 0, 0), -1)

            # Text info
            x_text, y_text = 5, 16
            cv2.putText(overlay, "SHAPING", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            y_text += 16
            if getattr(self, "_last_ball", None) is not None:
                xb, yb = self._last_ball
                cv2.putText(overlay, f"ball=({xb:.2f},{yb:.2f}) frames_on_right={self._frames_on_right}", (x_text, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
                y_text += 16

            act_str = "NONE"
            if horiz_action == 1:
                act_str = "RIGHT"
            elif horiz_action == 2:
                act_str = "LEFT"
            cv2.putText(overlay, f"horiz_action={act_str}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
            y_text += 16

            for k in ("k_side", "k_away", "k_toward"):
                if k in deltas:
                    val = deltas[k]
                    cv2.putText(overlay, f"{k}:{val:+.3f}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
                    y_text += 16
            cv2.putText(overlay, f"total:{total_delta:+.3f}", (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 180), 1, cv2.LINE_AA)

            tag_str = "+".join(tags) if tags else "none"
            self._save_dbg_rgb01(f"shape_{tag_str}", overlay.astype(np.float32) / 255.0)
        except Exception:
            # Avoid crashing training if saving fails
            pass

    def _detect_ball_xy(self, rgb_img01, save_debug: bool = False):
        """
        Detect the ball centroid (x_norm, y_norm) in an RGB image in [0,1] range.
        Optionally saves mask/overlay for debugging.
        """
        try:
            img8 = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
            # Yellow range; tune if needed based on render
            lower = np.array([20, 100, 110], dtype=np.uint8)
            upper = np.array([40, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            # Morph cleanup
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                if save_debug and self._should_dbg_detect():
                    self._save_dbg_mask("ball_mask", mask)
                    overlay = img8.copy()
                    cv2.putText(overlay, "BALL: none", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1, cv2.LINE_AA)
                    self._save_dbg_rgb01("ball_overlay", overlay.astype(np.float32) / 255.0)
                return None
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M.get("m00", 0) == 0:
                return None
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            h, w = mask.shape[:2]
            x_norm, y_norm = (cx / w), (cy / h)

            if save_debug and self._should_dbg_detect():
                self._save_dbg_mask("ball_mask", mask)
                overlay = img8.copy()
                cv2.drawContours(overlay, [c], -1, (0, 255, 255), 1)
                cv2.circle(overlay, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(overlay, f"BALL: ({x_norm:.2f},{y_norm:.2f})", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1, cv2.LINE_AA)
                self._save_dbg_rgb01("ball_overlay", overlay.astype(np.float32) / 255.0)

            return (x_norm, y_norm)
        except Exception:
            return None

    def _detect_right_paddle_xy(self, rgb_img01, save_debug: bool = False):
        """
        Detect the right paddle centroid (x_norm, y_norm) by searching only on the rightmost ROI.
        Optionally saves mask/overlay for debugging.
        """
        try:
            img8 = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = img8.shape[:2]
            x0 = int(0.60 * w)  # search on right 40%
            roi = img8[:, x0:w, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            # Orange-ish paddle (tune if needed)
            lower = np.array([5, 100, 80], dtype=np.uint8)
            upper = np.array([25, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                if save_debug and self._should_dbg_detect():
                    full_mask = np.zeros((h, w), np.uint8); full_mask[:, x0:w] = mask
                    self._save_dbg_mask("paddleR_mask", full_mask)
                    overlay = img8.copy()
                    cv2.putText(overlay, "PADDLE R: none", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
                    self._save_dbg_rgb01("paddleR_overlay", overlay.astype(np.float32) / 255.0)
                return None
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M.get("m00", 0) == 0:
                return None
            cx = M["m10"] / M["m00"] + x0  # shift to full image coords
            cy = M["m01"] / M["m00"]
            x_norm, y_norm = (cx / w), (cy / h)

            if save_debug and self._should_dbg_detect():
                full_mask = np.zeros((h, w), np.uint8); full_mask[:, x0:w] = mask
                self._save_dbg_mask("paddleR_mask", full_mask)
                overlay = img8.copy()
                cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                cv2.putText(overlay, f"PADDLE R: ({x_norm:.2f},{y_norm:.2f})", (5, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1, cv2.LINE_AA)
                self._save_dbg_rgb01("paddleR_overlay", overlay.astype(np.float32) / 255.0)

            return (x_norm, y_norm)
        except Exception:
            return None

    def _preprocess(self, obs):
        """
        Preprocess observation to match training format.

        CRITICAL FIX: Unity ML-Agents outputs images already normalized to [0, 1].
        We should NOT divide by 255 again!
        """
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)  # (H, W, C)

        # The court should be most of the frame
        # Try a MINIMAL crop - just remove obvious UI areas
        h, w = obs.shape[:2]

        # Remove only top UI bar (score) and bottom text
        # Start with conservative values
        crop_top = int(h * 0.27)  # Remove top ~30%
        crop_bottom = int(h * 0.95)  # Keep to 95%
        crop_left = int(w * 0.09)  # Remove left ~11%
        crop_right = int(w * 0.91)  # Keep to 89%

        obs = obs[crop_top:crop_bottom, crop_left:crop_right]
        if self._debug_frame_count % 10000 == 0:
            cv2.imwrite(f"{self._debug_dir}/after_crop_{self._debug_frame_count:04d}.png",
                        (obs * 255).astype(np.uint8))
        # Then resize
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        if self._debug_frame_count % 10000 == 0:
            cv2.imwrite(f"{self._debug_dir}/after_crop_R_{self._debug_frame_count:04d}.png",
                        (obs * 255).astype(np.uint8))

        # Detect ball/paddle on RGB before converting to grayscale
        rgb01 = obs.copy()
        self._last_rgb01 = rgb01
        save_dbg = self._should_dbg_detect()
        self._last_ball = self._detect_ball_xy(rgb01, save_debug=save_dbg)
        if self._use_paddle_detection:
            self._last_paddle = self._detect_right_paddle_xy(rgb01, save_debug=save_dbg)
        # Combined overlay with markers
        if save_dbg:
            overlay = (np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8)
            if self._last_ball is not None:
                xb, yb = self._last_ball
                cv2.circle(overlay, (int(xb * overlay.shape[1]), int(yb * overlay.shape[0])), 5, (0, 0, 255), -1)
            if getattr(self, "_last_paddle", None) is not None:
                xp, yp = self._last_paddle
                cv2.circle(overlay, (int(xp * overlay.shape[1]), int(yp * overlay.shape[0])), 5, (255, 0, 0), -1)
            self._save_dbg_rgb01("det_overlay", overlay.astype(np.float32) / 255.0)

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        else:
            obs = obs.transpose(2, 0, 1)  # (C, H, W)

        # ✅ FIXED: Don't divide by 255 - Unity already outputs [0, 1]
        obs = obs.astype(np.float32)

        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()

        # DEBUG: Inspect the full observation dict on first reset
        if not hasattr(self, "_debugged_obs_space"):
            self._debugged_obs_space = True
            print("=== Unity obs dict summary on reset ===")
            for agent_id, od in obs_dict.items():
                try:
                    if isinstance(od, dict):
                        for k, v in od.items():
                            try:
                                arr = np.asarray(v)
                                if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                                    vmin = float(np.min(arr))
                                    vmax = float(np.max(arr))
                                    print(f"  agent {agent_id} key '{k}': shape={arr.shape} dtype={arr.dtype} min={vmin:.4f} max={vmax:.4f}")
                                else:
                                    print(f"  agent {agent_id} key '{k}': type={type(v).__name__} shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
                            except Exception as e:
                                print(f"  agent {agent_id} key '{k}': error inspecting value: {e}")
                    else:
                        try:
                            arr = np.asarray(od)
                            if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
                                vmin = float(np.min(arr))
                                vmax = float(np.max(arr))
                                print(f"  agent {agent_id}: shape={arr.shape} dtype={arr.dtype} min={vmin:.4f} max={vmax:.4f}")
                            else:
                                print(f"  agent {agent_id}: type={type(od).__name__} shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
                        except Exception as e:
                            print(f"  agent {agent_id}: error inspecting obs: {e}")
                except Exception as e:
                    print(f"  agent {agent_id}: error inspecting obs: {e}")

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        if self._save_debug_frames and self._debug_frame_count < 100:
            self._save_frame(obs, "reset")

        # Debug first observation
        print(f"Reset: obs range = [{obs.min():.3f}, {obs.max():.3f}]")

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        if self.left_agent is not None and hasattr(self.left_agent, "reset"):
            self.left_agent.reset()

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def step(self, action):
        # Compute left agent action if provided
        actions = {}
        if self.left_agent is not None:
            # Provide stacked frames to opponent; resize to expected if available
            if len(self.frames) > 0:
                current_stack = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)
            else:
                current_stack = None
            left_input = current_stack
            try:
                if left_input is not None and hasattr(self.left_agent, "expected_shape"):
                    ec, eh, ew = self.left_agent.expected_shape
                    # Align channels (stack size)
                    if left_input.shape[0] != ec:
                        if left_input.shape[0] > ec:
                            left_input = left_input[-ec:, :, :]
                        else:
                            pad = np.repeat(left_input[-1:, :, :], ec - left_input.shape[0], axis=0)
                            left_input = np.concatenate([left_input, pad], axis=0)
                    # Align spatial dimensions
                    if left_input.shape[1] != eh or left_input.shape[2] != ew:
                        hwc = np.transpose(left_input, (1, 2, 0))
                        resized = cv2.resize(hwc, (ew, eh), interpolation=cv2.INTER_AREA)
                        left_input = np.transpose(resized.astype(np.float32), (2, 0, 1))
            except Exception as e:
                # Fallback to latest frame if any error occurs
                left_input = self.frames[-1] if len(self.frames) > 0 else None
            left_action = self.left_agent.act(left_input)
            actions[self.agent_other] = left_action

        # Right agent action from the learning policy
        actions[self.agent] = action

        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        if self._debug_frame_count % 1000 == 0:
            # Check if ball exists in observation dict
            print(f"Step {self._debug_frame_count}: obs_dict keys = {obs_dict.keys()}")
            print(f"  Agent obs shape: {obs_dict[self.agent_obs]['observation'][0].shape}")
            print(
                f"  Agent obs range: [{obs_dict[self.agent_obs]['observation'][0].min():.3f}, {obs_dict[self.agent_obs]['observation'][0].max():.3f}]")

        # Debug rewards
        if (rewards[self.agent] != 0 or rewards[self.agent_other] != 0):
            print(f"Rewards: agent={rewards[self.agent]:.2f}, opponent={rewards[self.agent_other]:.2f}")

        # Simple episodic reward scheme (Option A)
        event = rewards[self.agent] - rewards[self.agent_other]
        done = terminations[self.agent] or terminations[self.agent_other]

        reward = -self._step_penalty if self._step_penalty > 0 else 0.0

        if self._use_simple_rewards and done:
            if event > 0:
                reward += self._win_bonus
            elif event < 0:
                reward -= self._loss_penalty
            else:
                # Treat timeouts/no-contact (no scorer) as a loss to discourage idling
                reward -= self._loss_penalty
        elif not self._use_simple_rewards:
            # Fallback to original difference-style shaping if desired
            reward = event - (self._step_penalty if self._step_penalty > 0 else 0.0)

        # Dense ball-based shaping to encourage recovery on our (right) half
        if getattr(self, "_use_ball_shaping", False) and self._last_ball is not None:
            x_b, _ = self._last_ball
            applied = []
            deltas = {}
            total_delta = 0.0
            # Penalize time while the ball stays on our half
            if x_b > 0.5:
                self._frames_on_right = min(self._frames_on_right + 1, 10000)
                reward -= self._k_side
                applied.append("k_side")
                deltas["k_side"] = -float(self._k_side)
                total_delta += -float(self._k_side)
            else:
                self._frames_on_right = 0
            # If ball is behind us near the right wall, discourage moving left and reward moving right
            try:
                horiz_action = int(action[1]) if hasattr(action, "__len__") else 0  # 0 none, 1 right, 2 left
            except Exception:
                horiz_action = 0
            if x_b > 0.8:
                if horiz_action == 2:
                    reward -= self._k_away
                    applied.append("k_away")
                    deltas["k_away"] = -float(self._k_away)
                    total_delta += -float(self._k_away)
                elif horiz_action == 1:
                    reward += self._k_toward
                    applied.append("k_toward")
                    deltas["k_toward"] = float(self._k_toward)
                    total_delta += float(self._k_toward)
            # Save overlay only when shaping is applied this frame
            if getattr(self, "_dbg_shaping", False) and len(applied) > 0:
                ordered_tags = [t for t in ("k_side", "k_away", "k_toward") if t in applied]
                self._save_shaping_overlay(ordered_tags, horiz_action, deltas, total_delta)

        if self._log_reward_components:
            print(f"Reward components: event={event:.3f}, done={done}, final={reward:.3f}")

        if done:
            print("Episode terminated. Resetting left agent if applicable.")
            if self.left_agent is not None and hasattr(self.left_agent, "reset"):
                self.left_agent.reset()

        # Increment frame counter at end to align indices across all saves
        self._debug_frame_count += 1
        return stacked_obs, reward, terminations[self.agent], False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for processing stacked grayscale frames.
    """

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
