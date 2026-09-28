# mylib.py V2
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

        self._save_debug_frames = False
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

        # ============================================================
        # NEW REWARD SYSTEM - Context-Aware Strategic Approach
        # ============================================================
        # Based on design doc: new_reward_system.md
        # Core philosophy:
        # 1. WIN POINTS (primary signal +10/-10)
        # 2. Reward ball contact and successful returns
        # 3. Strategic positioning based on game state
        # 4. Purposeful movement only (no random action spam)
        # 5. NO unfair time-based penalties
        # ============================================================

        # 1. MAIN SIGNALS (unchanged)
        self._win_bonus = 10.0
        self._loss_penalty = 10.0

        # 2. BALL CONTACT & RETURN REWARDS (NEW - core actions)
        self._use_ball_contact_reward = True
        self._ball_contact_reward = 0.5       # Reward for hitting the ball
        self._return_success_bonus = 0.3      # Reward for getting ball back to opponent
        self._frames_since_contact = 999      # Track recent contact
        self._contact_memory_frames = 5       # How long to remember contact

        # 3. STRATEGIC POSITIONING (REPLACES ball tracking)
        self._use_strategic_positioning = True
        self._ready_position_reward_max = 0.01    # When ball on opponent side
        self._intercept_reward_max = 0.02         # When ball on our side
        self._depth_quality_bonus = 0.01          # For good court depth
        self._optimal_ready_x = 0.70
        self._optimal_ready_y = 0.50
        self._ready_zone_radius = 0.25
        self._optimal_depth_min = 0.65
        self._optimal_depth_max = 0.88
        self._intercept_zone_radius = 0.3

        # 4. PURPOSEFUL MOVEMENT (REPLACES random movement reward)
        self._use_purposeful_movement = True
        self._purposeful_movement_coef = 0.005
        self._last_position_quality = 0.0

        # 5. WALL SAFETY (graduated penalty system)
        self._use_wall_penalty = True
        self._wall_warning_threshold = 0.88   # Soft warning zone
        self._wall_danger_threshold = 0.92    # Danger zone
        self._wall_warning_penalty = 0.01
        self._wall_danger_penalty_max = 0.05

        # 6. BALL SIDE TRACKING (informational only, NO time penalty)
        # The real penalty comes from losing the point naturally
        self._ball_on_our_side_threshold = 0.5
        self._frames_ball_on_our_side = 0
        self._max_frames_one_side = 300  # 5 seconds at 60fps (for info only)
        self._use_urgency_nudge = False      # Optional gentle nudge after 4s
        self._urgency_nudge_bonus = 0.005
        self._urgency_threshold = 240        # 4 seconds (80% of limit)

        # REMOVED: Ball stagnation penalty (was unfair)
        # REMOVED: Movement encouragement (encouraged jittering)

        # DISABLED: Static position rewards (these caused camping)
        self._use_position_reward = False
        self._use_front_penalty = False
        self._use_forward_exposed_penalty = False
        self._k_good_position = 0.0
        self._front_penalty = 0.0
        self._forward_exposed_penalty = 0.0
        self._optimal_x_min = 0.60
        self._optimal_x_max = 0.85
        self._front_line_threshold = 0.55

        # Detection (for position tracking and debugging)
        self._use_paddle_detection = True
        self._last_paddle = None
        self._estimated_paddle_x = 0.75  # Fallback: assume agent starts center-right
        self._last_ball = None
        self._last_rgb01 = None

        # Debugging controls
        self._log_reward_components = False
        self._dbg_detect = False
        self._dbg_detect_max_steps = 1000
        self._dbg_detect_every = 0
        self._dbg_position = False
        self._dbg_position_every = 100
        self._dbg_position_max_steps = 10000

        # Step counter for timing analysis
        self._episode_step_count = 0
        self._total_step_count = 0

    def _save_frame(self, frame, tag):
        # frame is (1,H,W) after grayscale preprocess; convert to uint8 PNG
        img = (frame* 255).astype(np.uint8)[0]
        path = os.path.join(self._debug_dir, f"{tag}_{self._debug_frame_count:04d}.png")
        cv2.imwrite(path, img)

    def _save_position_debug(self):
        """
        Save a debug overlay showing paddle position relative to optimal zone.
        Shows good positioning (green) vs. bad positioning (red/yellow).
        """
        try:
            base = getattr(self, "_last_rgb01", None)
            if not isinstance(base, np.ndarray):
                return

            overlay = (np.clip(base, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = overlay.shape[:2]

            # Draw zone boundaries with clear colors
            front_line_px = int(self._front_line_threshold * w)  # 0.55 - too far forward
            optimal_min_px = int(self._optimal_x_min * w)        # 0.65 - optimal zone start
            optimal_max_px = int(self._optimal_x_max * w)        # 0.80 - optimal zone end
            wall_danger_px = int(self._wall_danger_threshold * w)  # 0.90 - wall danger

            # Draw vertical lines for all zones
            cv2.line(overlay, (front_line_px, 0), (front_line_px, h), (255, 0, 255), 3)  # front line (magenta)
            cv2.line(overlay, (optimal_min_px, 0), (optimal_min_px, h), (0, 255, 0), 3)  # optimal start (green)
            cv2.line(overlay, (optimal_max_px, 0), (optimal_max_px, h), (0, 255, 0), 3)  # optimal end (green)
            cv2.line(overlay, (wall_danger_px, 0), (wall_danger_px, h), (0, 0, 255), 3)  # wall danger (red)

            # Semi-transparent zone highlights
            zone_overlay = overlay.copy()
            # Optimal zone (green tint)
            cv2.rectangle(zone_overlay, (optimal_min_px, 0), (optimal_max_px, h), (0, 255, 0), -1)
            # Front penalty zone (magenta tint)
            cv2.rectangle(zone_overlay, (0, 0), (front_line_px, h), (255, 0, 255), -1)
            # Wall danger zone (red tint)
            cv2.rectangle(zone_overlay, (wall_danger_px, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.85, zone_overlay, 0.15, 0, overlay)

            # Draw paddle position and ball
            x_p = None
            y_p = h // 2  # Default y position for estimated paddle
            is_estimated = False
            status = "UNKNOWN"
            color = (200, 200, 200)
            reward_text = ""

            if getattr(self, "_last_paddle", None) is not None:
                x_p, y_p = self._last_paddle
            else:
                # Use estimated position as fallback
                x_p = getattr(self, "_estimated_paddle_x", 0.75)
                is_estimated = True

            if x_p is not None:
                paddle_px = int(x_p * w)
                paddle_py = int(y_p * h)

                # Determine status based on new thresholds
                if x_p < self._front_line_threshold:
                    status = "TOO_FAR_FORWARD"
                    color = (255, 0, 255)  # magenta
                    reward_text = f"(-{self._front_penalty})"
                elif self._optimal_x_min <= x_p <= self._optimal_x_max:
                    status = "OPTIMAL ZONE"
                    color = (0, 255, 0)  # green
                    reward_text = f"(+{self._k_good_position})"
                elif x_p > self._wall_danger_threshold:
                    status = "WALL DANGER"
                    color = (0, 0, 255)  # red
                    reward_text = f"(-{self._wall_penalty})"
                elif x_p > self._optimal_x_max:
                    status = "BETWEEN ZONES"
                    color = (255, 255, 0)  # yellow
                    reward_text = "(0.0)"
                else:
                    status = "SUBOPTIMAL"
                    color = (255, 255, 0)  # yellow
                    reward_text = "(0.0)"

                # Draw paddle with status color
                if is_estimated:
                    # Dashed/outlined circle for estimated position
                    cv2.circle(overlay, (paddle_px, paddle_py), 8, color, 2)
                    cv2.circle(overlay, (paddle_px, paddle_py), 12, (255, 255, 0), 2)
                    cv2.putText(overlay, "EST", (paddle_px - 15, paddle_py - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
                else:
                    # Solid circle for detected position
                    cv2.circle(overlay, (paddle_px, paddle_py), 8, color, -1)
                    cv2.circle(overlay, (paddle_px, paddle_py), 10, (255, 255, 255), 2)

            # Draw ball
            if getattr(self, "_last_ball", None) is not None:
                x_b, y_b = self._last_ball
                ball_px = int(x_b * w)
                ball_py = int(y_b * h)
                cv2.circle(overlay, (ball_px, ball_py), 6, (0, 255, 255), -1)
                cv2.circle(overlay, (ball_px, ball_py), 8, (255, 255, 255), 2)

            # Add text annotations with legend
            y_text = 20
            cv2.putText(overlay, f"POSITION DEBUG - Step {self._debug_frame_count}",
                       (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y_text += 30

            if x_p is not None:
                cv2.putText(overlay, f"Paddle X: {x_p:.3f} - {status} {reward_text}",
                           (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                y_text += 30

            # Zone legend
            cv2.putText(overlay, "ZONES:",
                       (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            y_text += 20
            cv2.putText(overlay, f"Front Penalty: x < {self._front_line_threshold:.2f} (-{self._front_penalty})",
                       (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y_text += 18
            cv2.putText(overlay, f"Optimal Zone: [{self._optimal_x_min:.2f}, {self._optimal_x_max:.2f}] (+{self._k_good_position})",
                       (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            y_text += 18
            cv2.putText(overlay, f"Wall Danger: x > {self._wall_danger_threshold:.2f} (-{self._wall_penalty})",
                       (5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            # Save with status in filename
            tag = f"position_{status.lower()}"
            path = os.path.join(self._debug_dir, f"{tag}_{self._debug_frame_count:06d}.png")
            cv2.imwrite(path, overlay)

        except Exception as e:
            # Don't crash training if debug saving fails
            pass

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

            for k in ("k_side", "k_away", "k_toward", "k_right_gated", "k_wall", "k_pre_oob"):
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

    def _detect_ball_contact(self, prev_ball, curr_ball, paddle_pos):
        """
        Detect if agent successfully hit the ball this frame.

        Conditions:
        1. Ball was on our side and moving toward us
        2. Ball suddenly changes direction toward opponent
        3. Paddle was close to ball when this happened
        """
        if prev_ball is None or curr_ball is None or paddle_pos is None:
            return False

        prev_x, prev_y = prev_ball
        curr_x, curr_y = curr_ball
        paddle_x, paddle_y = paddle_pos

        # Ball must have been on our side
        if prev_x <= 0.5:
            return False

        # Ball must now be moving toward opponent (x decreasing)
        ball_moving_left = curr_x < prev_x
        if not ball_moving_left:
            return False

        # Paddle must be close to ball
        distance_to_ball = np.sqrt((paddle_x - curr_x)**2 + (paddle_y - curr_y)**2)
        if distance_to_ball > 0.15:
            return False

        # All conditions met - successful hit!
        return True

    def _calculate_position_quality(self, paddle_pos, ball_state):
        """
        Calculate position quality score [0, 1] based on strategic positioning.

        Args:
            paddle_pos: (x, y) normalized paddle position
            ball_state: dict with 'x', 'y', 'on_opponent_side'

        Returns:
            quality score between 0 and 1
        """
        if paddle_pos is None:
            return 0.0

        paddle_x, paddle_y = paddle_pos
        ball_x = ball_state.get('x', 0.5)
        ball_y = ball_state.get('y', 0.5)
        on_opponent_side = ball_state.get('on_opponent_side', True)

        if on_opponent_side:
            # Quality = proximity to ready position
            ready_dist = np.sqrt((paddle_x - self._optimal_ready_x)**2 +
                               (paddle_y - self._optimal_ready_y)**2)
            quality = max(0.0, 1.0 - ready_dist / self._ready_zone_radius)
        else:
            # Quality = ability to intercept ball trajectory
            # Simple version: vertical alignment + depth quality
            vertical_dist = abs(paddle_y - ball_y)
            vertical_quality = max(0.0, 1.0 - vertical_dist / self._intercept_zone_radius)

            # Depth quality
            in_optimal_depth = (self._optimal_depth_min <= paddle_x <= self._optimal_depth_max)
            depth_quality = 1.0 if in_optimal_depth else 0.5

            quality = vertical_quality * depth_quality

        return quality

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
        Uses multiple color ranges to improve detection reliability.
        """
        try:
            img8 = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = img8.shape[:2]
            x0 = int(0.50 * w)  # search on right 50% (was 60%, expand search area)
            roi = img8[:, x0:w, :]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            # Try multiple color ranges for paddle detection (orange, red-orange, yellow-orange)
            lower1 = np.array([0, 80, 60], dtype=np.uint8)    # Red-orange
            upper1 = np.array([15, 255, 255], dtype=np.uint8)
            lower2 = np.array([10, 80, 60], dtype=np.uint8)   # Orange
            upper2 = np.array([30, 255, 255], dtype=np.uint8)

            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
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
        prev_ball = getattr(self, "_last_ball", None)
        self._last_ball = self._detect_ball_xy(rgb01, save_debug=save_dbg)
        self._prev_ball = prev_ball
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

        # Reset episode step counter
        self._episode_step_count = 0
        print(f"\n{'='*60}")
        print(f"EPISODE RESET - Total steps so far: {self._total_step_count}")
        print(f"{'='*60}\n")

        obs_dict = self.env.reset()

        # Store raw observation for left agent (before preprocessing)
        self._last_raw_obs = obs_dict[self.agent_obs]['observation'][0]

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
            # Get raw RGB observation from Unity for left agent
            # Left agent will do its own preprocessing (matching TeamX approach)
            # We need to get this BEFORE calling env.step, so we use the last observation
            # Actually, we should pass the observation AFTER env.step to be consistent
            # For now, pass None and let left agent handle it, or we can store raw obs
            # Better approach: store the raw observation before preprocessing
            left_input = getattr(self, '_last_raw_obs', None)
            left_action = self.left_agent.act(left_input)
            actions[self.agent_other] = left_action

        # Right agent action from the learning policy
        actions[self.agent] = action

        obs_dict, rewards, terminations, infos = self.env.step(actions)

        # Store raw observation for left agent (before preprocessing)
        self._last_raw_obs = obs_dict[self.agent_obs]['observation'][0]

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        if self._debug_frame_count % 1000 == 0:
            # Check if ball exists in observation dict
            print(f"Step {self._debug_frame_count}: obs_dict keys = {obs_dict.keys()}")
            print(f"  Agent obs shape: {obs_dict[self.agent_obs]['observation'][0].shape}")
            print(
                f"  Agent obs range: [{obs_dict[self.agent_obs]['observation'][0].min():.3f}, {obs_dict[self.agent_obs]['observation'][0].max():.3f}]")

        # Extract game events
        event = rewards[self.agent] - rewards[self.agent_other]
        done = terminations[self.agent] or terminations[self.agent_other]

        # Increment step counters
        self._episode_step_count += 1
        self._total_step_count += 1

        # Print every step with event information
        print(f"Step {self._episode_step_count:4d} | Total: {self._total_step_count:6d} | "
              f"Event: {event:+.1f} | Agent_R: {rewards[self.agent]:+.1f} | Agent_L: {rewards[self.agent_other]:+.1f} | "
              f"Done: {done}")

        # ============================================================
        # NEW REWARD CALCULATION - Context-Aware Strategic System
        # ============================================================
        reward = 0.0
        reward_components = {}  # For debugging

        # 1. MAIN REWARD: Points scored/lost (strongest signal)
        if event != 0:
            if event > 0:
                reward += self._win_bonus
                reward_components['win_point'] = self._win_bonus
                self._frames_ball_on_our_side = 0
                print(f"\n{'='*70}")
                print(f"🎯 WE SCORED! Episode step {self._episode_step_count}")
                print(f"   Rally duration: ~{self._episode_step_count} steps")
                print(f"   Estimated rally time: ~{self._episode_step_count / 60:.1f}s (60 fps)")
                print(f"   Reward: +{self._win_bonus}")
                print(f"{'='*70}\n")
            elif event < 0:
                reward -= self._loss_penalty
                reward_components['lose_point'] = -self._loss_penalty
                self._frames_ball_on_our_side = 0
                print(f"\n{'='*70}")
                print(f"❌ OPPONENT SCORED! Episode step {self._episode_step_count}")
                print(f"   Rally duration: ~{self._episode_step_count} steps")
                print(f"   Estimated rally time: ~{self._episode_step_count / 60:.1f}s (60 fps)")
                print(f"   Reward: -{self._loss_penalty}")
                print(f"{'='*70}\n")

        # Get paddle position
        paddle_x, paddle_y = None, None
        if self._last_paddle is not None:
            paddle_x, paddle_y = self._last_paddle
        else:
            paddle_x = self._estimated_paddle_x
            paddle_y = 0.5  # Assume center if unknown

        # Update estimated position based on action (for fallback)
        try:
            horiz_action = int(action[1]) if hasattr(action, "__len__") else 0
            vert_action = int(action[0]) if hasattr(action, "__len__") else 0
            move_speed = 0.03
            if horiz_action == 1:  # Moving right
                self._estimated_paddle_x = min(0.95, self._estimated_paddle_x + move_speed)
            elif horiz_action == 2:  # Moving left
                self._estimated_paddle_x = max(0.50, self._estimated_paddle_x - move_speed)
        except Exception:
            pass

        # 2. BALL CONTACT DETECTION & REWARD (NEW - highest priority shape reward)
        if self._use_ball_contact_reward and self._last_ball is not None:
            prev_ball = getattr(self, '_prev_ball', None)
            paddle_pos = (paddle_x, paddle_y) if paddle_x is not None else None

            if self._detect_ball_contact(prev_ball, self._last_ball, paddle_pos):
                reward += self._ball_contact_reward
                reward_components['ball_contact'] = self._ball_contact_reward
                self._frames_since_contact = 0
                if self._log_reward_components:
                    print(f"  [CONTACT] Successfully hit the ball! +{self._ball_contact_reward}")
            else:
                self._frames_since_contact += 1

        # 3. RETURN SUCCESS BONUS (NEW - reward getting ball back)
        if self._last_ball is not None:
            prev_ball = getattr(self, '_prev_ball', None)
            if prev_ball is not None:
                prev_x, _ = prev_ball
                curr_x, _ = self._last_ball
                # Ball crossed from our side to opponent's side
                if prev_x > 0.5 and curr_x <= 0.5:
                    # Check if we recently hit it
                    if self._frames_since_contact <= self._contact_memory_frames:
                        reward += self._return_success_bonus
                        reward_components['return_success'] = self._return_success_bonus
                        if self._log_reward_components:
                            print(f"  [RETURN] Successfully returned ball! +{self._return_success_bonus}")

        # 4. STRATEGIC POSITIONING REWARD (NEW - replaces ball tracking)
        if self._use_strategic_positioning and self._last_ball is not None and paddle_x is not None:
            ball_x, ball_y = self._last_ball
            ball_on_opponent_side = (ball_x < 0.5)

            if ball_on_opponent_side:
                # STATE 1: Ball on opponent's side - reward ready position
                ready_dist = np.sqrt((paddle_x - self._optimal_ready_x)**2 +
                                   (paddle_y - self._optimal_ready_y)**2)
                if ready_dist < self._ready_zone_radius:
                    ready_reward = self._ready_position_reward_max * (1.0 - ready_dist / self._ready_zone_radius)
                    reward += ready_reward
                    reward_components['ready_position'] = ready_reward
            else:
                # STATE 2: Ball on our side - reward intercept positioning
                vertical_dist = abs(paddle_y - ball_y)
                if vertical_dist < self._intercept_zone_radius:
                    intercept_reward = self._intercept_reward_max * (1.0 - vertical_dist / self._intercept_zone_radius)
                    reward += intercept_reward
                    reward_components['intercept_position'] = intercept_reward

                # Depth quality bonus
                if self._optimal_depth_min <= paddle_x <= self._optimal_depth_max:
                    reward += self._depth_quality_bonus
                    reward_components['depth_quality'] = self._depth_quality_bonus

        # 5. PURPOSEFUL MOVEMENT REWARD (NEW - replaces random movement)
        if self._use_purposeful_movement and self._last_ball is not None and paddle_x is not None:
            ball_x, ball_y = self._last_ball
            ball_state = {
                'x': ball_x,
                'y': ball_y,
                'on_opponent_side': (ball_x < 0.5)
            }

            # Calculate position quality before and after
            current_quality = self._calculate_position_quality((paddle_x, paddle_y), ball_state)
            improvement = current_quality - self._last_position_quality

            if improvement > 0.01:  # Meaningful improvement
                movement_reward = self._purposeful_movement_coef * improvement
                reward += movement_reward
                reward_components['purposeful_movement'] = movement_reward
            elif improvement < -0.01:  # Made position worse
                movement_penalty = self._purposeful_movement_coef * abs(improvement)
                reward -= movement_penalty
                reward_components['bad_movement'] = -movement_penalty

            self._last_position_quality = current_quality

        # 6. WALL SAFETY PENALTY (graduated system)
        if paddle_x is not None and self._use_wall_penalty:
            if self._wall_warning_threshold < paddle_x <= self._wall_danger_threshold:
                # Soft warning zone
                reward -= self._wall_warning_penalty
                reward_components['wall_warning'] = -self._wall_warning_penalty
            elif paddle_x > self._wall_danger_threshold:
                # Danger zone with scaling
                scale = (paddle_x - self._wall_danger_threshold) / (1.0 - self._wall_danger_threshold)
                wall_penalty = self._wall_danger_penalty_max * scale
                reward -= wall_penalty
                reward_components['wall_danger'] = -wall_penalty
                if self._log_reward_components:
                    print(f"  [WALL] x={paddle_x:.3f} in danger zone")

        # 7. BALL SIDE TRACKING (informational only, optional urgency nudge)
        if self._last_ball is not None:
            ball_x, _ = self._last_ball
            if ball_x > self._ball_on_our_side_threshold:
                self._frames_ball_on_our_side += 1
                # Optional gentle urgency nudge after 4 seconds
                if self._use_urgency_nudge and self._frames_ball_on_our_side > self._urgency_threshold:
                    # Small bonus for moving toward ball
                    if self._last_ball is not None and paddle_x is not None:
                        moving_toward_ball = abs(paddle_x - ball_x) < 0.3  # Simple check
                        if moving_toward_ball:
                            reward += self._urgency_nudge_bonus
                            reward_components['urgency_nudge'] = self._urgency_nudge_bonus
            else:
                self._frames_ball_on_our_side = 0


        # Periodic position debugging
        if self._dbg_position and self._debug_frame_count < self._dbg_position_max_steps:
            if self._debug_frame_count % self._dbg_position_every == 0:
                self._save_position_debug()

        # Log reward breakdown if enabled
        if self._log_reward_components and len(reward_components) > 0:
            components_str = ", ".join([f"{k}={v:+.4f}" for k, v in reward_components.items()])
            print(f"  [Reward Breakdown] {components_str} | Total={reward:.4f}")

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
