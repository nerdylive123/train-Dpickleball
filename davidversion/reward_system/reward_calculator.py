"""
Unified Reward Calculator
Integrates all reward components with proper weighting and validation.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .ball_tracker import BallTracker
from .contact_detector import ContactDetector
from .position_evaluator import PositionEvaluator
from .oob_detector import OOBDetector


class RewardCalculator:
    """
    Central reward calculation with all fixes applied.
    Addresses all major issues from weakness analysis.
    """

    def __init__(self):
        # Main signals (unchanged)
        self.win_bonus = 10.0
        self.loss_penalty = 10.0

        # Contact and return rewards
        self.ball_contact_reward = 0.5
        self.return_success_bonus = 0.3
        self.return_validation_frames = 5  # Stricter than before

        # Strategic positioning
        self.ready_position_reward_max = 0.01
        self.intercept_reward_max = 0.02
        self.depth_quality_bonus = 0.01

        # Purposeful movement
        self.purposeful_movement_coef = 0.005
        self.min_meaningful_improvement = 0.01

        # Wall safety
        self.wall_warning_threshold = 0.88
        self.wall_danger_threshold = 0.92
        self.wall_warning_penalty = 0.01
        self.wall_danger_penalty_max = 0.05

        # OOB handling
        self.oob_let_go_bonus = 0.2
        self.oob_hit_penalty = 0.3

        # Components
        self.ball_tracker = BallTracker(history_length=10)
        self.contact_detector = ContactDetector()
        self.position_evaluator = PositionEvaluator()
        self.oob_detector = OOBDetector()

        # State tracking for return validation (Fix #2)
        self.last_return_frame = -999
        self.contact_validation_window = []  # Track contact events for validation

    def calculate_reward(self,
                        paddle_pos: Optional[Tuple[float, float]],
                        event_reward: float,
                        current_frame: int,
                        action: Optional[np.ndarray] = None,
                        log_components: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total reward for current frame.

        Args:
            paddle_pos: (x, y) normalized paddle position
            event_reward: Win/loss event reward (±10 or 0)
            current_frame: Current frame number
            action: Agent's action (for movement analysis)
            log_components: Whether to return component breakdown

        Returns:
            (total_reward, components_dict) tuple
        """
        reward = 0.0
        components = {}

        # 1. MAIN REWARD: Points scored/lost
        # CRITICAL: When point is scored/lost, return ONLY that signal (no dilution)
        if event_reward != 0:
            reward = event_reward  # Use assignment, not +=
            if event_reward > 0:
                components['win_point'] = event_reward
            else:
                components['lose_point'] = event_reward

            # Return immediately - no shaping rewards when point decided
            if log_components:
                return reward, components
            return reward, {}

        # Only calculate shaping rewards if no point was scored
        # This ensures clean ±10 signals for wins/losses

        # 2. BALL CONTACT DETECTION (with robust checks)
        contact_detected = self.contact_detector.detect_contact(
            self.ball_tracker, paddle_pos, current_frame
        )

        if contact_detected:
            reward += self.ball_contact_reward
            components['ball_contact'] = self.ball_contact_reward

            # Track contact for return validation
            self.contact_validation_window.append({
                'frame': current_frame,
                'ball_pos': self.ball_tracker.get_current_position(),
                'validated': False
            })

        # Update contact frame counter
        self.contact_detector.update_frame_counter()

        # 3. RETURN SUCCESS BONUS (with validation - Fix #2)
        return_reward = self._calculate_return_success_reward(current_frame)
        if return_reward > 0:
            reward += return_reward
            components['return_success'] = return_reward

        # 4. STRATEGIC POSITIONING (velocity-aware - Fix #6)
        positioning_reward = self._calculate_positioning_reward(paddle_pos)
        if positioning_reward != 0:
            reward += positioning_reward
            if positioning_reward > 0:
                components['strategic_position'] = positioning_reward

        # 5. PURPOSEFUL MOVEMENT (smoothed - Fix #4)
        movement_reward = self._calculate_movement_reward(paddle_pos)
        if movement_reward != 0:
            reward += movement_reward
            if movement_reward > 0:
                components['purposeful_movement'] = movement_reward
            else:
                components['bad_movement'] = movement_reward

        # 6. WALL SAFETY (graduated penalties)
        wall_penalty = self._calculate_wall_penalty(paddle_pos)
        if wall_penalty != 0:
            reward += wall_penalty
            components['wall_penalty'] = wall_penalty

        # 7. OOB HANDLING (Fix #3B)
        oob_reward = self._calculate_oob_reward(action, contact_detected)
        if oob_reward != 0:
            reward += oob_reward
            components['oob_handling'] = oob_reward

        if log_components:
            return reward, components
        return reward, {}

    def _calculate_return_success_reward(self, current_frame: int) -> float:
        """
        Calculate return success bonus with validation.
        Prevents false positives from opponent serves and double bounces.
        """
        curr_pos = self.ball_tracker.get_current_position()
        prev_pos = self.ball_tracker.get_previous_position()

        if curr_pos is None or prev_pos is None:
            return 0.0

        # Check if ball crossed from our side to opponent's side
        prev_x, _ = prev_pos
        curr_x, _ = curr_pos

        if not (prev_x > 0.5 and curr_x <= 0.5):
            return 0.0

        # Ball crossed net! Now validate:
        # 1. Did we have recent contact?
        if not self.contact_detector.had_recent_contact(self.return_validation_frames):
            return 0.0

        # 2. Validate the contact was legitimate (not a double bounce)
        # Check if ball trajectory makes sense
        vel = self.ball_tracker.get_velocity()
        if vel is None or vel[0] >= 0:  # Ball should be moving left
            return 0.0

        # 3. Mark contact as validated
        for contact_event in self.contact_validation_window:
            if current_frame - contact_event['frame'] <= self.return_validation_frames:
                if not contact_event['validated']:
                    contact_event['validated'] = True
                    self.last_return_frame = current_frame
                    return self.return_success_bonus

        return 0.0

    def _calculate_positioning_reward(self, paddle_pos: Optional[Tuple[float, float]]) -> float:
        """
        Calculate strategic positioning reward with velocity awareness.
        """
        if paddle_pos is None:
            return 0.0

        # Get base position quality
        quality = self.position_evaluator.calculate_position_quality(
            paddle_pos, self.ball_tracker
        )

        # Apply urgency multiplier based on ball velocity
        urgency = self.ball_tracker.calculate_urgency_multiplier()

        # Scale reward based on game state
        ball_on_opponent = not self.ball_tracker.is_ball_on_our_side()

        if ball_on_opponent:
            base_reward = self.ready_position_reward_max
        else:
            base_reward = self.intercept_reward_max

        # Final reward scaled by quality and urgency
        reward = base_reward * quality * urgency

        return reward

    def _calculate_movement_reward(self, paddle_pos: Optional[Tuple[float, float]]) -> float:
        """
        Calculate purposeful movement reward.
        Only rewards movements that improve position quality.
        """
        if paddle_pos is None:
            return 0.0

        improvement = self.position_evaluator.calculate_movement_quality_change(
            paddle_pos, self.ball_tracker
        )

        # Only reward meaningful changes (prevents noise exploitation)
        if improvement > self.min_meaningful_improvement:
            return self.purposeful_movement_coef * improvement
        elif improvement < -self.min_meaningful_improvement:
            return self.purposeful_movement_coef * improvement  # Penalty

        return 0.0

    def _calculate_wall_penalty(self, paddle_pos: Optional[Tuple[float, float]]) -> float:
        """
        Calculate graduated wall penalty.
        """
        if paddle_pos is None:
            return 0.0

        paddle_x, _ = paddle_pos

        if self.wall_warning_threshold < paddle_x <= self.wall_danger_threshold:
            # Soft warning zone
            return -self.wall_warning_penalty
        elif paddle_x > self.wall_danger_threshold:
            # Danger zone with scaling
            scale = (paddle_x - self.wall_danger_threshold) / (1.0 - self.wall_danger_threshold)
            return -self.wall_danger_penalty_max * scale

        return 0.0

    def _calculate_oob_reward(self,
                             action: Optional[np.ndarray],
                             contact_detected: bool) -> float:
        """
        Calculate OOB handling reward.
        Rewards letting OOB balls go, penalizes hitting them.
        """
        if not self.oob_detector.is_ball_going_oob(self.ball_tracker):
            return 0.0

        # Ball is going OOB
        if contact_detected:
            # BAD: Hit a ball that was going out
            return -self.oob_hit_penalty

        # GOOD: Not chasing the OOB ball (agent staying calm)
        # Check if agent is relatively stationary
        if action is not None:
            action_magnitude = np.linalg.norm(action)
            if action_magnitude < 0.3:  # Small/no action
                return self.oob_let_go_bonus

        return 0.0

    def update_ball_position(self, ball_pos: Optional[Tuple[float, float]]) -> None:
        """Update ball tracker with new position."""
        self.ball_tracker.update(ball_pos)

    def reset(self) -> None:
        """Reset all components for new episode."""
        self.ball_tracker.clear()
        self.contact_detector.reset()
        self.position_evaluator.reset()
        self.contact_validation_window.clear()
        self.last_return_frame = -999

    def get_debug_info(self) -> Dict[str, any]:
        """Get debug information about current state."""
        return {
            'ball_pos': self.ball_tracker.get_current_position(),
            'ball_velocity': self.ball_tracker.get_velocity(),
            'frames_since_contact': self.contact_detector.get_frames_since_contact(),
            'position_quality': self.position_evaluator.get_last_quality(),
            'urgency_multiplier': self.ball_tracker.calculate_urgency_multiplier(),
            'oob_status': self.oob_detector.get_oob_status_string(self.ball_tracker),
        }

