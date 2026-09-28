"""
Out of Bounds Detection
Fixes: No OOB handling (Issue #3B)
"""

import numpy as np
from typing import Optional, Tuple
from .ball_tracker import BallTracker


class OOBDetector:
    """
    Detects if ball is heading out of bounds.
    Helps agent learn to let OOB balls go instead of chasing them.
    """

    def __init__(self,
                 y_min: float = 0.0,
                 y_max: float = 1.0,
                 x_max: float = 1.0,
                 prediction_frames: int = 20):
        """
        Args:
            y_min: Minimum valid Y coordinate
            y_max: Maximum valid Y coordinate
            x_max: Maximum valid X coordinate (back wall)
            prediction_frames: How many frames ahead to predict
        """
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max
        self.prediction_frames = prediction_frames

    def predict_trajectory_oob(self, ball_tracker: BallTracker) -> Tuple[bool, Optional[str]]:
        """
        Predict if ball will go out of bounds based on current trajectory.

        Args:
            ball_tracker: BallTracker with current ball state

        Returns:
            (is_oob, reason) tuple where:
                is_oob: True if ball predicted to go OOB
                reason: 'top', 'bottom', 'back_wall', or None
        """
        pos = ball_tracker.get_current_position()
        vel = ball_tracker.get_velocity()

        if pos is None or vel is None:
            return False, None

        ball_x, ball_y = pos
        vx, vy = vel

        # Simple linear prediction
        for frame in range(1, self.prediction_frames + 1):
            pred_x = ball_x + vx * frame
            pred_y = ball_y + vy * frame

            # Check boundaries
            if pred_y < self.y_min:
                return True, 'bottom'
            if pred_y > self.y_max:
                return True, 'top'
            if pred_x > self.x_max:
                return True, 'back_wall'

            # If ball crosses net, stop prediction (opponent's problem)
            if pred_x < 0.5:
                break

        return False, None

    def is_ball_going_oob(self, ball_tracker: BallTracker) -> bool:
        """
        Simple check if ball is going OOB.

        Returns:
            True if ball predicted to go out of bounds
        """
        is_oob, _ = self.predict_trajectory_oob(ball_tracker)
        return is_oob

    def calculate_oob_reward_modifier(self,
                                      ball_tracker: BallTracker,
                                      agent_moving_toward_ball: bool,
                                      contact_detected: bool) -> float:
        """
        Calculate reward modifier for OOB situations.

        Args:
            ball_tracker: BallTracker instance
            agent_moving_toward_ball: Is agent moving toward the ball?
            contact_detected: Did agent just hit the ball?

        Returns:
            Reward modifier (positive for correct behavior, negative for wrong)
        """
        is_oob, reason = self.predict_trajectory_oob(ball_tracker)

        if not is_oob:
            return 0.0

        reward = 0.0

        # Ball is going OOB
        if contact_detected:
            # BAD: Agent hit a ball that was going OOB
            reward -= 0.3
        elif not agent_moving_toward_ball:
            # GOOD: Agent is letting it go
            reward += 0.2

        return reward

    def get_oob_status_string(self, ball_tracker: BallTracker) -> str:
        """Get human-readable OOB status."""
        is_oob, reason = self.predict_trajectory_oob(ball_tracker)
        if is_oob:
            return f"OOB_{reason.upper()}"
        return "IN_BOUNDS"

