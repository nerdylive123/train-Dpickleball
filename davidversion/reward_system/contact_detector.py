"""
Robust Ball Contact Detection
Fixes: False positive contact detection (Issue #1)
"""

import numpy as np
from typing import Optional, Tuple
from .ball_tracker import BallTracker


class ContactDetector:
    """
    Detects when agent successfully hits the ball.
    Prevents false positives from wall bounces, natural arcs, and opponent shots.
    """

    def __init__(self,
                 proximity_threshold: float = 0.12,
                 min_velocity_magnitude: float = 0.005,
                 min_velocity_change_ratio: float = 0.3):
        """
        Args:
            proximity_threshold: Max distance between paddle and ball for contact
            min_velocity_magnitude: Minimum ball velocity to consider (filters slow drift)
            min_velocity_change_ratio: Minimum ratio of velocity change to detect impact
        """
        self.proximity_threshold = proximity_threshold
        self.min_velocity_magnitude = min_velocity_magnitude
        self.min_velocity_change_ratio = min_velocity_change_ratio

        self.frames_since_contact = 999
        self.last_contact_frame = -999

    def detect_contact(self,
                      ball_tracker: BallTracker,
                      paddle_pos: Optional[Tuple[float, float]],
                      current_frame: int) -> bool:
        """
        Detect if agent successfully hit the ball this frame.

        Uses multiple signals to prevent false positives:
        1. Ball must have been on our side
        2. Ball velocity must reverse direction (X component)
        3. Velocity magnitude must change significantly (impact signature)
        4. Paddle must be very close to ball at impact
        5. Ball must be moving away from paddle after contact

        Args:
            ball_tracker: BallTracker instance with history
            paddle_pos: (x, y) paddle position, or None
            current_frame: Current frame number

        Returns:
            True if contact detected, False otherwise
        """
        # Need paddle position
        if paddle_pos is None:
            return False

        # Need sufficient ball history
        curr_pos = ball_tracker.get_current_position()
        prev_pos = ball_tracker.get_previous_position()

        if curr_pos is None or prev_pos is None:
            return False

        # Get velocities before and after potential contact
        prev_vel = ball_tracker.get_velocity_at_frame(1)
        curr_vel = ball_tracker.get_velocity()

        if prev_vel is None or curr_vel is None:
            return False

        # 1. Ball must have been on our side (right side, x > 0.5)
        prev_x, prev_y = prev_pos
        curr_x, curr_y = curr_pos

        if prev_x <= 0.5:
            return False

        # 2. Check for velocity reversal in X direction
        prev_vx, prev_vy = prev_vel
        curr_vx, curr_vy = curr_vel

        # Filter out very slow ball movement (natural drift)
        if abs(prev_vx) < self.min_velocity_magnitude:
            return False

        # X velocity must reverse (ball was moving right/stationary, now moving left)
        # or ball was moving left slowly, now moving left faster
        velocity_reversed = (prev_vx * curr_vx) < 0  # Different signs

        if not velocity_reversed:
            # Check if velocity increased significantly in left direction
            # (e.g., ball bouncing back from wall then we hit it harder left)
            if curr_vx >= prev_vx:  # Not moving more left
                return False

        # 3. Velocity magnitude must change significantly (impact signature)
        prev_speed = np.sqrt(prev_vx**2 + prev_vy**2)
        curr_speed = np.sqrt(curr_vx**2 + curr_vy**2)

        velocity_change_ratio = abs(curr_speed - prev_speed) / (prev_speed + 1e-6)

        if velocity_change_ratio < self.min_velocity_change_ratio:
            return False

        # 4. Paddle must be very close to ball
        paddle_x, paddle_y = paddle_pos
        distance_to_ball = np.sqrt((paddle_x - curr_x)**2 + (paddle_y - curr_y)**2)

        if distance_to_ball > self.proximity_threshold:
            return False

        # 5. Ball should be moving away from paddle after contact
        # Vector from paddle to ball
        paddle_to_ball = (curr_x - paddle_x, curr_y - paddle_y)

        # Dot product: positive if ball moving away from paddle
        dot_product = paddle_to_ball[0] * curr_vx + paddle_to_ball[1] * curr_vy

        if dot_product <= 0:  # Ball moving toward paddle or parallel
            return False

        # All checks passed - this is a real contact!
        self.frames_since_contact = 0
        self.last_contact_frame = current_frame
        return True

    def update_frame_counter(self) -> None:
        """Increment frames since last contact."""
        self.frames_since_contact += 1

    def get_frames_since_contact(self) -> int:
        """Get number of frames since last detected contact."""
        return self.frames_since_contact

    def had_recent_contact(self, max_frames: int = 5) -> bool:
        """Check if contact occurred within last N frames."""
        return self.frames_since_contact <= max_frames

    def reset(self) -> None:
        """Reset contact tracking."""
        self.frames_since_contact = 999
        self.last_contact_frame = -999

