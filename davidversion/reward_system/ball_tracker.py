"""
Ball Tracking with Velocity Estimation
Fixes: Missing velocity awareness (Issue #6)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


class BallTracker:
    """
    Tracks ball position history and calculates velocity.
    Prevents false positives from natural ball arcs and physics.
    """

    def __init__(self, history_length: int = 10):
        """
        Args:
            history_length: Number of frames to track for velocity estimation
        """
        self.history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length - 1)

    def update(self, ball_pos: Optional[Tuple[float, float]]) -> None:
        """
        Update ball position history.

        Args:
            ball_pos: (x, y) normalized ball position, or None if not detected
        """
        if ball_pos is not None:
            self.history.append(ball_pos)

            # Calculate velocity if we have at least 2 positions
            if len(self.history) >= 2:
                prev_pos = self.history[-2]
                curr_pos = self.history[-1]
                velocity = (
                    curr_pos[0] - prev_pos[0],
                    curr_pos[1] - prev_pos[1]
                )
                self.velocity_history.append(velocity)

    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """Get most recent ball position."""
        return self.history[-1] if len(self.history) > 0 else None

    def get_previous_position(self) -> Optional[Tuple[float, float]]:
        """Get second most recent ball position."""
        return self.history[-2] if len(self.history) >= 2 else None

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Get current ball velocity (change per frame).

        Returns:
            (vx, vy) velocity tuple, or None if insufficient data
        """
        if len(self.velocity_history) == 0:
            return None

        # Use average of last 3 velocities for smoothing
        recent_vels = list(self.velocity_history)[-3:]
        avg_vx = np.mean([v[0] for v in recent_vels])
        avg_vy = np.mean([v[1] for v in recent_vels])

        return (avg_vx, avg_vy)

    def get_velocity_at_frame(self, frames_ago: int) -> Optional[Tuple[float, float]]:
        """
        Get velocity at specific frame in history.

        Args:
            frames_ago: How many frames back (0 = most recent)
        """
        idx = -(frames_ago + 1)
        if abs(idx) <= len(self.velocity_history):
            return self.velocity_history[idx]
        return None

    def estimate_time_to_net(self) -> Optional[float]:
        """
        Estimate frames until ball crosses net (x=0.5).

        Returns:
            Number of frames, or None if ball not moving toward net or insufficient data
        """
        pos = self.get_current_position()
        vel = self.get_velocity()

        if pos is None or vel is None:
            return None

        ball_x, _ = pos
        vx, _ = vel

        # Ball must be on right side and moving left
        if ball_x <= 0.5 or vx >= 0:
            return None

        # Simple linear estimation
        distance_to_net = ball_x - 0.5
        frames_to_net = distance_to_net / abs(vx)

        return frames_to_net

    def calculate_urgency_multiplier(self) -> float:
        """
        Calculate urgency multiplier based on time until ball arrives.

        Returns:
            Multiplier for positioning rewards (1.0 to 3.0)
        """
        frames_to_net = self.estimate_time_to_net()

        if frames_to_net is None:
            return 1.0

        # High urgency if ball arriving soon
        if frames_to_net < 10:  # Less than 0.17 seconds at 60fps
            return 3.0
        elif frames_to_net < 30:  # Less than 0.5 seconds
            return 1.5
        else:
            return 1.0

    def is_ball_on_our_side(self, our_side_threshold: float = 0.5) -> bool:
        """Check if ball is on our side (right side for right agent)."""
        pos = self.get_current_position()
        if pos is None:
            return False
        return pos[0] > our_side_threshold

    def get_position_n_frames_ago(self, n: int) -> Optional[Tuple[float, float]]:
        """Get ball position from n frames ago."""
        idx = -(n + 1)
        if abs(idx) <= len(self.history):
            return self.history[idx]
        return None

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()
        self.velocity_history.clear()

