"""
Position Quality Evaluation
Fixes: Static positioning trap (Issue #3), Noise exploitation (Issue #4)
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque
from .ball_tracker import BallTracker


class PositionEvaluator:
    """
    Evaluates agent positioning quality based on game state.
    Uses smoothing to prevent noise exploitation.
    """

    def __init__(self,
                 optimal_ready_x: float = 0.70,
                 optimal_ready_y: float = 0.50,
                 ready_zone_radius: float = 0.25,
                 optimal_depth_min: float = 0.65,
                 optimal_depth_max: float = 0.88,
                 intercept_zone_radius: float = 0.3,
                 smoothing_window: int = 5):
        """
        Args:
            optimal_ready_x: X position for ready stance when ball on opponent side
            optimal_ready_y: Y position for ready stance
            ready_zone_radius: Acceptable distance from ready position
            optimal_depth_min: Minimum good depth position
            optimal_depth_max: Maximum good depth position
            intercept_zone_radius: Distance tolerance for intercept positioning
            smoothing_window: Frames to smooth quality scores over
        """
        self.optimal_ready_x = optimal_ready_x
        self.optimal_ready_y = optimal_ready_y
        self.ready_zone_radius = ready_zone_radius
        self.optimal_depth_min = optimal_depth_min
        self.optimal_depth_max = optimal_depth_max
        self.intercept_zone_radius = intercept_zone_radius

        # Smoothing to prevent noise exploitation (Fix #4)
        self.quality_history = deque(maxlen=smoothing_window)
        self.last_quality = 0.0

    def calculate_position_quality(self,
                                   paddle_pos: Optional[Tuple[float, float]],
                                   ball_tracker: BallTracker,
                                   apply_smoothing: bool = True) -> float:
        """
        Calculate position quality score [0, 1] based on strategic positioning.

        Different evaluation based on ball location:
        - Ball on opponent side: Reward ready position
        - Ball on our side: Reward intercept positioning and depth

        Args:
            paddle_pos: (x, y) normalized paddle position
            ball_tracker: BallTracker with current ball state
            apply_smoothing: Whether to apply temporal smoothing

        Returns:
            Quality score between 0 and 1
        """
        if paddle_pos is None:
            return 0.0

        ball_pos = ball_tracker.get_current_position()
        if ball_pos is None:
            return 0.0

        paddle_x, paddle_y = paddle_pos
        ball_x, ball_y = ball_pos

        # Determine game state
        ball_on_opponent_side = (ball_x < 0.5)

        if ball_on_opponent_side:
            # STATE 1: Ready position quality
            quality = self._calculate_ready_position_quality(paddle_x, paddle_y)
        else:
            # STATE 2: Intercept position quality
            quality = self._calculate_intercept_quality(
                paddle_x, paddle_y, ball_x, ball_y, ball_tracker
            )

        # Apply smoothing to prevent noise exploitation
        if apply_smoothing:
            self.quality_history.append(quality)
            quality = np.mean(list(self.quality_history))

        self.last_quality = quality
        return quality

    def _calculate_ready_position_quality(self, paddle_x: float, paddle_y: float) -> float:
        """
        Calculate quality when ball is on opponent's side.
        Reward being in ready position for quick reactions.
        """
        # Distance to optimal ready position
        ready_dist = np.sqrt(
            (paddle_x - self.optimal_ready_x)**2 +
            (paddle_y - self.optimal_ready_y)**2
        )

        # Quality decreases with distance from ready position
        if ready_dist > self.ready_zone_radius:
            return 0.0

        quality = 1.0 - (ready_dist / self.ready_zone_radius)
        return max(0.0, min(1.0, quality))

    def _calculate_intercept_quality(self,
                                     paddle_x: float,
                                     paddle_y: float,
                                     ball_x: float,
                                     ball_y: float,
                                     ball_tracker: BallTracker) -> float:
        """
        Calculate quality when ball is on our side.
        Reward positioning to intercept ball trajectory.
        """
        # Component 1: Vertical alignment with ball (or predicted position)
        # Use velocity to predict where ball will be
        vel = ball_tracker.get_velocity()

        if vel is not None and abs(vel[0]) > 0.01:
            # Predict ball position a few frames ahead
            prediction_frames = 5
            predicted_y = ball_y + vel[1] * prediction_frames
            target_y = np.clip(predicted_y, 0.0, 1.0)
        else:
            target_y = ball_y

        vertical_dist = abs(paddle_y - target_y)
        vertical_quality = max(0.0, 1.0 - (vertical_dist / self.intercept_zone_radius))

        # Component 2: Depth quality (not too forward, not at wall)
        if self.optimal_depth_min <= paddle_x <= self.optimal_depth_max:
            depth_quality = 1.0
        elif paddle_x < self.optimal_depth_min:
            # Too far forward
            dist_forward = self.optimal_depth_min - paddle_x
            depth_quality = max(0.0, 1.0 - dist_forward / 0.2)
        else:
            # Too far back
            dist_back = paddle_x - self.optimal_depth_max
            depth_quality = max(0.0, 1.0 - dist_back / 0.12)

        # Combine components
        quality = vertical_quality * 0.7 + depth_quality * 0.3
        return max(0.0, min(1.0, quality))

    def calculate_movement_quality_change(self,
                                         paddle_pos: Optional[Tuple[float, float]],
                                         ball_tracker: BallTracker) -> float:
        """
        Calculate how much the movement improved/degraded position quality.

        Args:
            paddle_pos: Current paddle position
            ball_tracker: BallTracker instance

        Returns:
            Quality change (positive = improvement, negative = degradation)
        """
        current_quality = self.calculate_position_quality(
            paddle_pos, ball_tracker, apply_smoothing=False
        )
        improvement = current_quality - self.last_quality

        return improvement

    def get_last_quality(self) -> float:
        """Get the last calculated position quality."""
        return self.last_quality

    def reset(self) -> None:
        """Reset position quality history."""
        self.quality_history.clear()
        self.last_quality = 0.0

