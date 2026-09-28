"""
Computer Vision Detection Utilities
Ball and paddle detection with debugging support.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class BallDetector:
    """Detects yellow ball in RGB images."""

    def __init__(self,
                 hsv_lower: Tuple[int, int, int] = (20, 100, 110),
                 hsv_upper: Tuple[int, int, int] = (40, 255, 255)):
        """
        Args:
            hsv_lower: Lower HSV threshold for yellow
            hsv_upper: Upper HSV threshold for yellow
        """
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

    def detect(self, rgb_img01: np.ndarray, save_debug: bool = False) -> Optional[Tuple[float, float]]:
        """
        Detect ball centroid in RGB image [0,1] range.

        Args:
            rgb_img01: RGB image in float [0, 1] range
            save_debug: Whether to save debug visualizations

        Returns:
            (x_norm, y_norm) tuple or None if not detected
        """
        try:
            img8 = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)

            # Create mask
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Get largest contour
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M.get("m00", 0) == 0:
                return None

            # Calculate centroid
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            h, w = mask.shape[:2]
            x_norm = cx / w
            y_norm = cy / h

            return (x_norm, y_norm)

        except Exception:
            return None


class PaddleDetector:
    """Detects orange paddle in RGB images."""

    def __init__(self,
                 roi_x_start: float = 0.50,
                 hsv_ranges: list = None):
        """
        Args:
            roi_x_start: Start of ROI as fraction of width (search right side only)
            hsv_ranges: List of (lower, upper) HSV range tuples for paddle colors
        """
        self.roi_x_start = roi_x_start

        if hsv_ranges is None:
            # Default: multiple orange/red-orange ranges
            self.hsv_ranges = [
                (np.array([0, 80, 60], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
                (np.array([10, 80, 60], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
            ]
        else:
            self.hsv_ranges = hsv_ranges

    def detect(self, rgb_img01: np.ndarray, save_debug: bool = False) -> Optional[Tuple[float, float]]:
        """
        Detect right paddle centroid in RGB image [0,1] range.
        Only searches rightmost portion of image.

        Args:
            rgb_img01: RGB image in float [0, 1] range
            save_debug: Whether to save debug visualizations

        Returns:
            (x_norm, y_norm) tuple or None if not detected
        """
        try:
            img8 = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = img8.shape[:2]

            # ROI: right portion only
            x0 = int(self.roi_x_start * w)
            roi = img8[:, x0:w, :]

            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            # Combine multiple color ranges
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in self.hsv_ranges:
                mask_i = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, mask_i)

            # Morphological cleanup
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Get largest contour
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M.get("m00", 0) == 0:
                return None

            # Calculate centroid (shift back to full image coordinates)
            cx = M["m10"] / M["m00"] + x0
            cy = M["m01"] / M["m00"]

            x_norm = cx / w
            y_norm = cy / h

            return (x_norm, y_norm)

        except Exception:
            return None


class DetectionDebugger:
    """Handles debug visualization for detections."""

    def __init__(self, debug_dir: str = "debug_frames"):
        """
        Args:
            debug_dir: Directory to save debug images
        """
        self.debug_dir = debug_dir
        self.frame_count = 0

    def save_detection_overlay(self,
                               rgb_img01: np.ndarray,
                               ball_pos: Optional[Tuple[float, float]],
                               paddle_pos: Optional[Tuple[float, float]],
                               tag: str = "detection") -> None:
        """
        Save an overlay image showing detected ball and paddle positions.

        Args:
            rgb_img01: RGB image in [0, 1] range
            ball_pos: (x, y) ball position or None
            paddle_pos: (x, y) paddle position or None
            tag: Filename tag
        """
        try:
            import os
            os.makedirs(self.debug_dir, exist_ok=True)

            overlay = (np.clip(rgb_img01, 0.0, 1.0) * 255).astype(np.uint8)
            h, w = overlay.shape[:2]

            # Draw ball
            if ball_pos is not None:
                xb, yb = ball_pos
                ball_px = int(xb * w)
                ball_py = int(yb * h)
                cv2.circle(overlay, (ball_px, ball_py), 6, (0, 255, 255), -1)
                cv2.circle(overlay, (ball_px, ball_py), 8, (255, 255, 255), 2)
                cv2.putText(overlay, f"Ball: ({xb:.2f},{yb:.2f})",
                          (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw paddle
            if paddle_pos is not None:
                xp, yp = paddle_pos
                paddle_px = int(xp * w)
                paddle_py = int(yp * h)
                cv2.circle(overlay, (paddle_px, paddle_py), 8, (255, 128, 0), -1)
                cv2.circle(overlay, (paddle_px, paddle_py), 10, (255, 255, 255), 2)
                cv2.putText(overlay, f"Paddle: ({xp:.2f},{yp:.2f})",
                          (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)

            # Save
            path = os.path.join(self.debug_dir, f"{tag}_{self.frame_count:06d}.png")
            cv2.imwrite(path, overlay)

        except Exception:
            pass  # Don't crash if debug save fails

    def increment_frame(self) -> None:
        """Increment frame counter."""
        self.frame_count += 1

