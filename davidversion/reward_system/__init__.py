"""
Modular Reward System for Pickleball RL Agent
Addresses weaknesses identified in reward_system_weaknesses.md
"""

from .ball_tracker import BallTracker
from .contact_detector import ContactDetector
from .position_evaluator import PositionEvaluator
from .reward_calculator import RewardCalculator
from .oob_detector import OOBDetector

__all__ = [
    'BallTracker',
    'ContactDetector',
    'PositionEvaluator',
    'RewardCalculator',
    'OOBDetector',
]

