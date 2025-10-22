# Modular Reward System for Pickleball RL

This directory contains a refactored, modular reward system that addresses all critical weaknesses identified in the original implementation.

## Architecture

The reward system is split into focused, testable modules:

```
reward_system/
├── __init__.py              # Module exports
├── ball_tracker.py          # Ball position & velocity tracking
├── contact_detector.py      # Robust contact detection
├── position_evaluator.py    # Strategic positioning evaluation
├── oob_detector.py          # Out-of-bounds detection
├── detection_utils.py       # CV detection utilities
├── reward_calculator.py     # Unified reward computation
└── README.md               # This file
```

## Key Improvements

### 1. **Velocity Tracking** (Fixes Issue #6)
`BallTracker` maintains ball position history and calculates velocity:
- Smoothed velocity estimation over 3 frames
- Urgency calculation based on time-to-net
- Prevents false positives from slow ball drift

### 2. **Robust Contact Detection** (Fixes Issue #1)
`ContactDetector` uses multiple signals to prevent false positives:
- ✅ Ball must be on our side
- ✅ Velocity must REVERSE direction (not just change)
- ✅ Velocity magnitude must change significantly
- ✅ Paddle must be very close to ball
- ✅ Ball must move AWAY from paddle after contact

**Prevents:**
- Wall bounce false positives
- Natural arc false positives
- Opponent shot false positives

### 3. **OOB Detection** (Fixes Issue #3B)
`OOBDetector` predicts ball trajectory:
- Rewards letting OOB balls go (+0.2)
- Penalizes hitting OOB balls (-0.3)
- Prevents agent from chasing bad balls

### 4. **Smoothed Position Quality** (Fixes Issue #4)
`PositionEvaluator` uses temporal smoothing:
- Rolling average over 5 frames
- Prevents noise exploitation
- State-aware evaluation (ready vs intercept)

### 5. **Validated Returns** (Fixes Issue #2)
`RewardCalculator` tracks contact events:
- Only rewards returns with validated contact
- Checks ball trajectory makes sense
- Prevents false positives from opponent serves

## Usage

### Integration with Gym Wrapper

```python
from reward_system import RewardCalculator
from reward_system.detection_utils import BallDetector, PaddleDetector

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, ...):
        # ... existing code ...
        
        # Initialize reward system
        self.reward_calc = RewardCalculator()
        
        # Initialize detectors
        self.ball_detector = BallDetector()
        self.paddle_detector = PaddleDetector()
    
    def step(self, action):
        # ... get observations ...
        
        # Detect ball and paddle
        ball_pos = self.ball_detector.detect(rgb_frame)
        paddle_pos = self.paddle_detector.detect(rgb_frame)
        
        # Update reward system
        self.reward_calc.update_ball_position(ball_pos)
        
        # Calculate reward
        reward, components = self.reward_calc.calculate_reward(
            paddle_pos=paddle_pos,
            event_reward=event,  # ±10 or 0
            current_frame=self.frame_count,
            action=action,
            log_components=True
        )
        
        return obs, reward, done, truncated, info
    
    def reset(self, ...):
        # ... existing code ...
        self.reward_calc.reset()
        return obs, info
```

## Reward Components

| Component | Value | Trigger | Notes |
|-----------|-------|---------|-------|
| **Win Point** | +10.0 | Score | Primary signal |
| **Lose Point** | -10.0 | Opponent scores | Primary signal |
| **Ball Contact** | +0.5 | Hit ball | Robust detection |
| **Return Success** | +0.3 | Ball to opponent | Validated |
| **Strategic Position** | +0.01-0.02 | Good positioning | Velocity-aware |
| **Purposeful Movement** | ±0.005 | Quality improvement | Smoothed |
| **Wall Warning** | -0.01 | x > 0.88 | Graduated |
| **Wall Danger** | -0.05 | x > 0.92 | Scaled |
| **OOB Let Go** | +0.2 | Don't chase OOB | New |
| **OOB Hit** | -0.3 | Hit OOB ball | New |

## Testing

### Unit Tests
```python
# Test contact detection doesn't trigger on wall bounce
ball_tracker = BallTracker()
ball_tracker.update((0.85, 0.5))  # Ball at wall
ball_tracker.update((0.83, 0.5))  # Bounced back
detector = ContactDetector()
assert not detector.detect_contact(ball_tracker, paddle_pos=(0.82, 0.5), frame=1)
```

### Integration Tests
```python
# Test OOB handling
oob_detector = OOBDetector()
ball_tracker = BallTracker()
ball_tracker.update((0.7, 0.8))
ball_tracker.update((0.72, 0.85))
ball_tracker.update((0.74, 0.92))  # Heading OOB high
assert oob_detector.is_ball_going_oob(ball_tracker)
```

## Debug Information

Get debug info at any time:
```python
debug_info = reward_calc.get_debug_info()
print(debug_info)
# {
#     'ball_pos': (0.65, 0.45),
#     'ball_velocity': (-0.02, 0.01),
#     'frames_since_contact': 3,
#     'position_quality': 0.87,
#     'urgency_multiplier': 1.5,
#     'oob_status': 'IN_BOUNDS'
# }
```

## Performance Considerations

- **Memory:** Each component uses deques with fixed max length
- **CPU:** CV detection is the bottleneck (not reward calculation)
- **Frame rate:** Designed for 60 FPS gameplay

## Future Enhancements

1. **Shot Quality Assessment** - Differentiate power shots from defensive returns
2. **Opponent Modeling** - Track opponent tendencies for adaptive positioning
3. **Trajectory Prediction** - ML-based physics predictor for better anticipation
4. **Multi-Agent Self-Play** - Train against diverse strategies

## Issues Addressed

✅ **Issue #1:** False positive contact detection (wall bounces, arcs)  
✅ **Issue #2:** Return bonus false positives (opponent serves)  
✅ **Issue #3:** Static positioning trap, no OOB handling  
✅ **Issue #4:** Noise exploitation from detection jitter  
✅ **Issue #6:** Missing velocity awareness  

See `reward_system_weaknesses.md` for detailed analysis.

## License

Part of the Pickleball RL training system.

