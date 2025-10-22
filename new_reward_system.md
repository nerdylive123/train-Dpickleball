# New Reward System Design - Addressing Current Flaws

**Date:** October 22, 2025  
**Purpose:** Redesign reward shaping to encourage optimal pickleball strategy without the current system's flaws

---

## Executive Summary

The current reward system has three major categories of flaws:
1. **Movement Encouragement** - Rewards useless fidgeting
2. **Ball Stagnation** - Punishes players unfairly for time-based rules
3. **Ball Tracking** - Rewards proximity over strategic positioning

This document proposes a **context-aware reward system** that rewards **purposeful actions** and **strategic gameplay** rather than simple proximity and movement.

---

## Current System Flaws - Detailed Analysis

### 1. Movement Encouragement (+0.001 per action)

**Problem:** Rewards ANY non-zero action regardless of usefulness.

**Specific Issues:**
- Agent learns to jitter/wiggle constantly to farm +0.001 rewards
- Wrong direction movements still rewarded (moving left when should go right)
- Encourages action spam over deliberate positioning
- Over 300 steps (5 seconds), can accumulate +0.3 from useless twitching vs +10 from scoring

**Example Bad Behavior:**
```
Ball on opponent's side at (0.2, 0.5)
Agent should: Wait in ready position
Agent actually does: Up, Down, Up, Down repeatedly → +0.001 per step
```

### 2. Ball Stagnation Penalty (kicks in at 3.5 seconds)

**Problem:** Penalizes having ball on your side without considering context.

**Specific Issues:**
- Opponent's slow return punishes YOU when ball arrives
- Rule violation is at 5 seconds, but penalty starts at 3.5s (punishes legal play)
- No distinction between defensive patience and actual stalling
- No reward for successfully returning before time limit
- Penalty grows: -0.02 → -0.04 → -0.06... as time increases

**Example Bad Behavior:**
```
Rally timeline:
0s: Opponent hits high lob
1-4s: Ball travels slowly to your side (you get penalized during this!)
4s: You prepare perfect return shot
Result: Penalized for good strategy
```

### 3. Vertical Ball Tracking (+0.02 max)

**Problem:** Rewards vertical proximity to ball regardless of game context.

**Specific Issues:**
- Rewards tracking ball even when it's on opponent's side (should be in ready position)
- Agent chases balls going out of bounds instead of letting them go
- Encourages following ball vertically even when horizontally malpositioned
- No consideration of shot preparation needs

**Example Bad Behavior:**
```
Ball at (0.1, 0.9) - far on opponent's side, near top
Agent at (0.75, 0.5) - good center position
Agent learns: Move to y=0.9 for +0.02 reward
Should do: Stay centered and ready
```

### 4. Horizontal Ball Tracking (+0.01 max, only when ball_x > 0.5)

**Problem:** Only reactive, encourages defensive camping behind ball.

**Specific Issues:**
- No reward when ball on opponent's side (no anticipation learning)
- Formula `ideal_x = min(0.85, ball_x + 0.1)` keeps agent always behind ball
- Never encourages aggressive positioning or anticipation
- Narrow reward window (distance < 0.2) misses many good positions

**Example Bad Behavior:**
```
Ball at x=0.52 (just crossed net to your side)
Ideal position: 0.62
Agent at 0.75: distance=0.13 → Gets NO reward (but 0.13 < 0.2, so actually gets small reward)
Agent learns: Always chase ball_x + 0.1, never anticipate
```

---

## Proposed Solution: Context-Aware Strategic Reward System

### Core Philosophy

**OLD:** Reward proximity to ball and any movement  
**NEW:** Reward purposeful actions and strategic positioning based on game state

### Design Principles

1. **State-Aware**: Different rewards for different game phases (ball on our side vs their side)
2. **Action-Outcome**: Reward actions that lead to positive outcomes, not random movement
3. **Strategic Positioning**: Reward anticipation and court coverage, not just reaction
4. **Contact Reward**: Explicitly reward hitting the ball successfully
5. **Eliminate Time-Based Penalties**: Remove unfair stagnation penalties

---

## New Reward Components

### 1. Ball Contact Detection & Reward (NEW - HIGHEST PRIORITY)

**Goal:** Directly reward successful ball hits

**Implementation:**
```python
# Detect if agent hit the ball this frame
ball_contact_reward = +0.5  # Substantial reward for hitting ball

# Detection method:
# - Track ball velocity/direction change
# - If ball was on our side AND moving toward us
# - Then suddenly changes direction toward opponent
# - AND paddle was close to ball (< 0.15 distance)
# → Agent successfully hit the ball!

if self._detect_ball_contact(prev_ball, curr_ball, paddle_pos):
    reward += 0.5
```

**Why this works:**
- Directly reinforces the core game action (hitting ball)
- Sparse enough to not overwhelm main signal (+10 for scoring)
- Replaces vague "tracking" rewards with concrete action reward

**Prevents:**
- Passive play (waiting doesn't give this reward)
- Ball chasing without purpose (only hitting counts)

---

### 2. Strategic Positioning Reward (REPLACES ball tracking)

**Goal:** Reward being in good position based on ball location and game phase

**State 1: Ball on Opponent's Side (ball_x < 0.5)**
```python
# Reward center-court ready position
optimal_ready_x = 0.70  # Slightly back from net
optimal_ready_y = 0.50  # Center vertically

distance = sqrt((paddle_x - optimal_ready_x)^2 + (paddle_y - optimal_ready_y)^2)
if distance < 0.25:  # Within reasonable ready zone
    ready_reward = 0.01 * (1.0 - distance/0.25)
    reward += ready_reward
```

**State 2: Ball on Our Side (ball_x > 0.5)**
```python
# Reward intercepting ball's trajectory
# Calculate where ball will be in next few frames (simple linear projection)
predicted_ball_y = ball_y + ball_velocity_y * prediction_steps

# Reward being vertically aligned with predicted position
y_distance = abs(paddle_y - predicted_ball_y)
if y_distance < 0.3:
    intercept_reward = 0.02 * (1.0 - y_distance/0.3)
    reward += intercept_reward

# Reward being at appropriate depth (not too forward, not at wall)
optimal_depth_range = (0.65, 0.88)
if optimal_depth_range[0] <= paddle_x <= optimal_depth_range[1]:
    depth_reward = 0.01
    reward += depth_reward
```

**Why this works:**
- Different strategy for different game states
- Encourages anticipation (predicted position, not current)
- Rewards court coverage when ball is away
- Only shapes when contextually appropriate

**Prevents:**
- Chasing ball on opponent's side (State 1 handles this)
- Reactive-only positioning (prediction encourages anticipation)

---

### 3. Purposeful Movement Reward (REPLACES random movement reward)

**Goal:** Reward movement that improves position, not random actions

**Implementation:**
```python
# Calculate position quality BEFORE and AFTER action
position_quality_before = self._calculate_position_quality(prev_paddle, ball_state)
position_quality_after = self._calculate_position_quality(curr_paddle, ball_state)

improvement = position_quality_after - position_quality_before

if improvement > 0.01:  # Meaningful improvement
    purposeful_movement_reward = 0.005 * improvement
    reward += purposeful_movement_reward
elif improvement < -0.01:  # Made position worse
    bad_movement_penalty = 0.005 * abs(improvement)
    reward -= bad_movement_penalty
# If improvement ≈ 0: no reward (movement didn't help)
```

**Position Quality Function:**
```python
def _calculate_position_quality(self, paddle_pos, ball_state):
    """
    Returns quality score [0, 1] based on how well positioned the paddle is.
    Considers: ball location, ball velocity, optimal zones, wall dangers
    """
    if ball_state['on_opponent_side']:
        # Quality = proximity to ready position
        return 1.0 - distance_to_ready_position
    else:
        # Quality = ability to intercept ball
        intercept_distance = distance_to_ball_trajectory
        depth_quality = 1.0 if in_optimal_depth else 0.5
        return (1.0 - intercept_distance) * depth_quality
```

**Why this works:**
- Only rewards movements that actually improve position
- Penalizes movements that worsen position (moving wrong direction)
- Neutral for useless movements (jittering gives no reward)
- Can't be gamed by spamming actions

**Prevents:**
- Random wiggling/jittering
- Moving in wrong direction
- Action spam

---

### 4. Wall Safety (KEEP but adjust threshold)

**Current:** Penalty at x > 0.93  
**Proposed:** Graduated penalty system

```python
# Soft warning zone
if 0.88 < paddle_x <= 0.92:
    wall_warning_penalty = -0.01
    reward += wall_warning_penalty

# Danger zone
elif paddle_x > 0.92:
    wall_danger_penalty = -0.05 * (paddle_x - 0.92) / 0.08
    # Scales from -0.05 at x=0.92 to -0.05 at x=1.0
    reward += wall_danger_penalty
```

**Why this works:**
- Gentler gradient helps learning
- Clear warning before major penalty
- Still prevents OOB violations

---

### 5. Ball Side Awareness (REMOVE stagnation penalty, ADD different system)

**Problem with old system:** Time-based penalty punishes player for opponent's actions

**New approach:** No time penalty. Instead:

```python
# Track ball side for information only (no penalty)
# The real penalty comes from:
# 1. Not hitting ball (no contact reward)
# 2. Eventually losing the point (-10)

# Optional: Very gentle urgency signal after 4 seconds
if ball_on_our_side_time > 240:  # 4 seconds (80% of limit)
    # Small bonus for moving toward ball
    if moving_toward_ball:
        urgency_bonus = 0.005
        reward += urgency_bonus
```

**Why this works:**
- Doesn't punish for opponent's slow play
- Real penalty is losing the point (natural consequence)
- Optional gentle nudge only when actually approaching time limit
- Focuses on action (moving toward ball) not just presence

**Prevents:**
- Unfair penalties for opponent camping
- Punishing legal play (before 5 second mark)
- Blame game scenarios

---

### 6. Return Success Bonus (NEW)

**Goal:** Reward successfully getting ball back to opponent's side

**Implementation:**
```python
# Track if ball was on our side last frame and opponent's side this frame
if prev_ball_x > 0.5 and curr_ball_x <= 0.5:
    # Ball crossed back to opponent!
    # Check if we were the ones who hit it (contact happened recently)
    if self._frames_since_contact <= 5:
        return_success_bonus = +0.3
        reward += return_success_bonus
```

**Why this works:**
- Directly rewards the goal (returning ball)
- Works in combination with contact reward
- Substantial enough to matter but not overwhelming main signal

---

## Complete Reward Structure Summary

| Component | Value | Trigger | Purpose |
|-----------|-------|---------|---------|
| **Win Point** | +10.0 | Score a point | PRIMARY SIGNAL - ultimate goal |
| **Lose Point** | -10.0 | Opponent scores | PRIMARY SIGNAL - avoid this |
| **Ball Contact** | +0.5 | Hit the ball | Reward core action |
| **Return Success** | +0.3 | Ball goes back to opponent after our hit | Reward successful returns |
| **Strategic Position** | +0.01-0.02 | Good positioning for game state | Shape positioning |
| **Purposeful Movement** | ±0.005 | Movement improves/worsens position | Reward smart movement only |
| **Wall Warning** | -0.01 | x > 0.88 | Gentle warning |
| **Wall Danger** | -0.05 | x > 0.92 | Prevent OOB |
| **Urgency Nudge** | +0.005 | Moving toward ball after 4s on our side | Optional gentle reminder |

---

## Implementation Priority

### Phase 1: Critical Fixes (Implement First)
1. ✅ **Remove movement encouragement reward** (delete `self._k_movement`)
2. ✅ **Remove ball stagnation penalty** (delete time-based penalty)
3. ✅ **Add ball contact detection** (new method)
4. ✅ **Add ball contact reward** (+0.5)

### Phase 2: Strategic Improvements
5. ✅ **Implement state-aware positioning** (replace current tracking)
6. ✅ **Add return success bonus** (+0.3)
7. ✅ **Implement purposeful movement system**

### Phase 3: Polish
8. ✅ **Adjust wall penalty gradient**
9. ✅ **Add optional urgency nudge** (if needed)
10. ✅ **Tune all coefficients** based on training results

---

## Expected Behavioral Changes

### Before (Current System)
- Agent jitters constantly for movement rewards
- Chases ball even on opponent's side
- Gets punished for opponent's slow play
- Always reactive, never anticipatory
- Focuses on being near ball, not hitting it

### After (New System)
- Agent moves purposefully only when beneficial
- Maintains ready position when ball is away
- Anticipates ball trajectory and positions accordingly
- Actively tries to hit ball (contact reward)
- Celebrates successful returns (return bonus)
- No unfair time penalties

---

## Risk Mitigation

### Risk 1: Contact detection might be noisy
**Mitigation:** Use multiple signals (ball direction change + paddle proximity + velocity change)

### Risk 2: Position quality function too complex
**Mitigation:** Start with simple version, iterate based on behavior

### Risk 3: Rewards might still conflict
**Mitigation:** Careful coefficient tuning (contact >> positioning >> movement)

### Risk 4: No movement encouragement might cause freezing
**Mitigation:** Purposeful movement system rewards good movement, position quality changes with ball state

---

## Testing Plan

### Test 1: No More Jittering
**Observation:** Agent should stand still when ball is on opponent's side
**Success Metric:** <10% of frames have non-zero actions when ball_x < 0.3

### Test 2: Ball Contact Learning
**Observation:** Agent should actively try to hit ball
**Success Metric:** Increasing contact rate over training episodes

### Test 3: Anticipatory Positioning
**Observation:** Agent should move to intercept position before ball arrives
**Success Metric:** Lower average distance to ball when ball reaches our side

### Test 4: No Unfair Penalties
**Observation:** Agent's reward shouldn't decrease just from time passing
**Success Metric:** No negative rewards when ball_on_our_side_time < 300 frames (unless losing point)

---

## Coefficient Recommendations

Based on relative importance and expected frequency:

```python
# Main signals (untouched)
win_bonus = 10.0
loss_penalty = 10.0

# High-value shape rewards (core actions)
ball_contact_reward = 0.5        # ~20 hits per rally = +10 potential
return_success_bonus = 0.3       # ~15 returns per rally = +4.5 potential

# Medium-value shape rewards (positioning)
strategic_position_max = 0.02    # Every frame positioned = +1.2 per second
depth_quality_bonus = 0.01       # Every frame in good depth = +0.6 per second

# Low-value shape rewards (movement quality)
purposeful_movement_max = 0.005  # Only when improving position

# Penalties (safety)
wall_warning = -0.01             # Gentle
wall_danger_max = -0.05          # Strong but not overwhelming
```

**Rationale:**
- Contact rewards (0.5-0.8 total per hit) are meaningful but can't overwhelm ±10 main signal
- Positioning rewards (0.02-0.03 per frame) provide gentle guidance
- Movement rewards (0.005) are small enough to not encourage spam
- All shaping rewards combined over 300 steps ≈ ±5, which is meaningful but secondary to ±10 win/loss

---

## Success Criteria

After implementing new system, agent should:
1. ✅ Score points more consistently (higher win rate)
2. ✅ Maintain strategic positions (no random wandering)
3. ✅ Actively pursue ball contact (higher hit rate)
4. ✅ Show anticipatory movement (position before ball arrives)
5. ✅ No jittering or action spam
6. ✅ Learn faster (clearer reward signals)

---

## Conclusion

The new reward system shifts from **proximity-based shaping** to **action-outcome based shaping**. By explicitly rewarding ball contact, strategic positioning, and purposeful movement, we create clearer learning signals that align with optimal pickleball strategy.

Key improvements:
- **Contact Reward**: Makes hitting the ball the clear subgoal
- **State-Aware Positioning**: Different strategies for different game phases
- **Purposeful Movement**: Only rewards movement that helps
- **No Time Penalties**: Removes unfair blame scenarios
- **Return Success**: Celebrates achieving the intermediate goal

This system should produce an agent that plays strategic, purposeful pickleball rather than one that jiggles near the ball waiting for points to happen.

