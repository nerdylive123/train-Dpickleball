# train-dPickleBall

Reinforcement learning for **dPickleBall**, a two-player Unity pickleball
environment. Trains a PPO agent from pixels using Stable-Baselines3 against a
built-in opponent, through a custom Gymnasium wrapper over ML-Agents.

## The setup

Two agents share one court. Agent 1 (right) is the one being trained; agent 0
(left) is the opponent.

**The environment has an observation asymmetry that shapes the whole design:**
only agent 0 emits camera observations, but agent 1 is the agent under control.
`SharedObsUnityGymWrapper` handles this — it reads frames from the observation
agent and applies them to the controlled agent, so the policy learns to play the
right-hand side from a view rendered for the left. Anything that assumes
"observation and action belong to the same agent" will silently train on the
wrong signal here.

Reward is zero-sum by construction:

```python
reward = rewards[agent_to_control] - rewards[opponent_agent]
```

Scoring against the opponent and conceding a point are therefore symmetric, and
a rally that scores for neither is worth nothing. The agent optimises margin, not
raw points.

## Pipeline

```
Unity build (dp.exe)
      │  ML-Agents side channels: StringSideChannel, CustomDataChannel (serve, p1, p2)
      ▼
UnityParallelEnv  →  SharedObsUnityGymWrapper
                       resize to 168×84  ·  optional grayscale  ·  frame stack
      ▼
SB3 PPO with CnnPolicy  →  CustomCNN feature extractor  →  checkpoints in runs/
```

### `CustomCNN`

A three-layer convolutional feature extractor (`8×8/4 → 4×4/2 → 3×3/1`, ReLU
throughout) subclassing SB3's `BaseFeaturesExtractor`. The flattened dimension is
computed with a single forward pass at construction rather than hard-coded, so
changing `--frame_stack`, `--img_size` or `--grayscale` does not require editing
the network.

### Frame stacking

Defaults to 8 frames in the training script (the wrapper's own default is 64).
A single frame cannot express ball velocity or direction, so stacking is what
makes the task learnable from pixels at all.

## Install

```bash
pip install stable-baselines3[extra] torch gym opencv-python numpy mlagents-envs
```

Needs a Unity training build of dPickleBall. It is not in this repository.

## Train

```bash
python train_simple_sb3.py \
  --executable "path/to/Pickleball_Build_Training/dp.exe" \
  --timesteps 10000 \
  --frame_stack 8 \
  --features_dim 256 \
  --out runs
```

| Flag | Default | What it does |
|---|---|---|
| `--executable` | a local Windows path | Unity training build |
| `--timesteps` | `10000` | total PPO timesteps |
| `--frame_stack` | `8` | frames stacked per observation |
| `--grayscale` | off | collapse to one channel |
| `--features_dim` | `256` | `CustomCNN` output width |
| `--out` | `runs` | checkpoint and log directory |

Each run writes to `runs/<timestamp>/`.

## Layout

```
train_simple_sb3.py       entry point — builds the env, configures PPO, trains
lib/
  unity_wrapper.py        SharedObsUnityGymWrapper: observation routing, preprocessing, reward
  custom_cnn.py           CustomCNN feature extractor
  train_agent.py          alternative training entry point
  test_trained_agent.py   load a checkpoint and play
CompabilityTesting/       competition harness — teamX/teamY scripts, shared-memory variants
complicated_ver/          RecurrentPPO right-agent training, shared env, callbacks, saved-model demo
davidversion/             reward-shaping experiments; reward_system/ is the modular reward package
mylib.py                  standalone copy of the shaped-reward wrapper
new_reward_system.md      design notes for the modular reward system
reward_system_weaknesses.md  analysis of the earlier reward shaping
PROJECT_SUMMARY.md        longer write-up of components
```

## Notes

- `--executable` defaults to a hard-coded Windows path from the original
  development machine. Pass your own; the default will not exist for you.
- `CompabilityTesting/` holds the competition-format scripts, including a
  shared-memory variant used when two independently trained agents play.
- `complicated_ver/` and `davidversion/` hold the reward-shaping work: ball and
  paddle detection from pixels, contact and out-of-bounds detection, and
  positioning rewards layered on top of the game's sparse score reward.
