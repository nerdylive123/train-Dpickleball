# dPickleBall Training Project Summary

## Overview

This project implements a reinforcement learning training pipeline for a Unity-based pickleball game called "dPickleBall". The system uses **Stable Baselines3 (SB3)** with **Proximal Policy Optimization (PPO)** to train an AI agent to play pickleball against an opponent.

## Project Structure

```
train-Dpickleball/
├── train_simple_sb3.py          # Main training script
├── lib/                         # Custom library package
│   ├── __init__.py             # Package initialization
│   ├── custom_cnn.py           # Custom CNN feature extractor
│   └── unity_wrapper.py        # Unity environment wrapper
├── runs/                       # Training output directory
└── __pycache__/               # Python cache files
```

## Core Components

### 1. Main Training Script (`train_simple_sb3.py`)

**Purpose**: Entry point for training the PPO agent on the dPickleBall environment.

**Key Features**:
- Command-line interface with configurable parameters
- Integration with Unity ML-Agents environment
- Custom side channels for game state communication
- Automatic model checkpointing and TensorBoard logging

**Command Line Arguments**:
- `--executable`: Path to Unity training build (default: local path)
- `--timesteps`: Total training timesteps (default: 10,000)
- `--frame_stack`: Number of frames to stack (default: 8)
- `--grayscale`: Enable grayscale observations
- `--features_dim`: CNN feature dimensions (default: 256)
- `--out`: Output directory for runs (default: "runs")

**Training Configuration**:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: CnnPolicy with custom feature extractor
- **Batch Size**: 128
- **Learning Rate**: 2.5e-4
- **Steps per Update**: 512
- **Device**: CUDA (GPU acceleration)

### 2. Unity Environment Wrapper (`lib/unity_wrapper.py`)

**Purpose**: Bridges Unity ML-Agents environment with gym interface for SB3 compatibility.

**Class**: `SharedObsUnityGymWrapper`

**Key Functionality**:

#### Agent Configuration
- **Agent to Control**: Agent 1 (right player) - the learning agent
- **Opponent Agent**: Agent 0 (left player) - the opponent
- **Observation Source**: Agent 0 provides visual observations for both agents

#### Observation Processing
- **Image Preprocessing**: Resize to configurable dimensions (default: 168x84)
- **Frame Stacking**: Combines multiple consecutive frames (default: 64 frames)
- **Grayscale Conversion**: Optional color-to-grayscale conversion
- **Normalization**: Pixel values normalized to [0, 1] range
- **Format**: Handles Unity's (C, H, W) to standard (H, W, C) conversion

#### Reward Engineering
- **Differential Reward**: `reward = agent_reward - opponent_reward`
- **Goal**: Encourages winning over the opponent rather than just scoring
- **Logging**: Prints rewards when scoring events occur

#### Observation Space
- **Type**: Box space with float32 values
- **Shape**: 
  - Grayscale: `(frame_stack, height, width)`
  - Color: `(frame_stack * channels, height, width)`
- **Range**: [0.0, 1.0]

### 3. Custom CNN Architecture (`lib/custom_cnn.py`)

**Purpose**: Specialized convolutional neural network for processing stacked frame observations.

**Class**: `CustomCNN` (extends `BaseFeaturesExtractor`)

**Architecture**:
```
Input: Stacked frames (e.g., 8×84×168 for grayscale)
│
├── Conv2d(in_channels, 32, kernel=8, stride=4) + ReLU
├── Conv2d(32, 64, kernel=4, stride=2) + ReLU  
├── Conv2d(64, 64, kernel=3, stride=1) + ReLU
├── Flatten()
└── Linear(computed_size, features_dim) + ReLU
│
Output: Feature vector (default: 512 dimensions)
```

**Key Features**:
- **Dynamic Shape Calculation**: Automatically computes flattened layer size
- **Configurable Output**: Adjustable feature dimensions
- **GPU Compatible**: Uses PyTorch tensors for CUDA acceleration
- **SB3 Integration**: Inherits from SB3's `BaseFeaturesExtractor`

### 4. Library Package (`lib/__init__.py`)

**Purpose**: Clean package interface for importing key components.

**Exports**:
- `SharedObsUnityGymWrapper`: Environment wrapper
- `CustomCNN`: Feature extractor

## Training Workflow

### 1. Environment Setup
```python
# Initialize Unity environment with custom channels
unity_env = UnityEnvironment(executable_path, side_channels=[...])

# Wrap with gym interface
env = SharedObsUnityGymWrapper(
    unity_env, 
    frame_stack=8, 
    img_size=(168, 84), 
    grayscale=True
)
```

### 2. Model Configuration
```python
# Configure PPO with custom CNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    # ... other hyperparameters
)
```

### 3. Training Execution
```python
# Train with specified timesteps
model.learn(total_timesteps=10000)

# Save trained model
model.save("ppo_dpickleball")
```

## Game Environment Details

### Unity Side Channels
- **StringSideChannel**: For string-based communication
- **CustomDataChannel**: For numerical game state (serve, scores)
- **Initial State**: `serve=212, p1=0, p2=0`

### Visual Observations
- **Source**: Camera view from the environment
- **Format**: RGB images from Unity cameras
- **Processing**: Resized, optionally converted to grayscale, and stacked
- **Temporal Component**: Multiple frames provide motion information

### Action Space
- **Type**: Determined by Unity environment configuration
- **Target**: Controls the right player (Agent 1)
- **Opponent**: Left player (Agent 0) behavior controlled by Unity

## Technical Requirements

### Dependencies
```bash
pip install stable-baselines3 torch gymnasium opencv-python numpy mlagents-envs
```

### Hardware Requirements
- **GPU**: CUDA-compatible for training acceleration
- **Memory**: Sufficient for frame stacking and CNN processing
- **Storage**: Space for model checkpoints and TensorBoard logs

## Output and Monitoring

### Directory Structure
```
runs/
└── YYYYMMDD_HHMMSS/          # Timestamped run directory
    ├── tb/                   # TensorBoard logs
    └── ppo_dpickleball.zip   # Saved model checkpoint
```

### Monitoring
- **TensorBoard**: Real-time training metrics and loss curves
- **Console Output**: Training progress and reward information
- **Model Checkpoints**: Automatic saving at training completion

## Usage Examples

### Basic Training
```bash
python train_simple_sb3.py --timesteps 50000
```

### Custom Configuration
```bash
python train_simple_sb3.py \
    --executable "path/to/unity/build.exe" \
    --timesteps 100000 \
    --frame_stack 4 \
    --grayscale \
    --features_dim 512 \
    --out "my_experiments"
```

## Key Design Decisions

1. **Shared Observations**: Uses opponent's visual input for both agents to ensure consistent observation quality
2. **Differential Rewards**: Competitive reward structure encourages beating the opponent
3. **Frame Stacking**: Provides temporal context for better decision making
4. **Modular Architecture**: Separates concerns between training script, environment wrapper, and neural network
5. **Configurable Parameters**: Allows experimentation with different hyperparameters and settings

This architecture provides a flexible and robust foundation for training reinforcement learning agents on the dPickleBall environment while maintaining clean separation of concerns and easy extensibility.
