#!/usr/bin/env python3
"""
Interactive demo to load and watch a saved model in action
"""

import os
import time
import torch
from stable_baselines3 import PPO
from shared_env import create_env, create_vectorized_env
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
device = "cuda" if torch.cuda.is_available() else "cpu"

def find_saved_model():
    """Find the most recent saved model"""
    model_files = []

    # Look for model files in current directory
    for file in os.listdir('.'):
        if file.endswith('.zip') and ('right_agent' in file or 'model' in file):
            model_files.append(file)

    if not model_files:
        print("❌ No saved models found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.zip'):
                print(f"  - {file}")
        return None

    # Sort by modification time, newest first
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print("📁 Found saved models:")
    for i, model_file in enumerate(model_files):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        mod_time = time.ctime(os.path.getmtime(model_file))
        print(f"  {i+1}. {model_file} ({size_mb:.1f} MB, modified: {mod_time})")

    return model_files[0]  # Return the newest


def get_model_env_params(model_path):
    """Load model and extract environment parameters from its observation space"""
    temp_model = PPO.load(model_path, device=device)

    obs_shape = temp_model.observation_space.shape
    print(f"📐 Model observation space: {obs_shape}")

    # Infer parameters from observation shape (frame_stack, height, width)
    frame_stack = obs_shape[0]
    height = obs_shape[1]
    width = obs_shape[2]
    img_size = (width, height)  # create_env expects (width, height)

    print(f"🔧 Inferred parameters: frame_stack={frame_stack}, img_size={img_size}")

    return {
        'frame_stack': frame_stack,
        'img_size': img_size,
        'grayscale': True,  # Assuming grayscale based on single channel per frame
    }


def interactive_model_demo():
    """Interactive demo to load and watch a model"""
    print("🎮 INTERACTIVE MODEL DEMO")
    print("=" * 50)

    model_path = find_saved_model()
    if not model_path:
        return False

    model_name = model_path.replace('.zip', '')
    print(f"\n🔄 Loading model: {model_name}")

    try:
        # Get environment parameters from the saved model
        print("🧠 Analyzing model observation space...")
        env_params = get_model_env_params(model_path)

        # Create environment with matching parameters
        print("🌍 Creating test environment...")
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=212, p1=0, p2=0)

        env = create_env(
            left_agent="predefined",
            side_channels=[string_channel, channel],
            no_graphics=False,  # Show graphics for demo!
            **env_params
        )

        # Load the model with the correctly configured environment
        print("🧠 Loading model with matching environment...")
        model = PPO.load(model_name, env=env, device=device)
        print(f"   ✅ Model loaded successfully!")
        print(f"   Device: {model.device}")
        print(f"   Policy: {type(model.policy).__name__}")
        print(f"   Environment observation space: {env.observation_space}")

        # Interactive demo loop
        print("\n🎯 Starting interactive demo...")
        print("Press Ctrl+C to stop the demo")
        print("-" * 50)

        # Target FPS (0 disables pacing)
        target_fps = int(os.getenv("DEMO_FPS", "60"))
        if target_fps > 0:
            print(f"⏱️  Capping to ~{target_fps} FPS (set DEMO_FPS=0 to uncap)")
        else:
            print("⏱️  No FPS cap (DEMO_FPS=0)")
        frame_time = 1.0 / target_fps if target_fps > 0 else 0.0

        obs, _ = env.reset()
        episode_count = 0
        step_count = 0
        episode_reward = 0.0
        total_reward = 0.0

        while True:
            try:
                t0 = time.perf_counter()
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=False)

                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)

                step_count += 1
                episode_reward += reward
                total_reward += reward

                # Print step info
                if reward != 0:
                    print(f"Step {step_count}: Action={action}, Reward={reward:.3f} ⭐")
                elif step_count % 50 == 0:
                    print(f"Step {step_count}: Action={action}, Reward={reward:.3f}")

                # Handle episode end
                if done:
                    episode_count += 1
                    print(f"\n🏆 EPISODE {episode_count} COMPLETED!")
                    print(f"   Episode reward: {episode_reward:.3f}")
                    print(f"   Episode steps: {step_count}")
                    print(f"   Total reward: {total_reward:.3f}")
                    print("-" * 40)

                    # Reset for next episode
                    obs, _ = env.reset()
                    episode_reward = 0.0
                    step_count = 0
                    time.sleep(1)

                # Pace to target FPS
                if target_fps > 0:
                    elapsed = time.perf_counter() - t0
                    remaining = frame_time - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            except KeyboardInterrupt:
                print(f"\n⏹️  Demo stopped by user")
                print(f"📊 Final stats:")
                print(f"   Episodes completed: {episode_count}")
                print(f"   Total steps: {step_count}")
                print(f"   Total reward: {total_reward:.3f}")
                break

        env.close()
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_inference_demo():
    """Demo batch inference with VecEnv"""
    print("\n🚀 BATCH INFERENCE DEMO")
    print("=" * 50)

    model_path = find_saved_model()
    if not model_path:
        return False

    model_name = model_path.replace('.zip', '')
    print(f"🔄 Loading model for batch inference: {model_name}")

    try:
        # Get environment parameters from the saved model
        print("🧠 Analyzing model observation space...")
        env_params = get_model_env_params(model_path)

        # Create VecEnv with matching parameters
        print("🌍 Creating vectorized environment...")
        vec_env = create_vectorized_env(
            n_envs=4,
            use_subproc=True,
            **env_params
        )

        # Load model with matching environment
        print("🧠 Loading model into VecEnv...")
        model = PPO.load(model_name, env=vec_env, device=device)
        print(f"   ✅ Model loaded for batch inference!")
        print(f"   Number of environments: {vec_env.num_envs}")

        # Run batch inference
        print("\n🎯 Running batch inference demo...")
        obs = vec_env.reset()

        for step in range(20):
            # Get actions for all environments
            actions, _states = model.predict(obs, deterministic=False)

            # Step all environments
            obs, rewards, dones, infos = vec_env.step(actions)

            # Print batch results
            print(f"Batch step {step+1}:")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones}")
            print()

        vec_env.close()
        return True

    except Exception as e:
        print(f"❌ Batch demo failed: {e}")
        return False


def model_analysis():
    """Analyze the loaded model"""
    print("\n🔬 MODEL ANALYSIS")
    print("=" * 50)

    model_path = find_saved_model()
    if not model_path:
        return False

    model_name = model_path.replace('.zip', '')

    try:
        # Get environment parameters from the saved model
        print("🧠 Analyzing model observation space...")
        env_params = get_model_env_params(model_path)

        # Create minimal environment for analysis
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=212, p1=0, p2=0)

        env = create_env(
            left_agent="predefined",
            side_channels=[string_channel, channel],
            no_graphics=True,
            **env_params
        )

        # Load model with matching environment
        model = PPO.load(model_name, env=env, device=device)

        print(f"📋 Model Information:")
        print(f"   Model file: {model_path}")
        print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        print(f"   Device: {model.device}")
        print(f"   Policy type: {type(model.policy).__name__}")

        # Analyze policy architecture
        if hasattr(model.policy, 'features_extractor'):
            features_extractor = model.policy.features_extractor
            print(f"   Features extractor: {type(features_extractor).__name__}")
            if hasattr(features_extractor, 'features_dim'):
                print(f"   Features dimension: {features_extractor.features_dim}")

        # Get observation and action spaces
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Test inference speed
        print(f"\n⚡ Testing inference speed...")
        obs, _ = env.reset()

        # Warmup
        for _ in range(5):
            model.predict(obs, deterministic=True)

        # Time inference
        start_time = time.time()
        for _ in range(100):
            action, _states = model.predict(obs, deterministic=True)
        end_time = time.time()

        avg_time_ms = (end_time - start_time) * 1000 / 100
        print(f"   Average inference time: {avg_time_ms:.2f} ms")
        print(f"   Inference rate: {1000/avg_time_ms:.1f} FPS")

        env.close()
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def main():
    """Main demo function with user choices"""
    print("🎮 SAVED MODEL DEMO SUITE")
    print("=" * 60)

    while True:
        print("\nChoose a demo:")
        print("1. 🎯 Interactive Demo (watch model play with graphics)")
        print("2. 🚀 Batch Inference Demo (VecEnv)")
        print("3. 🔬 Model Analysis")
        print("4. 🔄 All Demos")
        print("5. ❌ Exit")

        try:
            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                print("\n" + "="*60)
                interactive_model_demo()
            elif choice == '2':
                print("\n" + "="*60)
                batch_inference_demo()
            elif choice == '3':
                print("\n" + "="*60)
                model_analysis()
            elif choice == '4':
                print("\n" + "="*60)
                model_analysis()
                batch_inference_demo()
                interactive_model_demo()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
