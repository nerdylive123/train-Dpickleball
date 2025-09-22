import torch
import numpy as np
import os
import json
from collections import deque
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

from callback.inferenceTime import InferenceTimerCallback
from mylib import PickleballCNN, ImprovedPickleballCNN
from shared_env import create_env, create_vectorized_env

# Performance knobs for faster inference on CUDA
# try:
#     torch.backends.cudnn.benchmark = True
#     if hasattr(torch.backends, 'cuda'):
#         torch.backends.cuda.matmul.allow_tf32 = True
#     if hasattr(torch.backends, 'cudnn'):
#         torch.backends.cudnn.allow_tf32 = True
#     if hasattr(torch, 'set_float32_matmul_precision'):
#         torch.set_float32_matmul_precision('high')
# except Exception:
#     pass


class SimpleEpisodeTracker(BaseCallback):
    """
    Enhanced episode reward tracker that properly detects game completion (21 points)
    """

    def __init__(self, check_freq=10, log_dir='./training_logs/', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

        # Episode tracking
        self.episode_rewards = []
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.step_count = 0

        # Running window for averages
        self.reward_window = deque(maxlen=check_freq)

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_progress.txt')

        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write("Episode | Episode Reward | Mean Reward (last 10) | Timesteps\n")
            f.write("-" * 60 + "\n")

        print(f"Episode tracking enabled - reporting every episode and summary every {check_freq} episodes")
        print(f"Progress log: {self.log_file}")

    def _on_step(self) -> bool:
        self.step_count += 1

        # Get the current reward from the environment (first env if vectorized)
        reward = 0.0
        try:
            if 'rewards' in self.locals:
                r = self.locals['rewards']
                if isinstance(r, (list, np.ndarray)) and len(r) > 0:
                    reward = float(r[0])
                elif isinstance(r, (float, int)):
                    reward = float(r)
            else:
                vals = getattr(self.training_env, 'get_attr', lambda name: [0.0])('_last_reward')
                if vals and vals[0] is not None:
                    reward = float(vals[0])
        except Exception:
            pass

        self.current_episode_reward += reward

        # Print step rewards for debugging
        if abs(reward) > 0.01:
            print(f"Step {self.step_count}: Reward = {reward:.3f}, Episode Reward = {self.current_episode_reward:.3f}")

        # Check if episode is done (first env if vectorized)
        done = False
        try:
            if 'dones' in self.locals:
                d = self.locals['dones']
                if isinstance(d, (list, np.ndarray)) and len(d) > 0:
                    done = bool(d[0])
                elif isinstance(d, (bool, np.bool_)):
                    done = bool(d)
        except Exception:
            pass

        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.reward_window.append(self.current_episode_reward)

            window_list = list(self.reward_window)
            current_mean = np.mean(window_list) if len(window_list) > 0 else 0.0

            print(f"\n🏆 EPISODE {self.episode_count} COMPLETED! 🏆")
            print(f"Episode Reward: {self.current_episode_reward:.3f}")
            print(f"Mean Reward (last {len(self.reward_window)} episodes): {current_mean:.3f}")
            print(f"Total Timesteps: {self.num_timesteps}")
            print("-" * 50)

            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_count:>7} | {self.current_episode_reward:>14.3f} | {current_mean:>18.3f} | {self.num_timesteps:>9d}\n")

            if self.episode_count % self.check_freq == 0:
                self._report_detailed_progress()

            self.current_episode_reward = 0.0

        return True

    def _report_detailed_progress(self):
        if len(self.reward_window) == 0:
            return

        rewards_array = np.array(list(self.reward_window), dtype=np.float32)
        mean_reward = float(np.mean(rewards_array))
        std_reward = float(np.std(rewards_array))
        min_reward = float(np.min(rewards_array))
        max_reward = float(np.max(rewards_array))

        episode_range = f"{self.episode_count - self.check_freq + 1:3d}-{self.episode_count:3d}"

        print("\n" + "=" * 70)
        print(f"📊 TRAINING PROGRESS SUMMARY - Episodes {episode_range} 📊")
        print(f"Mean Episode Reward: {mean_reward:8.3f} (±{std_reward:.3f})")
        print(f"Reward Range: [{min_reward:7.3f}, {max_reward:7.3f}]")
        print(f"Total Episodes Completed: {self.episode_count}")
        print(f"Total Timesteps: {self.num_timesteps}")
        print(f"Episodes per {self.check_freq} games: {self.check_freq}")

        if len(self.episode_rewards) >= self.check_freq * 2:
            prev_window = list(self.episode_rewards)[-self.check_freq * 2:-self.check_freq]
            prev_mean = float(np.mean(prev_window)) if len(prev_window) > 0 else 0.0
            improvement = mean_reward - prev_mean
            if improvement > 0.1:
                trend = "📈 IMPROVING"
            elif improvement < -0.1:
                trend = "📉 DECLINING"
            else:
                trend = "➡️ STABLE"
            print(f"Trend: {trend} (Δ{improvement:+.3f})")

        print("=" * 70 + "\n")

        self._save_detailed_data()

    def _save_detailed_data(self):
        data = {
            'episode_count': int(self.episode_count),
            'total_timesteps': int(self.num_timesteps),
            'all_episode_rewards': list(self.episode_rewards),
            'recent_stats': {
                'episodes': f"{self.episode_count - self.check_freq + 1}-{self.episode_count}",
                'mean_reward': float(np.mean(list(self.reward_window))) if len(self.reward_window) > 0 else 0.0,
                'std_reward': float(np.std(list(self.reward_window))) if len(self.reward_window) > 0 else 0.0,
                'min_reward': float(np.min(list(self.reward_window))) if len(self.reward_window) > 0 else 0.0,
                'max_reward': float(np.max(list(self.reward_window))) if len(self.reward_window) > 0 else 0.0,
            }
        }

        json_file = os.path.join(self.log_dir, 'episode_data.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self):
        if not self.episode_rewards:
            return {
                'total_episodes': 0,
                'overall_mean_reward': 0.0,
                'recent_mean_reward': 0.0,
                'improvement': 0.0,
            }

        total_episodes = len(self.episode_rewards)
        mean_reward = float(np.mean(list(self.episode_rewards)))

        recent_rewards = list(self.episode_rewards)[-self.check_freq:] if len(self.episode_rewards) >= self.check_freq else list(self.episode_rewards)
        recent_mean = float(np.mean(recent_rewards)) if len(recent_rewards) > 0 else 0.0

        return {
            'total_episodes': total_episodes,
            'overall_mean_reward': mean_reward,
            'recent_mean_reward': recent_mean,
            'improvement': recent_mean - mean_reward if total_episodes > self.check_freq else 0.0
        }


def train_right_agent(n_envs=2, use_vecenv=True, low_latency=False,
                      frame_stack=None, img_size=None, grayscale=True):
    MODEL_SAVE_PATH = "right_agent_model"

    # Defaults tuned for latency when not explicitly provided
    if frame_stack is None:
        frame_stack = 6 if low_latency else 8
    if img_size is None:
        # Updated for better pickleball gameplay with spin detection
        img_size = (192, 108) if low_latency else (256, 144)  # 16:9 aspect ratio, better resolution

    # Create log directory
    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment based on configuration
    if use_vecenv and n_envs > 1:
        print(f"🚀 Using VecEnv with {n_envs} parallel environments for faster training!")
        env = create_vectorized_env(
            n_envs=n_envs,
            left_agent="predefined",
            no_graphics=True,
            use_subproc=True,
            frame_stack=frame_stack,
            img_size=img_size,
            grayscale=grayscale,
        )
        n_steps_per_env = 256 if low_latency else 512
        batch_size = 128 if low_latency else 64
    else:
        print("Using single environment (standard training)")
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=212, p1=0, p2=0)

        env = create_env(
            left_agent="predefined",
            side_channels=[string_channel, channel],
            no_graphics=True,
            frame_stack=frame_stack,
            img_size=img_size,
            grayscale=grayscale,
        )
        n_steps_per_env = 512 if low_latency else 1024
        batch_size = 64 if low_latency else 32

    # Choose features extractor
    if low_latency:
        features_extractor_class = PickleballCNN
        features_extractor_kwargs = dict(features_dim=256, use_amp=True, use_channels_last=True)
        net_arch = [128]
    else:
        # Use ImprovedPickleballCNN for better parameter efficiency
        features_extractor_class = ImprovedPickleballCNN
        features_extractor_kwargs = dict(
            features_dim=384,
            use_amp=True,
            use_temporal_compression=True,
            use_batchnorm=True
        )
        net_arch = [256]  # Smaller since CNN head is already efficient

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
        net_arch=net_arch,
    )

    model = RecurrentPPO(
        env=env,
        policy="CnnLstmPolicy",
        policy_kwargs=policy_kwargs,
        verbose=2,
        n_steps=n_steps_per_env,
        batch_size=batch_size,
        n_epochs=5 if low_latency else 10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05 if low_latency else 0.01,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "auto",
    )

    episode_tracker = SimpleEpisodeTracker(check_freq=10, log_dir=log_dir, verbose=1)
    infTime = InferenceTimerCallback(target_ms=10, window=2000, verbose=1)

    print(f"Starting training on device: {model.device}")
    if use_vecenv and n_envs > 1:
        print(f"🔥 VECTORIZED TRAINING with {n_envs} environments:")
        print(f"- Total steps per update: {n_steps_per_env * n_envs}")
        print(f"- Steps per environment: {n_steps_per_env}")
    print("Training hyperparameters:")
    print(f"- n_steps: {model.n_steps}")
    print(f"- batch_size: {model.batch_size}")
    print(f"- n_epochs: {model.n_epochs}")
    print(f"- learning_rate: {model.learning_rate}")
    print(f"- clip_range: {model.clip_range}")
    print(f"- ent_coef: {model.ent_coef}")
    print(f"- frame_stack: {frame_stack}, img_size: {img_size}, grayscale: {grayscale}")
    print(f"- Features: {features_extractor_class.__name__}, net_arch: {net_arch}")

    try:
        model.learn(total_timesteps=500_000, callback=[episode_tracker, infTime], progress_bar=True)
        model.save(MODEL_SAVE_PATH)

        summary = episode_tracker.get_summary()
        print("\n" + "=" * 50)
        print("TRAINING COMPLETE")
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Overall Mean Reward: {summary['overall_mean_reward']:.3f}")
        print(f"Recent Mean Reward: {summary['recent_mean_reward']:.3f}")
        if summary['improvement'] != 0:
            trend = "improvement" if summary['improvement'] > 0 else "decline"
            print(f"Overall {trend}: {summary['improvement']:+.3f}")
        print(f"Model saved: {MODEL_SAVE_PATH}.zip")
        print(f"Training logs: {log_dir}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(MODEL_SAVE_PATH + "_interrupted")
        print(f"Model saved: {MODEL_SAVE_PATH}_interrupted.zip")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    train_right_agent()