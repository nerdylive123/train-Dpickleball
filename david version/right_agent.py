import torch
import numpy as np
import os
import json
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

from callback.inferenceTime import InferenceTimerCallback
from custom_cnn import CustomCNN
from shared_env import create_env


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

        # Get the current reward from the environment
        reward = 0
        try:
            # Try to get reward from the VecEnv
            if hasattr(self.training_env, 'buf_rews') and len(self.training_env.buf_rews) > 0:
                reward = float(self.training_env.buf_rews[-1][0])
            elif hasattr(self.training_env, 'get_attr'):
                # Try to get from wrapped environment
                env_rewards = self.training_env.get_attr('_last_reward')
                if env_rewards and env_rewards[0] is not None:
                    reward = float(env_rewards[0])
        except:
            pass

        self.current_episode_reward += reward

        # Print step rewards for debugging
        if abs(reward) > 0.01:
            print(f"Step {self.step_count}: Reward = {reward:.3f}, Episode Total = {self.current_episode_reward:.3f}")

        # Check if episode is done
        done = False
        try:
            if hasattr(self.training_env, 'buf_dones') and len(self.training_env.buf_dones) > 0:
                done = bool(self.training_env.buf_dones[-1][0])
            elif 'dones' in self.locals:
                dones = self.locals['dones']
                done = bool(dones[0]) if isinstance(dones, (list, np.ndarray)) else bool(dones)
        except:
            pass

        if done:
            # Episode finished - a game has ended (someone reached 21 points)
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.reward_window.append(self.current_episode_reward)

            # Calculate current mean reward
            current_mean = np.mean(list(self.reward_window)) if len(self.reward_window) > 0 else 0.0

            # Print episode completion
            print(f"\n🏆 EPISODE {self.episode_count} COMPLETED! 🏆")
            print(f"Episode Reward: {self.current_episode_reward:.3f}")
            print(f"Mean Reward (last {len(self.reward_window)} episodes): {current_mean:.3f}")
            print(f"Total Timesteps: {self.num_timesteps}")
            print("-" * 50)

            # Save to log file
            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_count:>7} | {self.current_episode_reward:>14.3f} | {current_mean:>18.3f} | {self.num_timesteps:>9d}\n")

            # Report detailed progress every N episodes
            if self.episode_count % self.check_freq == 0:
                self._report_detailed_progress()

            # Reset for next episode
            self.current_episode_reward = 0.0

        return True

    def _report_detailed_progress(self):
        """Report detailed training progress summary"""
        if len(self.reward_window) == 0:
            return

        # Calculate statistics
        rewards_array = np.array(list(self.reward_window))
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        min_reward = np.min(rewards_array)
        max_reward = np.max(rewards_array)

        # Create report
        episode_range = f"{self.episode_count - self.check_freq + 1:3d}-{self.episode_count:3d}"

        print("\n" + "=" * 70)
        print(f"📊 TRAINING PROGRESS SUMMARY - Episodes {episode_range} 📊")
        print(f"Mean Episode Reward: {mean_reward:8.3f} (±{std_reward:.3f})")
        print(f"Reward Range: [{min_reward:7.3f}, {max_reward:7.3f}]")
        print(f"Total Episodes Completed: {self.episode_count}")
        print(f"Total Timesteps: {self.num_timesteps}")
        print(f"Episodes per {self.check_freq} games: {self.check_freq}")

        # Check for improvement trend
        if len(self.episode_rewards) >= self.check_freq * 2:
            prev_window = self.episode_rewards[-self.check_freq * 2:-self.check_freq]
            prev_mean = np.mean(prev_window)
            improvement = mean_reward - prev_mean
            if improvement > 0.1:
                trend = "📈 IMPROVING"
            elif improvement < -0.1:
                trend = "📉 DECLINING"
            else:
                trend = "➡️ STABLE"
            print(f"Trend: {trend} (Δ{improvement:+.3f})")

        print("=" * 70 + "\n")


        # Save detailed JSON data
        self._save_detailed_data()

    def _save_detailed_data(self):
        """Save all episode data to JSON file"""
        window_list = list(self.reward_window)
        data = {
            'episode_count': self.episode_count,
            'total_timesteps': int(self.num_timesteps),
            'all_episode_rewards': self.episode_rewards,
            'recent_stats': {
                'episodes': f"{self.episode_count - self.check_freq + 1}-{self.episode_count}",
                'mean_reward': float(np.mean(window_list)) if window_list else 0.0,
                'std_reward': float(np.std(window_list)) if window_list else 0.0,
                'min_reward': float(np.min(window_list)) if window_list else 0.0,
                'max_reward': float(np.max(window_list)) if window_list else 0.0
            }
        }

        json_file = os.path.join(self.log_dir, 'episode_data.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self):
        """Get training summary"""
        # Always return a consistent dict shape
        if not self.episode_rewards:
            return {
                'total_episodes': 0,
                'overall_mean_reward': 0.0,
                'recent_mean_reward': 0.0,
                'improvement': 0.0,
            }

        total_episodes = len(self.episode_rewards)
        mean_reward = np.mean(self.episode_rewards)

        # Get recent performance
        recent_rewards = self.episode_rewards[-self.check_freq:] if len(
            self.episode_rewards) >= self.check_freq else self.episode_rewards
        recent_mean = np.mean(recent_rewards)

        return {
            'total_episodes': total_episodes,
            'overall_mean_reward': mean_reward,
            'recent_mean_reward': recent_mean,
            'improvement': recent_mean - mean_reward if total_episodes > self.check_freq else 0
        }


def train_right_agent():
    MODEL_SAVE_PATH = "right_agent_model"

    # Create log directory
    log_dir = "./training_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 1. Create the side channels
    string_channel = StringSideChannel()
    channel = CustomDataChannel()
    channel.send_data(serve=212, p1=0, p2=0)

    # 2. Initialize the Unity Environment with the side channels
    # unity_env = UnityEnvironment(
    #     ENV_PATH,
    #     worker_id=1,
    #     no_graphics=False,
    #     side_channels=[string_channel, channel]
    # )

    # 3. Wrap with SharedObsUnityGymWrapper (frame-stack + grayscale preprocessing)
    # env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True)
    env = create_env(side_channels=[string_channel, channel])

    # 4. Use the refactored custom CNN features extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # 4. Improved PPO hyperparameters for better learning
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=2,  # More verbose output
        n_steps=2048,  # Increased for more stable updates
        batch_size=64,  # Increased batch size
        n_epochs=4,  # Reduced epochs to prevent overfitting
        gamma=0.99,  # Standard discount factor
        gae_lambda=0.95,
        clip_range=0.1,  # Reduced clipping for more stable updates
        ent_coef=0.005,  # Reduced entropy for less random actions
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        learning_rate=2.5e-4,  # Standard learning rate
        use_sde=False,  # Disable stochastic domain randomization
        sde_sample_freq=-1,
        normalize_advantage=True,  # Normalize advantages
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 5. Create episode tracker with more frequent reporting
    episode_tracker = SimpleEpisodeTracker(
        check_freq=10,  # Report every 10 episodes for better monitoring
        log_dir=log_dir,
        verbose=1
    )

    print(f"Starting training on device: {model.device}")
    print("Improved hyperparameters for better learning:")
    print(f"- n_steps: {model.n_steps}")
    print(f"- batch_size: {model.batch_size}")
    print(f"- n_epochs: {model.n_epochs}")
    print(f"- learning_rate: {model.learning_rate}")
    print(f"- clip_range: {model.clip_range}")
    print(f"- ent_coef: {model.ent_coef}")
    print("Episode reward tracking enabled with frequent reporting")

    try:
        # Train with episode tracking and progress callback
        model.learn(
            total_timesteps=500_000,  # Reduced for faster testing
            callback=[episode_tracker, InferenceTimerCallback()],
            progress_bar=True  # Show the progress bar
        )

        # Save model
        model.save(MODEL_SAVE_PATH)

        # Print final summary
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