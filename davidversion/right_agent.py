# right_agent.py
import os
import re
import shutil
import torch
import numpy as np
from sb3_contrib import RecurrentPPO
from shared_env import make_vector_env
from mylib import CustomCNN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from collections import defaultdict
from collections import deque


class CopyToOpponentPoolCallback(BaseCallback):
    def __init__(self, opponent_pool_dir: str, save_freq: int = 5_000, name_prefix: str = "right_agent"):
        super().__init__()
        self.opponent_pool_dir = opponent_pool_dir
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        os.makedirs(self.opponent_pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            step_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            tmp_path = step_name
            self.model.save(tmp_path)
            src = tmp_path + ".zip"
            dst = os.path.join(self.opponent_pool_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                print(f"[OpponentPool] Saved checkpoint to pool: {os.path.basename(dst)}")
            except Exception as e:
                print(f"[OpponentPool] Failed to copy {src} -> {dst}: {e}")
        return True


class ObservationMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int = 5000, buffer_size: int = 4096):
        super().__init__()
        self.check_freq = check_freq
        self.buf = deque(maxlen=buffer_size)
        self.first_check = True

    def _grab_obs(self):
        for k in ("new_obs", "obs", "observations"):
            if k in self.locals and self.locals[k] is not None:
                return self.locals[k]
        return None

    def _on_step(self) -> bool:
        obs = self._grab_obs()
        if obs is not None:
            self.buf.append(np.array(obs, copy=True))  # (n_envs, C, H, W)

        if self.first_check or (self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0):
            self._print_stats()
            self.first_check = False
        return True

    def _print_stats(self):
        if len(self.buf) == 0:
            print("\n" + "="*60)
            print("No observations collected yet.")
            print("="*60)
            return
        batch = np.concatenate(list(self.buf), axis=0)  # (N, C, H, W)
        print("\n" + "="*60)
        hdr = "INITIAL OBSERVATION CHECK" if self.first_check else f"OBSERVATION CHECK at step {self.num_timesteps}"
        print(hdr)
        print("="*60)
        print(f"Observation batch shape: {batch.shape}")
        print(f"Observation dtype: {batch.dtype}")
        print(f"Observation range: [{batch.min():.4f}, {batch.max():.4f}]")
        print(f"Observation mean: {batch.mean():.4f}")
        print(f"Observation std: {batch.std():.4f}")
        if batch.max() < 0.1:
            print("WARNING: Observations are very dark (max < 0.1) -> check preprocessing/normalization")
        elif batch.std() < 0.01:
            print("WARNING: Very low variance in observations -> images may be too uniform")
        print("="*60)

class PolicyDiversityCallback(BaseCallback):
    def __init__(self, check_freq: int = 10000, n_samples: int = 100, buffer_size: int = 4096, deterministic: bool = False):
        super().__init__()
        self.check_freq = check_freq
        self.n_samples = n_samples
        self.buf = deque(maxlen=buffer_size)
        self.deterministic = deterministic

    def _grab_obs(self):
        for k in ("new_obs", "obs", "observations"):
            if k in self.locals and self.locals[k] is not None:
                return self.locals[k]
        return None

    def _on_step(self) -> bool:
        obs = self._grab_obs()
        if obs is not None:
            self.buf.append(np.array(obs, copy=True))

        if self.num_timesteps > 0 and self.num_timesteps % self.check_freq == 0:
            self._check_diversity()
        return True

    def _check_diversity(self):
        print("\n" + "="*60)
        print(f"POLICY DIVERSITY CHECK at step {self.num_timesteps}")
        print("="*60)
        if len(self.buf) == 0:
            print("No observations in buffer yet.")
            print("="*60)
            return

        pool = np.concatenate(list(self.buf), axis=0)  # (N, C, H, W)
        N = pool.shape[0]
        K = min(self.n_samples, N)
        idx = np.random.choice(N, size=K, replace=False)
        obs_batch = pool[idx]

        # RecurrentPPO: call policy with episode_starts
        episode_starts = np.ones((obs_batch.shape[0],), dtype=bool)
        actions, _ = self.model.policy.predict(
            obs_batch, state=None, episode_start=episode_starts, deterministic=self.deterministic
        )

        counts = defaultdict(int)
        for a in np.asarray(actions):
            counts[tuple(np.atleast_1d(a).tolist())] += 1

        unique = len(counts)
        top_act, top_cnt = max(counts.items(), key=lambda x: x[1])
        top_pct = 100.0 * top_cnt / K

        print(f"Unique actions: {unique}/{K}")
        print(f"Most common action: {top_act} ({top_pct:.1f}%)")
        if unique < 5 or top_pct > 90:
            print("WARNING: Low action diversity (may be stuck policy or poor observations)")
        print("\nTop 5 actions:")
        for act, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {act}: {cnt} times ({100.0*cnt/K:.1f}%)")
        print("="*60)

class TrainingStatsCallback(BaseCallback):
    """Log detailed training statistics."""

    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode statistics
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    if 'episode' in self.locals.get('infos', [{}])[i]:
                        ep_info = self.locals['infos'][i]['episode']
                        self.episode_rewards.append(ep_info['r'])
                        self.episode_lengths.append(ep_info['l'])

        # Log statistics periodically
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])

            print(f"\n[Step {self.num_timesteps}] Training Stats:")
            print(f"  Mean reward (last 100 eps): {mean_reward:.2f}")
            print(f"  Mean length (last 100 eps): {mean_length:.1f}")
            print(f"  Total episodes: {len(self.episode_rewards)}")

        return True


def _find_latest_checkpoint(pool_dir: str, name_prefix: str = "right_agent") -> tuple[str | None, int]:
    """Return (path, timestep) of the checkpoint with the highest step count, or (None, 0) if none."""
    if not os.path.isdir(pool_dir):
        return None, 0
    best_path = None
    best_steps = 0
    pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)_steps\.zip$")
    interrupted_pattern = re.compile(rf"^{re.escape(name_prefix)}_interruptedx\.zip$")

    for fname in os.listdir(pool_dir):
        # Check for interrupted checkpoint
        if interrupted_pattern.match(fname):
            print(" Found: ", fname)
            interrupted_path = os.path.join(pool_dir, fname)
            # If we find an interrupted checkpoint, prioritize it over numbered ones
            # since it's likely the most recent
            return interrupted_path, 0

        # Check for numbered checkpoint
        m = pattern.match(fname)
        if not m:
            continue
        steps = int(m.group(1))
        if steps > best_steps:
            best_steps = steps
            best_path = os.path.join(pool_dir, fname)
    return best_path, best_steps


def train_right_agent():
    NAME_PREFIX = "right_agent"
    OPPONENT_POOL_PATH = r"opponent_pool"
    os.makedirs(OPPONENT_POOL_PATH, exist_ok=True)

    # Training curriculum parameters
    num_generations = 20
    timesteps_per_generation = 500_000

    # Common policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    cumulative_steps = 0
    latest_model_path, latest_steps = _find_latest_checkpoint(OPPONENT_POOL_PATH, NAME_PREFIX)
    cumulative_steps = latest_steps

    print(f"\n{'=' * 60}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Generations: {num_generations}")
    print(f"Steps per generation: {timesteps_per_generation:,}")
    print(f"Starting from step: {cumulative_steps:,}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'=' * 60}\n")

    for gen in range(num_generations):
        print(f"\n{'#' * 60}")
        print(f"GENERATION {gen + 1}/{num_generations}")
        print(f"{'#' * 60}\n")

        # Fresh environment each generation with dynamic opponents from the pool
        env = make_vector_env(
            num_envs=5,
            no_graphics=False,
            left_agent="predefined",
            opponent_pool_dir=OPPONENT_POOL_PATH,
        )

        # Load latest model if present, else create new
        if latest_model_path is not None and os.path.exists(latest_model_path):
            print(f"[Gen {gen + 1}] Loading model: {latest_model_path}")
            model = RecurrentPPO.load(
                latest_model_path,
                env=env,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"[Gen {gen + 1}] Model loaded successfully, continuing from {cumulative_steps:,} steps")
        else:
            print(f"[Gen {gen + 1}] No existing model found. Initializing new model.")
            model = RecurrentPPO(
                "CnnLstmPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=2,
                n_steps=1024,
                batch_size=128,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,  # Increased from 0.001 for more exploration
                learning_rate=3e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        try:
            # Create callbacks for monitoring and checkpointing
            callbacks = [
                ObservationMonitorCallback(check_freq=5000),
                PolicyDiversityCallback(check_freq=10000, n_samples=100),
                TrainingStatsCallback(log_freq=5000),
                CopyToOpponentPoolCallback(
                    opponent_pool_dir=OPPONENT_POOL_PATH,
                    save_freq=50000,
                    name_prefix=NAME_PREFIX
                )
            ]

            # Train
            print(f"\n[Gen {gen + 1}] Starting training for {timesteps_per_generation:,} steps...")
            model.learn(
                total_timesteps=timesteps_per_generation,
                progress_bar=True,
                callback=callbacks,
                reset_num_timesteps=False
            )

            # # Save VecNormalize stats if present
            # if isinstance(env, VecNormalize):
            #     vecnorm_path = "vecnormalize.pkl"
            #     env.save(vecnorm_path)
            #     print(f"[Gen {gen + 1}] Saved VecNormalize stats to {vecnorm_path}")

            # Update cumulative steps and save generation checkpoint into pool
            cumulative_steps += timesteps_per_generation
            gen_ckpt_name = f"{NAME_PREFIX}_{cumulative_steps}_steps.zip"
            gen_ckpt_path = os.path.join(OPPONENT_POOL_PATH, gen_ckpt_name)
            model.save(gen_ckpt_path[:-4])  # save() expects path without .zip
            print(f"[Gen {gen + 1}] Saved checkpoint: {gen_ckpt_path}")

            # Prepare for next generation: use the checkpoint just saved
            latest_model_path = gen_ckpt_path

        except KeyboardInterrupt:
            print(f"\n[Gen {gen + 1}] Training interrupted by user")
            model.save("./opponent_pool/right_agent_interrupted.zip")
            break
        except Exception as e:
            print(f"[Gen {gen + 1}] Training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.close()

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"Final step count: {cumulative_steps:,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train_right_agent()