# right_agent.py
import os
import re
import shutil
import torch
from sb3_contrib import RecurrentPPO
from shared_env import make_vector_env
from mylib import CustomCNN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class CopyToOpponentPoolCallback(BaseCallback):
    def __init__(self, opponent_pool_dir: str, save_freq: int = 200_000, name_prefix: str = "right_agent"):
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
            except Exception as e:
                print(f"[OpponentPool] Failed to copy {src} -> {dst}: {e}")
        return True


def _find_latest_checkpoint(pool_dir: str, name_prefix: str = "right_agent") -> tuple[str | None, int]:
    """Return (path, timestep) of the checkpoint with the highest step count, or (None, 0) if none."""
    if not os.path.isdir(pool_dir):
        return None, 0
    best_path = None
    best_steps = 0
    pattern = re.compile(rf"^{re.escape(name_prefix)}_(\d+)_steps\.zip$")
    for fname in os.listdir(pool_dir):
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

    for gen in range(num_generations):
        # Fresh environment each generation with dynamic opponents from the pool
        env = make_vector_env(
            num_envs=1,
            no_graphics=False,
            left_agent="predefined",
            opponent_pool_dir=OPPONENT_POOL_PATH,
        )

        # Load latest model if present, else create new
        if latest_model_path is not None and os.path.exists(latest_model_path):
            print(f"[Gen {gen+1}] Loading model: {latest_model_path}")
            model = RecurrentPPO.load(latest_model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            print(f"[Gen {gen+1}] No existing model found. Initializing new model.")
            model = RecurrentPPO(
                "CnnLstmPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=2,
                n_steps=1024,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                learning_rate=3e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        try:
            # Optional: also update pool mid-generation
            copy_pool_cb = CopyToOpponentPoolCallback(opponent_pool_dir=OPPONENT_POOL_PATH, save_freq=200_000, name_prefix=NAME_PREFIX)
            model.learn(total_timesteps=timesteps_per_generation, progress_bar=True, callback=copy_pool_cb, reset_num_timesteps=False)

            # Save VecNormalize stats if present
            if isinstance(env, VecNormalize):
                env.save("vecnormalize.pkl")

            # Update cumulative steps and save generation checkpoint into pool
            cumulative_steps += timesteps_per_generation
            gen_ckpt_name = f"{NAME_PREFIX}_{cumulative_steps}_steps.zip"
            gen_ckpt_path = os.path.join(OPPONENT_POOL_PATH, gen_ckpt_name)
            model.save(gen_ckpt_path[:-4])  # save() expects path without .zip
            print(f"[Gen {gen+1}] Saved checkpoint: {gen_ckpt_path}")

            # Prepare for next generation: use the checkpoint just saved
            latest_model_path = gen_ckpt_path
        except Exception as e:
            print(f"[Gen {gen+1}] Training failed: {e}")
        finally:
            env.close()


if __name__ == "__main__":
    train_right_agent()