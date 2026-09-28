# shared_env.py
import os
import random
import time
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # Added DummyVecEnv for single-env testing
from stable_baselines3.common.vec_env import VecNormalize
from mlagents_envs.side_channel import SideChannel
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv  # Add missing import
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
try:
    from davidversion.mylib_v2 import SharedObsUnityGymWrapper, CustomCNN
    print("Using new modular reward system")
except:
    from davidversion.mylib import SharedObsUnityGymWrapper, CustomCNN
    print("Using old reward system")
from left_agent import LeftAgent, ModelLeftAgent, RobustModelLeftAgent


def create_env(left_agent="predefined", side_channels=None, worker_id=None, no_graphics=False):
    ENV_PATH = r"E:\DpickleBallEnv\PickleBallFinal\Pickleball_Build_Training\dp.exe"
    if not os.path.exists(ENV_PATH):
        raise FileNotFoundError(f"Unity executable not found at {ENV_PATH}")
    if worker_id is None:
        worker_id = random.randint(1, 1000)
    try:
        unity_env = UnityEnvironment(ENV_PATH, worker_id=worker_id, no_graphics=no_graphics,
                                     side_channels=side_channels or [])
    except Exception as e:
        raise RuntimeError(f"Failed to initialize UnityEnvironment with path {ENV_PATH}: {e}")

    if isinstance(left_agent, str) and left_agent != "predefined":
        # Instantiate robust model-based left opponent
        left_agent_instance = RobustModelLeftAgent(left_agent, 'cuda', debug=False,
                                                   debug_dir='davidversion/debug_frames/left')
    elif left_agent == "predefined":
        left_agent_instance = LeftAgent()
    else:
        left_agent_instance = None
    env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True, left_agent=left_agent_instance)
    return env


def make_vector_env(num_envs=30, base_worker_id=1, no_graphics=False, left_agent="predefined", opponent_pool_dir: str |
                                                                                                              None = None):
    # Build a list of opponent model paths if a pool directory is provided
    opponent_models = []
    if opponent_pool_dir is not None and os.path.isdir(opponent_pool_dir):
        for name in os.listdir(opponent_pool_dir):
            if name.lower().endswith(".zip") and name.startswith("left_agent_"):
                opponent_models.append(os.path.join(opponent_pool_dir, name))

    def pick_opponent():
        if opponent_models:
            return random.choice(opponent_models)
        return "predefined"

    def make_thunk(rank):
        def _init():
            time.sleep(0.5 * rank)
            string_channel = StringSideChannel()
            data_channel = CustomDataChannel()
            data_channel.send_data(serve=212, p1=0, p2=0)
            side_channels = [string_channel, data_channel]
            opponent = "opponent_pool/left_agent_600000_steps"
            env = create_env(side_channels=side_channels, worker_id=base_worker_id + rank * 5,
                             no_graphics=no_graphics, left_agent=opponent)
            return env

        return _init

    if num_envs == 1:
        vec_env_class = DummyVecEnv
        vec_env = vec_env_class([make_thunk(i) for i in range(num_envs)])
    else:
        vec_env_class = SubprocVecEnv
        vec_env = vec_env_class([make_thunk(i) for i in range(num_envs)], start_method="spawn")

    vec_env = VecMonitor(vec_env)
    # Apply normalization wrapper for more stable learning; clip rewards moderately
    # vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return vec_env
