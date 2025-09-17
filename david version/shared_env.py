from typing import Optional, List
from functools import partial

from mlagents_envs.side_channel import SideChannel
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

from mylib import SharedObsUnityGymWrapper
from mlagents_envs.environment import UnityEnvironment
from left_agent import LeftAgent
import random
import os

def create_env(left_agent="predefined", side_channels: Optional[List[SideChannel]] = None, worker_id=None,
               no_graphics=True, frame_stack=64, img_size=(168, 84), grayscale=True):
    """
    Create and return a SharedObsUnityGymWrapper environment.
    Args:
        left_agent (str): Type of left agent to use. Default is "predefined".
        side_channels (List[SideChannel], optional): List of side channels for Unity environment communication.
        worker_id (int, optional): Worker ID for Unity environment to avoid port conflicts. If None, a random ID will be assigned.
        no_graphics (bool): Whether to run the Unity environment in no-graphics mode. Default is True.
        frame_stack (int): Number of frames to stack for observations.
        img_size (tuple[int,int]): (W, H) resize for frames.
        grayscale (bool): Whether to convert observations to grayscale.
    Returns:
        SharedObsUnityGymWrapper: The wrapped Unity environment.
    """
    ENV_PATHS = [
        r"E:\DpickleBallEnv\PickleBallFinal\Pickleball_Build_Training\dp.exe",
        r"C:\Users\David\DPICKLEBALL COMPETITIONS\PickleBallFinal\Pickleball_Build_Training\dp.exe",
        r"C:\Users\Vanessa\Downloads\dpickleball\Pickleball_Build_Training\dp.exe"
    ]

    ENV_PATH = next((p for p in ENV_PATHS if os.path.exists(p)), ENV_PATHS[0])

    # Use a random worker_id if not specified to avoid conflicts
    if worker_id is None:
        worker_id = random.randint(1, 100)

    # Initialize the Unity Environment
    unity_env = UnityEnvironment(ENV_PATH, worker_id=worker_id, no_graphics=no_graphics, side_channels=side_channels or [])

    if left_agent == "predefined":
        left_agent_instance = LeftAgent()
    else:
        left_agent_instance = None  # Placeholder for other left agent logic

    env = SharedObsUnityGymWrapper(
        unity_env,
        frame_stack=frame_stack,
        img_size=img_size,
        grayscale=grayscale,
        left_agent=left_agent_instance,
    )
    return env


def make_env_factory(env_id: int, left_agent="predefined", no_graphics=True, base_worker_id=1000,
                     frame_stack=64, img_size=(168, 84), grayscale=True):
    """
    Factory function to create environments for VecEnv.
    Each environment gets its own worker_id and side channels to avoid conflicts.

    Args:
        env_id (int): Environment ID (0, 1, 2, ...)
        left_agent (str): Type of left agent to use
        no_graphics (bool): Whether to run without graphics
        base_worker_id (int): Base worker ID to avoid conflicts
        frame_stack (int): Number of frames to stack
        img_size (tuple[int,int]): (W,H) resize
        grayscale (bool): Grayscale observations

    Returns:
        callable: Function that creates an environment
    """
    def _init():
        # Create unique side channels for each environment
        string_channel = StringSideChannel()
        channel = CustomDataChannel()
        channel.send_data(serve=212, p1=0, p2=0)

        # Calculate unique worker_id for each environment
        worker_id = base_worker_id + env_id

        env = create_env(
            left_agent=left_agent,
            side_channels=[string_channel, channel],
            worker_id=worker_id,
            no_graphics=no_graphics,
            frame_stack=frame_stack,
            img_size=img_size,
            grayscale=grayscale,
        )
        return env

    return _init


def create_vectorized_env(n_envs=4, left_agent="predefined", no_graphics=True, use_subproc=True,
                          frame_stack=64, img_size=(168, 84), grayscale=True):
    """
    Create a vectorized environment with multiple parallel environments.

    Args:
        n_envs (int): Number of parallel environments
        left_agent (str): Type of left agent to use
        no_graphics (bool): Whether to run without graphics
        use_subproc (bool): Whether to use SubprocVecEnv (recommended) or DummyVecEnv
        frame_stack (int): Number of frames to stack
        img_size (tuple[int,int]): (W,H) resize
        grayscale (bool): Grayscale observations

    Returns:
        VecEnv: Vectorized environment
    """
    print(f"Creating {n_envs} parallel environments...")

    # Create environment factory functions
    env_fns = []
    base_worker_id = random.randint(1000, 9000)  # Random base to avoid conflicts

    for i in range(n_envs):
        env_fn = make_env_factory(
            env_id=i,
            left_agent=left_agent,
            no_graphics=no_graphics,
            base_worker_id=base_worker_id,
            frame_stack=frame_stack,
            img_size=img_size,
            grayscale=grayscale,
        )
        env_fns.append(env_fn)

    # Create vectorized environment
    if use_subproc:
        # SubprocVecEnv runs each environment in a separate process (recommended)
        vec_env = SubprocVecEnv(env_fns)
        print(f"✅ Created SubprocVecEnv with {n_envs} environments (worker IDs: {base_worker_id}-{base_worker_id + n_envs - 1})")
    else:
        # DummyVecEnv runs all environments in the same process (for debugging)
        vec_env = DummyVecEnv(env_fns)
        print(f"✅ Created DummyVecEnv with {n_envs} environments (worker IDs: {base_worker_id}-{base_worker_id + n_envs - 1})")

    return vec_env
