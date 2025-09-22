from typing import Optional, List

from mlagents_envs.side_channel import SideChannel

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

    env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True, left_agent=left_agent_instance)
    return env
