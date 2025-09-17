from typing import Optional, List

from mlagents_envs.side_channel import SideChannel

from mylib import SharedObsUnityGymWrapper
from mlagents_envs.environment import UnityEnvironment
from left_agent import LeftAgent

def create_env(left_agent="predefined", side_channels: Optional[List[SideChannel]] = None, worker_id=None):
    import random
    import os

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
    unity_env = UnityEnvironment(ENV_PATH, worker_id=worker_id, no_graphics=False, side_channels=side_channels or [])

    if left_agent == "predefined":
        left_agent_instance = LeftAgent()
    else:
        left_agent_instance = None  # Placeholder for other left agent logic

    env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True, left_agent=left_agent_instance)
    return env
