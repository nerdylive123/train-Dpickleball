from mylib import SharedObsUnityGymWrapper
from mlagents_envs.environment import UnityEnvironment
from left_agent import LeftAgent

def create_env(left_agent="predefined", worker_id=None):
    import random
    ENV_PATH = r"E:\DpickleBallEnv\PickleBallFinal\Pickleball_Build_Training\dp.exe"

    # Use a random worker_id if not specified to avoid conflicts
    if worker_id is None:
        worker_id = random.randint(1, 100)

    # Initialize the Unity Environment
    unity_env = UnityEnvironment(ENV_PATH, worker_id=worker_id, no_graphics=False)

    if left_agent == "predefined":
        left_agent_instance = LeftAgent()
    else:
        left_agent_instance = None  # Placeholder for other left agent logic

    env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True, left_agent=left_agent_instance)
    return env
