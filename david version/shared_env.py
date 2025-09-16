from mylib import SharedObsUnityGymWrapper
from mlagents_envs.environment import UnityEnvironment
from left_agent import LeftAgent

def create_env(left_agent="predefined"):
    ENV_PATH = r"C:\Users\David\DPICKLEBALL COMPETITIONS\PickleBallFinal\Pickleball_Build_Training\dp.exe"
    unity_env = UnityEnvironment(ENV_PATH, worker_id=1, no_graphics=False)

    if left_agent == "predefined":
        left_agent_instance = LeftAgent()
    else:
        left_agent_instance = None  # Placeholder for other left agent logic

    env = SharedObsUnityGymWrapper(unity_env, frame_stack=4, grayscale=True, left_agent=left_agent_instance)
    return env
