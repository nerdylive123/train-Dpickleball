from stable_baselines3 import PPO
from train_agent import PickleballGymWrapper
import time
import numpy as np

def test_trained_agent():
    # Create environment with graphics enabled, [GANTI PATHNYA SENDIRI]
    env = PickleballGymWrapper(
        env_path=r"C:\Users\Vanessa\Downloads\dpickleball\Pickleball_Build_Training\dp.exe",
        worker_id=5005,
        no_graphics=False  # Enable graphics for visualization
    )
    
    print("Loading trained model...")
    model = PPO.load("models/pickeball_ppo")
    
    print("Starting agent test...")
    obs, _ = env.reset()
    total_reward = 0
    episode = 0
    steps = 0
    
    movement_map = {
        str([0, 0, 0]): "STAY",
        str([1, 0, 0]): "RIGHT",
        str([2, 0, 0]): "LEFT",
        str([0, 1, 0]): "UP",
        str([0, 2, 0]): "DOWN",
        str([0, 0, 1]): "HIT"
    }
    
    try:
        while True:
            # Add randomness to exploration
            if np.random.random() < 0.1:  # 10% random actions
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)  # Use stochastic actions
            
            action_desc = movement_map.get(str(action.tolist()), "UNKNOWN")
            
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps % 10 == 0:
                print(f"Episode {episode}, Step {steps}:")
                print(f"  Action: {action} ({action_desc})")
                print(f"  Reward: {reward}")
                print(f"  Total Reward: {total_reward}")
                print("-" * 40)
            
            if done:
                print(f"Episode {episode} finished with total reward: {total_reward}")
                obs, _ = env.reset()
                total_reward = 0
                episode += 1
                steps = 0
            
            time.sleep(0.05)  # Reduced delay for smoother visualization
            
    except KeyboardInterrupt:
        print("\nStopping agent test...")
    finally:
        env.close()

if __name__ == "__main__":
    test_trained_agent()
