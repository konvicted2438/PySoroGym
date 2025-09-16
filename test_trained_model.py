from stable_baselines3 import PPO
from PysoroGym.envs import PushBoxEnv
import time

def test_model(model_path="models/best_model"):
    """Test a trained model with rendering"""
    # Create environment with rendering
    env = PushBoxEnv(render_mode="human")
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run episodes
    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Initial box position: {info['box_position']}")
        print(f"Target position: {info['target_position']}")
        
        while not done:
            # Predict action using trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Print progress
            if steps % 50 == 0:
                print(f"  Step {steps}: Distance = {info['distance_to_target']:.3f}")
            
            if terminated:
                print(f"  Success! Reached target in {steps} steps")
            elif truncated:
                print(f"  Episode truncated after {steps} steps")
        
        print(f"  Total reward: {total_reward:.2f}")
        time.sleep(1)  # Pause between episodes
    
    env.close()

if __name__ == "__main__":
    # Test with best model or specific checkpoint
    test_model("models/best_model")
    # test_model("models/push_box_model_50000_steps")