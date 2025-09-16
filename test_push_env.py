from PysoroGym.envs import PushBoxEnv
import numpy as np
import time

# Create environment with rendering
env = PushBoxEnv(render_mode="human")

# Reset
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial box position: {info['box_position']}")
print(f"Target position: {info['target_position']}")

# Run with random actions
for i in range(500):
    # Random action
    action = env.action_space.sample()  # Scale down for gentler movements

    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print progress every 50 steps
    if i % 50 == 0:
        print(f"Step {i}: Distance to target: {info['distance_to_target']:.3f}, Reward: {reward:.3f}")
    
    if terminated:
        print(f"Success! Box reached target in {i} steps")
        break
    
    if truncated:
        print("Episode truncated")
        break

# Hold for a moment before closing
time.sleep(2)
env.close()