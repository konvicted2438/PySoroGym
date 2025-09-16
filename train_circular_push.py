import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PysoroGym.envs import CircularPushBoxEnv
import os
import torch

def train():
    """Train the soft robot to push a box in a circular path"""
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([lambda: CircularPushBoxEnv(render_mode="human")])
    
    # Force CPU to avoid the warning
    device = torch.device("cpu")
    
    # Create model with tuned hyperparameters for this task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="./logs/",
        device=device
    )
    
    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="circular_push_model"
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=300000,  # More steps for complex trajectory
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/circular_push_final")
    print("Training complete!")
    
    env.close()

if __name__ == "__main__":
    train()