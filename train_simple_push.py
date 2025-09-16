import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from PysoroGym.envs import PushBoxEnv
import os
import torch

def train():
    """Train the soft robot to push a box to target"""
    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create environments
    # Training env without rendering for speed
    train_env = DummyVecEnv([lambda: PushBoxEnv(render_mode=None)])
    
    # Evaluation env with rendering
    eval_env = PushBoxEnv(render_mode="human")
    
    # Use CPU
    device = torch.device("cpu")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=128,        # Reduced for quick validation
        batch_size=64,
        n_epochs=4,         # Reduced for quick validation
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/",
        device=device
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="push_box_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=128,      # Evaluate after one rollout
        deterministic=True,
        render=True
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=256, # Run for a short duration (2 rollouts)
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("models/push_box_final")
    print("Training complete!")
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    train()