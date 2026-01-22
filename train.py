import gymnasium as gym
from stable_baselines3 import PPO
from nano_env import NanoBotEnv
import os

# 1. Create the environment
env = NanoBotEnv()

# 2. Define the AI Model (The Brain)
# MlpPolicy = Multi-Layer Perceptron (Standard Neural Network)
# verbose=1 allows us to see the training progress log
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("--- STARTING TRAINING ---")
print("The AI is now experimenting with magnetic fields...")

# 3. Train the model
# 50,000 timesteps is usually enough for this simple 2D physics
model.learn(total_timesteps=200000)

print("--- TRAINING COMPLETE ---")

# 4. Save the brain
model.save("nano_ai_brain")
print("Model saved as nano_ai_brain.zip")