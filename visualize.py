import gymnasium as gym
from stable_baselines3 import PPO
from nano_env import NanoBotEnv
import matplotlib.pyplot as plt
import numpy as np

env = NanoBotEnv()
model = PPO.load("nano_ai_brain")

# 1. Reset and Grab Target
obs, info = env.reset()
real_target = env.target_pos 

# 2. Run Sim
done = False
path_x, path_y = [obs[0]], [obs[1]]
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    path_x.append(obs[0])
    path_y.append(obs[1])

# 3. Create the Scent Map (The Background)
# Grid of 100x100 points and calculate the smell at every point
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
# Calculate distance from every point to the target
dist = np.sqrt((X - real_target[0])**2 + (Y - real_target[1])**2)
# Apply the same formula used in the environment (Make sure this matches your env!)
Z = np.exp(-dist / 45.0) 

# 4. Plot
plt.figure(figsize=(7, 6))
plt.title("AI-Guided Chemotaxis (Inflammation Heatmap)")
plt.xlim(0, 100)
plt.ylim(0, 100)

# Draw the Heatmap (The "Pain Field")
contour = plt.contourf(X, Y, Z, 20, cmap='Reds', alpha=0.3)
plt.colorbar(contour, label="Chemical Concentration (pH/Cytokines)")

# Draw Elements
plt.scatter([path_x[0]], [path_y[0]], c='green', s=100, label="Start", edgecolors='black')
plt.scatter([real_target[0]], [real_target[1]], c='red', s=100, label="Pain Source", edgecolors='black')
plt.plot(path_x, path_y, c='blue', lw=2, label="Path")

# Draw Flow
plt.arrow(10, 90, 0, -15, head_width=3, color='grey', alpha=0.8)
plt.text(12, 80, "Blood Flow", color='grey')

plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
