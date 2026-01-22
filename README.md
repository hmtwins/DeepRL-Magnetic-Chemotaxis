# AI-Guided Magnetic Nanobot for Targeted Drug Delivery

### Project Overview
This project demonstrates an autonomous navigation system for magnetic nanoparticles in the human vascular system. Using **Deep Reinforcement Learning (PPO)**, the AI learns to steer a nanobot against pulsatile blood flow to locate specific pain regions (inflammation) via **chemotaxis** (chemical gradient sensing).

### The Technology
*   **Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines3.
*   **Physics Engine:** Custom environment simulating **Hemodynamics** (blood flow), **Brownian Motion** (thermal noise), and **Low Reynolds Number** fluid dynamics.
*   **Sensing:** The agent utilizes a multi-sensor array to detect biochemical gradients (e.g., pH/Cytokines) without GPS coordinates, mimicking biological leukocyte navigation.

### File Structure
*   `nano_env.py`: The physics and biology simulation environment (Gymnasium).
*   `train.py`: The training script that utilizes the PPO algorithm to create the neural network.
*   `visualize.py`: Runs a simulation using the trained brain and generates a heatmap of the drug delivery path.
*   `nano_ai_brain.zip`: The pre-trained neural network model.

### How to Run
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
2. **Train the AI (Optional):**
  ```bash
   python train.py
  ```
3. **Run the Visualization:**
   ```bash
   python visualize.py
