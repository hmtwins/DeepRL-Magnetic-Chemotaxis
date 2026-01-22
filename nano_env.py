import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NanoBotEnv(gym.Env):
    def __init__(self):
        super(NanoBotEnv, self).__init__()
        
        self.AREA_SIZE = 100.0
        self.TARGET_RADIUS = 5.0
        self.MAX_STEPS = 500
        
        # Action: Magnetic Force X, Y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- THE BIG CHANGE: SENSORS NOT COORDINATES ---
        # The AI no longer knows WHERE the target is.
        # It only knows: [My_X, My_Y, Concentration_Center, Concentration_Left, Concentration_Right]
        # It has to compare these 3 numbers to figure out which way is "Pain"
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def _get_pain_concentration(self, pos):
        # Simulating a Chemical Gradient (Gaussian Field)
        # Closer to target = Higher value (High pH/Inflammation)
        dist = np.linalg.norm(pos - self.target_pos)
        # Max signal is 1.0, drops off as you get further away
        signal = np.exp(-dist / 45.0) 
        return signal

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bot_pos = np.array([50.0, 10.0], dtype=np.float32)
        
        # Randomize pain region every time
        self.target_pos = np.array([np.random.uniform(20, 80), np.random.uniform(70, 90)], dtype=np.float32)
        
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # --- PHYSICS (Standard) ---
        heartbeat = np.sin(self.current_step * 0.1)
        flow_y = -1.5 * (abs(heartbeat) + 0.5)
        flow_vector = np.array([0.0, flow_y]) 
        
        mag_force = action * 5.0 # Strong magnet
        noise = np.random.normal(0, 0.2, size=2)
        
        self.bot_pos += flow_vector + mag_force + noise

        # --- REWARD SYSTEM ---
        dist = np.linalg.norm(self.bot_pos - self.target_pos)
        
        # Calculate Current Chemical Signal
        current_signal = self._get_pain_concentration(self.bot_pos)
        
        # Reward 1: Survival
        reward = -0.1
        
        # Reward 2: "Getting Warmer"
        # If signal is high, give huge reward. 
        # The AI learns: High Signal = Good.
        reward += current_signal * 10.0 

        terminated = False
        truncated = False
        
        if dist < self.TARGET_RADIUS:
            reward += 100.0
            terminated = True
            
        if (self.bot_pos[0] < 0 or self.bot_pos[0] > self.AREA_SIZE or 
            self.bot_pos[1] < 0 or self.bot_pos[1] > self.AREA_SIZE):
            reward -= 50.0
            terminated = True
            
        self.current_step += 1
        if self.current_step >= self.MAX_STEPS:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # --- SIMULATE SENSORS ---
        # The bot samples the water in 3 places
        
        # 1. Center (Where I am)
        signal_center = self._get_pain_concentration(self.bot_pos)
        
        # 2. Left Sensor (Offset by -2 on X)
        pos_left = self.bot_pos + np.array([-2.0, 0.0])
        signal_left = self._get_pain_concentration(pos_left)
        
        # 3. Right Sensor (Offset by +2 on X)
        pos_right = self.bot_pos + np.array([2.0, 0.0])
        signal_right = self._get_pain_concentration(pos_right)
        
        # Notice: target_pos is NOT returned here. The AI is blind to the location.
        return np.array([
            self.bot_pos[0], self.bot_pos[1], 
            signal_center, signal_left, signal_right
        ], dtype=np.float32)