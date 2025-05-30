
#!/usr/bin/env python3

import math
import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import logging
import dubins

class TrajectoryEnv(gym.Env):
    def __init__(self, problem, scenario):
        self.problem = problem 
        self.scenario = scenario
        self.state = None
        self.action_space = None
        self.observation_space = None
        
        self.desired_heading = 0.0
        self.desired_height = 0.0
        self.history_length = 10

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.previous_states = np.zeros((self.history_length,6), dtype=np.float32)
        self.previous_input = np.zeros((self.history_length,2), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state":          gym.spaces.Box(-np.inf, np.inf, shape=(6,),      dtype=np.float32),
            "previous_state": gym.spaces.Box(-np.inf, np.inf, shape=(self.history_length, 6),    dtype=np.float32),
            "previous_input": gym.spaces.Box(-np.inf, np.inf, shape=(self.history_length 2),    dtype=np.float32),
        })
        
        self.initial_conditions = {
            'traj.phase0.timeseries.alpha': 0.0,
            'traj.phase0.timeseries.bank': 0.0,
            'traj.phase0.timeseries.height': 100.0,
            'traj.phase0.timeseries.heading': 0.0,
            'traj.phase0.timeseries.roll': 0.0,
            'traj.phase0.timeseries.pitch': 0.0,
        }
        
        self.pose    = np.zeros(6, dtype=np.float32)        # [x, y, m, v, h, heading]
        self.actions = np.zeros(2, dtype=np.float32)


    def observe(self):
        self.previous_states = np.roll(self.previous_states, 1, 0)
        self.previous_states[0] = self.pose.astype(np.float32, copy=False)

        self.previous_input  = np.roll(self.previous_input, 1, 0)
        self.previous_input[0] = self.actions.astype(np.float32, copy=False)

        # (3) build dict – nothing changes shape or dtype after this
        return {
            "state":          pose32,              # (6,)      float32
            "previous_state": self.previous_states, # (H, 6)   float32
            "previous_input": self.previous_input,  # (H, 2)   float32
        }
        
    def reset(self):
        # Reset the environment to an initial state
        self.state = self.problem.initial_state()
        return self.state
    
    def reward(self):

        head = self.problem.get_val('traj.phase0.timeseries.heading')[-1]
        height = self.problem.get_val('traj.phase0.timeseries.height')[-1]
        
        height_error = abs(height - self.desired_height)
        heading_error = abs(head - self.desired_heading)
        
        total_error = height_error + heading_error
        
        return total_error


    def compute_inputs(self, action):
        alpha_ff = float(action[0])
        bank_ff = float(action[1])
        
        delta_alpha = self.problem.get_val('traj.phase0.timeseries.alpha_rate')[-1]
        delta_bank = self.problem.get_val('traj.phase0.timeseries.bank_rate')[-1]
        
        alpha = alpha_ff + delta_alpha
        bank = bank_ff + delta_bank
        
        return alpha, bank
        
    def step(self, action):
        # 1. blend feed-forward with last rates
        alpha_cmd, bank_cmd = self.compute_inputs(action)
        self.actions[:] = [alpha_cmd, bank_cmd]

        # 2. push controls & **run one forward pass only**
        self.problem.set_val('traj.phase0.controls:alpha', alpha_cmd)
        self.problem.set_val('traj.phase0.controls:bank',  bank_cmd)
        self.scenario.__init_conditions(self.problem, initial_conditions=self.initial_conditions)
        self.initial_conditions = self.pose.copy()
        
        # 3. refresh cached pose from latest time-series sample
        ts = 'traj.phase0.timeseries.states'
        self.pose[:] = np.array([
            self.problem.get_val(f'{ts}:x')[-1],
            self.problem.get_val(f'{ts}:y')[-1],
            self.problem.get_val(f'{ts}:theta')[-1],
            self.problem.get_val(f'{ts}:wL')[-1],
            self.problem.get_val(f'{ts}:wR')[-1],
        ], dtype=np.float32)

        obs     = self.observe()
        reward  = self.reward()
        terminated = False                       # define a goal flag if needed
        truncated  = False                       # you said no max-time
        return obs, reward, terminated, truncated, {}
    
    
    
"""
obs, _ = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
print("rolled 5 steps without buffer error ✅")
"""