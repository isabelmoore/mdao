import numpy as np
from math import * 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime
from pathlib import Path
import openmdao.api as om


'''
This script is designed to help in reducing the amount of major iterations for SNOPT. 

Originally, this was designed by reducing errors for aircraft parameters at various points in
the trajectory. However, this is not good for future development and  current work, as different input decks 
expected different behaviors, causing the "tuned" reward or error model to drastically change per deck.

Thus, what we know is that the major iterations is meant to be minimized and that certain values of the 
initial conditions leads to more successfull outcomes. Rather than "beating around the bush" and "slapping
on bandaids" and endless runs of "with vs without the driver" to save in computation energy, we will run WITH
the driver and delievery in these points:
    - run with driver
    - to save in compuational energy, run these in parallel
    - since we know we have some good values, apply a Perturbation Optimization as per run
        https://www.worldscientific.com/doi/epdf/10.1142/S021759591950009X

'''

class TrajectoryEnv():
    def __init__(self, input_deck, problem, scenario):
        super(TrajectoryEnv, self).__init__()
        self.input_deck = input_deck
        self.problem = problem
        self.scenario = scenario
        self.pose = np.zeros(8, dtype=np.float32)
        self.action = np.zeros(6, dtype=np.float32)
        self.h_ideal_term = self.input_deck["trajectory_phases"]["terminal"]["constraints"]["boundary"]["h"]["equals"]
        self.gam_bound_boost = 0.0
        self.ts = "traj_vehicle_0"
        self.alpha_boost_1 = self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0]
        self.alpha_boost_2 = self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1]
        print(f"initial: {self.alpha_boost_1, self.alpha_boost_2}")
        
    def update_state(self):
        '''
        Updates pose with metrics from terminal and boost phases
        '''

        try:
            h_boost_min = float(min(self.problem.get_val(f"{self.ts}.phases.boost_11.timeseries.h", units="m")))
            range_boost = float(self.problem.get_val(f"{self.ts}.phases.boost_11.timeseries.range", units="m")[-1])

            self.pose = [h_boost_min, range_boost]
        except Exception as e:
            print(f"[ERROR] Failed to update state: {e}")  

    def reward(self):
        '''
        Computes penalties based on trajectory metrics
        '''

        print()
        print("Rewarding...")
        h_boost, range_boost = self.pose

        range_boost_penalty = (range_boost / 1000) ** 2
        h_boost_penalty = (h_boost / 1000) ** 2
        total_error = -(range_boost_penalty + h_boost_penalty)
        
        print(f"Raw Penalties: \n\tBoost Range: {range_boost} \n\tBoost Height: {h_boost}")        
        print(f"Total Penalty: \t{total_error}")
        print(f"Params: \t{self.alpha_boost_1, self.alpha_boost_2}")

        return total_error
    
    def objective_function(self, params):
        '''
        Updates control parameters and evaluates the resulting trajectory. Feeds results back to optimization
        '''

        alpha_boost_1, alpha_boost_2 = params
        self.alpha_boost_1 = alpha_boost_1
        self.alpha_boost_2 = alpha_boost_2
        
        self.problem.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = self.alpha_boost_1
        self.problem.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = self.alpha_boost_2
        
        try:
            self.scenario.setup()
        except Exception as e:
            print("[WARNING] Alpha outside of bounds:", e)

        self.update_state()
        return self.reward()

    def step(self):
        '''
        Executes the optimization process using the Nelder-Mead method. Identifies optimal control parameters.
        '''
        
        print("Stepping...")

        self.update_state()
        initial_guess = [self.alpha_boost_1, self.alpha_boost_2]
        result = minimize(self.objective_function, initial_guess, method='Nelder-Mead', tol=1e-3, options={'maxiter': 100})
        self.actions = np.array(result.x, dtype=np.float32).flatten()
    
def train(input_deck, p, scenario):
    '''
    Wrapper function that initializes the optimizer and returns optimal initial conditions
    '''
    
    env = TrajectoryEnv(input_deck, p, scenario)
    env.step()

    optimal_params = env.actions  
    return optimal_params
