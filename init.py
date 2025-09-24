import numpy as np
from math import * 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime
from pathlib import Path
import openmdao.api as om
import dymos as dm

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

    def run(self):
        '''
        Computes penalties based on trajectory metrics
        '''

        print()
        print("Rewarding...")
        scenario = self.scenario
        problem = self.problem

        dm.run_problem(scenario.problem, run_driver=True, simulate=False)
        # om.n2(scenario.p, outfile="n2_post_run.html")

        with open(self.problem.get_outputs_dir() / "SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
            SNOPT_history = f.read()

        # Define where the exit code information starts
        exit_code_start = SNOPT_history.rfind("SNOPTC EXIT")
        exit_code_end = SNOPT_history.find("\n", exit_code_start)

        # Extract the exit code line
        exit_code = int((SNOPT_history[exit_code_start:exit_code_end]).split()[2])

        # TO DO: fix this tomatch that of major iterattions, not minor
        before_exit = SNOPT_history[:exit_code_start].splitlines()
        last_row_line = before_exit[-1].strip()
        last_major_iter = int(last_row_line.split()[0])

        print("Exit code:", exit_code)
        print("Last major iteration:", last_major_iter)
    
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



        print("Initial Conditions Before: ")
        print(f"\tBoost Alphas: \n\t\t ", p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls" ]["alpha"], " deg")

        # train model to find optimal parameters for initial conditions
        env = TrajectoryEnv(input_deck, p, scenario)
        # env.step()s
        # alpha_boost_1, alpha_boost_2 = env.actions

        # # update initial conditions with the optimized results
        # p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = 0.0010003987
        # p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = 0.009254456

        print("Initial Conditions After: ")
        print(f"\tBoost Alphas: \n\t\t ", p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls" ]["alpha"], " deg")

        scenario.setup()
        problem_report_dir = p.get_reports_dir()

        try:
            if input_deck["optimizer_settings"]["feasibility_driver"]:
                print("\nRunning Feasibility Driver")
                # run and simulate afterwards, see if it errors out, if it does just make the traj report without sim data
                scenario.p.driver.opt_settings["Problem Type"] = "Feasible point"
                dm.run_problem(scenario.p, run_driver=True, simulate=False, make_plots=False)
                scenario.p.driver.opt_settings["Problem Type"] = "Minimize"
        except:
            pass


        print("\nRunning Optimality Driver")

        dm.run_problem(scenario.p, run_driver=True, simulate=True, make_plots=True, simulate_kwargs={"method": "RK45"})  
              
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
