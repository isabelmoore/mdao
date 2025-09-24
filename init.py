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
import os, sys, time, copy, shutil
from pathlib import Path
from datetime import datetime
from yaml import safe_load, dump
import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.visualization.timeseries.bokeh_timeseries_report import make_timeseries_report
import matplotlib.pyplot as plt
import tempfile
import multiprocessing
from functools import partial
from pathlib import Path
from bs4 import BeautifulSoup
import scipy.stats.qmc as qmc
import pandas as pd
from itertools import product
import sqlite3

paths_to_modules = [
    "scenarios",
    "scenarios/concepts",
    "scenarios/configs",
    "src/models/aero/datcom",
    "src/models/layout",
    "src/models/loads",
    "src/models/power",
    "src/models/prop/solid",
    "src/models/prop/inlet_tables",
    "src/models/radar",
    "src/models/structure",
    "src/models/thermal",
    "src/models/warhead",
    "src/models/warhead/bf_warhead",
    "src/models/weapondatalink",
    "src/models/weight",
    "tools/helpers/",
    "tools/plotters",
    "tools/ppt",
    "tools/datcom_nn",
]
os.chdir(r"/home/imoore/misslemdao")

project_dir = Path(__file__).resolve().parent.parent  

for path in paths_to_modules:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dymos_generator import dymos_generator

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


    
    def run(self, case, input_deck):
        casenum= case[0]
        azimuth= case[1]
        range_ = case[2]
        
        print(f"Running Trajectory Tests for case: {casenum}")
        print(f"Processing case data: \n{case}") 

        problem_name = f'case_{casenum}'
        p = om.Problem(name=problem_name)

        scenario = dymos_generator(problem=p, input_deck=input_deck)
        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["azimuth"]["equals"] = azimuth
        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["range"]["equals"] = range_

        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2
        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
        scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2

        scenario.setup()

        print("\nRunning Optimality Driver")
        
        try:
            dm.run_problem(scenario.p, run_driver=True, simulate=True, simulate_kwargs={"method": "RK45"})

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
            return last_major_iter

        except Exception as e:
            print(f"Error during trajectory test for case {casenum}: {e}")
            return None

    def make_noise():
        return 
    def run_parallel_envs(self, 
        input_deck: dict,
        seed_alphas: tuple[float, float],
        *,
        n_candidates: int = 24,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        method: str = "sobol",
        max_workers: int = 8,
        logdir: Path = Path("snopt_logs")
        ):
        '''
        Executes the optimization process using the Nelder-Mead method. Identifies optimal control parameters.
        '''

        
        
        print("Stepping...")

        self.update_state()
        
        initial_guess = [self.alpha_boost_1, self.alpha_boost_2]
        result = minimize(self.run, initial_guess, method='Nelder-Mead', tol=1e-3, options={'maxiter': 100})
        self.actions = np.array(result.x, dtype=np.float32).flatten()
