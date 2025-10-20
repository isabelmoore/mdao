'''
Purpose of this file is to create and access in between the inputdeck information 
and generating the SNOPT output of the alpha and bank vectors (boost_11 and terminal)
'''

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
import warnings
import numpy as np
from typing import Any, Tuple, Dict, List, Union
from bayes_opt import BayesianOptimization
from dataclasses import dataclass
from itertools import product
import warnings
warnings.filterwarnings("ignore")
import os, sys
from pathlib import Path
from datetime import datetime
from yaml import safe_load
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from functools import partial
import sqlite3
import io
import contextlib
from bayes_opt.util import UtilityFunction
from prettytable import PrettyTable
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import imageio
import multiprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
import yaml
from yaml import safe_load, dump

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


script_dir = Path(__file__).resolve().parent.parent.parent.parent
os.chdir(script_dir)

for path in paths_to_modules:
    if str(path) not in sys.path:
        sys.path.append(str(path))

sys.path.append(str(Path.cwd())) 
BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
date = datetime.now().strftime('%Y_%m_%d')
timing = datetime.now().strftime('%H%M%S')

from dymos_generator import dymos_generator

def generate_design_space(variables, num_samples_azimuth, num_samples_range, seed=42):
    # print("Generating design space...")
    
    # azimuth_bounds = variables["azimuth"]['bounds']
    # range_bounds = variables["range"]['bounds']
    
    # azimuth_data = np.linspace(azimuth_bounds[0], azimuth_bounds[1], num_samples_azimuth)
    # range_data = np.linspace(range_bounds[0], range_bounds[1], num_samples_range)

    # print(azimuth_data)
    # print(range_data)
    
    # azimuth_grid, range_grid = np.meshgrid(azimuth_data, range_data)
    
    # # Reshape to list of tuples
    # casenums = np.arange(num_samples_azimuth*num_samples_range)
    # print("Case Numbers:", len(casenums))
    # initial_grid = list(zip(casenums,azimuth_grid.ravel(), range_grid.ravel()))
    

    # Latin Hyper Cube Implementation
    print("Generating design space...")

    azimuth_bounds = variables["azimuth"]['bounds']
    range_bounds = variables["range"]['bounds']

    # Determine total number of samples
    total_samples = num_samples_azimuth * num_samples_range

    # Latin Hypercube Sampling
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    lh_samples = sampler.random(n=total_samples)

    # Scale the samples to the appropriate bounds
    lh_samples_scaled = qmc.scale(lh_samples, 
                                [azimuth_bounds[0], range_bounds[0]], 
                                [azimuth_bounds[1], range_bounds[1]])

    # Create an index (case numbers) for each sample
    casenums = np.arange(total_samples)
    print("Case Numbers:", len(casenums))

    # Reshape to list of tuples: (case number, azimuth value, range value)
    design_space = list(zip(casenums, lh_samples_scaled[:, 0], lh_samples_scaled[:, 1]))
    print("Design space generated:", design_space)
    return design_space

def store_results(results, db_filename):

    """
    Storing Multi-valued Attributes in Relational Databases
    When you have data that consists of lists or arrays (like multiple values for alpha and bank), 
    the best practice in relational databases is to normalize your data. This means creating separate 
    tables to hold related data instead of trying to fit multiple values into a single column.
    """
    connection = sqlite3.connect(db_filename)
    cursor = connection.cursor()
    
    # Clear previous data
    cursor.execute('DROP TABLE IF EXISTS boost_11_timeseries')
    cursor.execute('DROP TABLE IF EXISTS terminal_timeseries')
    cursor.execute('DROP TABLE IF EXISTS results')

    # Create tables
    cursor.execute('''
        CREATE TABLE results (
            case_num INTEGER PRIMARY KEY,
            azimuth REAL,
            range REAL,
            status INTEGER,
            iterations INTEGER     
        )
    ''')

    cursor.execute('''
        CREATE TABLE boost_11_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_num INTEGER,
            alpha REAL,
            bank REAL,
            velocity REAL,
            FOREIGN KEY(case_num) REFERENCES results(case_num)
        )
    ''')

    cursor.execute('''
        CREATE TABLE terminal_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_num INTEGER,
            alpha REAL,
            bank REAL,
            velocity REAL,
            FOREIGN KEY(case_num) REFERENCES results(case_num)
        )
    ''')

    # Inserting results into results table
    for index, result in enumerate(results):
        cursor.execute('''
            INSERT INTO results (case_num, azimuth, range, status, iterations)
            VALUES (?, ?, ?, ?, ?)
        ''', (index + 1, result['azimuth'], result['range'], result['status'], result['iterations']))

        # Insert boost_11 timeseries
        boost_alpha_list = result['boost_11']['alpha']  
        boost_bank_list = result['boost_11']['bank']  
        boost_vel_list = result['boost_11']['velocity']  

        for alpha, bank, velocity in zip(boost_alpha_list, boost_bank_list, boost_vel_list):
            cursor.execute('''
                INSERT INTO boost_11_timeseries (case_num, alpha, bank, velocity)
                VALUES (?, ?, ?, ?)
            ''', (index + 1, alpha, bank, velocity))


        # Insert terminal timeseries
        terminal_alpha_list = result['terminal']['alpha']  
        terminal_bank_list = result['terminal']['bank'] 
        terminal_vel_list = result['terminal']['velocity']

        for alpha, bank, velocity in zip(terminal_alpha_list, terminal_bank_list):
            cursor.execute('''
                INSERT INTO terminal_timeseries (case_num, alpha, bank, velocity)
                VALUES (?, ?, ?, ?)
            ''', (index + 1, alpha, bank, velocity))

    connection.commit()
    connection.close()

def merge_input_decks(base, child):
    for key, value in base.items():
        if key not in child:
            child[key] = value
        elif isinstance(value, dict) and isinstance(child[key], dict):
            merge_input_decks(value, child[key])
    return child

def run_traj_test(case, input_deck, report_dir):
    casenum = case[0]

    print(f"Running Trajectory Tests for case: {casenum}")
    print(f"Processing case data: \n{case}") 
    print("HHAHAHAHAH")
    azimuth= case[1]
    range_ = case[2]
    problem_name = str(report_dir / 'cases' / f'case_{casenum}')    

    # scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
    # scenario.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2
    # scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][0] = azimuth / 2
    # scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["initial_conditions"]["controls"]["bank"][1] = azimuth / 2

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):

                p = om.Problem(name=problem_name)
                print("0")

                scenario = dymos_generator(problem=p, input_deck=input_deck)
                print("1")
                scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["azimuth"]["equals"] = azimuth
                scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["range"]["equals"] = range_

                print("2")

                scenario.setup()
                print("3")

                dm.run_problem(scenario.p, run_driver=False, simulate=True)
                # om.n2(scenario.p, outfile="n2_post_run.html")
                print("4")

                with open(p.get_outputs_dir() / "SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
                    SNOPT_history = f.read()

                exit_code_start = SNOPT_history.rfind("SNOPTC EXIT")
                exit_code_end = SNOPT_history.find("\n", exit_code_start)
                exit_code = int((SNOPT_history[exit_code_start:exit_code_end]).split()[2])
                iter_code_start = SNOPT_history.rfind("No. of iterations")
                iter_code_end = SNOPT_history.find("\n", iter_code_start)

                iterations = int((SNOPT_history[iter_code_start:iter_code_end]).split()[3])

                exit_code = 0
                print("Exit Code", exit_code)
                if exit_code != 0:
                    status = 0 # FAILURE
                else:
                    status = 1 # SUCCESS
                result = {
                    'azimuth': azimuth,
                    'range': range_,
                    'boost_11': {
                        'alpha': [item for sublist in p.get_val("traj_vehicle_0.boost_11.timeseries.alpha", units="deg").tolist() for item in sublist], 
                        'bank': [item for sublist in p.get_val("traj_vehicle_0.boost_11.timeseries.bank", units="deg").tolist() for item in sublist], 
                        'velocity': [item for sublist in p.get_val("traj_vehicle_0.boost_11.timeseries.v", units="ft/s").tolist() for item in sublist], 
                    },
                    'terminal': {
                        'alpha': [item for sublist in p.get_val("traj_vehicle_0.terminal.timeseries.alpha", units="deg").tolist() for item in sublist], 
                        'bank': [item for sublist in p.get_val("traj_vehicle_0.terminal.timeseries.bank", units="deg").tolist() for item in sublist], 
                        'velocity': [item for sublist in p.get_val("traj_vehicle_0.terminal.timeseries.v", units="ft/s").tolist() for item in sublist], 
                    },
                    'status': status,
                    'iterations': iterations
                }
    except Exception as e:
        print(f"Error during trajectory test for case {casenum}: {e}")
        result = {
            'azimuth': azimuth,
            'range': range_,
            'boost_11': {
                'alpha': [], 
                'bank': [], 
                'velocity': [], 
            },
            'terminal': {
                'alpha': [], 
                'bank':  [], 
                'velocity': [], 
            },
            'status': 0,
            'iterations': 0
        }

    return result

def main(num_samples_azimuth, num_samples_range, num_processors):
    start_time = time.time()
    
    input_deck_base = "base_input_deck.yml"
    input_deck_filename = 'one_stage_one_pulse_traj_ann.yml'

    input_deck_path = os.path.join("scenarios/input_decks/", input_deck_filename)
    input_deck_base_path = os.path.join("scenarios/input_decks/", input_deck_base)

    with open(input_deck_path, "r") as f:
        input_deck_child = safe_load(f)
    with open(input_deck_base_path) as base_file:
        base = safe_load(base_file)
    if "base" in input_deck_child:
        del input_deck_child["base"]

    input_deck = merge_input_decks(base, input_deck_child)

    variables = {
        "azimuth": {'bounds': np.array([-90, 90])},
        "range": {'bounds': np.array([10, 60])},
    }

    # Create the design space
    cases = generate_design_space(variables, num_samples_azimuth, num_samples_range)
    
    report_dir = REPORTS_DIR / f"{input_deck_filename.replace('.yml', '')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=num_processors) as pool:
        print("Created pool...")
        # Map the function to the list of traj doe values
        task_partial = partial(run_traj_test, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task_partial, cases), total=len(cases), desc="Processing Samples"))

    print("Results finished...")
    pool.close()
    pool.join()


    db_filename = f'trajectory_results_azimuth{num_samples_azimuth}_range{num_samples_range}_cores{num_processors}_date_{datetime.now().strftime("%Y_%m_%d")}_time_{datetime.now().strftime("%H%M")}.db'    
    store_results(results, db_filename)

    connection = sqlite3.connect(db_filename)
    
    results_df = pd.read_sql_query('SELECT * FROM results', connection)
    print("Results Table:")
    print(results_df)
    print("\n")  

    boost11_df = pd.read_sql_query('SELECT * FROM boost_11_timeseries', connection)
    print("Boost 11 Timeseries Table:")
    print(boost11_df)
    print("\n")  

    terminal_df = pd.read_sql_query('SELECT * FROM terminal_timeseries', connection)
    print("Terminal Timeseries Table:")
    print(terminal_df)

    connection.close()


    elapsed_time = time.time() - start_time
    print(f"Data generation and processing completed in {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    main(num_samples_azimuth=2, num_samples_range=2, num_processors=24) # 500^2 = 250_000 samples, 20 cores (max)



-------------------------------------------


def simulate_trajectory(case, input_deck, input_params, report_dir):
    """
    Simulate the trajectory of the vehicle based on the input parameters.

    Parameters:
    case (Tuple): case parameters including case ID and specific parameter values.
    input_deck (dict): loaded input deck containing configuration settings.
    input_params (List[Union[NumericSpec, CategoricalSpec]]): List of specifications for input parameters.
    report_dir (Path): Directory for saving output files.

    Returns:
    Dict[str, Union[int, float, str]]: A dictionary containing the simulation results.
    """
    casenum = case[0]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                problem_name = str(report_dir / 'cases' / f'case_{casenum}')
                p = om.Problem(name=problem_name)
                scenario = dymos_generator(problem=p, input_deck=input_deck)
                for i, spec in enumerate(input_params):
                    value = case[i + 1]
                    set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)
                scenario.setup()
                dm.run_problem(
                    scenario.p,
                    run_driver=True,
                    simulate=True,
                )
                final_range = p.get_val("traj_vehicle_0.terminal.timeseries.x", units="NM")[-1, 0]
        result = {
            'input_params': case,
            'range': final_range,
            'status': 1,
            'comments': "SUCCESS"
        }
    except Exception as e:
        result = {
            'input_params': case,
            'range': 0,
            'status': 0,
            'comments': f"Error: {e}"
        }
    return result

def run_search(input_deck_filename: str = '',
        num_processors: int = os.cpu_count() / 2,
        n_samples: int = 100,
        input_params: List[Union[NumericSpec, CategoricalSpec]] = [],
        db_filename: str = "",
        buildLHS_bool: bool = False,
        optimize_bool: bool = False,
):
    """
    run_search function that orchestrates the loading of the input deck, 
    performs Bayesian optimization if specified, and updates the input deck 
    with the best parameters based on the optimization results.
    
    Parameters:
    input_deck_filename (str): filename of the YAML input deck.
    num_processors (int): number of processors to use for parallel processing.
    n_samples (int): number of samples for optimization.
    input_params (List[Union[NumericSpec, CategoricalSpec]]): List of specification objects for input parameters.
    db_filename (str): path to the database file for results persistence.
    optimize_bool (bool): Flag to determine if optimization should be performed.
    create_LH_bool (bool): Flag to indicate whether to create Latin Hypercube samples.
    
    Returns:
    Dict[str, float]: A dictionary containing the best parameters after optimization.
    """
    # Load input deck configuration
    input_deck_base = "base_input_deck.yml"
    input_deck_path = os.path.join("scenarios/input_decks/", input_deck_filename)
    input_deck_base_path = os.path.join("scenarios/input_decks/", input_deck_base)

    with open(input_deck_path, "r") as f:
        input_deck_child = safe_load(f)
    with open(input_deck_base_path) as base_file:
        base = safe_load(base_file)
    if "base" in input_deck_child:
        del input_deck_child["base"]

    input_deck = merge_input_decks(base, input_deck_child)

    # Determine suffix for the database filename based on given controls
    db_suffix = ""

    if buildLHS_bool and optimize_bool:
        db_suffix = "_buildLHS_optimize"
    elif buildLHS_bool:
        db_suffix = "_buildLHS"
    elif optimize_bool:
        db_suffix = "_optimize"
    else:
        db_suffix = "_loadedDB"

    if not buildLHS_bool:
        results, n_samples = load_db(db_filename=db_filename, optimize_bool=False)

    # Establish the reports directory
    report_dir = REPORTS_DIR / f"{input_deck_filename.replace('.yml', '')}_date_{date}_time_{timing}{db_suffix}_{n_samples}"
    report_dir.mkdir(parents=True, exist_ok=True)


    # --- Processes as Defined by User --- 
    if optimize_bool:
        bayesian_optimization_batch(
            input_deck,
            input_params,
            n_samples=n_samples,
            n_iter=10,
            batch_size=10,
            num_processors=num_processors,
            buildLHS_bool=buildLHS_bool,
            db_filename=db_filename
        )
    
    if buildLHS_bool:
        full_input_samples = GridSampling(input_params, n_samples=n_samples)
        with multiprocessing.Pool(processes=num_processors) as pool:
            print("Created pool...")
            task_partial = partial(simulate_trajectory, input_deck=input_deck, input_params=input_params, report_dir=report_dir)
            results = list(tqdm(pool.imap(task_partial, full_input_samples), total=len(full_input_samples), desc="Processing Samples"))
        results, _ = save_db(input_params, results, optimize_bool, n_samples, report_dir)
    else:
        results, _ = save_db(input_params, results, optimize_bool, n_samples, report_dir)

    for result in results:
        for spec in input_params:
            if result.get(spec.name) is None:
                raise ValueError(
                    f"Error: For case {result.get('case_id')}, the parameter '{spec.name}' is None. "
                    "This may be due to a misspelling in one of the input deck parameter's path. Aborting execution."
                )
