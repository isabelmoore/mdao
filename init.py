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


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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


# TO DO: match with home directory of repo
os.chdir(r"/home/imoore/misslemdao")

project_dir = Path(__file__).resolve().parent.parent  

for path in paths_to_modules:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dymos_generator import dymos_generator


def generate_design_space(variables, num_samples_azimuth, num_samples_range, seed=42):
    print("Generating design space...")
    
    azimuth_bounds = variables["azimuth"]['bounds']
    range_bounds = variables["range"]['bounds']
    
    azimuth_data = np.linspace(azimuth_bounds[0], azimuth_bounds[1], num_samples_azimuth)
    range_data = np.linspace(range_bounds[0], range_bounds[1], num_samples_range)

    print(azimuth_data)
    print(range_data)
    
    azimuth_grid, range_grid = np.meshgrid(azimuth_data, range_data)
    
    # Reshape to list of tuples
    casenums = np.arange(num_samples_azimuth*num_samples_range)
    print("Case Numbers:", len(casenums))
    initial_grid = list(zip(casenums,azimuth_grid.ravel(), range_grid.ravel()))
    

    # Latin Hyper Cube Implementation
    # num_vars = 2  
    # LH_generator = qmc.LatinHypercube(num_vars, seed=seed, scramble=False)

    # LH_samples = LH_generator.random(num_samples)
    
    # LH_scaled = qmc.scale(LH_samples, np.array([azimuth_bounds[0], range_bounds[0]]), 
    #                           np.array([azimuth_bounds[1], range_bounds[1]]))

    # combined_grid = np.vstack((initial_grid, LH_scaled))  # Combine both sources

    # df = pd.DataFrame(data=initial_grid, columns=['azimuth', 'range'])
    
    print("Design space generated.")
    return initial_grid

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
            FOREIGN KEY(case_num) REFERENCES results(case_num)
        )
    ''')

    cursor.execute('''
        CREATE TABLE terminal_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_num INTEGER,
            alpha REAL,
            bank REAL,
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

        for alpha, bank in zip(boost_alpha_list, boost_bank_list):
            cursor.execute('''
                INSERT INTO boost_11_timeseries (case_num, alpha, bank)
                VALUES (?, ?, ?)
            ''', (index + 1, alpha, bank))


        # Insert terminal timeseries
        terminal_alpha_list = result['terminal']['alpha']  
        terminal_bank_list = result['terminal']['bank']    

        for alpha, bank in zip(terminal_alpha_list, terminal_bank_list):
            cursor.execute('''
                INSERT INTO terminal_timeseries (case_num, alpha, bank)
                VALUES (?, ?, ?)
            ''', (index + 1, alpha, bank))

    connection.commit()
    connection.close()
def merge_input_decks(base, child):
    for key, value in base.items():
        if key not in child:
            child[key] = value
        elif isinstance(value, dict) and isinstance(child[key], dict):
            merge_input_decks(value, child[key])
    return child

def run_traj_test(case, input_deck):
    
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

    try:     
        dm.run_problem(scenario.p, run_driver=True, simulate=False, restart=r"/home/imoore/misslemdao/tools/traj_ann/dymos_solution.db")
        # om.n2(scenario.p, outfile="n2_post_run.html")

        with open(p.get_outputs_dir() / "SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
            SNOPT_history = f.read()

        # Define where the exit code information starts
        exit_code_start = SNOPT_history.rfind("SNOPTC EXIT")
        exit_code_end = SNOPT_history.find("\n", exit_code_start)

        # Extract the exit code line
        exit_code = int((SNOPT_history[exit_code_start:exit_code_end]).split()[2])

        # TO DO: fix this tomatch that of major iterattions, not minor
        iter_code_start = SNOPT_history.rfind("No. of iterations")
        iter_code_end = SNOPT_history.find("\n", iter_code_start)

        iterations = int((SNOPT_history[iter_code_start:iter_code_end]).split()[3])


        print("Exit Code", exit_code)
        if exit_code != 0:
            status = 0 # FAILURE
        else:
            status = 1 # SUCCESS
        result = {
            'azimuth': azimuth,
            'range': range,
            'boost_11': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
            },
            'terminal': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
            },
            'status': status,
            'iterations': iterations
        }
    except Exception as e:
        print(f"Error during trajectory test for case {casenum}: {e}")
        result = {
            'azimuth': azimuth,
            'range': range,
            'boost_11': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "boost_11"), units="deg").tolist() for item in sublist], 
            },
            'terminal': {
                'alpha': [item for sublist in p.get_val("%s.%s.%s.timeseries.alpha" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
                'bank': [item for sublist in p.get_val("%s.%s.%s.timeseries.bank" % ("traj_vehicle_0", "phases", "terminal"), units="deg").tolist() for item in sublist], 
            },
            'status': 0,
            'iterations': 0
        }

    return result

def main(num_samples_azimuth, num_samples_range, num_processors):

    project_root = Path(__file__).parents[2]  
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    start_time = time.time()
    
    input_deck_base = "base_input_deck.yml"
    input_deck_filename = 'one_stage_one_pulse_traj_ann.yml'

    input_deck_path = os.path.join("scenarios/input_decks/", input_deck_filename)
    input_deck_base_path = os.path.join("scenarios/input_decks/", input_deck_base)

    with open(input_deck_path, "r") as f:
        input_deck_child = safe_load(f)

    # Load base and child files
    with open(input_deck_base_path) as base_file:
        base = safe_load(base_file)

    # Merge base into child 
    if "base" in input_deck_child:
        del input_deck_child["base"]

    # input_deck = merge_input_decks(base, input_deck_child)
    input_deck = input_deck_child
    variables = {
        "azimuth": {'bounds': np.array([-90, 90])},
        "range": {'bounds': np.array([10, 60])},
    }
    
    # Create output directory
    dataset_name = "altitude_mach_data"
    base_temp_dir = Path("./datasets") / dataset_name
    os.makedirs(base_temp_dir, exist_ok=True)

    # Create the design space
    cases = generate_design_space(variables, num_samples_azimuth, num_samples_range)


    with multiprocessing.Pool(processes=num_processors) as pool:
        print("Created pool...")
        # Map the function to the list of traj doe values
        task_partial = partial(run_traj_test, input_deck=input_deck)
        results = pool.map(task_partial, cases)

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
    main(num_samples_azimuth=2, num_samples_range=2, num_processors=2) # 500^2 = 250_000 samples, 20 cores (max)
