
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
    
    # Reshape to tuple of tuples: (case number, azimuth value, range value)
    design_space = tuple(
        (case, azimuth, range_val) 
        for case, azimuth, range_val in zip(casenums, lh_samples_scaled[:, 0], lh_samples_scaled[:, 1])
    )
    
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


def simulate_trajectory(case, input_deck, report_dir):
    casenum = case[0]
    azimuth = case[1]
    range_ = case[2]
    print(f"Running Trajectory Tests for case: {casenum}")
    print(f"Processing case data: \n{case}") 
    try:
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
        #     warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
        #     buf = io.StringIO()
        #     with contextlib.redirect_stdout(buf):
        problem_name = str(report_dir / 'cases' / f'case_{casenum}')
        p = om.Problem(name=problem_name)
        scenario = dymos_generator(problem=p, input_deck=input_deck)
        scenario.setup()
        dm.run_problem(
            scenario.p,
            run_driver=True,
            simulate=True,
        )
        status=1
        result = {
            'azimuth': azimuth,
            'range': range_,
            'status': status,
            'comments': "SUCCESS"
        }
    except Exception as e:
        status=0
        result = {
            'azimuth': azimuth,
            'range': range_,
            'status': status,
            'comments': f"Error: {e}"
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
    full_input_samples = generate_design_space(variables, num_samples_azimuth, num_samples_range)
    print(type(full_input_samples))
    print(full_input_samples)    
    report_dir = REPORTS_DIR / f"{input_deck_filename.replace('.yml', '')}_date_{date}_time_{timing}"
    report_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=num_processors) as pool:
        print("Created pool...")
        task_partial = partial(simulate_trajectory, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task_partial, full_input_samples), total=len(full_input_samples), desc="Processing Samples"))
    print("Results finished...")
    print(results)

Processing Samples: 100%|████████████| 4/4 [00:08<00:00,  2.13s/it]
Results finished...
[{'azimuth': -34.82802218501835, 'range': 17.014019503099348, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 324 of file misdat.for (unit = 2, file = 'fort.2')\nFortran runtime error: File cannot be deleted\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4a6299\n#4  0x457bac\n#5  0x40186e\n#6  0x4e11ef\n#7  0x401e9d\n"}, {'azimuth': -83.63690639601221, 'range': 38.78289963675795, 'status': 0, 'comments': "Error: <model> <class Group>: Indices for aliases ['traj_vehicle_0.terminal.range[final]'] are overlapping constraint/objective 'traj_vehicle_0.phases.terminal.timeseries.range'."}, {'azimuth': 40.76201934505576, 'range': 47.804720604540556, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 324 of file misdat.for (unit = 2, file = 'fort.2')\nFortran runtime error: File cannot be deleted\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4a6299\n#4  0x457bac\n#5  0x40186e\n#6  0x4e11ef\n#7  0x401e9d\n"}, {'azimuth': 55.74871341043411, 'range': 25.174196184038077, 'status': 0, 'comments': "Error: 'run_datcom' <class EXEdatcom>: Error calling compute(), return_code = 2\nError Output:\nAt line 324 of file misdat.for (unit = 2, file = 'fort.2')\nFortran runtime error: File cannot be deleted\n\nError termination. Backtrace:\n#0  0x4a4f1a\n#1  0x4a5189\n#2  0x4a5c1f\n#3  0x4a6299\n#4  0x457bac\n#5  0x40186e\n#6  0x4e11ef\n#7  0x401e9d\n"}]
(py311) [imoore@sequoia traj_generator]$ 

