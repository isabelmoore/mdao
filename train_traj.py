
def simulate_trajectory(case, input_deck, report_dir):
    casenum = case[0]
    azimuth = case[1]
    range_ = case[2]
    datcom_version='MD0909.exe'
    original_dir = os.getcwd()

    case_dir = report_dir / f'cases/case_{casenum}'
    case_dir.mkdir(parents=True, exist_ok=True)
    temp_exe_path = Path(case_dir, datcom_version)

    Path(case_dir, datcom_version)

    src_exe_path = os.path.join(original_dir, "tools", "datcom", datcom_version)
    shutil.copyfile(src_exe_path, temp_exe_path)

    problem_name = str(case_dir)       
    os.chdir(case_dir)        
    p = om.Problem(name=problem_name)

    scenario = dymos_generator(problem=p, input_deck=input_deck)
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["azimuth"]["equals"] = azimuth
    scenario.p.model_options["vehicle_0"]["trajectory_phases"]["terminal"]["constraints"]["boundary"]["range"]["equals"] = range_
    scenario.setup()
    dm.run_problem(
        scenario.p,
        run_driver=True,
        simulate=False,


def run_search(input_deck_filename: str = '',
        num_processors= 16,
        num_samples: int = 10,
):
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

    # Establish the reports directory
    report_dir = REPORTS_DIR / f"{input_deck_filename.replace('.yml', '')}_date_{date}_time_{timing}"
    report_dir.mkdir(parents=True, exist_ok=True)

    variables = {
        "azimuth": {'bounds': np.array([-90, 90])},
        "range": {'bounds': np.array([10, 60])},
    }

    # Create the design space
    cases = generate_design_space(variables, num_samples)
    print(cases)
    with mp.Pool(processes=num_processors, initializer=_init_mp_context) as pool:
        task = partial(simulate_trajectory, input_deck=input_deck, report_dir=report_dir)
        results = list(tqdm(pool.imap(task, cases, chunksize=1),
                            total=len(cases),
                            desc="Processing Samples"))

#-----------------------------------------------------------------------------------------------------


def modify_input_deck(input_deck_filename,case,original_dir,og_dict,temp_dir):
    input_deck_path = os.path.join(original_dir, "scenarios/input_decks/", input_deck_filename)
    with open(input_deck_path, "r") as f:
        input_deck = safe_load(f)
    
    stages = [0]
    for stage_idx in stages:
        # modify relative file name for aero data directory
        relative_aero_dir = input_deck["aero_dir"]["stage_" + str(stage_idx + 1)] # ["aero_data_dir"]["stage_" + str(stage_idx + 1)]
        absolute_aero_dir = os.path.join(os.path.dirname(__file__), relative_aero_dir)
        
        input_deck["aero_dir"]["stage_" + str(stage_idx + 1)] = temp_dir
    
    for field, value in case.items():
        if field != 'case_name':
            if not replace_key_value(field, input_deck, og_dict[field]['subkey'], value, og_dict[field]['index']):
                raise ValueError(f"doe variable ='{field}' is not defined in the input deck")
        
    return input_deck,input_deck_path,absolute_aero_dir

def run_traj_test(case, og_dict, base_temp_dir,input_deck_filename, datcom_version = 'MD0909.exe'):

    casename = case["case_name"]

    temp_dir = tempfile.mkdtemp(dir=base_temp_dir)

    original_dir = os.getcwd()
    
    temp_exe_path = Path(temp_dir, datcom_version)
    os.chdir(temp_dir)  # Change to the temporary directory

    input_deck,input_deck_path,absolute_aero_dir = modify_input_deck(input_deck_filename,case,original_dir,og_dict,os.getcwd())
    input_deck_optimized = copy.deepcopy(input_deck)
    
    # aero_file = Path(absolute_aero_dir, input_deck["aero_data_options"]["aero_data_filename"])
    # temp_aero_path = Path(temp_dir, input_deck["aero_data_options"]["aero_data_filename"])
    
    # if aero_file.suffix == '.mat':
    #     shutil.copyfile(aero_file, temp_aero_path)
    # else:
    shutil.copyfile(os.path.join(original_dir, "tools/datcom" + f'/{datcom_version}'), temp_exe_path)
    
    problem_name = f'{casename.tolist()[0]}'
    # =============================================================================
    # Create and run scenario
    # =============================================================================
    p = om.Problem(name=problem_name)
    vehicle_num = 0 # default vehicle num because missilemdao only runs one vehicle
    multiple_vehicles = False
    scenario = dymos_generator(problem=p, input_deck=input_deck, vehicle_num=vehicle_num, multiple_vehicles = multiple_vehicles)

    scenario.setup()

    dm.run_problem(scenario.p, run_driver=True, simulate=False, make_plots=True)
    
        
    # =============================================================================
    # Create input deck for frozen design
    # =============================================================================
    # create an input deck with all design vars assigned to optimized values, and design variables and design constraints turned off
    # this creates an input deck with a "frozen" design
    for design_var in input_deck_optimized["design_variables"].keys():
        units = input_deck_optimized["design_variables"][design_var]["units"]
        input_deck_optimized["design_variables"][design_var]["opt"] = False
        # may have extra unused design vars in deck, which will not be in problem. Just skip them if not found
        try:
            input_deck_optimized["design_variables"][design_var]["value"] = scenario.p.model.indep_vars.get_val(design_var, units=units).item()
        except:
            pass
    # turn off all design constraints, as those cannot be affected by design variables that are off
    for constr in input_deck_optimized["design_constraints"].keys():
        input_deck_optimized["design_constraints"][constr]["enforce"] = False


    problem_report_dir = os.path.join(base_temp_dir, problem_name)
    # if "optimized" not in input_deck_filename:
    #     with open(os.path.join(problem_report_dir, "optimized_" + input_deck_filename), "w") as outfile:
    #         dump(input_deck_optimized, outfile, sort_keys=False)
    # shutil.copy(input_deck_path, os.path.join(problem_report_dir, input_deck_filename))
    try:
        shutil.copy("dymos_solution.db", os.path.join(problem_report_dir, "dymos_solution.db"))
        shutil.copy("dymos_simulation.db", os.path.join(problem_report_dir, "dymos_simulation.db"))
    except:
        pass
    os.chdir(original_dir)  # Change back to the original directory
    shutil.rmtree(temp_dir, ignore_errors=True)  # Remove the temporary directory

if __name__ == '__main__':
    start_time = time.time()
    # Pick datcom version executable
    datcom_version = 'MD0909.exe'
    # Specify the input deck
    input_deck_filename = 'one_stage_one_pulse.yml'
    # stick in reports directory
    base_temp_dir = os.path.join(os.path.dirname(__file__), "reports")
    # or stick files somewhere else
    # base_temp_dir = r'file path here'
    num_samples = 4
    variables = {
        "mach": {'bounds': np.array([0.3, 1.0]),
              'subkey': 'lower',
              'index': None},
        "h": {'bounds': np.array([30000, 50000.0]),
                'subkey': None,
                'index': 0},
    }

    cases = generate_design_space(variables, num_samples)
    cases_df = case_label(cases)
    case = np.array_split(cases_df, num_samples)
    pool = multiprocessing.Pool(processes=num_samples)

    # Map the function to the list of traj doe values
    task_partial = partial(run_traj_test, 
                           og_dict = variables, 
                           base_temp_dir = base_temp_dir, 
                           input_deck_filename = input_deck_filename,
                           datcom_version = datcom_version)
    results = pool.map(task_partial, case)
