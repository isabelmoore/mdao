def objective_wrapper_factory(input_params, input_deck):
    """
    Returns an objective function for BayesianOptimization.
    :param categorical_values: a tuple of categorical values in order.
    :param input_params: the complete list of parameter spec objects.
    """
    # Filter the numeric specs from input_params.
    numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    
    def objective_function(**kwargs):
        numeric_values = tuple(kwargs[spec.name] for spec in numeric_specs)
        case = (0,) + numeric_values
                

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                problem_name = f"case_{case[0]}"
                p = om.Problem(name=problem_name)
                scenario = dymos_generator(problem=p, input_deck=input_deck)
                
                for i, spec in enumerate(input_params):
                    value = case[i + 1]
                    set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)

                try:
                    scenario.setup()
                    dm.run_problem(scenario.p, run_driver=False, simulate=True)
                    rng_arr = p.get_val("traj_vehicle_0.terminal.timeseries.x", units="NM")
                    final_range = rng_arr[-1, 0]
                    print("Final range:", final_range)
                    return final_range
                except Exception as e:
                    print("Exception in simulation:", e)
                    return 0.0  # Penalize failures by returning 0
    return objective_function

    
def evaluate_candidate(candidate, f):
    """
    Evaluate a candidate using the provided function.
    
    :param candidate: A dictionary of parameters for the optimization.
    :param f: The objective function to evaluate the candidate.
    """
    return f(**candidate)

def batch_evaluate(optimizer, n_candidates, pool):
    # Gather a batch of candidate points
    candidates = []
    for _ in range(n_candidates):
        candidate = optimizer.suggest()  # use optimizer.suggest()
        if candidate is None:
            break
        candidates.append(candidate)

    if not candidates:
        return [], []
    
    # Use the top-level evaluate_candidate function with pool.map
    results = pool.map(partial(evaluate_candidate, f=optimizer._f), candidates)
    return candidates, results
def save_db(input_params, results):
    # Create a unique database filename with the current date/time.
    db_filename = f"results_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.db"
    db_path = Path(db_filename)

    # Construct table schema and insertion dynamically.
    # Define the columns based on input_params. We'll store numeric values as REAL, and categorical values as TEXT.
    columns = []
    for spec in input_params:
        if isinstance(spec, NumericSpec):
            columns.append((spec.name, "REAL"))
        elif isinstance(spec, CategoricalSpec):
            columns.append((spec.name, "TEXT"))

    # Build the SQL for creating the table.
    # We also add columns for 'range' and 'status'.
    create_table_sql = "CREATE TABLE IF NOT EXISTS results (case_id INTEGER PRIMARY KEY, " + \
        ", ".join([f"{col} {ctype}" for col, ctype in columns] + ["range REAL", "status REAL"]) + ")"

    # Connect (or create) the SQLite database.
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    # Build the dynamic INSERT statement.
    # Get the column names (same order as in our 'columns' list).
    col_names = [col for col, ctype in columns]
    all_cols = ["case_id"] + col_names + ["range", "status"]
    placeholders = ", ".join(["?"] * (len(all_cols)))
    insert_sql = f"INSERT INTO results ({', '.join(all_cols)}) VALUES ({placeholders})"

    # Insert each result into the database.
    for i, result in enumerate(results):
        params_tuple = result['input_params']
        # Build a list of parameter values based on the columns from input_params.
        # Assumes params_tuple ordering: the first element (index 0) might be skipped.
        # The code below extracts each parameter using j+1 as in your print loop.
        param_values = [params_tuple[j+1] for j in range(len(input_params))]
        # Build the complete tuple: case_id, parameter values..., range, and status.
        values = [i] + param_values + [result.get('range'), result.get('status')]
        cursor.execute(insert_sql, values)

    conn.commit()
    conn.close()
    print(f"Results saved to {db_path}")
    
def main(num_processors):
    project_root = Path(__file__).parents[2]  
    import sys
    sys.path.append('/home/imoore/misslemdao') 
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    start_time = time.time()
    
    input_deck_base = "base_input_deck.yml"
    input_deck_filename = 'SFRJ_EK103A_variable.yml'

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
    input_deck = merge_input_decks(base, input_deck_child)

    input_params = [
        NumericSpec(
            name='boost_alpha_0',
            bounds=(-30, 30),
            deck_path=["trajectory_phases", "boost_11", "initial_conditions", "controls", "alpha", 0]
        ),
        NumericSpec(
            name='boost_alpha_1',
            bounds=(-30, 30),
            deck_path=["trajectory_phases", "boost_11", "initial_conditions", "controls", "alpha", 1]
        ),
        CategoricalSpec(
            name='motor_1_pulse_1_propellant',
            types=["TP-H-3402A"],
            deck_path=["motor_options", "motor_1_pulse_1_propellant"]
        )
    ]

    from bayes_opt import BayesianOptimization
    from itertools import product

    # numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    # categorical_specs = [spec for spec in input_params if isinstance(spec, CategoricalSpec)]
    # pbounds = {spec.name: spec.bounds for spec in numeric_specs}

    # # Generate categorical combinations
    # cat_choices = [spec.types for spec in categorical_specs]
    # categorical_combinations = list(product(*cat_choices)) if categorical_specs else [()]
    # categorical_values = categorical_combinations[0]
    # # Create a multiprocessing pool

    # with multiprocessing.Pool(processes=num_processors) as pool_eval:
    #     for categorical_values in categorical_combinations:
    #         print(f"Starting Bayesian Optimization for categorical values: {categorical_values}")
            
    #         obj_function = objective_wrapper_factory(categorical_values, input_params, input_deck)
    #         optimizer = BayesianOptimization(f=obj_function, pbounds=pbounds, random_state=1)
            
    #         # Increase initial trials for better model performance
    #         optimizer.maximize(init_points=15, n_iter=0)
    #         print("done running maximizer")
    #         # Optimize with batch evaluation
    #         batch_size = 16  # Adjusted size to utilize more processing capacity
    #         n_batches = 10   # Keep total count of batches

    #         for batch_num in range(n_batches):
    #             candidates, batch_results = batch_evaluate(optimizer, batch_size, pool_eval)

    #             # Using zip safely ensures candidates and results are linked correctly even when empty lists emerge.
    #             for candidate, target in zip(candidates, batch_results):
    #                 optimizer.register(params=candidate, target=target)

    #             # Streamline logging. Must evaluate how much information is necessary.
    #             print(f"After batch {batch_num + 1}, best so far: {optimizer.max}")

    #     pool_eval.close()
    #     pool_eval.join()
    #     print("Final optimization results:")
    #     print(optimizer.max)
    # exit()
    full_input_samples = GridSampling(input_params, n_samples=50)
    print(full_input_samples)

    # with multiprocessing.Pool(processes=num_processors) as pool:
    #     print("Created pool...")
    #     task_partial = partial(run_traj_test, input_deck=input_deck, input_params=input_params)
    #     results = pool.map(task_partial, full_input_samples)

    # for i, result in enumerate(results):
    #     print(f"Case {i}:")
    #     params_tuple = result['input_params']
    #     for j, spec in enumerate(input_params):
    #         print(f"  {spec.name}: {params_tuple[j+1]}")
    #     print(f"  Range: {result['range']}")
    #     print(f"  Status: {result['comments']}")
    #     print("-" * 40)
    # import sqlite3
    
