def main():
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
            types=["TP-H-3402A", "TP-H-3395A"],
            deck_path=["motor_options", "motor_1_pulse_1_propellant"]
        )
    ]

    from bayes_opt import BayesianOptimization
    from itertools import product

    numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    categorical_specs = [spec for spec in input_params if isinstance(spec, CategoricalSpec)]

    pbounds = { spec.name: spec.bounds for spec in numeric_specs }

    # Create a generator for all categorical combinations.
    if categorical_specs:
        # Build the list of possible values for each categorical spec.
        cat_choices = [spec.types for spec in categorical_specs]
        categorical_combinations = list(product(*cat_choices))
    else:
        # If no categoricals, use an empty tuple.
        categorical_combinations = [()]

    # For each distinct categorical combination, run Bayesian optimization.
    for categorical_values in categorical_combinations:
        # For example, if you have a single categorical input, categorical_values is a 1-tuple.
        print(f"Starting Bayesian Optimization for categorical values: {categorical_values}")
        obj_function = objective_wrapper_factory(categorical_values)
        optimizer = BayesianOptimization(
            f=obj_function,
            pbounds=pbounds,
            random_state=1
        )
        # You may adjust init_points and n_iter as desired.
        optimizer.maximize(init_points=5, n_iter=10)
        
        print(f"Optimization results for categorical values {categorical_values}:")
        best = optimizer.max
        print(f"  Best performance (range): {best['target']}")
        print(f"  Best numeric parameters: {best['params']}")
        print("-" * 50)


def objective_wrapper_factory(categorical_values):
    """
    Returns a function that can be passed to BayesianOptimization.
    categorical_values: a tuple of values corresponding to categorical_specs in order.
    """
    def objective_function(**kwargs):
        # Build a tuple of numeric values in the order of numeric_specs.
        problem_name = f'case_{casenum}'
        p = om.Problem(name=problem_name)

        scenario = dymos_generator(problem=p, input_deck=input_deck)
        for i, spec in enumerate(input_params):
            value = case[i + 1]
            set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)
        print(f"Running Trajectory Tests for case: {casenum}")
        print(f"Processing case data: {case}")

        try:
            scenario.setup()

            dm.run_problem(
                scenario.p,
                run_driver=False,
                simulate=True,
            )
            range = p.get_val("traj_vehicle_0.terminal.timeseries.x", units="NM")[-1, 0]
            result = {
                'input_params': case,
                'range': range,
                'status': "SUCCESS"
            }
        except Exception as e:
            # Allow the error to occur and capture the exception message.
            result = {
                'input_params': case,
                'range': 0,
                'status': f"Error: {e}"
            }
        return result
    return objective_function
    

    exit()
