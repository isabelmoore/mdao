def simulate_trajectory(case, input_deck, input_params, casenum):
    """
    Run the simulation for a given case and return a dictionary with the results.
    'case' is a list where case[1:] corresponds to the values for input_params in order.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                problem_name = f'case_{casenum}'
                p = om.Problem(name=problem_name)
                scenario = dymos_generator(problem=p, input_deck=input_deck)
                # Set parameters based on the provided NumericSpec objects.
                for i, spec in enumerate(input_params):
                    # case[0] can be a dummy placeholder; values start at index 1.
                    value = case[i + 1]
                    set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)
                scenario.setup()
                dm.run_problem(
                    scenario.p,
                    run_driver=False,
                    simulate=True,
                )
                # Use a different name instead of 'range' to avoid shadowing the built-in range.
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
def eval_candidate(candidate, input_params, input_deck, casenum):
    """
    Convert candidate dict to simulation 'case' and evaluate target metric.
    This function is defined at module level so it can be pickled.
    """
    # Build the case list: index 0 is a dummy placeholder and then follow the candidate values
    case = [None] + [candidate[spec.name] for spec in input_params]
    sim_result = simulate_trajectory(case, input_deck, input_params, casenum)
    return sim_result["range"]
def bayesian_optimization_batch(input_deck, input_params, casenum,
                                param_bounds, init_points=20, n_iter=3,
                                batch_size=10, num_processors=24):
    """
    Run batched Bayesian optimization using your expensive simulation.
    """
    # Create a Bayesian optimizer with a dummy objective (since we register results manually).
    optimizer = BayesianOptimization(
        f=lambda **params: 0,
        pbounds=param_bounds,
        random_state=42,
    )
    # Create a utility function; here we use UCB.
    acq  = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)


    # Build a list of parameter keys.
    keys = list(param_bounds.keys())
    lows = np.array([param_bounds[key][0] for key in keys])
    highs = np.array([param_bounds[key][1] for key in keys])
    
    # Generate the initial random candidates.
    random_points = np.random.uniform(lows, highs, size=(init_points, len(keys)))
    initial_candidates = [{key: point[i] for i, key in enumerate(keys)} for point in random_points]

    # Create a partial function that already has input_params, input_deck, and casenum bound.
    eval_func = partial(eval_candidate, input_params=input_params,
                        input_deck=input_deck, casenum=casenum)
    
    # Make sure you pass eval_func (not eval_candidate) to pool.imap.
    with multiprocessing.Pool(processes=num_processors) as pool:
        initial_results = list(tqdm(pool.imap(eval_func, initial_candidates),
                                    total=len(initial_candidates),
                                    desc="Initial evaluations"))
    
    # Register these observations with the Bayesian optimizer.
    for cand, result in zip(initial_candidates, initial_results):
        optimizer.register(params=cand, target=result)
        print(f"Input: {cand}, Output (Target): {result}")
    # orig_bounds = param_bounds.copy()

    # Track the Gaussian Process for updates
    # Extract parameters and corresponding targets in the correct shape
    X_train = np.array([list(cand.values()) for cand in initial_candidates])
    y_train = np.array(initial_results)

    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))  # Initialize the GP model
    gp.fit(X_train, y_train)  # Fit the model with correct shapes


    # Iteratively evaluate batches of candidate points.
    for it in range(n_iter):
        batch_candidates = []
        for _ in range(batch_size):
            candidate = optimizer.suggest(acq)
            batch_candidates.append(candidate)

        # Evaluate the batch in parallel
        with multiprocessing.Pool(processes=batch_size) as pool:
            batch_results = list(tqdm(pool.imap(eval_func, batch_candidates),
                                    total=len(batch_candidates),
                                    desc=f"Batch iteration {it + 1}"))

        # Count how many results are zero.
        zero_count = sum(1 for result in batch_results if result == 0)
        print(f"Iteration {it + 1} complete. {zero_count} out of {len(batch_results)} evaluated to zero.")
        
        # Register the results with the optimizer
        for cand, result in zip(batch_candidates, batch_results):
            optimizer.register(params=cand, target=result)

            # Print the inputs (parameters) and outputs (results)
            print(f"Input: {cand}, Output (Target): {result}")

        # Update the Gaussian Process model with the new results
        X_train = np.vstack((X_train, np.array([list(cand.values()) for cand in batch_candidates])))
        y_train = np.concatenate((y_train, batch_results))  # Concatenate results
        gp.fit(X_train, y_train)  # Refit the GP model with updated data

        # Calculate y_max from the maximum observed target value
        y_max = optimizer.max['target'] if optimizer.max is not None else max(batch_results) if batch_results else 0
        print("y_max:", y_max)

        # Create the plot for the current iteration
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        X1, X2 = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))

        # Generate predictions and the acquisition function
        Z_gp_mean, _ = gp.predict(X_train, return_std=True)
        Z_acquisition = acq.utility(gp, y_max)  # Utilize gp and y_max correctly

        # Create contour plots
        axs[0].contourf(X1, X2, Z_gp_mean.reshape(X1.shape), levels=100, cmap='viridis')
        axs[0].set_title('Gaussian Process Predicted Mean')

        axs[1].contourf(X1, X2, np.random.rand(*X1.shape), levels=100, cmap='viridis')
        axs[1].set_title('Random Landscape (Simulated)')

        axs[2].contourf(X1, X2, Z_acquisition.reshape(X1.shape), levels=100, cmap='viridis')
        axs[2].set_title('Acquisition Function')

        plt.tight_layout()
        frame_filename = f"frame_{it}.png"
        plt.savefig(frame_filename)
        plt.close(fig)

        # Create GIF from frames
    with imageio.get_writer('bayesian_optimization.gif', mode='I', duration=1) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    # Optionally: Clean up temporary frame files
    for frame in frames:
        os.remove(frame)

    print("GIF created: bayesian_optimization.gif")
    
    print("Optimization finished.")
    return optimizer




(py311) [imoore@sequoia init_cond_opt]$ python train_init_perturbation.py
Number of CPU cores: 32
Number of available CPUs: 32
Optimal number of processors: 16
Initial evaluations: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:57<00:00,  9.58s/it]
Input: {'boost_alpha_0': 28.695925891169175, 'boost_alpha_1': -9.556887081250682, 'boost_t_dur': 11.80999262535104, 'r_throat': 1.445287691093599}, Output (Target): 0
Input: {'boost_alpha_0': -3.4892858710217958, 'boost_alpha_1': -7.195443491698793, 'boost_t_dur': 9.419355508289609, 'r_throat': 3.4049632842784656}, Output (Target): 0
Input: {'boost_alpha_0': -28.763460283957766, 'boost_alpha_1': -1.1035959196202292, 'boost_t_dur': 26.641716730734153, 'r_throat': 2.9667750733155733}, Output (Target): 0
Input: {'boost_alpha_0': -11.042055776890574, 'boost_alpha_1': -0.1718530638254947, 'boost_t_dur': 22.49357472726761, 'r_throat': 1.1265597986061453}, Output (Target): 0
Input: {'boost_alpha_0': -8.581387395810406, 'boost_alpha_1': 21.759193615370037, 'boost_t_dur': 24.420395844557696, 'r_throat': 3.5660126158198215}, Output (Target): 0
Input: {'boost_alpha_0': -17.319699070375258, 'boost_alpha_1': 11.887322252673634, 'boost_t_dur': 16.327332078124485, 'r_throat': 0.28380333182442574}, Output (Target): 72.53444396918908
Batch iteration 1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:53<00:00, 26.77s/it]
Iteration 1 complete. 1 out of 2 evaluated to zero.
Input: {'boost_alpha_0': -17.531767529880867, 'boost_alpha_1': 11.748033692651894, 'boost_t_dur': 16.811666693112283, 'r_throat': 0.1865250675090119}, Output (Target): 83.6805900691833
Input: {'boost_alpha_0': -27.046883120394863, 'boost_alpha_1': -28.891007999206412, 'boost_t_dur': 18.414666265866213, 'r_throat': 3.646752041252828}, Output (Target): 0
y_max: 83.6805900691833
Traceback (most recent call last):
  File "/home/imoore/misslemdao/tools/traj_ann/init_cond_opt/train_init_perturbation.py", line 758, in <module>
    main(16)
  File "/home/imoore/misslemdao/tools/traj_ann/init_cond_opt/train_init_perturbation.py", line 684, in main
    optimizer = bayesian_optimization_batch(input_deck, input_params, casenum,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/imoore/misslemdao/tools/traj_ann/init_cond_opt/train_init_perturbation.py", line 587, in bayesian_optimization_batch
    Z_acquisition = acq.utility(gp, y_max)  # Utilize gp and y_max correctly
                    ^^^^^^^^^^^^^^^^^^^^^^
TypeError: UtilityFunction.utility() missing 1 required positional argument: 'y_max'
(py311) [imoore@sequoia init_cond_opt]$ 
