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
def bayesian_optimization_batch(
    input_deck,
    input_params,
    casenum,
    param_bounds,
    init_points=20,
    n_iter=3,
    batch_size=10,
    num_processors=24,
    plot_params=None,
    grid_points=100,
    gif_name='bayesian_optimization.gif',
):
    """
    Run batched Bayesian optimization using your expensive simulation.

    Parameters
    ----------
    input_deck, input_params, casenum : as in your environment
    param_bounds : dict[str, tuple[float, float]]
        E.g., {"boost_alpha_0": (-30, 30), "boost_alpha_1": (-30, 30), "boost_t_dur": (5, 30), "r_throat": (0.1, 4.0)}
    init_points : int
        Number of random initial evaluations.
    n_iter : int
        Number of batch iterations.
    batch_size : int
        Candidates per iteration (evaluated in parallel).
    num_processors : int
        Parallel processes for simulation.
    plot_params : tuple[str, str] | None
        Two parameter names to project/plot. If None, uses the first two keys.
    grid_points : int
        Resolution of the 2D plotting grid.
    gif_name : str
        Output GIF filename.

    Returns
    -------
    optimizer : BayesianOptimization
        The fitted optimizer with registered observations.
    """

    # === Set up optimizer with a dummy objective (we register targets manually) ===
    optimizer = BayesianOptimization(
        f=lambda **params: 0.0,
        pbounds=param_bounds,
        # random_state=42,
    )

    # Acquisition function (UCB is fine; adjust kappa/xi to taste)
    acq = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # Parameter bookkeeping
    keys = list(param_bounds.keys())
    if len(keys) == 0:
        raise ValueError("param_bounds is empty.")

    # Figure out which two params to plot (if any)
    if plot_params is not None:
        if len(plot_params) != 2:
            raise ValueError("plot_params must be a tuple of exactly two parameter names.")
        p1, p2 = plot_params
        if p1 not in keys or p2 not in keys:
            raise ValueError("plot_params must be keys in param_bounds.")
    else:
        # default to first two if available
        if len(keys) >= 2:
            p1, p2 = keys[0], keys[1]
        else:
            p1, p2 = None, None  # Not enough params to do 2D plots
    import numpy as np
    lows = np.array([param_bounds[k][0] for k in keys], dtype=float)
    highs = np.array([param_bounds[k][1] for k in keys], dtype=float)

    # === Initial random candidates ===
    rng = np.random.default_rng(42)
    random_points = rng.uniform(lows, highs, size=(init_points, len(keys)))
    initial_candidates = [{k: float(pt[i]) for i, k in enumerate(keys)} for pt in random_points]

    # === Parallel evaluation (initial) ===
    eval_func = partial(eval_candidate, input_params=input_params, input_deck=input_deck, casenum=casenum)
    # with multiprocessing.Pool(processes=num_processors) as pool:
    #     initial_results = list(
    #         tqdm(pool.imap(eval_func, initial_candidates),
    #              total=len(initial_candidates),
    #              desc="Initial evaluations")
    #     )
    db_filename = "results_2025_10_09_121703.db"  # update to the actual file
    db_path = Path(db_filename)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()
    conn.close()
    initial_results = [dict(row) for row in rows]
    zero_count = sum(1 for r in initial_results if r["status"] == 0)

    print(f"Iteration complete. {zero_count} out of {len(initial_results)} evaluated to zero.")
    
    # Register initial observations
    for cand, result in zip(initial_candidates, initial_results):
        target_value = float(result["range"])
        optimizer.register(params=cand, target=target_value)

    X_train = np.array([list(c.values()) for c in initial_candidates], dtype=float)
    y_train = np.array([float(r["range"]) for r in initial_results], dtype=float)

    # GP for visualization (independent of optimizer's internal GP)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
    if len(X_train) > 0:
        gp.fit(X_train, y_train)

    # For GIF frames
    frames = []
    successful_cases = {spec.name: [] for spec in input_params}
    successful_targets = []
    
    # === Batched iterations ===
    for it in range(n_iter):
        # Suggest a batch of candidates
        batch_candidates = [optimizer.suggest(acq) for _ in range(batch_size)]

        # Evaluate the batch in parallel (use num_processors)
        with multiprocessing.Pool(processes=num_processors) as pool:
            batch_results = list(
                tqdm(pool.imap(eval_func, batch_candidates),
                     total=len(batch_candidates),
                     desc=f"Batch iteration {it + 1}")
            )

        # Diagnostics
        zero_count = sum(1 for r in batch_results if r == 0)
        print(f"Iteration {it + 1} complete. {zero_count} out of {len(batch_results)} evaluated to zero.")

        # Register observations
        for cand, result in zip(batch_candidates, batch_results):
            target = float(result)
            optimizer.register(params=cand, target=target)
            if target != 0:  # Print only non-zero results
                # print(f"Input: {cand}, Output (Target): {target}")
                successful_targets.append(target)  
                for spec in input_params:
                    successful_cases[spec.name].append(cand[spec.name])
        print(successful_cases)
        n_params = len(successful_cases)
        print(n_params)

        # Create subplots for each parameter defined in param_bounds
        fig, axs = plt.subplots(n_params, 1, figsize=(10, 5 * n_params), sharex=True)

        # Ensure axs is iterable if there's only one parameter
        if n_params == 1:
            axs = [axs]

        # When plotting, for each parameter we now zip the candidate values (xvals) with successful_targets
        for i, (name, xvals) in enumerate(successful_cases.items()):
            ax = axs[i]  # Get the current axis for the subplot
            yvals = []
            x_aligned = []

            # Zip each candidate's x-value for this parameter with its corresponding target value.
            for x, target in zip(xvals, successful_targets):
                try:
                    x_val = float(x)
                    y_val = float(target)
                    x_aligned.append(x_val)
                    yvals.append(y_val)
                except (KeyError, TypeError, ValueError):
                    continue

            print(f"Parameter: {name}, x_aligned: {x_aligned}, yvals: {yvals}")

            if not x_aligned or not yvals:
                print(f"No aligned data available for parameter '{name}'. Skipping plot.")
                continue

            # Convert to numpy arrays
            x_arr = np.array(x_aligned, dtype=float)
            y_arr = np.array(yvals, dtype=float)

            # Debugging array sizes
            print(f"x_arr size: {x_arr.size}, y_arr size: {y_arr.size}, x_arr: {x_arr}, y_arr: {y_arr}")

            # For a scatter plot using our x and y data
            sc = ax.scatter(x_arr, y_arr, marker='o', label='Successful Cases')
            ax.set_title(f"Successful Cases for {name}")
            ax.set_ylabel("Target Value (Range)")
            ax.set_xlabel(name)

            # Set x-axis limits using bounds defined in param_bounds
            if name in param_bounds:
                lower, upper = param_bounds[name]
                print(f"{name} bounds: {lower}, {upper}")
                ax.set_xlim(lower, upper)

            # Set y-axis limits based on data
            if y_arr.size > 0:
                ymin, ymax = np.min(y_arr), np.max(y_arr)
                ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))

            ax.grid(True)
            ax.legend()


        axs[-1].set_xlabel("Parameter Value")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # Adjust this value as needed
        frame_filename = f"iteration_{it + 1}.png"
        plt.savefig(frame_filename, dpi=200)
        print(f"Saved scatter grid to {frame_filename}")

        # Update GP train set and refit visualization GP
        if len(batch_candidates) > 0:
            X_batch = np.array([list(c.values()) for c in batch_candidates], dtype=float)
            y_batch = np.array(batch_results, dtype=float)
            X_train = np.vstack((X_train, X_batch))
            y_train = np.concatenate((y_train, y_batch))
            gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
            gp.fit(X_train, y_train)

        # # Best-so-far target
        # if optimizer.max is not None and 'target' in optimizer.max:
        #     y_max = float(optimizer.max['target'])
        # else:
        #     y_max = float(np.max(y_train)) if y_train.size > 0 else 0.0
        # print("y_max:", y_max)
        # Set y_max to the maximum of the current batch results
        y_max = max(batch_results) if batch_results else 0.0  # fallback in case of no results
        print("y_max:", y_max)
        # ----------------------------------------------------------------------------------
        # Visualization (2D projection over p1, p2). If fewer than 2 params, skip plotting.
        # ----------------------------------------------------------------------------------
        if p1 is not None and p2 is not None:
            # Fix remaining params at current best (fallback to midpoint if no best yet)
            if optimizer.max is not None and 'params' in optimizer.max:
                fixed = {k: float(optimizer.max['params'][k]) for k in keys if k not in (p1, p2)}
            else:
                fixed = {k: float((param_bounds[k][0] + param_bounds[k][1]) / 2.0)
                         for k in keys if k not in (p1, p2)}

            # 2D mesh over p1, p2
            p1_lin = np.linspace(param_bounds[p1][0], param_bounds[p1][1], grid_points)
            p2_lin = np.linspace(param_bounds[p2][0], param_bounds[p2][1], grid_points)
            X1, X2 = np.meshgrid(p1_lin, p2_lin)

            # Build full-dim X_query for the grid
            X_query = []
            for a, b in zip(X1.ravel(), X2.ravel()):
                row = []
                for k in keys:
                    if k == p1:
                        row.append(a)
                    elif k == p2:
                        row.append(b)
                    else:
                        row.append(fixed[k])
                X_query.append(row)
            X_query = np.array(X_query, dtype=float)  # (grid_points^2, D)

            # GP predictions + acquisition on the grid
            if X_train.size > 0:
                Z_mu, Z_std = gp.predict(X_query, return_std=True)
                Z_acq = acq.utility(X_query, gp, y_max)  # <-- correct signature
            else:
                # If somehow we have nothing trained, show blanks
                Z_mu = np.zeros(X_query.shape[0], dtype=float)
                Z_std = np.zeros(X_query.shape[0], dtype=float)
                Z_acq = np.zeros(X_query.shape[0], dtype=float)

            # Plot: mean, std, acquisition
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            im0 = axs[0].contourf(X1, X2, Z_mu.reshape(X1.shape), levels=50)
            axs[0].set_title('GP Predicted Mean')
            axs[0].set_xlabel(p1); axs[0].set_ylabel(p2)
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].contourf(X1, X2, Z_std.reshape(X1.shape), levels=50)
            axs[1].set_title('GP Predictive Std')
            axs[1].set_xlabel(p1); axs[1].set_ylabel(p2)
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].contourf(X1, X2, Z_acq.reshape(X1.shape), levels=50)
            axs[2].set_title(f'Acquisition ({acq.kind.upper()})')
            axs[2].set_xlabel(p1); axs[2].set_ylabel(p2)
            fig.colorbar(im2, ax=axs[2])

            plt.tight_layout()
            frame_filename = f"frame_{it:03d}.png"
            plt.savefig(frame_filename, dpi=120)
            plt.close(fig)
            frames.append(frame_filename)

    # ----------------------------------------------------------------------------------
    # Build GIF from frames (if any)
    # ----------------------------------------------------------------------------------
    if frames:
        with imageio.get_writer(gif_name, mode='I', duration=1.0) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))

        # Clean up frame files
        for frame in frames:
            try:
                os.remove(frame)
            except OSError:
                pass

        print(f"GIF created: {gif_name}")

    print("Optimization finished.")
    return optimizer
