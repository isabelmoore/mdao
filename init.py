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
        random_state=42,
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

    lows = np.array([param_bounds[k][0] for k in keys], dtype=float)
    highs = np.array([param_bounds[k][1] for k in keys], dtype=float)

    # === Initial random candidates ===
    rng = np.random.default_rng(42)
    random_points = rng.uniform(lows, highs, size=(init_points, len(keys)))
    initial_candidates = [{k: float(pt[i]) for i, k in enumerate(keys)} for pt in random_points]

    # === Parallel evaluation (initial) ===
    eval_func = partial(eval_candidate, input_params=input_params, input_deck=input_deck, casenum=casenum)
    with mp.Pool(processes=num_processors) as pool:
        initial_results = list(
            tqdm(pool.imap(eval_func, initial_candidates),
                 total=len(initial_candidates),
                 desc="Initial evaluations")
        )

    # Register initial observations
    for cand, result in zip(initial_candidates, initial_results):
        optimizer.register(params=cand, target=float(result))
        print(f"Input: {cand}, Output (Target): {result}")

    # Build training arrays for our GP visualization model
    X_train = np.array([list(c.values()) for c in initial_candidates], dtype=float)
    y_train = np.array(initial_results, dtype=float)

    # GP for visualization (independent of optimizer's internal GP)
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
    if len(X_train) > 0:
        gp.fit(X_train, y_train)

    # For GIF frames
    frames = []

    # === Batched iterations ===
    for it in range(n_iter):
        # Suggest a batch of candidates
        batch_candidates = [optimizer.suggest(acq) for _ in range(batch_size)]

        # Evaluate the batch in parallel (use num_processors)
        with mp.Pool(processes=num_processors) as pool:
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
            optimizer.register(params=cand, target=float(result))
            print(f"Input: {cand}, Output (Target): {result}")

        # Update GP train set and refit visualization GP
        if len(batch_candidates) > 0:
            X_batch = np.array([list(c.values()) for c in batch_candidates], dtype=float)
            y_batch = np.array(batch_results, dtype=float)
            X_train = np.vstack((X_train, X_batch))
            y_train = np.concatenate((y_train, y_batch))
            gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
            gp.fit(X_train, y_train)

        # Best-so-far target
        if optimizer.max is not None and 'target' in optimizer.max:
            y_max = float(optimizer.max['target'])
        else:
            y_max = float(np.max(y_train)) if y_train.size > 0 else 0.0
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
