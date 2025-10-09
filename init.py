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



    numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    categorical_specs = [spec for spec in input_params if isinstance(spec, CategoricalSpec)]
    pbounds = {spec.name: spec.bounds for spec in numeric_specs}

    domain = [{'name': spec.name, 'type': 'continuous', 'domain': spec.bounds} for spec in numeric_specs]   
    obj_function = objective_wrapper_factory(input_params, input_deck)


    full_input_samples = GridSampling(input_params, n_samples=10)
    print(full_input_samples)

    with multiprocessing.Pool(processes=num_processors) as pool:
        print("Created pool...")
        task_partial = partial(run_traj_test, input_deck=input_deck, input_params=input_params)
        results = pool.map(task_partial, full_input_samples)

    for i, result in enumerate(results):
        print(f"Case {i}:")
        params_tuple = result['input_params']
        for j, spec in enumerate(input_params):
            print(f"  {spec.name}: {params_tuple[j+1]}")
        print(f"  Range: {result['range']}")
        print(f"  Status: {result['comments']}")
        print("-" * 40)
    
    df = save_db(input_params, results)
    Xcols = [spec.name for spec in numeric_specs]

    # Now extract features and outcomes dynamically.
    X = df[Xcols].to_numpy(float)
    y_range = df["range"].to_numpy(float)  
    # Assuming the database column "status" was used to store the binary success indicator.
    y_succ = df["status"].astype(int).to_numpy()

    # ---- Build surrogate models based on the dynamic parameters.
    # Set up the Gaussian process regressor using only successful runs.
    ker_r = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(Xcols)), nu=2.5) + WhiteKernel(1e-6)
    reg = Pipeline([
        ("sc", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=ker_r, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=0))
    ])
    # Build a classifier using all runs.
    ker_c = 1.0 * Matern(length_scale=np.ones(len(Xcols)), nu=1.5)
    clf = Pipeline([
        ("sc", StandardScaler()),
        ("gp", GaussianProcessClassifier(kernel=ker_c, random_state=0, max_iter_predict=200))
    ])

    # Fit the regressor on only the successful runs.
    mask_ok = y_succ == 1
    reg.fit(X[mask_ok], y_range[mask_ok])
    clf.fit(X, y_succ)

    # ---- Define Expected Improvement (EI) for a maximization problem.
    best_y = y_range[mask_ok].max() if mask_ok.any() else 0.0

    def ei(mu, sigma, best, xi=0.01):
        # Avoid division by zero.
        sigma = np.maximum(sigma, 1e-12)
        z = (mu - best - xi) / sigma
        return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    # ---- Score candidate points by combining EI with the probability of success.
    def score_candidates(C):
        mu, std = reg.predict(C, return_std=True)
        ei_val = ei(mu, std, best=best_y)
        p_ok = clf.predict_proba(C)[:, 1]
        return ei_val * p_ok, mu, p_ok

    # ---- Propose new candidate points by searching the domain.
    def suggest(n_suggestions=5, n_samples=20000, bounds=None):
        if bounds is None:
            # Generate bounds from the numeric_specs (assumes all have the same structure).
            bounds = [spec.bounds for spec in numeric_specs]
        lows  = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        # Sample uniformly.
        C = np.random.rand(n_samples, len(bounds)) * (highs - lows) + lows
        scores, mu, p = score_candidates(C)
        idx = np.argsort(-scores)[:n_suggestions]
        return C[idx], scores[idx], mu[idx], p[idx]

    cand, s, mu_vals, p_vals = suggest(n_suggestions=10)
    df_candidates = pd.DataFrame(cand, columns=Xcols)
    df_candidates = df_candidates.assign(score=s, pred_range=mu_vals, p_success=p_vals)
    print(df_candidates)
    exit()
