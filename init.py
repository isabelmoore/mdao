
@dataclass
class NumericSpec:
    name: str
    bounds: Tuple[float, float]
    deck_path: List[Union[str, int]]

@dataclass
class CategoricalSpec:
    name: str
    types: List[str]
    deck_path: List[Union[str, int]]

def set_nested_value(dic: Dict, keys: List[Union[str, int]], value) -> None:
    d = dic
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value

def merge_input_decks(base, child):
    for key, value in base.items():
        if key not in child:
            # If the key is missing in the child, use the value from the base
            child[key] = value
        elif isinstance(value, dict) and isinstance(child[key], dict):
            # If both values are dictionaries, merge them recursively
            merge_input_decks(value, child[key])
    return child
    
# phase 0: creating region
class LatinHypercube:
    def __init__(self, numeric_specs, n_samples):
        self.specs = numeric_specs
        self.n_samples = n_samples
        self.rng = np.random.default_rng()
        
    # Full joint LHS sampling for numeric parameters
    def sample(self) -> List[dict[str, float]]:
        names = [s.name for s in self.specs]
        bounds = [s.bounds for s in self.specs]
        d = len(self.specs)
        cut = np.linspace(0, 1, self.n_samples + 1)
        u = self.rng.random((self.n_samples, d))
        pts01 = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])
        
        # Shuffle each column independently
        for j in range(d):
            self.rng.shuffle(pts01[:, j])
        
        samples = []
        for i in range(self.n_samples):
            row = {}
            for j, (name, (lo, hi)) in enumerate(zip(names, bounds)):
                row[name] = float(lo + pts01[i, j] * (hi - lo))
            samples.append(row)
        return samples

    # Independent per-variable LHS sampling (if needed)
    def sample_per_variable(self) -> dict[str, list[float]]:
        results = {}
        for spec in self.specs:
            lo, hi = spec.bounds
            cut = np.linspace(0, 1, self.n_samples + 1)
            u = self.rng.random(self.n_samples)
            pts01 = cut[:-1] + u * (cut[1:] - cut[:-1])
            self.rng.shuffle(pts01)
            results[spec.name] = [float(lo + p * (hi - lo)) for p in pts01]
        return results

# # phase 0: for catigorical values
def GridSampling(input_params, n_samples):
    numeric_samples = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    categorical_specs = [spec for spec in input_params if isinstance(spec, CategoricalSpec)]
    numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]

    # Use numeric_specs for LatinHypercube sampling
    lhc = LatinHypercube(numeric_specs, n_samples=n_samples)
    numeric_samples = lhc.sample()

    cat_choices = [spec.types for spec in categorical_specs]
    cat_combinations = list(product(*cat_choices))

    # Convert each tuple to a dictionary (using the categorical name as key)
    categorical_samples = []
    for comb in cat_combinations:
        sample = {}
        for i, spec in enumerate(categorical_specs):
            sample[spec.name] = comb[i]
        categorical_samples.append(sample)

    # Combine numeric samples with every categorical combination
    full_input_samples = []
    for num_sample in numeric_samples:
        for cat_sample in categorical_samples:
            sample = num_sample.copy()
            sample.update(cat_sample)
            full_input_samples.append(sample)
    
    keys = tuple(spec.name for spec in input_params)
    
    # Create a tuple that begins with a case id (i) followed by the values in the defined order.
    tuple_samples = tuple(
        (i,) + tuple(sample[k] for k in keys)
        for i, sample in enumerate(full_input_samples)
    )
    
    return tuple_samples


def save_db(input_params, results):
    db_filename = f"results_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}.db"
    db_path = Path(db_filename)

    columns = []
    for spec in input_params:
        if isinstance(spec, NumericSpec):
            columns.append((spec.name, "REAL"))
        elif isinstance(spec, CategoricalSpec):
            columns.append((spec.name, "TEXT"))

    create_table_sql = "CREATE TABLE IF NOT EXISTS results (case_id INTEGER PRIMARY KEY, " + \
        ", ".join([f"{col} {ctype}" for col, ctype in columns] + ["range REAL", "status REAL"]) + ")"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(create_table_sql)

    col_names = [col for col, ctype in columns]
    all_cols = ["case_id"] + col_names + ["range", "status"]
    placeholders = ", ".join(["?"] * (len(all_cols)))
    insert_sql = f"INSERT INTO results ({', '.join(all_cols)}) VALUES ({placeholders})"

    for i, result in enumerate(results):
        params_tuple = result['input_params']
        param_values = [params_tuple[j+1] for j in range(len(input_params))]
        values = [i] + param_values + [result.get('range'), result.get('status')]
        cursor.execute(insert_sql, values)

    conn.commit()
    df = pd.read_sql_query("SELECT * FROM results", conn)

    conn.close()
    print(f"Results saved to {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # This converts each row to a dictionary-like object.
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()

    conn.close()

    # Convert rows to standard dictionaries.
    results = [dict(row) for row in rows]
    return results

# phase 1: finding feasibility region (wrapper)
def run_traj_test(case, input_deck, input_params):
    casenum = case[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openmdao")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="openmdao")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                problem_name = f'case_{casenum}'
                p = om.Problem(name=problem_name)

                scenario = dymos_generator(problem=p, input_deck=input_deck)
                for i, spec in enumerate(input_params):
                    value = case[i + 1]
                    set_nested_value(scenario.p.model_options["vehicle_0"], spec.deck_path, value)

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

def load_db_optimize(db_filename):
    db_path = Path(db_filename)
    initial_results: list[float] = []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()
    conn.close()

    # Map DB rows to (candidate, result). We only keep rows that cover all params.
    loaded_candidates = []
    loaded_results = []
    for row in rows:
        if "range" in row.keys():
            cand = {}
            ok = True
            for k in keys:
                if k in row.keys():
                    cand[k] = float(row[k])
                else:
                    ok = False
                    break
            if ok:
                loaded_candidates.append(cand)
                loaded_results.append(float(row["range"]))

    # Only use DB warm-start if we have at least as many entries as init_points
    if len(loaded_results) >= init_points:
        initial_candidates = loaded_candidates[:init_points]
        initial_results = loaded_results[:init_points]
    return initial_results

def unpack(results):
    rows = []
    for i, r in enumerate(results):
        row = {"case_id": i}
        for key in r.keys():
            if key not in ['status', 'case_id', 'comments']:
                row[key] = r[key]
        row["range"] = r.get("range", None)  
        row["success"] = int(r.get("success", r.get("status", 0)))  
        
        rows.append(row)
    return rows

def plot_range(results):
    rows = unpack(results)
    success_x, success_y, success_range = [], [], []
    unsuccess_x, unsuccess_y = [], []
    ranges = []
    for row in rows:
        x_val = row["boost_alpha_0"]
        y_val = row["boost_alpha_1"]
        rng = row["range"]
        status = row["success"]
        if status == 1:  
            success_x.append(x_val)
            success_y.append(y_val)
            success_range.append(rng)
        else:  
            unsuccess_x.append(x_val)
            unsuccess_y.append(y_val)
        ranges.append(rng)
    plt.figure(figsize=(12, 8))

    sc = plt.scatter(success_x, success_y, c=success_range, cmap='viridis', marker='o', label='Successful Cases')
    plt.scatter(unsuccess_x, unsuccess_y, color='red', marker='o', label='Unsuccessful Cases')

    cbar = plt.colorbar(sc)
    cbar.set_label('Actual Range')

    plt.xlabel("Boost Alpha 0 (X)")
    plt.ylabel("Boost Alpha 1 (Y)")
    plt.title("Scatter Plot of Alpha Values (Successful: Gradient Range, Unsuccessful: Red)")
    plt.legend()
    plt.grid(True)
    plt.savefig("testing_actual_ranges.png")

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
    casenum: int,
    param_bounds: dict[str, tuple[float, float]],
    init_input,
    n_iter: int = 3,
    batch_size: int = 10,
    num_processors: int = 24,
    plot_params: tuple[str, str] | None = None,
    grid_points: int = 100,
    retrain=False, 
    db_filename = str,
):

    # --- Validate bounds ---
    keys = list(param_bounds.keys())
    if not keys:
        raise ValueError("param_bounds is empty.")

    # --- Which two to plot ---
    if plot_params is not None:
        if len(plot_params) != 2:
            raise ValueError("plot_params must be a tuple of exactly two parameter names.")
        p1, p2 = plot_params
        if p1 not in keys or p2 not in keys:
            raise ValueError("plot_params must be keys in param_bounds.")
    else:
        p1, p2 = (keys[0], keys[1]) if len(keys) >= 2 else (None, None)

    # --- Optimizer scaffold (we register manually) ---
    optimizer = BayesianOptimization(
        f=lambda **params: 0.0,           # dummy; we register actual results
        pbounds=param_bounds,
        verbose=0,
    )

    # Acquisition function (UCB works well; tune kappa/xi as needed)
    acq = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # --- Random initial candidates ---
    rng = np.random.default_rng(42)
    lows = np.array([param_bounds[k][0] for k in keys], dtype=float)
    highs = np.array([param_bounds[k][1] for k in keys], dtype=float)
    random_points = rng.uniform(lows, highs, size=(init_points, len(keys)))
    initial_candidates = [{k: float(pt[i]) for i, k in enumerate(keys)} for pt in random_points]

    # --- Build evaluator ---
    eval_func = partial(
        eval_candidate,
        input_params=input_params,
        input_deck=input_deck,
        casenum=casenum
    )

    if retrain:
        with multiprocessing.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            initial_results = list(
                tqdm(pool.imap(eval_func, initial_candidates),
                        total=len(initial_candidates),
                        desc="Initial evaluations")
            )
    else:

        # --- Try warm-start from DB if present; else actually evaluate initial points ---
        initial_results = load_db_optimize()



    zero_count = sum(1 for r in initial_results if float(r) == 0.0)
    print(f"Initial set complete. {zero_count} / {len(initial_results)} evaluated to zero.")

    # --- Register initial observations with BO ---
    for cand, target_value in zip(initial_candidates, initial_results):
        optimizer.register(params=cand, target=float(target_value))

    X_train = np.array([list(c.values()) for c in initial_candidates], dtype=float)
    y_train = np.array([float(v) for v in initial_results], dtype=float)

    # --- Visualization GP (separate from BO's internal GP) ---
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
    if X_train.shape[0] >= 1:
        gp.fit(X_train, y_train)

    # --- Success caches for per-parameter scatter ---
    successful_cases_initial = {spec.name: [] for spec in input_params}
    successful_targets_initial: list[float] = []
    for cand, val in zip(initial_candidates, initial_results):
        if float(val) != 0.0:
            successful_targets_initial.append(float(val))
            for spec in input_params:
                successful_cases_initial[spec.name].append(cand[spec.name])

    # --- Frames for GIF (2D surfaces) ---
    frames: list[str] = []

    successful_cases = {spec.name: [] for spec in input_params}
    successful_targets: list[float] = []
    for it in range(n_iter):
        # Suggest a batch
        batch_candidates = [optimizer.suggest(acq) for _ in range(batch_size)]

        # Evaluate in parallel
        with multiprocessing.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            batch_results = list(
                tqdm(pool.imap(eval_func, batch_candidates),
                     total=len(batch_candidates),
                     desc=f"Batch {it + 1}")
            )
        print("batch_results:", batch_results)
        zero_count = sum(1 for r in batch_results if float(r) == 0.0)
        print(f"Iteration {it + 1} complete. {zero_count} / {len(batch_results)} evaluated to zero.")

        # Register + success caches
        for cand, result in zip(batch_candidates, batch_results):
            target = float(result)
            optimizer.register(params=cand, target=target)
            if target != 0.0:
                successful_targets.append(target)
                for spec in input_params:
                    successful_cases[spec.name].append(cand[spec.name])

        # -- Per-parameter scatter (skip gracefully if nothing successful yet) --
        n_params = len(successful_cases)
        has_any = any(len(v) > 0 for v in successful_cases.values())
        if n_params > 0 and has_any:
            fig, axs = plt.subplots(n_params, 1, figsize=(10, 4.5 * n_params), sharex=False)
            if n_params == 1:
                axs = [axs]

            # Each param list is aligned with successful_targets by construction
            for ax, (name, xvals) in zip(axs, successful_cases_initial.items()):
                x_arr = np.asarray(xvals, dtype=float)
                y_arr = np.asarray(successful_targets_initial, dtype=float)
                m = min(len(x_arr), len(y_arr))
                if m == 0:
                    continue
                x_arr, y_arr = x_arr[:m], y_arr[:m]

                ax.scatter(x_arr, y_arr, marker='o', color='r', label='Initial')
                ax.set_title(f"Successful Cases for {name}")
                ax.set_ylabel("Target (Range)")
                ax.set_xlabel(name)

                if name in param_bounds:
                    lo, hi = param_bounds[name]
                    ax.set_xlim(lo, hi)

                if y_arr.size:
                    ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
                    pad = 0.1 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
                    # ax.set_ylim(ymin - pad, ymax + pad)

                ax.grid(True)
                ax.legend()

            for ax, (name, xvals) in zip(axs, successful_cases.items()):
                x_arr = np.asarray(xvals, dtype=float)
                y_arr = np.asarray(successful_targets, dtype=float)
                m = min(len(x_arr), len(y_arr))
                if m == 0:
                    continue
                x_arr, y_arr = x_arr[:m], y_arr[:m]

                ax.scatter(x_arr, y_arr, marker='o', color='b', label='Batches')
                ax.set_title(f"Successful Cases for {name}")
                ax.set_ylabel("Target (Range)")
                ax.set_xlabel(name)

                if name in param_bounds:
                    lo, hi = param_bounds[name]
                    ax.set_xlim(lo, hi)

                if y_arr.size:
                    ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
                    pad = 0.1 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
                    # ax.set_ylim(ymin - pad, ymax + pad)

                ax.grid(True)
                ax.legend()

            plt.tight_layout()
            frame_filename = f"iteration_{it + 1}.png"
            plt.savefig(frame_filename, dpi=200)
            plt.close(fig)
            print(f"Saved scatter grid to {frame_filename}")

        # -- Update visualization GP with batch --
        if batch_candidates:
            X_batch = np.array([list(c.values()) for c in batch_candidates], dtype=float)
            y_batch = np.array([float(v) for v in batch_results], dtype=float)
            X_train = np.vstack((X_train, X_batch))
            y_train = np.concatenate((y_train, y_batch))
            gp.fit(X_train, y_train)

        # Best-so-far y_max for acquisition plots (global, not just batch)
        y_max = float(np.max(y_train)) if y_train.size else 0.0
        print("Best-so-far y_max:", y_max)

        # -- 2D projection (p1,p2) if available and we have training data --
        if p1 is not None and p2 is not None and X_train.size > 0:
            # Fix others at current best params (fallback to midpoints)
            if optimizer.max is not None and 'params' in optimizer.max:
                fixed = {k: float(optimizer.max['params'][k]) for k in keys if k not in (p1, p2)}
            else:
                fixed = {k: float((param_bounds[k][0] + param_bounds[k][1]) / 2.0)
                         for k in keys if k not in (p1, p2)}

            p1_lin = np.linspace(param_bounds[p1][0], param_bounds[p1][1], grid_points)
            p2_lin = np.linspace(param_bounds[p2][0], param_bounds[p2][1], grid_points)
            X1, X2 = np.meshgrid(p1_lin, p2_lin)

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
            X_query = np.array(X_query, dtype=float)

            Z_mu, Z_std = gp.predict(X_query, return_std=True)
            Z_acq = acq.utility(X_query, gp, y_max)

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

    # --- Build GIF from frames (if any) ---
    if frames:
        with imageio.get_writer(gif_name, mode='I', duration=1.0) as writer:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        # Best-effort cleanup
        for frame in frames:
            try:
                os.remove(frame)
            except OSError:
                pass
        print(f"GIF created: {gif_name}")

    print("Optimization finished.")
    return optimizer

def main(input_deck_filename, num_processors, n_samples, input_params, db_filename):
    input_deck_base = "base_input_deck.yml"

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

    input_deck = merge_input_decks(base, input_deck_child)

    numeric_specs = [spec for spec in input_params if isinstance(spec, NumericSpec)]
    categorical_specs = [spec for spec in input_params if isinstance(spec, CategoricalSpec)]
    param_bounds = {spec.name: spec.bounds for spec in numeric_specs}


    if optimize:
        optimizer = bayesian_optimization_batch(input_deck, input_params, casenum=1,
                                param_bounds, init_iters=500, n_iter=10,
                                batch_size=10, num_processors=num_processors, retrain=retrain,
                                db_filename=db_filename)
    
    if retrain:
        full_input_samples = GridSampling(input_params, n_samples=n_samples)

        with multiprocessing.Pool(processes=num_processors) as pool:
            print("Created pool...")
            task_partial = partial(run_traj_test, input_deck=input_deck, input_params=input_params)
            results = list(tqdm(pool.imap(task_partial, full_input_samples), total=len(full_input_samples), desc="Processing Samples"))
        results = save_db(input_params, results)

    else:
        db_path = Path(db_filename)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()
        conn.close()
        results = [dict(row) for row in rows]

    # --- Filter + sort successful results ---
    successful_results = [res for res in results if res.get('status') == 1]
    sorted_successful = sorted(successful_results, key=lambda r: r.get('range', float('-inf')), reverse=True)

    print("Top 10 Successful Cases by Range:")
    table = PrettyTable()

    if sorted_successful:
        # Build table headers from the first row, excluding some keys
        sample_case = sorted_successful[0]
        excluded_keys = {'status', 'case_id', 'comments'}
        field_names = [k for k in sample_case.keys() if k not in excluded_keys]
        table.field_names = field_names

        # Add up to 10 rows, formatting floats
        for result in sorted_successful[:10]:
            row = []
            for key in field_names:
                val = result.get(key)
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            table.add_row(row)

        print(table)
    else:
        print("No successful results found (status == 1). Skipping table and plots.")

    try:
        numeric_param_names = [spec.name for spec in input_params if isinstance(spec, NumericSpec)]
    except NameError:
        # Fallback: attempt to infer numeric columns heuristically (float/int) from the first success
        if sorted_successful:
            candidate_keys = [k for k, v in sorted_successful[0].items() if k not in {'status', 'case_id', 'comments', 'range'}]
            # Keep keys whose values are numeric in most rows
            numeric_param_names = []
            for k in candidate_keys:
                vals = [r.get(k) for r in sorted_successful[:50]]
                nums = [x for x in vals if isinstance(x, (int, float))]
                if len(nums) >= max(3, int(0.6 * len(vals))):
                    numeric_param_names.append(k)
        else:
            numeric_param_names = []

    # Prepare X (param values) and Y (target) aligned with sorted_successful order
    successful_targets = np.array([float(r['range']) for r in sorted_successful], dtype=float)

    # Collect values for each numeric parameter
    successful_cases = {name: [] for name in numeric_param_names}
    for r in sorted_successful:
        for name in numeric_param_names:
            v = r.get(name)
            if v is not None:
                try:
                    successful_cases[name].append(float(v))
                except (TypeError, ValueError):
                    # Skip non-numeric entries
                    pass

    successful_cases = {k: v for k, v in successful_cases.items() if len(v) > 0}
    if not successful_cases:
        print("No numeric parameter data available to plot. Skipping plots.")
    else:
        n_params = len(successful_cases)

        # Create one subplot per parameter; handle the single-axes case
        fig, axs = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=False)
        if n_params == 1:
            axs = [axs]
        # Loop over each parameter and plot the scatter with a gradient based on y value
        for ax, (name, xvals) in zip(axs, successful_cases.items()):
            yvals = []
            x_aligned = []
            for x, r in zip(xvals, sorted_successful):
                try:
                    y = float(r['range'])
                    x_aligned.append(float(x))
                    yvals.append(y)
                except (KeyError, TypeError, ValueError):
                    continue
            if len(x_aligned) == 0:
                print(f"No aligned data to plot for parameter '{name}'.")
                continue
            x_arr = np.array(x_aligned, dtype=float)
            y_arr = np.array(yvals, dtype=float)
            
            # Create scatter plot with color gradient mapped to y_arr.
            sc = ax.scatter(x_arr, y_arr, marker='o', c=y_arr, cmap='viridis_r', label='Successful Cases')
            
            ax.set_title(f"Successful Cases for {name}")
            ax.set_ylabel("Target Value (Range)")
            # Determine smart axis limits
            xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
            ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
            if xmax > xmin:
                ax.set_xlim(xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin))
            if ymax > ymin:
                ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
            ax.grid(True)
            ax.legend()
            
            # Add a colorbar to show the mapping of the gradient from lowest to highest y values
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Target Value (Range)")
            # Determine the highest target value
            idx_max = np.argmax(y_arr)
            max_val = y_arr[idx_max]
            max_x = x_arr[idx_max]
            
            # Annotate the highest value on the plot.
            ax.annotate(f"max: {max_x:.2f}, range: {max_val:.2f}",
                        xy=(max_x, max_val), 
                        xytext=(max_x, max_val + 0.05*(ymax-ymin)),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=9,
                        color='red')

        axs[-1].set_xlabel("Parameter Value")
        plt.tight_layout()
        plt.savefig("parameters.png", dpi=200)
        print("Saved scatter grid to parameters.png")
    plot_range(results)


if __name__ == '__main__':
    print("Number of CPU cores:", os.cpu_count())

    input_deck_filename = 'tomtom.yml'
    num_processors = 16
    n_samples = 300

    db_filename = "results_2025_10_09_121703.db"

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
        NumericSpec(
            name="boost_t_dur",
            bounds=(1, 30),
            deck_path=["trajectory_phases", "boost_11", "initial_conditions", "time", "t_duration"]
        ),
        NumericSpec(
            name="r_throat",
            bounds=(0.1, 5),
            deck_path=["design_variables", "r_throat_1", "value"]
        ),
        # CategoricalSpec(
        #     name='motor_1_pulse_1_propellant',
        #     types=["TP-H-3402A"],
        #     deck_path=["motor_options", "motor_1_pulse_1_propellant"]
        # )
    ]

    main(input_deck_filename, num_processors, n_samples, input_params, db_filename)
