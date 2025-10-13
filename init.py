from __future__ import annotations

def bayesian_optimization_batch(
    input_deck,
    input_params,
    casenum: int,
    param_bounds: dict[str, tuple[float, float]],
    init_points: int = 20,
    n_iter: int = 3,
    batch_size: int = 10,
    num_processors: int = 24,
    plot_params: tuple[str, str] | None = None,
    grid_points: int = 100,
    gif_name: str = "bayesian_optimization.gif",
):
    """
    Run batched Bayesian optimization using your expensive simulation.

    Parameters
    ----------
    input_deck, input_params, casenum : as in your environment
    param_bounds : dict[str, tuple[float, float]]
        e.g., {"boost_alpha_0": (-30, 30), "boost_alpha_1": (-30, 30), ...}
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
    import os
    import io
    import sqlite3
    from pathlib import Path
    from functools import partial
    import multiprocessing
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio
    from tqdm import tqdm
    from bayes_opt import BayesianOptimization, UtilityFunction
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

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

    # --- Try warm-start from DB if present; else actually evaluate initial points ---
    initial_results: list[float] = []
    used_db = False
    db_filename = "results_2025_10_09_121703.db"  # if you want to use it, keep it here
    db_path = Path(db_filename)

    if db_path.exists():
        try:
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
                used_db = True
        except Exception:
            # If anything goes wrong with DB, just fall back to real evals
            used_db = False
            initial_results = []

    if not used_db:
        with multiprocessing.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            initial_results = list(
                tqdm(pool.imap(eval_func, initial_candidates),
                     total=len(initial_candidates),
                     desc="Initial evaluations")
            )

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
    successful_cases = {spec.name: [] for spec in input_params}
    successful_targets: list[float] = []
    for cand, val in zip(initial_candidates, initial_results):
        if float(val) != 0.0:
            successful_targets.append(float(val))
            for spec in input_params:
                successful_cases[spec.name].append(cand[spec.name])

    # --- Frames for GIF (2D surfaces) ---
    frames: list[str] = []

    # =================================
    # Batched BO iterations
    # =================================
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
            for ax, (name, xvals) in zip(axs, successful_cases.items()):
                x_arr = np.asarray(xvals, dtype=float)
                y_arr = np.asarray(successful_targets, dtype=float)
                m = min(len(x_arr), len(y_arr))
                if m == 0:
                    continue
                x_arr, y_arr = x_arr[:m], y_arr[:m]

                ax.scatter(x_arr, y_arr, marker='o', label='Successful Cases')
                ax.set_title(f"Successful Cases for {name}")
                ax.set_ylabel("Target (Range)")
                ax.set_xlabel(name)

                if name in param_bounds:
                    lo, hi = param_bounds[name]
                    ax.set_xlim(lo, hi)

                if y_arr.size:
                    ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
                    pad = 0.1 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
                    ax.set_ylim(ymin - pad, ymax + pad)

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
            # bayes_opt.UtilityFunction accepts 2D array for X
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
    
