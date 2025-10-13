Number of CPU cores: 32
Number of available CPUs: 32
Optimal number of processors: 16
X_all [[-13.05989765  11.51769535  22.84203064   4.01161763]
 [-11.41796967   3.33469015  15.59384723   0.55582228]
 [-26.16910499 -29.52163176   7.30624659   0.49505921]
 ...
 [-14.34173033  27.2780365   12.43256928   1.27629065]
 [  0.03216307 -24.67179764  18.02422837   4.27246566]
 [-29.09399812 -17.80143111   6.26273762   0.15480806]]
Batch 1: 100%|██████████████████████████████████████████████| 10/10 [00:11<00:00,  1.11s/it]
batch_results 0 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 1: 10/10 infeasible; best-so-far y_max=187.591
Batch 2: 100%|██████████████████████████████████████████████| 10/10 [00:11<00:00,  1.12s/it]
batch_results 1 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 2: 10/10 infeasible; best-so-far y_max=187.591
Batch 3: 100%|██████████████████████████████████████████████| 10/10 [00:10<00:00,  1.08s/it]
batch_results 2 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 3: 10/10 infeasible; best-so-far y_max=187.591
Batch 4: 100%|██████████████████████████████████████████████| 10/10 [00:11<00:00,  1.16s/it]
batch_results 3 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 4: 10/10 infeasible; best-so-far y_max=187.591
Batch 5: 100%|██████████████████████████████████████████████| 10/10 [00:11<00:00,  1.15s/it]
batch_results 4 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 5: 10/10 infeasible; best-so-far y_max=187.591
Batch 6: 100%|██████████████████████████████████████████████| 10/10 [00:11<00:00,  1.15s/it]
batch_results 5 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 6: 10/10 infeasible; best-so-far y_max=187.591
Batch 7: 100%|██████████████████████████████████████████████| 10/10 [00:10<00:00,  1.09s/it]
batch_results 6 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 7: 10/10 infeasible; best-so-far y_max=187.591
Batch 8: 100%|██████████████████████████████████████████████| 10/10 [00:10<00:00,  1.05s/it]
batch_results 7 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 8: 10/10 infeasible; best-so-far y_max=187.591
Batch 9: 100%|██████████████████████████████████████████████| 10/10 [00:10<00:00,  1.09s/it]
batch_results 8 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 9: 10/10 infeasible; best-so-far y_max=187.591
Batch 10: 100%|█████████████████████████████████████████████| 10/10 [00:11<00:00,  1.13s/it]
batch_results 9 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Iter 10: 10/10 infeasible; best-so-far y_max=187.591
Top 10 Successful Cases by Range:



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

# --- deps you already use ---
import numpy as np
import sqlite3
from pathlib import Path
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression


# -------- utilities --------
def load_db_observations(db_path: str | Path, param_keys: list[str]):
    """
    Reads your 'results' table and returns aligned arrays:
    - X_all:  (N,D) parameter matrix in param_keys order
    - y_all:  (N,)  objective (range)
    - z_all:  (N,)  feasibility flag (1=success, 0=failure). Missing status -> 1.
    Skips rows missing required columns.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM results")
    rows = cur.fetchall()
    conn.close()

    X_list, y_list, z_list = [], [], []
    for row in rows:
        if "range" not in row.keys():
            continue
        ok = all(k in row.keys() for k in param_keys)
        if not ok:
            continue
        X_list.append([float(row[k]) for k in param_keys])
        y_list.append(float(row["range"]))
        z_list.append(int(row["status"]) if "status" in row.keys() else 1)

    if not X_list:
        raise ValueError("No valid rows in DB for provided param_keys.")

    X_all = np.array(X_list, dtype=float)
    y_all = np.array(y_list, dtype=float)
    z_all = np.array(z_list, dtype=int)
    return X_all, y_all, z_all


def select_diverse_subset(X: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """
    Greedy max-min over normalized space to pick k diverse indices.
    If len(X)<=k, returns all indices.
    """
    n = X.shape[0]
    if n <= k:
        return list(range(n))
    rng = np.random.default_rng(seed)
    # normalize per dim
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / span

    idxs = [int(rng.integers(n))]
    dists = np.full(n, np.inf)
    for _ in range(1, k):
        dists = np.minimum(dists, np.linalg.norm(Xn - Xn[idxs[-1]], axis=1))
        dists[idxs] = -1.0  # exclude chosen
        idxs.append(int(np.argmax(dists)))
    return idxs


def suggest_batch_feasible(
    acq: UtilityFunction,
    gp: GaussianProcessRegressor,
    clf: LogisticRegression,
    keys: list[str],
    bounds: dict[str, tuple[float, float]],
    y_max: float,
    batch_size: int = 10,
    n_probe: int = 4096,
    seed: int = 0,
) -> list[dict[str, float]]:
    """
    Probe many random points, score = acquisition * P(success), take top-k.
    """
    rng = np.random.default_rng(seed)
    lows  = np.array([bounds[k][0] for k in keys], dtype=float)
    highs = np.array([bounds[k][1] for k in keys], dtype=float)
    Xcand = rng.uniform(lows, highs, size=(n_probe, len(keys)))

    # Acquisition from visualization GP
    U = acq.utility(Xcand, gp, y_max).reshape(-1)

    # Feasibility probability
    if hasattr(clf, "predict_proba"):
        Pf = clf.predict_proba(Xcand)[:, 1]
    else:
        # fallback: scale decision_function to [0,1]
        s = clf.decision_function(Xcand)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        Pf = s
    Pf = np.clip(Pf, 0.0, 1.0)

    score = U * Pf
    top = np.argsort(-score)[:batch_size]
    return [dict(zip(keys, Xcand[i].tolist())) for i in top]


# -------- main loop (constrained BO) --------
def bayesian_optimization_batch(
    input_deck,
    input_params,
    casenum: int,
    param_bounds: dict[str, tuple[float, float]],
    *,
    db_path: str | Path = "results_2025_10_09_121703.db",
    init_points: int = 200,        # warm-start subset size (tune: 100–300)
    n_iter: int = 5,
    batch_size: int = 12,
    num_processors: int = 24,
    ucb_kappa: float = 3.0,        # a bit more exploration when feasibility is sparse
    probe_candidates: int = 4096,  # candidates scored per iteration
    seed: int = 42,
    evaluate_topups: bool = False, # set True if you want to actually run sims for random top-ups
):
    """
    Constrained BO with feasibility model:
      - Warm-start from DB (uses both successes & fails).
      - GP objective trained on successes only.
      - LogisticRegression feasibility model trained on all points.
      - Batch suggestion via acquisition * P(success).

    Returns
    -------
    optimizer : BayesianOptimization
    """
    keys = list(param_bounds.keys())
    if not keys:
        raise ValueError("param_bounds is empty.")

    # 0) Load DB → X_all, y_all, z_all
    X_all, y_all, z_all = load_db_observations(db_path, keys)
    print("X_all", X_all)

    # 1) Pick a diverse warm-start subset; bias toward successes when possible
    rng = np.random.default_rng(seed)
    succ_idx = np.where(z_all == 1)[0].tolist()
    fail_idx = np.where(z_all == 0)[0].tolist()

    # pick ~70% successes (or all if fewer), rest fails
    s_keep = min(int(0.7 * init_points), len(succ_idx))
    f_keep = min(init_points - s_keep, len(fail_idx))

    s_sel = select_diverse_subset(X_all[succ_idx], s_keep, seed=seed) if s_keep > 0 else []
    f_sel = select_diverse_subset(X_all[fail_idx], f_keep, seed=seed + 1) if f_keep > 0 else []

    warm_idx = ([succ_idx[i] for i in s_sel] +
                [fail_idx[i] for i in f_sel])
    if len(warm_idx) < init_points:
        # pad with random unused rows (any feasibility)
        pool = list(set(range(X_all.shape[0])) - set(warm_idx))
        rng.shuffle(pool)
        warm_idx += pool[: init_points - len(warm_idx)]

    X0, y0, z0 = X_all[warm_idx], y_all[warm_idx], z_all[warm_idx]

    # 2) Build BO optimizer; register ONLY successes
    optimizer = BayesianOptimization(f=lambda **p: 0.0, pbounds=param_bounds, verbose=0)
    for x, y, z in zip(X0, y0, z0):
        if z == 1:
            optimizer.register(params=dict(zip(keys, x.tolist())), target=float(y))

    # 3) Fit models
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), normalize_y=True)
    X_succ = X0[z0 == 1]
    y_succ = y0[z0 == 1]
    if X_succ.shape[0] >= 1:
        gp.fit(X_succ, y_succ)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    if X0.shape[0] >= 2:
        clf.fit(X0, z0)

    acq = UtilityFunction(kind="ucb", kappa=ucb_kappa, xi=0.0)

    # For evaluation
    eval_func = partial(eval_candidate, input_params=input_params, input_deck=input_deck, casenum=casenum)
    print(eval_func)
    # Optional: evaluate random top-ups if you want real y’s instead of placeholders
    if evaluate_topups and len(warm_idx) < init_points:
        need = init_points - len(warm_idx)
        lows  = np.array([param_bounds[k][0] for k in keys], float)
        highs = np.array([param_bounds[k][1] for k in keys], float)
        R = rng.uniform(lows, highs, size=(need, len(keys)))
        topup = [dict(zip(keys, r.tolist())) for r in R]
        with mp.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            y_eval = list(tqdm(pool.imap(eval_func, topup), total=len(topup), desc="Top-up evals"))
        X_add   = np.array([list(c.values()) for c in topup], float)
        y_add   = np.array(y_eval, float)
        z_add   = (y_add > 0.0).astype(int)
        # augment datasets
        X0 = np.vstack([X0, X_add]); y0 = np.concatenate([y0, y_add]); z0 = np.concatenate([z0, z_add])
        X_all, y_all, z_all = X0, y0, z0
        # update models
        X_succ = X0[z0 == 1]; y_succ = y0[z0 == 1]
        if X_succ.shape[0] >= 1:
            gp.fit(X_succ, y_succ)
        if X0.shape[0] >= 2:
            clf.fit(X0, z0)
        for x, y, z in zip(X_add, y_add, z_add):
            if z == 1:
                optimizer.register(params=dict(zip(keys, x.tolist())), target=float(y))

    # 4) Iterations
    for it in range(n_iter):
        y_max = float(np.max(y_succ)) if y_succ.size else 0.0

        # Constrained suggestion
        batch_candidates = [optimizer.suggest(acq) for _ in range(batch_size)]

        # Evaluate in parallel
        with multiprocessing.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            batch_results = list(
                tqdm(pool.imap(eval_func, batch_candidates),
                     total=len(batch_candidates),
                     desc=f"Batch {it + 1}")
            )
            
        print(f"batch_results {it}", batch_results)
        Xb = np.array([list(c.values()) for c in batch_candidates], float)
        yb = np.array(batch_results, float)
        zb = (yb > 0.0).astype(int)

        # Update feasibility dataset and refit classifier
        X_all = np.vstack([X_all, Xb])
        y_all = np.concatenate([y_all, yb])
        z_all = np.concatenate([z_all, zb])
        if X_all.shape[0] >= 2:
            clf.fit(X_all, z_all)

        # Update GP/BO with only successes
        if zb.any():
            X_add = Xb[zb == 1]; y_add = yb[zb == 1]
            X_succ = np.vstack([X_succ, X_add]) if X_succ.size else X_add
            y_succ = np.concatenate([y_succ, y_add]) if y_succ.size else y_add
            gp.fit(X_succ, y_succ)
            for x, y in zip(X_add, y_add):
                optimizer.register(params=dict(zip(keys, x.tolist())), target=float(y))

        zero_count = int((zb == 0).sum())
        print(f"Iter {it+1}: {zero_count}/{len(zb)} infeasible; "
              f"best-so-far y_max={float(np.max(y_succ)) if y_succ.size else 0.0:.3f}")

    return optimizer

    
