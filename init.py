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
def bayesopt_constrained_batch(
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
        batch_candidates = suggest_batch_feasible(
            acq=acq, gp=gp, clf=clf,
            keys=keys, bounds=param_bounds,
            y_max=y_max, batch_size=batch_size,
            n_probe=probe_candidates, seed=seed + it
        )

        # Evaluate in parallel
        with mp.Pool(processes=num_processors, maxtasksperchild=8) as pool:
            batch_results = list(
                tqdm(pool.imap(eval_func, batch_candidates),
                     total=len(batch_candidates),
                     desc=f"Batch {it + 1}")
            )
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
