# mixed_search.py
from __future__ import annotations
from itertools import product
from typing import Dict, List, Tuple, Iterable, Callable, Any
import numpy as np

# Optional: pip install bayesian-optimization
try:
    from bayes_opt import BayesianOptimization
    HAS_BO = True
except Exception:
    HAS_BO = False


# =========================
# Samplers
# =========================

class GridSampling:
    """
    Exhaustive grid over categorical variables.
    Example config:
        cat_space = {
            "tire_type": ["soft", "medium", "hard"],
            "guidance": ["A", "B"]
        }
    """
    def __init__(self, cat_space: Dict[str, List[Any]]):
        self.cat_space = cat_space
        self.keys = list(cat_space.keys())
        self._grid = list(product(*[cat_space[k] for k in self.keys]))

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for combo in self._grid:
            yield {k: v for k, v in zip(self.keys, combo)}

    def __len__(self) -> int:
        return len(self._grid)


class LatinHypercube:
    """
    Simple LHS for numeric warm-starts.
    bounds: dict name -> (lo, hi)
    n: number of samples
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]], n: int, seed: int = 0):
        self.bounds = bounds
        self.n = n
        self.seed = seed

    def samples(self) -> List[Dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        names = list(self.bounds.keys())
        dim = len(names)

        # Stratify each dimension
        cut = np.linspace(0, 1, self.n + 1)
        u = rng.random((self.n, dim))
        pts01 = (cut[:-1] + u * (cut[1:] - cut[:-1]))  # n x dim

        # Randomly permute per dimension
        for j in range(dim):
            rng.shuffle(pts01[:, j])

        # Scale to bounds
        out = []
        for i in range(self.n):
            row = {}
            for j, name in enumerate(names):
                lo, hi = self.bounds[name]
                row[name] = lo + pts01[i, j] * (hi - lo)
            out.append(row)
        return out


class Bayesian:
    """
    Thin wrapper around `bayesian-optimization` for numeric params.
    Uses the "penalty trick": objective returns large negative when infeasible,
    otherwise returns the scalar we want to maximize (here: `range_final`).
    """
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        init_points: int = 12,
        n_iter: int = 60,
        random_state: int = 42,
        acq: str = "ei",
        xi: float = 0.01,
        penalty: float = -1e9,
    ):
        if not HAS_BO:
            raise RuntimeError(
                "bayesian-optimization not installed. `pip install bayesian-optimization`"
            )
        self.bounds = bounds
        self.init_points = init_points
        self.n_iter = n_iter
        self.random_state = random_state
        self.acq = acq
        self.xi = xi
        self.penalty = penalty

    def run(
        self,
        objective: Callable[[Dict[str, float]], float],
        warm_starts: List[Dict[str, float]] | None = None,
        verbose: int = 2,
    ) -> Tuple[Dict[str, float], float]:
        # Make pbounds
        pbounds = {k: (float(lo), float(hi)) for k, (lo, hi) in self.bounds.items()}

        # Wrap objective for bayes_opt signature f(x1, x2, ...)
        names = list(self.bounds.keys())

        def f_wrapped(**kwargs):
            # kwargs contains names -> value
            x = {k: float(kwargs[k]) for k in names}
            val = objective(x)
            return float(val)

        opt = BayesianOptimization(
            f=f_wrapped,
            pbounds=pbounds,
            random_state=self.random_state,
            verbose=verbose,
        )

        # If warm starts are provided, register them
        if warm_starts:
            for ws in warm_starts:
                y = objective(ws)
                opt.register(params=ws, target=float(y))

        # Random init + guided
        if self.init_points > 0:
            opt.maximize(init_points=self.init_points, n_iter=0, acq=self.acq, xi=self.xi)
        if self.n_iter > 0:
            opt.maximize(n_iter=self.n_iter, acq=self.acq, xi=self.xi)

        best = opt.max
        return best["params"], float(best["target"])


# =========================
# Trajectory wrapper (your env)
# =========================

class Trajectory:
    """
    Orchestrates:
      - categorical grid search
      - numeric Bayesian optimization per categorical combo
    You provide:
      - set_categoricals: how to apply categorical settings to the sim
      - evaluate_numeric: given numeric dict, run once and return (feasible, range_final)
    """
    def __init__(
        self,
        set_categoricals: Callable[[Dict[str, Any]], None],
        evaluate_numeric: Callable[[Dict[str, float]], Tuple[bool, float]],
    ):
        self.set_categoricals = set_categoricals
        self.evaluate_numeric = evaluate_numeric

    def optimize(
        self,
        cat_space: Dict[str, List[Any]],
        num_bounds: Dict[str, Tuple[float, float]],
        lhs_warmup: int = 10,
        bo_init_points: int = 12,
        bo_iters: int = 60,
        seed: int = 0,
    ) -> Dict[str, Any]:
        grid = GridSampling(cat_space)
        lhs = LatinHypercube(num_bounds, n=lhs_warmup, seed=seed)
        warm = lhs.samples() if lhs_warmup > 0 else []

        best_overall = {
            "cat": None,
            "num": None,
            "range": -np.inf,
        }
        per_combo_results = []

        # Iterate over ALL categorical combinations
        for cat_combo in grid:
            # Apply categoricals to the environment
            self.set_categoricals(cat_combo)

            # Define BO objective: returns range (maximize) if feasible; large negative otherwise
            def objective(xnum: Dict[str, float]) -> float:
                feasible, rng = self.evaluate_numeric(xnum)
                return float(rng) if (feasible and np.isfinite(rng)) else -1e9

            # Run BO for this categorical combo
            bo = Bayesian(
                bounds=num_bounds,
                init_points=bo_init_points,
                n_iter=bo_iters,
                random_state=seed,
                acq="ei",
                xi=0.01,
                penalty=-1e9,
            )
            best_num, best_val = bo.run(objective, warm_starts=warm, verbose=1)

            per_combo_results.append({
                "categoricals": cat_combo,
                "best_numeric": best_num,
                "best_range": best_val,
            })

            if best_val > best_overall["range"]:
                best_overall = {
                    "cat": cat_combo,
                    "num": best_num,
                    "range": best_val,
                }

        return {
            "best_overall": best_overall,
            "per_combo": per_combo_results,
        }


# =========================
# Example wiring
# =========================

# You’ll adapt these two functions to your real environment.

def set_categoricals_in_env(cat_cfg: Dict[str, Any]) -> None:
    """
    Example: push categoricals into your OpenMDAO deck/problem.
    Replace with your real plumbing, e.g.:
       problem.model_options["vehicle_0"]["tire_type"] = cat_cfg["tire_type"]
    """
    # --- USER: IMPLEMENT THIS ---
    # e.g. env.set_option("tire_type", cat_cfg["tire_type"])
    pass


def evaluate_numeric_once(xnum: Dict[str, float]) -> Tuple[bool, float]:
    """
    One sim run:
      - Set numeric params (e.g., alphas)
      - Try scenario.setup()
      - If it runs, pull range_final, return (True, range_final)
      - If it fails/throws, return (False, -inf)
    """
    try:
        # --- USER: set numbers ---
        # env.set_alpha1(xnum["alpha1"]); env.set_alpha2(xnum["alpha2"]); ...
        # env.scenario.setup()
        # range_final = env.get_val("...range")[-1]
        # return True, float(range_final)
        raise NotImplementedError  # remove when you implement
    except Exception:
        return False, float("-inf")


if __name__ == "__main__":
    # Define categorical space (exhaustive search)
    cat_space = {
        "tire_type": ["soft", "medium", "hard"],
        "guidance": ["A", "B"],
        # add more categorical knobs here
    }

    # Define numeric bounds (BO over these)
    num_bounds = {
        "alpha1": (-30.0*np.pi/180.0, 30.0*np.pi/180.0),
        "alpha2": (-30.0*np.pi/180.0, 30.0*np.pi/180.0),
        # add more numeric knobs here
    }

    traj = Trajectory(set_categoricals=set_categoricals_in_env,
                      evaluate_numeric=evaluate_numeric_once)

    results = traj.optimize(
        cat_space=cat_space,
        num_bounds=num_bounds,
        lhs_warmup=12,        # warm-start samples per combo
        bo_init_points=8,     # random init evals for BO per combo
        bo_iters=40,          # BO guided steps per combo
        seed=42,
    )

    print("\n=== BEST OVERALL ===")
    print(results["best_overall"])
    # You also get per-combo bests:
    # for r in results["per_combo"]: print(r)
