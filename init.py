# bo_traj_opt.py
import numpy as np
from typing import Tuple, Dict, Any, List
from bayes_opt import BayesianOptimization, UtilityFunction

# ----------------------------
# Your environment (feasible-or-not + range)
# ----------------------------
class TrajectoryEnv:
    def __init__(self, input_deck, problem, scenario):
        super().__init__()
        self.input_deck = input_deck
        self.problem = problem
        self.scenario = scenario

        self.ts = "traj_vehicle_0"
        self.alpha_boost_1 = float(
            self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0]
        )
        self.alpha_boost_2 = float(
            self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1]
        )
        print(f"[init] alpha_boost = ({self.alpha_boost_1:.6g}, {self.alpha_boost_2:.6g})")

        # TODO: set realistic bounds for your two alphas (radians or degrees — match your deck!)
        self.alpha_bounds = [
            (-30.0 * np.pi / 180.0, 30.0 * np.pi / 180.0),  # alpha_boost_1
            (-30.0 * np.pi / 180.0, 30.0 * np.pi / 180.0),  # alpha_boost_2
        ]

        self.success_log: List[Dict[str, float]] = []
        self.best_params = np.array([self.alpha_boost_1, self.alpha_boost_2], dtype=float)
        self.best_range = -np.inf

    # ---- low-level helpers ----
    def _clip(self, params):
        out = []
        for (lo, hi), v in zip(self.alpha_bounds, params):
            out.append(float(np.clip(v, lo, hi)))
        return np.array(out, dtype=float)

    def _set_params(self, params):
        a1, a2 = map(float, params)
        self.alpha_boost_1, self.alpha_boost_2 = a1, a2
        alpha_arr = self.problem.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"]
        alpha_arr[0] = a1
        alpha_arr[1] = a2

    def _get_metrics(self) -> Tuple[float, float]:
        """Return (h_min, range_final). Only call after a successful setup."""
        h_ts = self.problem.get_val(f"{self.ts}.phases.boost_11.timeseries.h", units="m")
        r_ts = self.problem.get_val(f"{self.ts}.phases.boost_11.timeseries.range", units="m")
        h_min = float(np.min(h_ts))
        range_final = float(r_ts[-1])
        return h_min, range_final

    # ---- the only contract BO needs ----
    def evaluate(self, params) -> Tuple[bool, float]:
        """
        Try params once.
        Returns:
          (success_flag, range_if_success_else_-inf)
        """
        params = self._clip(params)
        try:
            self._set_params(params)
            self.scenario.setup()  # your heavy reconfigure + model run
            h_min, range_final = self._get_metrics()

            feasible = np.isfinite(range_final) and (range_final > 0.0) and np.isfinite(h_min)
            if feasible:
                self.success_log.append({"a1": params[0], "a2": params[1], "range": range_final})
                if range_final > self.best_range:
                    self.best_range = range_final
                    self.best_params = params.copy()
                return True, range_final
            else:
                return False, float("-inf")
        except Exception:
            # If the run throws (e.g., ODE blow-up, constraint violation), count as infeasible
            return False, float("-inf")


# ----------------------------
# BO wrapper (penalty trick)
# ----------------------------
def run_bo_on_env(
    env: TrajectoryEnv,
    init_points: int = 12,
    n_iter: int = 60,
    random_state: int = 42,
    acq_kind: str = "ei",     # 'ei' | 'ucb' | 'poi'
    kappa: float = 2.576,     # for UCB
    xi: float = 0.0           # for EI/POI
) -> Dict[str, Any]:
    """
    Maximizes final range among feasible runs by penalizing failures.
    """

    # Unpack bounds for the bayes_opt API
    (lo1, hi1), (lo2, hi2) = env.alpha_bounds
    pbounds = {"alpha1": (lo1, hi1), "alpha2": (lo2, hi2)}

    # Objective the optimizer sees: big negative when infeasible, 'range' otherwise
    def objective(alpha1: float, alpha2: float) -> float:
        ok, r = env.evaluate([alpha1, alpha2])
        return (float(r) if ok and np.isfinite(r) else -1e9)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=random_state,
        verbose=2,  # 0 silent, 1 minimal, 2 all
    )

    # Optional: control acquisition explicitly
    util = UtilityFunction(kind=acq_kind, kappa=kappa, xi=xi)

    # Phase 0: a bit of random exploration to find feasibility
    optimizer.maximize(init_points=init_points, n_iter=0, acq=acq_kind, kappa=kappa, xi=xi)

    # Phase 2: BO iterations (guided)
    optimizer.maximize(n_iter=n_iter, acq=acq_kind, kappa=kappa, xi=xi)

    best = optimizer.max  # {'target': best_range_or_penalty, 'params': {'alpha1':..., 'alpha2':...}}

    # In case best['target'] was penalized (unlikely after iterations), fall back to env.best_*
    best_range = best["target"]
    best_params = (best["params"]["alpha1"], best["params"]["alpha2"])

    if not np.isfinite(best_range) or best_range <= -1e8:
        best_range = env.best_range
        best_params = tuple(env.best_params.tolist())

    return {
        "best_params": best_params,
        "best_range": best_range,
        "success_log": env.success_log,  # list of dicts
        "optimizer": optimizer,
    }


# ----------------------------
# Example integration hook
# ----------------------------
def train(input_deck, problem, scenario):
    """
    Create env, run BO, return best alphas and the success log for analysis.
    """
    env = TrajectoryEnv(input_deck, problem, scenario)
    result = run_bo_on_env(
        env,
        init_points=12,  # 10-20 is fine; more if failures are frequent
        n_iter=60,       # tune to your run budget
        random_state=42,
        acq_kind="ei",   # 'ei' is a solid default
        xi=0.01          # small positive xi encourages mild exploration
    )

    a1, a2 = result["best_params"]
    print(f"[BO] best_range={result['best_range']:.6g} at (a1={a1:.6g}, a2={a2:.6g})")
    # Maintain your previous interface if needed:
    action_vec = np.array([a1, a2, 0, 0, 0, 0], dtype=np.float32)
    return action_vec, result["success_log"]
