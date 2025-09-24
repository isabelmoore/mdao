import numpy as np
from math import * 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from datetime import datetime
from pathlib import Path
import openmdao.api as om
import dymos as dm
import os, sys, time, copy, shutil
from pathlib import Path
from datetime import datetime
from yaml import safe_load, dump
import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.visualization.timeseries.bokeh_timeseries_report import make_timeseries_report
import matplotlib.pyplot as plt
import tempfile
import multiprocessing
from functools import partial
from pathlib import Path
from bs4 import BeautifulSoup
import scipy.stats.qmc as qmc
import pandas as pd
from itertools import product
import sqlite3

paths_to_modules = [
    "scenarios",
    "scenarios/concepts",
    "scenarios/configs",
    "src/models/aero/datcom",
    "src/models/layout",
    "src/models/loads",
    "src/models/power",
    "src/models/prop/solid",
    "src/models/prop/inlet_tables",
    "src/models/radar",
    "src/models/structure",
    "src/models/thermal",
    "src/models/warhead",
    "src/models/warhead/bf_warhead",
    "src/models/weapondatalink",
    "src/models/weight",
    "tools/helpers/",
    "tools/plotters",
    "tools/ppt",
    "tools/datcom_nn",
]
os.chdir(r"/home/imoore/misslemdao")

project_dir = Path(__file__).resolve().parent.parent  

for path in paths_to_modules:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dymos_generator import dymos_generator

'''
This script is designed to help in reducing the amount of major iterations for SNOPT. 

Originally, this was designed by reducing errors for aircraft parameters at various points in
the trajectory. However, this is not good for future development and  current work, as different input decks 
expected different behaviors, causing the "tuned" reward or error model to drastically change per deck.

Thus, what we know is that the major iterations is meant to be minimized and that certain values of the 
initial conditions leads to more successfull outcomes. Rather than "beating around the bush" and "slapping
on bandaids" and endless runs of "with vs without the driver" to save in computation energy, we will run WITH
the driver and delievery in these points:
    - run with driver
    - to save in compuational energy, run these in parallel
    - since we know we have some good values, apply a Perturbation Optimization as per run
        https://www.worldscientific.com/doi/epdf/10.1142/S021759591950009X

'''
class TrajectoryEnv:
    def __init__(self, input_deck):
        self.input_deck = input_deck
        self.ts = "traj_vehicle_0"

    # ---------- tiny utils ----------
    @staticmethod
    def _read_text(p: Path) -> str:
        if not p.exists(): return ""
        try: return p.read_text(encoding="utf-8", errors="ignore")
        except Exception: return p.read_bytes().decode("utf-8", errors="ignore")

    @staticmethod
    def _last_major_before_exit(txt: str) -> int | None:
        k = max(txt.rfind("SNOPTA EXIT"), txt.rfind("SNOPTC EXIT"))
        if k == -1: return None
        for line in reversed(txt[:k].splitlines()):
            s = line.strip()
            if not s: continue
            m = re.match(r"\s*(\d+)\b", s)
            if m: return int(m.group(1))
        return None

    # ---------- candidate sets (reduced randomness) ----------
    @staticmethod
    def _sobol_box_2d(seed_xy, n_pts, span_frac=(0.2, 0.2), bounds=None):
        x0, y0 = map(float, seed_xy)
        sx = max(abs(x0)*span_frac[0], 1e-3)
        sy = max(abs(y0)*span_frac[1], 1e-3)
        lo = np.array([x0 - sx, y0 - sy]); hi = np.array([x0 + sx, y0 + sy])
        try:
            from scipy.stats import qmc
            eng = qmc.Sobol(d=2, scramble=False)
            pts = eng.random_base2(int(np.ceil(np.log2(n_pts))))[:n_pts]
            cand = qmc.scale(pts, lo, hi)
            cands = [tuple(map(float, xy)) for xy in cand]
        except Exception:
            k = max(3, int(np.ceil(np.sqrt(n_pts))))
            xs = np.linspace(lo[0], hi[0], k); ys = np.linspace(lo[1], hi[1], k)
            cands = [(x, y) for x in xs for y in ys][:n_pts]
        if bounds:
            (x_lo, x_hi), (y_lo, y_hi) = bounds
            cands = [(x,y) for (x,y) in cands if x_lo <= x <= x_hi and y_lo <= y <= y_hi]
        cands = [(x0, y0)] + cands
        uniq = list(dict.fromkeys([(round(x,12), round(y,12)) for (x,y) in cands]))
        return [(x, y) for (x, y) in uniq]

    # ---------- single candidate evaluation (static for multiprocessing) ----------
    @staticmethod
    def _run_one_alpha_case(case_tuple, input_deck):
        """
        case_tuple = (id, alpha1, alpha2)
        Returns dict with 'cost' = last major iterations (lower is better).
        """
        cid, a1, a2 = case_tuple
        PENALTY_FAIL = 1e9
        try:
            p = om.Problem(name=f"case_{cid}")
            deck = copy.deepcopy(input_deck)
            scen = dymos_generator(problem=p, input_deck=deck)

            # Set alphas in the model options used by your generator
            mo = scen.p.model_options["vehicle_0"]
            mo["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = float(a1)
            mo["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = float(a2)

            scen.setup()
            dm.run_problem(scen.p, run_driver=True, simulate=False)

            outdir = scen.p.get_outputs_dir()
            txt = TrajectoryEnv._read_text(outdir / "SNOPT_summary.out") or \
                  TrajectoryEnv._read_text(outdir / "SNOPT_print.out")
            last_major = TrajectoryEnv._last_major_before_exit(txt) if txt else None
            cost = float(last_major) if last_major is not None else PENALTY_FAIL

            return {"case": cid, "alpha1": a1, "alpha2": a2, "last_major": last_major, "cost": cost, "ok": last_major is not None}
        except Exception as e:
            return {"case": cid, "alpha1": a1, "alpha2": a2, "last_major": None, "cost": PENALTY_FAIL, "ok": False, "error": str(e)}

    # ---------- public: build cases + run in parallel ----------
    def build_alpha_cases(self, seed_alphas, n_candidates=24, span_frac=(0.2,0.2), bounds=None, start_idx=0):
        pts = self._sobol_box_2d(seed_alphas, n_pts=n_candidates, span_frac=span_frac, bounds=bounds)
        return [(start_idx+i, float(a1), float(a2)) for i, (a1, a2) in enumerate(pts)]

    def run_parallel_envs_alpha(self, seed_alphas, *, n_candidates=24, span_frac=(0.2,0.2),
                                bounds=None, processes=8):
        cases = self.build_alpha_cases(seed_alphas, n_candidates=n_candidates, span_frac=span_frac, bounds=bounds)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(partial(TrajectoryEnv._run_one_alpha_case, input_deck=self.input_deck), cases)
        best = min(results, key=lambda r: r["cost"])
        return results, best


start_time = time.time()

input_deck_path = os.path.join(os.path.dirname(__file__), "scenarios/input_decks/", input_deck_filename)
input_deck_base_path = os.path.join(os.path.dirname(__file__), "scenarios/input_decks/", input_deck_base)

with open(input_deck_path, "r") as f:
    input_deck_child = safe_load(f)

# Load base and child files
with open(input_deck_base_path) as base_file:
    base = safe_load(base_file)

# Merge base into child (child takes precedence)
if "base" in input_deck_child:
    del input_deck_child["base"]  # Remove the 'base' key after processing if it exists

input_deck = merge_input_decks(base, input_deck_child)

input_deck_optimized = copy.deepcopy(input_deck)

problem_name = input_deck_filename.split(".")[0] + datetime.now().strftime("_date_%Y_%m_%d_time_%H%M")

# =============================================================================
# Create and run scenario
# =============================================================================
# make all reports globally go here
os.environ["OPENMDAO_WORKDIR"] = str(Path(Path(__file__).parent, "reports"))
p = om.Problem(name=problem_name, reports=["inputs", "n2", "optimizer"])

vehicle_num = 0  # default vehicle num because missilemdao only runs one vehicle
multiple_vehicles = False
scenario = dymos_generator(problem=p, input_deck=input_deck, vehicle_num=vehicle_num, multiple_vehicles=multiple_vehicles)

from train_init_cond import TrajectoryEnv

import warnings
warnings.filterwarnings("ignore")

print("Initial Conditions Before: ")
print(f"\tBoost Alphas: \n\t\t ", p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls" ]["alpha"], " deg")

# train model to find optimal parameters for initial conditions
env = TrajectoryEnv(input_deck, p, scenario)
