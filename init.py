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
import multiprocessing as mp 
from functools import partial

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
import os
print(os.path.dirname(os.path.realpath(__file__)))
os.chdir(r"C:\Users\N81446\misslemdao")
print(os.path.dirname(os.path.realpath(__file__)))

project_dir = Path(__file__).resolve().parent.parent  

for path in paths_to_modules:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from dymos_generator import dymos_generator

# ---------- case builder: simple grid around seed ----------
def build_alpha_cases(seed_a1, seed_a2, n1=3, n2=3, span1=1.0, span2=1.0, bounds=None):
    a1_vals = np.linspace(seed_a1 - span1, seed_a1 + span1, n1)
    a2_vals = np.linspace(seed_a2 - span2, seed_a2 + span2, n2)
    cases, cid = [], 0
    for a1 in a1_vals:
        for a2 in a2_vals:
            if bounds:
                (L1,H1),(L2,H2) = bounds
                if not (L1 <= a1 <= H1 and L2 <= a2 <= H2): continue
            cases.append((cid, float(a1), float(a2)))
            cid += 1
    return cases

# ---------- worker: build fresh Problem, run, parse major iterations ----------
def run_alpha_case(case_tuple, input_deck):
    cid, a1, a2 = case_tuple
    try:
        print("1")
        p = om.Problem(name=f"case_{cid}")
        deck = copy.deepcopy(input_deck)
        scen = dymos_generator(problem=p, input_deck=deck)

        # set alphas (where your generator reads them)
        try:
            print("2")

            scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = a1
            scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = a2
        except Exception:
            print("3")

            pass
        try:
            print("4")

            p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = a1
            p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = a2
        except Exception:
            print("5")

            pass

        scen.setup()
        dm.run_problem(scen.p, run_driver=True, simulate=False)

        # read SNOPT logs from THIS run
        outdir = scen.p.get_outputs_dir()

        txt = (outdir/"SNOPT_print.out").read_text(encoding="utf-8", errors="ignore")
        print("txt:",txt)
        # parse last major iteration (row before EXIT)
        last_major = None
        if txt:
            k = max(txt.rfind("SNOPTA EXIT"), txt.rfind("SNOPTC EXIT"))
            print(k)
            if k != -1:
                pre = txt[:k].splitlines()
                print(pre)
                for line in reversed(pre):
                    s = line.strip()
                    if not s: continue
                    m = re.match(r"\s*(\d+)\b", s)
                    if m:
                        last_major = int(m.group(1))
                        break

        return (cid, last_major)

    except Exception as e:
        print(f"Error during trajectory test for case {cid}: {e}")
        return (cid, None)

# ---------- main ----------
def main(n1=3, n2=3, span1=1.0, span2=1.0, processes=4):
    # load deck
    print("YESSS", os.getcwd())

    input_deck_path = os.path.join(os.getcwd(), "scenarios/input_decks/", "one_stage_one_pulse_traj_ann.yml")
    print(input_deck_path)
    with open(input_deck_path, "r") as f:
        input_deck = safe_load(f)
    
    print(input_deck)

    # seed alphas from deck
    seed_alpha = input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"]
    seed_a1, seed_a2 = float(seed_alpha[0]), float(seed_alpha[1])

    # build cases
    bounds = [(-5.0, 5.0), (0.0, 10.0)]  # optional clamp
    cases = build_alpha_cases(seed_a1, seed_a2, n1=n1, n2=n2, span1=span1, span2=span2, bounds=bounds)
    print(f"Built {len(cases)} cases around seed ({seed_a1}, {seed_a2})")

    # run parallel
    with mp.Pool(processes=processes) as pool:
        results = pool.map(partial(run_alpha_case, input_deck=input_deck), cases)

    # print ONLY iterations
    results.sort(key=lambda t: t[0])  # by case id
    print("\nSNOPT major iterations per case:")
    for cid, iters in results:
        print(f"case {cid}: {iters}")

if __name__ == "__main__":
    # Example: 3x3 grid, ±1.0 around each alpha, 4 workers
    main(n1=3, n2=3, span1=1.0, span2=1.0, processes=4)


: [0.5, 0.5, 0.5], 'opt': False, 'units': None}, 'LMAXL': {'val': [0.5, 0.5, 0.5], 'opt': False, 'units': None}, 'LFLATU': {'val': [0.0, 0.0, 0.0], 'opt': False, 'units': None}, 'LFLATL': {'val': [0.0, 0.0, 0.0], 'opt': False, 'units': None}}}}}
Built 9 cases around seed (0.0, 9.0)
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
1
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
C:\Users\N81446\misslemdao\tools\traj_ann\init_cond_opt
Error during trajectory test for case 0: No module named 'scenarios'
1
Error during trajectory test for case 1: No module named 'scenarios'
1
Error during trajectory test for case 2: No module named 'scenarios'
1
Error during trajectory test for case 3: No module named 'scenarios'
1
Error during trajectory test for case 4: No module named 'scenarios'
1
Error during trajectory test for case 5: No module named 'scenarios'
1
Error during trajectory test for case 6: No module named 'scenarios'
1
Error during trajectory test for case 7: No module named 'scenarios'
1
Error during trajectory test for case 8: No module named 'scenarios'

SNOPT major iterations per case:
case 0: None
case 1: None
case 2: None
case 3: None
case 4: None
case 5: None
case 6: None
case 7: None
case 8: None
