import os, sys, copy, re, time
from pathlib import Path
from datetime import datetime
from yaml import safe_load
import numpy as np
import openmdao.api as om
import dymos as dm
import multiprocessing as mp
from functools import partial

# --- repo setup (match your tree) ---
os.chdir(r"/home/imoore/misslemdao")
paths_to_modules = [
    "scenarios","scenarios/concepts","scenarios/configs",
    "src/models/aero/datcom","src/models/layout","src/models/loads","src/models/power",
    "src/models/prop/solid","src/models/prop/inlet_tables","src/models/radar",
    "src/models/structure","src/models/thermal","src/models/warhead","src/models/warhead/bf_warhead",
    "src/models/weapondatalink","src/models/weight","tools/helpers/","tools/plotters","tools/ppt","tools/datcom_nn",
]
for pth in paths_to_modules:
    if pth not in sys.path: sys.path.append(pth)
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
        p = om.Problem(name=f"case_{cid}")
        deck = copy.deepcopy(input_deck)
        scen = dymos_generator(problem=p, input_deck=deck)

        # set alphas (where your generator reads them)
        try:
            scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = a1
            scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = a2
        except Exception:
            pass
        try:
            p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = a1
            p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = a2
        except Exception:
            pass

        scen.setup()
        dm.run_problem(scen.p, run_driver=True, simulate=False)

        # read SNOPT logs from THIS run
        outdir = scen.p.get_outputs_dir()
        try:
            txt = (outdir/"SNOPT_summary.out").read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        if not txt:
            try:
                txt = (outdir/"SNOPT_print.out").read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = ""

        # parse last major iteration (row before EXIT)
        last_major = None
        if txt:
            k = max(txt.rfind("SNOPTA EXIT"), txt.rfind("SNOPTC EXIT"))
            if k != -1:
                pre = txt[:k].splitlines()
                for line in reversed(pre):
                    s = line.strip()
                    if not s: continue
                    m = re.match(r"\s*(\d+)\b", s)
                    if m:
                        last_major = int(m.group(1))
                        break

        return (cid, last_major)

    except Exception:
        return (cid, None)

# ---------- main ----------
def main(n1=3, n2=3, span1=1.0, span2=1.0, processes=4):
    # load deck
    input_deck_path = os.path.join("scenarios/input_decks/", "one_stage_one_pulse_traj_ann.yml")
    with open(input_deck_path, "r") as f:
        input_deck = safe_load(f)

    # seed alphas from deck
    seed_alpha = input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"]
    seed_a1, seed_a2 = float(seed_alpha[0]), float(seed_alpha[1])

    # build cases
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # optional clamp
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
