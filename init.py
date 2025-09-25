import os, sys, re, copy, multiprocessing as mp
from pathlib import Path
from yaml import safe_load
import numpy as np
import openmdao.api as om
import dymos as dm

class AlphaGridRunner:
    def __init__(self, repo_root: str | Path, deck_relpath: str, processes: int = 4, step: float = 0.25):
        self.repo_root = Path(repo_root).resolve()
        self.deck_path = (self.repo_root / deck_relpath).resolve()
        self.processes = int(processes)
        self.step = float(step)

        # module subpaths your generator needs
        self.paths_to_modules = [
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

        # load deck once (parent only)
        os.chdir(str(self.repo_root))
        with open(self.deck_path, "r") as f:
            self.input_deck = safe_load(f)

        seed = self.input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"]
        self.seed_a1, self.seed_a2 = float(seed[0]), float(seed[1])

    def build_3x3_cases(self):
        """(seed±step) × (seed±step) → 9 cases with ids 0..8."""
        offsets = (-self.step, 0.0, self.step)
        cases = []
        cid = 0
        for da1 in offsets:
            for da2 in offsets:
                cases.append((cid, self.seed_a1 + da1, self.seed_a2 + da2))
                cid += 1
        return cases

    @staticmethod
    def _worker(case_tuple, repo_root_str, paths_to_modules, input_deck):
        """Build fresh Problem, set alphas, run driver, parse major iterations; return (cid, a1, a2, last_major)."""
        # make worker self-sufficient (Windows spawn)
        os.chdir(repo_root_str)
        for sub in paths_to_modules:
            ap = str((Path(repo_root_str) / sub).resolve())
            if ap not in sys.path:
                sys.path.insert(0, ap)
        from dymos_generator import dymos_generator  # import AFTER path setup

        cid, a1, a2 = case_tuple

        p = om.Problem(name=f"case_{cid}")
        deck = copy.deepcopy(input_deck)
        scen = dymos_generator(problem=p, input_deck=deck)

        # set initial alphas (both common locations)
        scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = float(a1)
        scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = float(a2)
        p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = float(a1)
        p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = float(a2)

        scen.setup()
        dm.run_problem(scen.p, run_driver=True, simulate=False)

        # read SNOPT log and parse last major (line before EXIT)
        outdir = scen.p.get_outputs_dir()
        txt = (outdir / "SNOPT_print.out").read_text(encoding="utf-8", errors="ignore")

        k = max(txt.rfind("SNOPTA EXIT"), txt.rfind("SNOPTC EXIT"))
        assert k != -1, "SNOPT EXIT not found in log"
        pre = txt[:k].rstrip("\n")
        last_line = pre.splitlines()[-1].strip()
        last_major = int(last_line.split()[0])

        return (cid, float(a1), float(a2), last_major)

    def run(self):
        cases = self.build_3x3_cases()
        print(f"seed alphas: ({self.seed_a1}, {self.seed_a2})")
        print(f"built {len(cases)} cases with step {self.step}")

        ctx = mp.get_context("spawn")  # Windows-safe
        worker_fn = lambda c: AlphaGridRunner._worker(
            c,
            str(self.repo_root),
            self.paths_to_modules,
            self.input_deck
        )
        with ctx.Pool(processes=self.processes) as pool:
            results = pool.map(worker_fn, cases)

        results.sort(key=lambda r: r[0])
        print("\nSNOPT major iterations per case:")
        for cid, a1, a2, iters in results:
            print(f"case {cid}: a1={a1:.3f}, a2={a2:.3f} -> major_iters={iters}")
        return results

if __name__ == "__main__":
    # EXAMPLES:
    # runner = AlphaGridRunner(r"C:\Users\N81446\misslemdao",
    #                          r"scenarios\input_decks\one_stage_one_pulse_traj_ann.yml",
    #                          processes=4, step=0.25)
    runner = AlphaGridRunner(r"/home/imoore/misslemdao",
                             "scenarios/input_decks/one_stage_one_pulse_traj_ann.yml",
                             processes=4, step=0.25)
    runner.run()
