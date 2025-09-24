import re

class TrajectoryEnv:
    def __init__(self, input_deck):
        self.input_deck = input_deck
        self.ts = "traj_vehicle_0"

    # ---------------- keep these AS IS ----------------
    @staticmethod
    def _read_text(p: Path) -> str:
        if not p.exists(): return ""
        try: return p.read_text(encoding="utf-8", errors="ignore")
        except Exception: return p.read_bytes().decode("utf-8", errors="ignore")

    @staticmethod
    def _last_major_before_exit(txt: str) -> int | None:
        # first int on the last data line before SNOPTA/SNOPTC EXIT
        k = max(txt.rfind("SNOPTA EXIT"), txt.rfind("SNOPTC EXIT"))
        if k == -1: return None
        for line in reversed(txt[:k].splitlines()):
            s = line.strip()
            if not s: continue
            m = re.match(r"\s*(\d+)\b", s)
            if m: return int(m.group(1))
        return None
    # --------------------------------------------------

    @staticmethod
    def _cloud(seed, n_pts, span_frac=(0.25, 0.25), abs_min_span=0.5, bounds=None):
        """Small deterministic box around seed; Sobol if available, else tiny grid."""
        a1, a2 = map(float, seed)
        s1 = max(abs(a1)*span_frac[0], abs_min_span)
        s2 = max(abs(a2)*span_frac[1], abs_min_span)
        lo = np.array([a1 - s1, a2 - s2]); hi = np.array([a1 + s1, a2 + s2])
        pts = [(a1, a2)]
        try:
            from scipy.stats import qmc
            eng = qmc.Sobol(d=2, scramble=False)
            X = eng.random_base2(int(np.ceil(np.log2(n_pts))))[:n_pts]
            X = qmc.scale(X, lo, hi)
            pts += [tuple(map(float, x)) for x in X]
        except Exception:
            k = max(3, int(np.ceil(np.sqrt(n_pts))))
            xs = np.linspace(lo[0], hi[0], k); ys = np.linspace(lo[1], hi[1], k)
            pts += [(float(x), float(y)) for x in xs for y in ys][:n_pts]
        if bounds:
            (l1,h1),(l2,h2) = bounds
            pts = [(x,y) for (x,y) in pts if l1<=x<=h1 and l2<=y<=h2]
        # de-dup
        uniq = list(dict.fromkeys([(round(x,12), round(y,12)) for (x,y) in pts]))
        return [(x,y) for (x,y) in uniq]

    @staticmethod
    def _eval_alpha_case(case_id, a1, a2, input_deck):
        """Fresh Problem/scenario → run driver → parse last major → cost."""
        PENALTY_FAIL = 1e9
        try:
            p = om.Problem(name=f"case_{case_id}")
            deck = copy.deepcopy(input_deck)
            scen = dymos_generator(problem=p, input_deck=deck)

            # set alphas (both places people commonly use)
            try:
                scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = float(a1)
                scen.p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = float(a2)
            except Exception: pass
            try:
                p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][0] = float(a1)
                p.model_options["vehicle_0"]["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"][1] = float(a2)
            except Exception: pass

            scen.setup()
            dm.run_problem(scen.p, run_driver=True, simulate=False)

            outdir = scen.p.get_outputs_dir()
            txt = (TrajectoryEnv._read_text(outdir / "SNOPT_summary.out")
                   or TrajectoryEnv._read_text(outdir / "SNOPT_print.out"))
            last_major = TrajectoryEnv._last_major_before_exit(txt) if txt else None
            return float(last_major) if last_major is not None else PENALTY_FAIL
        except Exception:
            return PENALTY_FAIL

    def perturb_optimize(self, seed_alphas, *, rounds=3, per_round=10,
                         span_schedule=(0.5, 0.25, 0.1),
                         bounds=None, abs_min_span=0.5):
        """Serial, shrinking-box perturbation. Returns (best_alphas, best_cost)."""
        assert rounds == len(span_schedule)
        best = (float(seed_alphas[0]), float(seed_alphas[1]))
        best_cost = self._eval_alpha_case(0, best[0], best[1], self.input_deck)

        cid = 1
        for r in range(rounds):
            frac = span_schedule[r]
            cand = self._cloud(best, n_pts=per_round, span_frac=(frac, frac),
                               abs_min_span=abs_min_span, bounds=bounds)
            for (a1,a2) in cand[1:]:   # skip seed
                cost = self._eval_alpha_case(cid, a1, a2, self.input_deck)
                cid += 1
                if cost < best_cost:
                    best_cost, best = cost, (a1, a2)
        return best, best_cost
env = TrajectoryEnv(input_deck)

seed = tuple(input_deck["trajectory_phases"]["boost_11"]["initial_conditions"]["controls"]["alpha"])  # e.g., (0.0, 5.0)

best_alphas, best_cost = env.perturb_optimize(
    seed_alphas=seed,
    rounds=3,
    per_round=12,
    span_schedule=(0.5, 0.25, 0.1),     # shrink each round
    bounds=[(-5.0, 5.0), (0.0, 10.0)],  # clamp to valid values
    abs_min_span=0.5,                   # makes α1 explore even if seed is 0.0
)

print("best alphas:", best_alphas, "min major iters:", best_cost)
