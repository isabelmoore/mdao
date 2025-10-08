
class LatinHypercube:
    """
    Latin Hypercube Sampler for NumericSpec objects.

    Usage:
        lhs = LatinHypercube(numeric_specs, n_samples=16, seed=42)
        samples = lhs.sample()  # joint samples across all variables

        # or for independent per-variable sampling
        per_var = lhs.sample_per_variable()
    """

    def __init__(self, numeric_specs, n_samples: int, seed: int = 0):
        self.specs = numeric_specs
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    # --- full joint LHS sampling
    def sample(self) -> list[dict[str, float]]:
        names = [s.name for s in self.specs]
        bounds = [s.bounds for s in self.specs]
        d = len(self.specs)

        cut = np.linspace(0, 1, self.n_samples + 1)
        u = self.rng.random((self.n_samples, d))
        pts01 = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])

        # shuffle each column independently
        for j in range(d):
            self.rng.shuffle(pts01[:, j])

        samples = []
        for i in range(self.n_samples):
            row = {}
            for j, (name, (lo, hi)) in enumerate(zip(names, bounds)):
                row[name] = float(lo + pts01[i, j] * (hi - lo))
            samples.append(row)
        return samples

    # --- independent LHS per variable
    def sample_per_variable(self) -> dict[str, list[float]]:
        results = {}
        for spec in self.specs:
            lo, hi = spec.bounds
            cut = np.linspace(0, 1, self.n_samples + 1)
            u = self.rng.random(self.n_samples)
            pts01 = cut[:-1] + u * (cut[1:] - cut[:-1])
            self.rng.shuffle(pts01)
            results[spec.name] = [float(lo + p * (hi - lo)) for p in pts01]
        return results
