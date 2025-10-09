# ---------- build DataFrame (and optionally also call save_db for persistence) ----------
rows = []
for i, r in enumerate(results):
    row = {"case_id": i}
    params_tuple = r["input_params"]
    for j, spec in enumerate(input_params):
        row[spec.name] = params_tuple[j+1]
    row["range"] = r["range"]
    row["success"] = int(r.get("success", r.get("status", 0)))
    rows.append(row)

df = pd.DataFrame(rows)

# If you want both DB and df, modify save_db to return df or just call it separately:
# save_db(input_params, results)  # persist to SQLite if you want


# ---------- phase 2: fit surrogate(s) on LHS results ----------
Xcols = [s.name for s in numeric_specs]
X       = df[Xcols].to_numpy(float)
y_range = df["range"].to_numpy(float)
y_succ  = df["success"].to_numpy(int)

ker_r = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(Xcols)), nu=2.5) + WhiteKernel(1e-6)
reg = Pipeline([
    ("sc", StandardScaler()),
    ("gp", GaussianProcessRegressor(kernel=ker_r, normalize_y=True, n_restarts_optimizer=3, random_state=0)),
])

ker_c = 1.0 * Matern(length_scale=np.ones(len(Xcols)), nu=1.5)
clf = Pipeline([
    ("sc", StandardScaler()),
    ("gp", GaussianProcessClassifier(kernel=ker_c, random_state=0, max_iter_predict=200)),
])

mask_ok = (y_succ == 1)
if mask_ok.any():
    reg.fit(X[mask_ok], y_range[mask_ok])
    best_y = float(y_range[mask_ok].max())
else:
    # fallback to avoid crash
    reg.fit(X, y_range)
    best_y = float(y_range.max())

clf.fit(X, y_succ)

from scipy.stats import norm

def ei(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def score_candidates(C):
    mu, std = reg.predict(C, return_std=True)
    ei_val = ei(mu, std, best=best_y)
    p_ok   = clf.predict_proba(C)[:, 1]
    return ei_val * p_ok, mu, p_ok

def suggest(n_suggestions=5, n_samples=20000, bounds=None, seed=0):
    rng = np.random.default_rng(seed)
    if bounds is None:
        bounds = [s.bounds for s in numeric_specs]
    lows  = np.array([b[0] for b in bounds], float)
    highs = np.array([b[1] for b in bounds], float)
    C = rng.random((n_samples, len(bounds))) * (highs - lows) + lows
    scores, mu, p = score_candidates(C)
    idx = np.argsort(-scores)[:n_suggestions]
    return C[idx], scores[idx], mu[idx], p[idx]

cand, s, mu_vals, p_vals = suggest(n_suggestions=10)
df_candidates = pd.DataFrame(cand, columns=Xcols).assign(score=s, pred_range=mu_vals, p_success=p_vals)
print(df_candidates.sort_values("score", ascending=False))
