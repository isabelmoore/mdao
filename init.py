import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# ---- data
Xcols = ["boost_alpha_0", "boost_alpha_1"]   # your param columns
X = df[Xcols].to_numpy(float)
y_range = df["range"].to_numpy(float)
y_succ  = df["success"].astype(int).to_numpy()

# ---- models
ker_r = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-6)
reg = Pipeline([("sc", StandardScaler()),
                ("gp", GaussianProcessRegressor(kernel=ker_r, normalize_y=True, n_restarts_optimizer=3, random_state=0))])

ker_c = 1.0 * Matern(length_scale=np.ones(X.shape[1]), nu=1.5)
clf = Pipeline([("sc", StandardScaler()),
                ("gp", GaussianProcessClassifier(kernel=ker_c, random_state=0, max_iter_predict=200))])

# fit regressor only on successful runs (better fidelity)
mask_ok = y_succ == 1
reg.fit(X[mask_ok], y_range[mask_ok])
clf.fit(X, y_succ)

# ---- expected improvement
best_y = y_range[mask_ok].max() if mask_ok.any() else 0.0

def ei(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)

# ---- propose next points by maximizing EI * P(success)
def score_candidates(C):
    mu, std = reg.predict(C, return_std=True)
    ei_val = ei(mu, std, best=best_y)
    p_ok = clf.predict_proba(C)[:, 1]
    return ei_val * p_ok, mu, p_ok

# simple global search by sampling; you can swap for CMA-ES or L-BFGS restarts
def suggest(n_suggestions=5, n_samples=20000, bounds=[(-30,30),(-30,30)]):
    lows  = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    C = np.random.rand(n_samples, len(bounds)) * (highs - lows) + lows
    s, mu, p = score_candidates(C)
    idx = np.argsort(-s)[:n_suggestions]
    return C[idx], s[idx], mu[idx], p[idx]

cand, s, mu, p = suggest(n_suggestions=10, bounds=[(-30,30),(-30,30)])
print(pd.DataFrame(cand, columns=Xcols).assign(score=s, pred_range=mu, p_success=p))
