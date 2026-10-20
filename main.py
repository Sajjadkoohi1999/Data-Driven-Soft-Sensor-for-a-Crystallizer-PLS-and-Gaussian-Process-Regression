"""
=============================================================================
 SOFT SENSOR FOR A CRYSTALLIZER
 PLS + Gaussian Process Regression (GPR) with full preprocessing pipeline.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# 1. CRYSTALLIZER SIMULATOR
# ---------------------------------------------------------------------------
def simulate_crystallizer(n_samples=2500, seed=42):
    rng = np.random.default_rng(seed)

    T_jacket = rng.uniform(300, 320, n_samples)
    F_feed   = rng.uniform(0.5, 2.0, n_samples)
    C_feed   = rng.uniform(0.25, 0.45, n_samples)
    RPM      = rng.uniform(200, 600, n_samples)
    T_feed   = rng.uniform(320, 340, n_samples)
    tau      = rng.uniform(20, 80, n_samples)

    # ---- Physics ----
    k_g   = 2.0e-8
    C_sat = 0.20
    S = np.clip(C_feed - C_sat * (1.0 + 0.02 * (T_jacket - 310)), 1e-4, None)
    G = k_g * S**0.8 * (1.0 + 0.03 * (T_jacket - 310))
    L50 = 3.0 * G * (tau * 60.0) * 1e6 + rng.normal(0, 1.0, n_samples)

    # ---- Sensor measurements ----
    conductivity = 0.8*C_feed + 0.02*RPM/100 + rng.normal(0, 0.005, n_samples)
    dT           = T_feed - T_jacket + rng.normal(0, 1.0, n_samples)
    density      = 1000 + 250*C_feed + 0.05*RPM + rng.normal(0, 3, n_samples)

    X = pd.DataFrame({
        "T_jacket": T_jacket, "F_feed": F_feed, "C_feed": C_feed,
        "RPM": RPM, "T_feed": T_feed, "tau": tau,
        "conductivity": conductivity, "dT": dT, "density": density,
    })
    y = pd.Series(L50, name="L50_um")

    # ---- Inject dirty data ----
    for col, frac in [("conductivity", 0.05), ("density", 0.03),
                      ("T_jacket", 0.02), ("RPM", 0.02)]:
        idx = rng.choice(n_samples, int(frac * n_samples), replace=False)
        X.loc[idx, col] = np.nan

    for col, frac, scale in [("conductivity", 0.01, 8.0),
                             ("density", 0.01, 6.0),
                             ("F_feed", 0.008, 5.0)]:
        idx = rng.choice(n_samples, int(frac * n_samples), replace=False)
        X.loc[idx, col] += rng.normal(0, scale * X[col].std(), len(idx))

    X["conductivity"] += np.linspace(0, 0.03, n_samples)

    dup_idx = rng.choice(n_samples, 30, replace=False)
    X = pd.concat([X, X.iloc[dup_idx]], ignore_index=True)
    y = pd.concat([y, y.iloc[dup_idx]], ignore_index=True)
    return X, y


# ---------------------------------------------------------------------------
# 2. PREPROCESSING  (NO smoothing step — it was destroying the signal)
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS = {
    "T_jacket": (290, 330), "F_feed": (0.1, 3.0), "C_feed": (0.10, 0.60),
    "RPM": (100, 800), "T_feed": (310, 350), "tau": (5, 120),
    "conductivity": (0.0, 2.0), "dT": (-5, 60), "density": (900, 1400),
}

def remove_duplicates(X, y):
    df = X.copy(); df["__y__"] = y.values
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[prep] duplicates removed: {before - len(df)}")
    return df.drop(columns="__y__"), df["__y__"].rename(y.name)

def filter_physical(X, y):
    mask = pd.Series(True, index=X.index)
    for c, (lo, hi) in PHYSICAL_BOUNDS.items():
        if c in X.columns:
            mask &= X[c].between(lo, hi) | X[c].isna()
    print(f"[prep] non-physical rows removed: {(~mask).sum()}")
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

def remove_outliers(X, y, k=3.0):
    mask = pd.Series(False, index=X.index)
    for c in X.select_dtypes(include=np.number).columns:
        q1, q3 = X[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask |= (X[c] < q1 - k*iqr) | (X[c] > q3 + k*iqr)
    print(f"[prep] outliers removed: {mask.sum()}")
    return X[~mask].reset_index(drop=True), y[~mask].reset_index(drop=True)

def impute_missing(X, n_neighbors=5):
    n_missing = X.isna().sum().sum()
    if n_missing == 0:
        print("[prep] no missing values"); return X
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    print(f"[prep] imputed {n_missing} cells (KNN)")
    return X

def correct_drift(X, cols=("conductivity",), window=101):
    X = X.copy()
    for c in cols:
        if c in X.columns:
            base = X[c].rolling(window, center=True, min_periods=1).median()
            X[c] = X[c] - base + base.mean()
    print(f"[prep] drift corrected on {list(cols)}")
    return X

def remove_collinear(X, corr_thr=0.95):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > corr_thr)]
    if drop:
        print(f"[prep] collinear removed: {drop}")
        X = X.drop(columns=drop)
    return X

def preprocess_pipeline(X_raw, y_raw, test_size=0.25, random_state=0):
    X, y = remove_duplicates(X_raw, y_raw)
    X, y = filter_physical(X, y)
    X, y = remove_outliers(X, y)
    X    = impute_missing(X)
    X    = correct_drift(X)
    # NOTE: smoothing step removed — it was destroying i.i.d. input signals.
    X    = remove_collinear(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"[prep] final shapes: train={X_tr_s.shape}, test={X_te_s.shape}")
    print(f"[prep] features: {list(X_tr.columns)}")
    return dict(X_train=X_tr_s, X_test=X_te_s,
                y_train=y_tr.values, y_test=y_te.values,
                scaler=scaler, feature_names=list(X_tr.columns))


# ---------------------------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------------------------
def train_pls(X_train, y_train, max_components=None, cv=5):
    upper = X_train.shape[1]
    if max_components is None or max_components > upper:
        max_components = upper

    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    rmses = []
    for n in range(1, max_components + 1):
        p = cross_val_predict(PLSRegression(n_components=n), X_train, y_train, cv=kf)
        rmses.append(np.sqrt(mean_squared_error(y_train, p)))
    n_opt = int(np.argmin(rmses)) + 1
    print(f"[PLS] optimal n_components = {n_opt}  (CV-RMSE={rmses[n_opt-1]:.2f})")
    return PLSRegression(n_components=n_opt).fit(X_train, y_train)


def train_gpr(X_train, y_train, max_samples=500):
    n = X_train.shape[0]
    if n > max_samples:
        idx = np.random.default_rng(0).choice(n, max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=np.ones(X_train.shape[1]),
                       length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2)))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=0,
        alpha=1e-3,
    ).fit(X_train, y_train)

    print(f"[GPR] trained on {X_train.shape[0]} samples")
    print(f"[GPR] learned kernel: {gpr.kernel_}")
    return gpr


def pls_vip(pls_model, feature_names):
    t = np.asarray(pls_model.x_scores_)
    w = np.asarray(pls_model.x_weights_)
    q = np.asarray(pls_model.y_loadings_)

    if t.ndim == 1: t = t.reshape(-1, 1)
    if w.ndim == 1: w = w.reshape(-1, 1)
    if q.ndim == 1: q = q.reshape(-1, 1)

    p, h = w.shape
    s = np.zeros(h)
    for j in range(h):
        s[j] = float(t[:, j].T @ t[:, j]) * float(q[:, j].T @ q[:, j])
    total = s.sum()
    if total == 0:
        return dict(zip(feature_names, np.zeros(p)))

    vip = np.zeros(p)
    for i in range(p):
        w_sq = np.zeros(h)
        for j in range(h):
            col_norm = np.linalg.norm(w[:, j])
            if col_norm > 0:
                w_sq[j] = (w[i, j] / col_norm) ** 2
        vip[i] = float(np.sqrt(p * (s @ w_sq) / total))
    return dict(sorted(zip(feature_names, vip), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# 4. EVALUATION + PLOTS
# ---------------------------------------------------------------------------
def report(name, y_true, y_pred, y_std=None):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    line = f"{name:<5s} | R2={r2:6.4f} | RMSE={rmse:7.3f} um | MAE={mae:7.3f} um"
    if y_std is not None:
        cov = np.mean(np.abs(y_true - y_pred) <= 1.96 * y_std)
        line += f" | 95% coverage={cov:.2%}"
    print(line)

def plot_results(y_test, preds, out="soft_sensor_results.png"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    lims = [y_test.min(), y_test.max()]

    for name, (p, s) in preds.items():
        ax[0,0].scatter(y_test, p, s=8, alpha=0.5, label=name)
    ax[0,0].plot(lims, lims, 'r--')
    ax[0,0].set(xlabel="Measured L50 (um)", ylabel="Predicted",
                title="Parity plot"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)

    for name, (p, s) in preds.items():
        ax[0,1].hist(y_test - p, bins=40, alpha=0.5, label=name)
    ax[0,1].set(xlabel="Residual (um)", ylabel="Count",
                title="Residuals"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)

    idx = np.arange(120)
    ax[1,0].plot(idx, y_test[:120], 'k-', lw=2, label="True")
    for name, (p, s) in preds.items():
        ax[1,0].plot(idx, p[:120], '--', label=name)
        if s is not None:
            ax[1,0].fill_between(idx, p[:120]-1.96*s[:120],
                                 p[:120]+1.96*s[:120], alpha=0.15)
    ax[1,0].set(xlabel="Sample", ylabel="L50 (um)",
                title="Tracking"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)

    for name, (p, s) in preds.items():
        ax[1,1].scatter(y_test, np.abs(y_test - p), s=8, alpha=0.4, label=name)
    ax[1,1].set(xlabel="Measured L50 (um)", ylabel="|Error| (um)",
                title="Error vs magnitude"); ax[1,1].legend(); ax[1,1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[plot] saved to {out}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  CRYSTALLIZER SOFT SENSOR - PLS + GPR with preprocessing")
    print("=" * 65)

    X_raw, y_raw = simulate_crystallizer(n_samples=2500, seed=42)
    print(f"\n[data] raw shape: {X_raw.shape}, "
          f"missing={X_raw.isna().sum().sum()}, "
          f"duplicates={X_raw.duplicated().sum()}")
    print(f"[data] target L50: mean={y_raw.mean():.2f}, "
          f"std={y_raw.std():.2f}, min={y_raw.min():.2f}, max={y_raw.max():.2f}\n")

    print("--- Preprocessing ---")
    d = preprocess_pipeline(X_raw, y_raw)

    print("\n--- Training ---")
    pls = train_pls(d["X_train"], d["y_train"])
    gpr = train_gpr(d["X_train"], d["y_train"])

    y_pls        = pls.predict(d["X_test"]).ravel()
    y_gpr, s_gpr = gpr.predict(d["X_test"], return_std=True)

    print("\n--- Test-set performance ---")
    report("PLS", d["y_test"], y_pls)
    report("GPR", d["y_test"], y_gpr, y_std=s_gpr)

    print("\n--- PLS VIP (feature importance) ---")
    for k, v in pls_vip(pls, d["feature_names"]).items():
        print(f"  {k:<15s}: {v:.3f}")

    plot_results(d["y_test"], {"PLS": (y_pls, None), "GPR": (y_gpr, s_gpr)})
    print("\n[done]")


if __name__ == "__main__":
    main()