
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import joblib
import os

MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "pipeline.pkl")


def _make_regimes(n: int, n_regimes: int = 10) -> np.ndarray:
    """Create rolling regime labels for GroupKFold equivalent."""
    regime_size = n // n_regimes
    labels = np.repeat(np.arange(n_regimes), regime_size)
    if len(labels) < n:
        labels = np.concatenate([labels, np.full(n - len(labels), n_regimes - 1)])
    return labels[:n]


def _winsorize(y: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    lo, hi = np.percentile(y, low * 100), np.percentile(y, high * 100)
    return np.clip(y, lo, hi), lo, hi




def compute_dro_weights(X: np.ndarray, y: np.ndarray,
                        regimes: np.ndarray, eta: float = 0.15,
                        n_rounds: int = 2) -> np.ndarray:
   
    unique_g = np.unique(regimes)
    n_g = len(unique_g)
    q = np.ones(n_g) / n_g
    sample_weights = np.ones(len(X))

    kf = KFold(n_splits=min(5, n_g), shuffle=False)

    for round_idx in range(n_rounds):
        oof_preds = np.zeros(len(X))

        for tr_idx, va_idx in kf.split(X):
            dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=sample_weights[tr_idx])
            params = {
                "objective": "regression", "metric": "mse",
                "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5,
                "min_child_samples": 10, "verbose": -1, "n_jobs": -1,
                "seed": 42 + round_idx
            }
            model = lgb.train(params, dtrain, num_boost_round=100,
                              callbacks=[lgb.log_evaluation(0)])
            oof_preds[va_idx] = model.predict(X[va_idx])

        group_mse = np.zeros(n_g)
        for i, g in enumerate(unique_g):
            mask = regimes == g
            if mask.sum() > 0:
                group_mse[i] = np.mean((oof_preds[mask] - y[mask]) ** 2)

        q = q * np.exp(eta * group_mse)
        q = q / q.sum()
        for i, g in enumerate(unique_g):
            mask = regimes == g
            sample_weights[mask] = q[i] * n_g

    return sample_weights



def compute_proxy_score(preds: np.ndarray, targets: np.ndarray,
                        n_batches: int = 10, n_runs: int = 3) -> dict:
 
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    full_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    batch_r2_list = []
    for run in range(n_runs):
        rng = np.random.RandomState(run + 100)
        indices = rng.permutation(len(preds))
        batch_size = max(len(preds) // n_batches, 5)
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size if b < n_batches - 1 else len(preds)
            idx = indices[start:end]
            b_ss_res = np.sum((targets[idx] - preds[idx]) ** 2)
            b_ss_tot = np.sum((targets[idx] - targets[idx].mean()) ** 2)
            if b_ss_tot > 0:
                batch_r2_list.append(1 - b_ss_res / b_ss_tot)

    mean_batch_r2 = float(np.mean(batch_r2_list)) if batch_r2_list else 0.0
    proxy = 2 * mean_batch_r2 - full_r2

    return {
        "proxy_score": round(proxy, 6),
        "full_r2": round(full_r2, 6),
        "mean_batch_r2": round(mean_batch_r2, 6),
        "leakage_gap": round(full_r2 - mean_batch_r2, 6)
    }




def train_pipeline(feat: pd.DataFrame,
                   selected_features: list[str],
                   n_folds: int = 5) -> dict:
 
    X = feat[selected_features].values
    y = feat["TARGET"].values

    # Winsorize target
    y, lo, hi = _winsorize(y)

    n = len(X)
    regimes = _make_regimes(n, n_regimes=n_folds)

    # Volatility weights (inverse vol per regime → stable regimes weighted less)
    vol_weights = np.ones(n)
    unique_r = np.unique(regimes)
    regime_vols = {r: np.std(y[regimes == r]) for r in unique_r}
    median_vol = np.median(list(regime_vols.values()))
    for r in unique_r:
        rel_vol = regime_vols[r] / (median_vol + 1e-8)
        w = np.clip(1.0 / np.sqrt(max(rel_vol, 0.5)), 0.5, 2.0)
        vol_weights[regimes == r] = w
    vol_weights /= vol_weights.mean()

    # Importance weights (covariate shift: recent 20% of data = "test")
    split_idx = int(n * 0.8)
    X_old, X_new = X[:split_idx], X[split_idx:]
    X_domain = np.nan_to_num(np.vstack([X_old, X_new]), nan=0.0)
    y_domain = np.concatenate([np.zeros(len(X_old)), np.ones(len(X_new))])

    domain_model = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=15,
        n_estimators=100, verbose=-1, n_jobs=-1, random_state=42
    )
    domain_model.fit(X_domain, y_domain)
    train_proba = domain_model.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    raw_w = (train_proba + 1e-6) / (1 - train_proba + 1e-6) * (len(X_old) / len(X_new))
    imp_weights = np.sqrt(np.clip(raw_w, 0.2, 5.0))
    imp_weights /= imp_weights.mean()

    # DRO weights
    print("[model] Computing Group DRO weights (eta=0.15)...")
    X_clean = np.nan_to_num(X, nan=0.0)
    dro_weights = compute_dro_weights(X_clean, y, regimes, eta=0.15)

    # Huber delta
    huber_delta = float(np.std(y) * 1.5)

    base_params = {
        "objective": "regression", "metric": "mse",
        "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1,
    }

    model_configs = [
        {"name": "LGB_Stable",   "params": {**base_params, "seed": 42}, "weights": vol_weights,               "n_boost": 300},
        {"name": "LGB_ImpWt",    "params": {**base_params, "seed": 43}, "weights": imp_weights * vol_weights, "n_boost": 300},
        {"name": "LGB_DRO_soft", "params": {**base_params, "seed": 44}, "weights": dro_weights * vol_weights, "n_boost": 300},
        {"name": "LGB_Huber",    "params": {**base_params, "objective": "huber",
                                             "huber_delta": huber_delta, "seed": 45},
                                  "weights": vol_weights, "n_boost": 300},
    ]

    kf = KFold(n_splits=n_folds, shuffle=False)
    oof_preds = {cfg["name"]: np.zeros(n) for cfg in model_configs}
    trained_models = {cfg["name"]: [] for cfg in model_configs}  # list of fold models

    for cfg in model_configs:
        print(f"[model] Training {cfg['name']}...")
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_clean)):
            dtrain = lgb.Dataset(X_clean[tr_idx], label=y[tr_idx], weight=cfg["weights"][tr_idx])
            dval = lgb.Dataset(X_clean[va_idx], label=y[va_idx])
            m = lgb.train(
                cfg["params"], dtrain,
                num_boost_round=cfg["n_boost"],
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            )
            oof_preds[cfg["name"]][va_idx] = m.predict(X_clean[va_idx])
            trained_models[cfg["name"]].append(m)

        oof_r2 = 1 - np.sum((y - oof_preds[cfg["name"]]) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"[model] {cfg['name']} OOF R²: {oof_r2:.6f}")

    # Ridge meta-stacker
    print("[model] Training Ridge meta-stacker...")
    model_names = [cfg["name"] for cfg in model_configs]
    oof_stack = np.column_stack([oof_preds[n] for n in model_names])
    ridge_oof = np.zeros(n)

    for tr_idx, va_idx in kf.split(oof_stack):
        ridge = Ridge(alpha=1.0)
        ridge.fit(oof_stack[tr_idx], y[tr_idx])
        ridge_oof[va_idx] = ridge.predict(oof_stack[va_idx])

    # Fit final Ridge on all data
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(oof_stack, y)

    ridge_r2 = 1 - np.sum((y - ridge_oof) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"[model] Ridge meta OOF R²: {ridge_r2:.6f}")

    oof_preds["Ridge_Meta"] = ridge_oof
    model_names.append("Ridge_Meta")

    # Proxy-score blend optimization
    print("[model] Optimizing blend via proxy score...")
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    best_proxy = -np.inf
    best_blend_weights = np.ones(len(model_names)) / len(model_names)
    best_proxy_stats = {}

    from itertools import product as iproduct
    for w_combo in iproduct(weight_options, repeat=len(model_names)):
        w_sum = sum(w_combo)
        if abs(w_sum - 1.0) > 0.01 or w_sum == 0:
            continue
        wts = np.array(w_combo) / w_sum
        blended = sum(wts[i] * oof_preds[name] for i, name in enumerate(model_names))
        stats = compute_proxy_score(blended, y)
        if stats["proxy_score"] > best_proxy:
            best_proxy = stats["proxy_score"]
            best_blend_weights = wts
            best_proxy_stats = stats

    print(f"[model] Best proxy={best_proxy:.6f} | Blend: "
          f"{dict(zip(model_names, best_blend_weights.round(3)))}")

    # Per-model OOF R² for dashboard display
    model_r2 = {}
    for name in model_names:
        ss_res = np.sum((y - oof_preds[name]) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        model_r2[name] = round(float(1 - ss_res / ss_tot), 6)

    return {
        "trained_models": trained_models,
        "final_ridge": final_ridge,
        "model_names": model_names,
        "blend_weights": dict(zip(model_names, best_blend_weights.tolist())),
        "proxy_stats": best_proxy_stats,
        "model_r2": model_r2,
        "selected_features": selected_features,
        "target_winsor": (lo, hi),
        "n_train": n,
    }


def predict(pipeline: dict, X_new: np.ndarray) -> dict:
    """
    Runs inference using the trained pipeline.
    Returns blended prediction + per-model predictions.
    """
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)

    model_names = pipeline["model_names"]
    blend_weights = pipeline["blend_weights"]
    trained_models = pipeline["trained_models"]
    final_ridge = pipeline["final_ridge"]

    per_model_preds = {}
    base_names = [n for n in model_names if n != "Ridge_Meta"]

    for name in base_names:
        fold_preds = np.mean([m.predict(X_new) for m in trained_models[name]], axis=0)
        per_model_preds[name] = fold_preds

    # Ridge meta prediction
    stack_input = np.column_stack([per_model_preds[n] for n in base_names])
    per_model_preds["Ridge_Meta"] = final_ridge.predict(stack_input)

    # Weighted blend
    final_pred = sum(
        blend_weights[name] * per_model_preds[name]
        for name in model_names
    )

    # Center predictions
    final_pred -= final_pred.mean()

    return {
        "prediction": final_pred.tolist(),
        "per_model": {k: v.tolist() for k, v in per_model_preds.items()},
        "blend_weights": blend_weights,
    }


def save_pipeline(pipeline: dict):
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_CACHE_PATH)
    print(f"[model] Pipeline saved to {MODEL_CACHE_PATH}")


def load_pipeline() -> dict | None:
    if os.path.exists(MODEL_CACHE_PATH):
        pipeline = joblib.load(MODEL_CACHE_PATH)
        print(f"[model] Pipeline loaded from cache")
        return pipeline
    return None
