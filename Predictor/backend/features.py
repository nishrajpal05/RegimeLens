
import numpy as np
import pandas as pd
from scipy import stats


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

   
    log_ret = np.log(close / close.shift(1))

    
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_lag{lag}"] = log_ret.shift(lag)

   
    log_vol = np.log(volume.replace(0, np.nan) / volume.shift(1).replace(0, np.nan))
    for lag in [1, 2, 3, 5, 10]:
        feat[f"vol_lag{lag}"] = log_vol.shift(lag)

  
    for window in [5, 10, 20]:
        feat[f"vol_std{window}"] = log_ret.rolling(window).std().shift(1)

    
    feat["hl_ratio"] = ((high - low) / close).shift(1)          
    feat["co_ratio"] = ((close - open_) / open_).shift(1)        

  
    for window in [5, 10, 20]:
        sma = close.rolling(window).mean()
        feat[f"mom_sma{window}"] = ((close - sma) / sma).shift(1)

   
    for window in [5, 10, 14]:
        up = (log_ret > 0).astype(float)
        feat[f"up_ratio{window}"] = up.rolling(window).mean().shift(1)

    
    feat["day_of_week"] = df.index.dayofweek.astype(float) / 4.0   

    feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)


    feat["is_quarter_end"] = df.index.month.isin([3, 6, 9, 12]).astype(float)

    
    feat["TARGET"] = log_ret.shift(-1)  
   
    feat.dropna(inplace=True)

    print(f"[features] Built {len(feat.columns)-1} features, {len(feat)} rows "
          f"(dropped {len(df)-len(feat)} rows with NaN)")

    return feat



def so3t_normalize(feat: pd.DataFrame,
                   session_col: str = "day_of_week",
                   n_bins: int = 5) -> tuple[pd.DataFrame, dict]:
    
    feat = feat.copy()
    feature_cols = [c for c in feat.columns
                    if c not in ["TARGET", session_col]]

    bin_edges = np.percentile(feat[session_col].values,
                               np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    feat["_bin"] = np.digitize(feat[session_col].values, bin_edges) - 1
    feat["_bin"] = feat["_bin"].clip(0, n_bins - 1)

    bin_stats = {}
    for b in range(n_bins):
        mask = feat["_bin"] == b
        if mask.sum() > 5:
            means = feat.loc[mask, feature_cols].mean()
            stds = feat.loc[mask, feature_cols].std().clip(lower=1e-8)
            target_mean = feat.loc[mask, "TARGET"].mean()
        else:
            means = pd.Series(0.0, index=feature_cols)
            stds = pd.Series(1.0, index=feature_cols)
            target_mean = 0.0

        bin_stats[b] = {"means": means, "stds": stds, "target_mean": target_mean}

        feat.loc[mask, feature_cols] = (
            (feat.loc[mask, feature_cols] - means) / stds
        )
     
        feat.loc[mask, "TARGET"] -= target_mean

    feat.drop(columns=["_bin"], inplace=True)
    print(f"[features] SO3_T-style normalization applied with {n_bins} session bins")
    return feat, bin_stats



def compute_regime_stability(feat: pd.DataFrame,
                              n_regimes: int = 20) -> pd.Series:
    
    feature_cols = [c for c in feat.columns if c != "TARGET"]
    n = len(feat)
    regime_size = n // n_regimes

    regime_labels = np.repeat(np.arange(n_regimes), regime_size)
    # Pad last rows into final regime
    if len(regime_labels) < n:
        regime_labels = np.concatenate([
            regime_labels,
            np.full(n - len(regime_labels), n_regimes - 1)
        ])
    regime_labels = regime_labels[:n]

    regimes = pd.Series(regime_labels, index=feat.index)
    unique_regimes = np.unique(regime_labels)

    # Compute per-regime correlations
    regime_corrs = pd.DataFrame(index=unique_regimes, columns=feature_cols, dtype=float)
    for g in unique_regimes:
        mask = regimes == g
        if mask.sum() > 5:
            for f in feature_cols:
                vals = feat.loc[mask, f].values
                tgt = feat.loc[mask, "TARGET"].values
                valid = ~(np.isnan(vals) | np.isnan(tgt))
                if valid.sum() > 3:
                    corr, _ = stats.spearmanr(vals[valid], tgt[valid])
                    regime_corrs.loc[g, f] = corr if not np.isnan(corr) else 0.0
                else:
                    regime_corrs.loc[g, f] = 0.0

    regime_corrs = regime_corrs.astype(float).fillna(0.0)

    # Compute stability score per feature
    stability_scores = {}
    for f in feature_cols:
        corrs = regime_corrs[f].values
        median_abs = np.median(np.abs(corrs))
        sign_consistency = np.abs(np.mean(np.sign(corrs[corrs != 0]))) if (corrs != 0).any() else 0.0
        q75, q25 = np.percentile(corrs, 75), np.percentile(corrs, 25)
        iqr = q75 - q25
        inv_iqr = min(1.0 / (iqr + 1e-8) / 100.0, 1.0)
        score = 0.40 * median_abs + 0.35 * sign_consistency + 0.25 * inv_iqr
        stability_scores[f] = score

    stability = pd.Series(stability_scores).sort_values(ascending=False)
    print(f"[features] Regime stability computed over {n_regimes} regimes")
    print(f"[features] Top 5 stable features: {stability.head(5).index.tolist()}")
    return stability



def adversarial_auc(train_feat: pd.DataFrame,
                    recent_feat: pd.DataFrame) -> tuple[float, pd.Series]:
   
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    feature_cols = [c for c in train_feat.columns if c != "TARGET"]

    X_train_adv = train_feat[feature_cols].values
    X_recent_adv = recent_feat[feature_cols].values

    X = np.vstack([X_train_adv, X_recent_adv])
    y = np.concatenate([np.zeros(len(X_train_adv)), np.ones(len(X_recent_adv))])

  
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=42)

    model = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=15, n_estimators=100,
        verbose=-1, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, y_pred)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print(f"[features] Adversarial AUC = {auc:.4f}")
    return auc, importances
