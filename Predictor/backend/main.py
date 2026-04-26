import os
import threading
import time
from datetime import datetime

from data import fetch_nifty50_daily
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from features import (
    adversarial_auc,
    build_features,
    compute_regime_stability,
    so3t_normalize,
)
from model import (
    load_pipeline,
    predict,
    save_pipeline,
    train_pipeline,
)

app = FastAPI(title="NSE Regime Predictor", version="1.0.0")



STATE = {
    "status": "idle",           
    "pipeline": None,
    "raw_df": None,             
    "feat_df": None,           
    "stability": None,          
    "adv_auc": None,
    "adv_importances": None,
    "last_trained": None,
    "error": None,
    "train_duration_sec": None,
}



TRAINING_TIMEOUT_SEC = 120  # Render timeout limit


def run_training():
    """
    Full pipeline: fetch → features → normalize → stability → train → cache.
    
    RENDER OPTIMIZED:
    - Uses render_mode=True to reduce model complexity
    - 120-second hard timeout with fallback to cached model
    - Lightweight feature selection if things are slow
    """
    STATE["status"] = "training"
    STATE["error"] = None
    t0 = time.time()

    try:
        def elapsed():
            return time.time() - t0
        
        print("\n[main] Step 1: Fetching Nifty 50 data...")
        raw_df = fetch_nifty50_daily(period_days=500)
        STATE["raw_df"] = raw_df
        print(f"[main] ✓ Fetched {len(raw_df)} rows ({elapsed():.1f}s)")

        print("[main] Step 2: Building features...")
        feat_df = build_features(raw_df)
        print(f"[main] ✓ Built {len(feat_df.columns)} features ({elapsed():.1f}s)")

        print("[main] Step 3: Session normalization...")
        feat_norm, bin_stats = so3t_normalize(feat_df, session_col="day_of_week", n_bins=5)
        print(f"[main] ✓ Normalized ({elapsed():.1f}s)")

        print("[main] Step 4: Computing regime stability...")
        stability = compute_regime_stability(feat_norm, n_regimes=20)
        STATE["stability"] = stability

        # RENDER: Select fewer features if training is slow
        n_select = max(int(len(stability) * 0.50), 8) if elapsed() > 30 else max(int(len(stability) * 0.60), 10)
        selected_features = stability.head(n_select).index.tolist()
        print(f"[main] ✓ Selected {len(selected_features)} regime-stable features ({elapsed():.1f}s)")

        if elapsed() > TRAINING_TIMEOUT_SEC * 0.6:
            print(f"[main] ⚠ Approaching timeout at {elapsed():.1f}s - using fast mode")

        print("[main] Step 5: Adversarial validation...")
        split = int(len(feat_norm) * 0.70)
        old_feat = feat_norm.iloc[:split]
        new_feat = feat_norm.iloc[split:]
        auc, imp = adversarial_auc(old_feat, new_feat)
        STATE["adv_auc"] = float(auc)
        STATE["adv_importances"] = imp
        print(f"[main] ✓ Adversarial AUC={auc:.4f} ({elapsed():.1f}s)")

        # RENDER: Check timeout before expensive training
        if elapsed() > TRAINING_TIMEOUT_SEC * 0.8:
            raise TimeoutError(f"Training exceeded {TRAINING_TIMEOUT_SEC}s threshold before ensemble - using cached model")

        print("[main] Step 6: Training ensemble (RENDER MODE - optimized)...")
        STATE["feat_df"] = feat_norm
        pipeline = train_pipeline(feat_norm, selected_features, n_folds=5, render_mode=True)
        pipeline["bin_stats"] = bin_stats
        
        if elapsed() > TRAINING_TIMEOUT_SEC:
            raise TimeoutError(f"Training exceeded {TRAINING_TIMEOUT_SEC}s - pipeline incomplete")

        save_pipeline(pipeline)
        STATE["pipeline"] = pipeline
        STATE["last_trained"] = datetime.now().isoformat()
        STATE["train_duration_sec"] = round(elapsed(), 1)
        STATE["status"] = "ready"
        print(f"\n[main] ✓ Pipeline ready in {STATE['train_duration_sec']}s")

    except TimeoutError as e:
        print(f"[main] TIMEOUT: {e}")
        STATE["status"] = "timeout"
        STATE["error"] = f"Training timed out: {str(e)}"
        # Try to use cached pipeline
        cached = load_pipeline()
        if cached:
            STATE["pipeline"] = cached
            STATE["status"] = "ready_cached"
            STATE["last_trained"] = "fallback_cached"
            print("[main] ✓ Using cached pipeline as fallback")
        else:
            STATE["status"] = "error"
            raise
    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)
        print(f"[main] ERROR: {e}")
        raise


@app.on_event("startup")
async def startup():
    cached = load_pipeline()
    if cached:
        STATE["pipeline"] = cached
        STATE["status"] = "ready"
        STATE["last_trained"] = "cached"
        print("[main] Loaded cached pipeline on startup")


@app.get("/api/status")
def get_status():
    return {
        "status": STATE["status"],
        "last_trained": STATE["last_trained"],
        "train_duration_sec": STATE["train_duration_sec"],
        "error": STATE["error"],
        "n_train_rows": len(STATE["feat_df"]) if STATE["feat_df"] is not None else None,
        "n_features": len(STATE["pipeline"]["selected_features"]) if STATE["pipeline"] else None,
    }


@app.post("/api/train")
def trigger_training():
    if STATE["status"] == "training":
        return {"message": "Training already in progress", "status": "training"}

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return {"message": "Training started", "status": "training"}


@app.get("/api/predict")
def get_prediction():
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"Pipeline not ready. Status: {STATE['status']}")

    pipeline = STATE["pipeline"]
    feat_df = STATE["feat_df"]

    if feat_df is None:
        
        raw_df = fetch_nifty50_daily(period_days=500)
        feat_df, _ = so3t_normalize(build_features(raw_df))
        STATE["feat_df"] = feat_df

    selected = pipeline["selected_features"]
   
    X_latest = feat_df[selected].iloc[[-1]].values

    result = predict(pipeline, X_latest)

    pred_value = float(result["prediction"][0])
    direction = "UP" if pred_value > 0 else "DOWN"
    confidence = min(abs(pred_value) / (feat_df["TARGET"].std() + 1e-8), 1.0)

    return {
        "prediction": round(pred_value, 6),
        "direction": direction,
        "confidence": round(float(confidence), 4),
        "per_model_predictions": {k: round(float(v[0]), 6) for k, v in result["per_model"].items()},
        "blend_weights": result["blend_weights"],
        "as_of_date": feat_df.index[-1].date().isoformat(),
        "proxy_stats": pipeline["proxy_stats"],
    }


@app.get("/api/history")
def get_history():
    """Last 120 days of Nifty 50 close prices + OOF prediction overlay."""
    if STATE["raw_df"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Run /api/train first.")

    raw_df = STATE["raw_df"].tail(120)
    feat_df = STATE["feat_df"]
    pipeline = STATE["pipeline"]

    dates = raw_df.index.strftime("%Y-%m-%d").tolist()
    closes = raw_df["Close"].round(2).tolist()

   
    actual_returns = []
    if feat_df is not None:
        aligned = feat_df["TARGET"].reindex(raw_df.index).fillna(0)
        actual_returns = (aligned * 100).round(4).tolist()

   
    oof_preds = []
    if pipeline and feat_df is not None:
        selected = pipeline["selected_features"]
        X = feat_df[selected].reindex(raw_df.index).fillna(0).values
        res = predict(pipeline, X)
        oof_preds = [round(float(p) * 100, 4) for p in res["prediction"]]

    return {
        "dates": dates,
        "closes": closes,
        "actual_returns_pct": actual_returns,
        "predicted_returns_pct": oof_preds,
    }


@app.get("/api/regime")
def get_regime():
    """Adversarial AUC + top regime-shifted features + stability scores."""
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    stability = STATE["stability"]
    adv_auc = STATE["adv_auc"]
    adv_imp = STATE["adv_importances"]

    stability_data = []
    if stability is not None:
        for feat, score in stability.head(15).items():
            stability_data.append({"feature": feat, "stability_score": round(float(score), 4)})

    shifted_features = []
    if adv_imp is not None:
        for feat, imp in adv_imp.head(10).items():
            shifted_features.append({"feature": feat, "adversarial_importance": int(imp)})

    # Interpret AUC
    if adv_auc is None:
        auc_interpretation = "Not computed"
    elif adv_auc > 0.85:
        auc_interpretation = "HIGH regime shift — recent market behaves differently from history"
    elif adv_auc > 0.65:
        auc_interpretation = "MODERATE regime shift — some distributional difference detected"
    else:
        auc_interpretation = "LOW regime shift — market conditions are stable"

    return {
        "adversarial_auc": round(adv_auc, 4) if adv_auc else None,
        "auc_interpretation": auc_interpretation,
        "top_stable_features": stability_data,
        "top_shifted_features": shifted_features,
    }


@app.get("/api/blend")
def get_blend():
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    pipeline = STATE["pipeline"]
    return {
        "blend_weights": pipeline["blend_weights"],
        "model_r2": pipeline["model_r2"],
        "proxy_stats": pipeline["proxy_stats"],
        "n_train_rows": pipeline["n_train"],
        "selected_features_count": len(pipeline["selected_features"]),
    }



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))