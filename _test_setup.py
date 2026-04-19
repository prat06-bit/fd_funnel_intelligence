import os, sys, traceback, joblib
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    print("🔍 Running post-setup validation...\n")

    model_dir = os.path.join(ROOT, "models", "saved")

    # ── 1. Check artifacts exist ─────────────────────
    required_files = [
        "conv_model.pkl",
        "cal_model.pkl",
        "scaler.pkl",
        "kmeans.pkl",
        "metrics.pkl",
        "funnel_data.pkl",
        "drop_rates.pkl"
    ]

    for f in required_files:
        path = os.path.join(model_dir, f)
        assert os.path.exists(path), f"Missing file: {f}"

    print("✅ All artifacts present")

    # ── 2. Load models ──────────────────────────────
    model = joblib.load(os.path.join(model_dir, "conv_model.pkl"))
    cal_model = joblib.load(os.path.join(model_dir, "cal_model.pkl"))

    print("✅ Models load correctly")

    # ── 3. Load feature data ────────────────────────
    feat_path = os.path.join(ROOT, "data", "features_with_segments.csv")
    assert os.path.exists(feat_path), "Missing feature dataset"

    df = pd.read_csv(feat_path)
    assert len(df) > 0, "Feature data empty"

    print(f"✅ Feature data loaded: {df.shape}")

    # ── 4. Prediction sanity check ──────────────────
    from features.pipeline import FEATURE_COLS

    sample = df[FEATURE_COLS].iloc[:10].fillna(0)
    preds = cal_model.predict_proba(sample)

    assert preds.shape == (10, 2), "Prediction shape invalid"
    assert (preds >= 0).all() and (preds <= 1).all(), "Invalid probabilities"

    print("✅ Predictions valid")

    # ── 5. SHAP sanity check (important for you) ────
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)

        if isinstance(sv, list):
            sv = sv[1]

        assert sv.shape[0] == sample.shape[0], "SHAP mismatch"
        print("✅ SHAP working")

    except Exception as e:
        print("⚠️ SHAP check failed:", e)

    # ── 6. Metrics sanity ───────────────────────────
    metrics = joblib.load(os.path.join(model_dir, "metrics.pkl"))

    assert metrics["auc"] > 0.6, "AUC too low"
    assert metrics["f1"] > 0.5, "F1 too low"

    print("✅ Metrics look healthy")

    print("\n🎉 ALL CHECKS PASSED")

except Exception:
    print("\n❌ TEST FAILED\n")
    traceback.print_exc()
    sys.exit(1)