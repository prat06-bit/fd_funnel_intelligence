import os
import warnings
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, classification_report, precision_recall_curve,
    f1_score, accuracy_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.pipeline import FEATURE_COLS, TARGET_COL, STAGE_ORDER


#  Segment label assignment 

SEGMENT_LABELS_LIST = [
    "🟢 High-Value Converters",
    "🔵 Exploring Newcomers",
    "🟡 Hesitant Comparers",
    "🔴 Dormant Drop-offs",
]


def _assign_segment_labels(seg_profiles: list) -> dict:
    """Assign meaningful labels by sorting on conversion rate & engagement."""
    sorted_profiles = sorted(
        seg_profiles,
        key=lambda x: (-x["conversion_rate"], -x["avg_events"]),
    )
    mapping = {}
    for i, p in enumerate(sorted_profiles):
        mapping[p["segment"]] = SEGMENT_LABELS_LIST[i]
    return mapping


# ── Main training function ────────────────────────────────────────────────────

def train_and_save(feature_df: pd.DataFrame, events_df: pd.DataFrame,
                   model_dir: str = "models/saved") -> dict:
    """
    Train all models and save artifacts.
    Returns dict of all trained artifacts for immediate use.
    """
    os.makedirs(model_dir, exist_ok=True)

    X = feature_df[FEATURE_COLS].copy()
    y = feature_df[TARGET_COL].copy()

    # Handle any remaining NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    results = {}

    #  MODEL 1: Conversion Predictor (LightGBM or XGBoost fallback)     
    print("\n  [1/4] Training conversion predictor...")

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    if HAS_LIGHTGBM:
        conv_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            scale_pos_weight=spw,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
    else:
        conv_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    conv_model.fit(X_train, y_train)

    # Calibrate probabilities (Platt scaling)
    cal_model = CalibratedClassifierCV(conv_model, cv=3, method="sigmoid")
    cal_model.fit(X_train, y_train)

    y_proba = cal_model.predict_proba(X_test)[:, 1]
    y_proba_raw = conv_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    auc_raw = roc_auc_score(y_test, y_proba_raw)

    # Find optimal threshold via F1
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_proba)
    f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_thresh = float(thresh_arr[f1s[:-1].argmax()])
    y_pred = (y_proba >= best_thresh).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(conv_model, X, y, cv=cv, scoring="roc_auc")

    print(f"    AUC: {auc:.4f} (calibrated) | {auc_raw:.4f} (raw)")
    print(f"    F1:  {report['1']['f1-score']:.4f} | Threshold: {best_thresh:.3f}")
    print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Baseline comparison ────────────────────────────────────────────────
    print("\n  [1b] Training baseline (Logistic Regression)...")
    scaler_lr = StandardScaler()
    X_train_sc = scaler_lr.fit_transform(X_train)
    X_test_sc = scaler_lr.transform(X_test)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_train_sc, y_train)
    lr_proba = lr_model.predict_proba(X_test_sc)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_f1 = f1_score(y_test, lr_pred)
    print(f"    Baseline AUC: {lr_auc:.4f} | F1: {lr_f1:.4f}")

    model_comparison = {
        "baseline": {"name": "Logistic Regression", "auc": round(lr_auc, 4), "f1": round(lr_f1, 4)},
        "primary": {
            "name": "LightGBM" if HAS_LIGHTGBM else "XGBoost",
            "auc": round(auc, 4),
            "f1": round(report["1"]["f1-score"], 4),
        },
        "improvement_auc": round((auc - lr_auc) / lr_auc * 100, 1),
        "improvement_f1": round((report["1"]["f1-score"] - lr_f1) / max(lr_f1, 0.01) * 100, 1),
    }

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  MODEL 2: Drop-off Stage Predictor (XGBoost multiclass)             ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    print("\n  [2/4] Training drop-off stage predictor...")

    # Only train on non-converted users (they have a drop-off stage)
    non_converted = feature_df[feature_df[TARGET_COL] == 0].copy()
    if len(non_converted) > 50:
        X_drop = non_converted[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_drop = non_converted["max_stage_idx"]  # which stage they dropped at

        # Filter to only stages with enough samples
        stage_counts = y_drop.value_counts()
        valid_stages = stage_counts[stage_counts >= 10].index
        mask = y_drop.isin(valid_stages)
        X_drop = X_drop[mask]
        y_drop = y_drop[mask]

        if len(y_drop.unique()) >= 2:
            X_drop_train, X_drop_test, y_drop_train, y_drop_test = train_test_split(
                X_drop, y_drop, test_size=0.20, random_state=42, stratify=y_drop,
            )

            drop_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )
            drop_model.fit(X_drop_train, y_drop_train, verbose=False)
            drop_pred = drop_model.predict(X_drop_test)
            drop_f1 = f1_score(y_drop_test, drop_pred, average="weighted")
            drop_acc = accuracy_score(y_drop_test, drop_pred)
            print(f"    Drop-off F1 (weighted): {drop_f1:.4f} | Accuracy: {drop_acc:.4f}")
        else:
            drop_model = None
            drop_f1 = 0
            drop_acc = 0
            print("    [WARN] Not enough stage diversity for multiclass model")
    else:
        drop_model = None
        drop_f1 = 0
        drop_acc = 0
        print("    [WARN] Not enough non-converted users for drop-off model")

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  MODEL 3: Behavioral Segmentation (KMeans)                          ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    print("\n  [3/4] Training behavioral segmentation...")

    seg_features = [
        "total_events", "total_time_in_funnel", "avg_scroll_depth",
        "funnel_attempts_feat", "evening_session_ratio", "max_stage_idx",
        "time_on_details", "kyc_attempts", "device_mobile", "income_num",
    ]
    X_seg = feature_df[seg_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    feature_df_copy = feature_df.copy()
    feature_df_copy["segment"] = kmeans.fit_predict(X_seg_scaled)

    # Build segment profiles
    seg_profiles = []
    for seg in range(4):
        s = feature_df_copy[feature_df_copy["segment"] == seg]
        seg_profiles.append({
            "segment": int(seg),
            "size": int(len(s)),
            "conversion_rate": float(s["converted"].mean()),
            "avg_events": float(s["total_events"].mean()),
            "avg_time_in_funnel": float(s["total_time_in_funnel"].mean()),
            "avg_scroll_depth": float(s["avg_scroll_depth"].mean()),
            "avg_details_time": float(s["time_on_details"].mean()),
            "avg_kyc_attempts": float(s["kyc_attempts"].mean()),
            "mobile_pct": float(s["device_mobile"].mean()),
            "reentry_rate": float(s["has_reentry"].mean()),
            "avg_income": float(s["income_num"].mean()),
            "city_tier_dist": s["city_tier"].value_counts(normalize=True).to_dict(),
        })

    seg_label_map = _assign_segment_labels(seg_profiles)
    feature_df_copy["segment_name"] = feature_df_copy["segment"].map(seg_label_map)

    for p in seg_profiles:
        p["label"] = seg_label_map[p["segment"]]

    print(f"    4 segments created")
    for p in sorted(seg_profiles, key=lambda x: -x["conversion_rate"]):
        # Strip emoji for Windows terminal compat
        clean_label = p['label'].encode('ascii', 'ignore').decode('ascii').strip()
        print(f"      {clean_label}: {p['size']} users, {p['conversion_rate']:.1%} conv")

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  SHAP EXPLAINABILITY                                                ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    print("\n  [4/4] Computing SHAP values...")

    shap_values_test = None
    explainer = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(conv_model)
            shap_values_test = explainer.shap_values(X_test)
            # For binary classification, shap_values may be a list [neg, pos]
            if isinstance(shap_values_test, list):
                shap_values_test = shap_values_test[1]
            print(f"    [OK] SHAP computed for {len(X_test)} test samples")
        except Exception as e:
            print(f"    [WARN] SHAP failed: {e}")
    else:
        print("    [WARN] SHAP not installed (pip install shap)")

    # Feature importance (from tree, as fallback / supplement)
    feat_imp = pd.Series(
        conv_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    # Global SHAP importance
    if shap_values_test is not None:
        shap_importance = pd.Series(
            np.abs(shap_values_test).mean(axis=0), index=FEATURE_COLS
        ).sort_values(ascending=False)
    else:
        shap_importance = feat_imp  # fallback

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  FUNNEL ANALYTICS (from raw events)                                 ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    total_users = events_df["user_id"].nunique()
    funnel_counts = {}
    for stage in ["landing", "fd_view", "details", "kyc", "deposit"]:
        funnel_counts[stage] = int(events_df[events_df["stage"] == stage]["user_id"].nunique())

    stage_labels = {
        "landing": "Landing Page",
        "fd_view": "FD Product View",
        "details": "FD Details",
        "kyc": "KYC Verification",
        "deposit": "FD Deposited",
    }
    funnel_data = {
        "stages": list(stage_labels.values()),
        "values": [funnel_counts.get(s, 0) for s in stage_labels.keys()],
        "stage_keys": list(stage_labels.keys()),
    }

    # Stage-wise drop-off rates
    drop_rates = {}
    stage_list = list(stage_labels.keys())
    for i in range(1, len(stage_list)):
        prev_count = funnel_counts.get(stage_list[i - 1], 1)
        curr_count = funnel_counts.get(stage_list[i], 0)
        drop_rates[stage_list[i]] = round(1 - curr_count / max(prev_count, 1), 4)

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  SAVE ALL ARTIFACTS                                                 ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    print("\n  Saving artifacts...")

    joblib.dump(conv_model,     os.path.join(model_dir, "conv_model.pkl"))
    joblib.dump(cal_model,      os.path.join(model_dir, "cal_model.pkl"))
    joblib.dump(scaler,         os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(kmeans,         os.path.join(model_dir, "kmeans.pkl"))
    joblib.dump(seg_profiles,   os.path.join(model_dir, "seg_profiles.pkl"))
    joblib.dump(seg_label_map,  os.path.join(model_dir, "seg_label_map.pkl"))
    joblib.dump(feat_imp,       os.path.join(model_dir, "feat_imp.pkl"))
    joblib.dump(shap_importance, os.path.join(model_dir, "shap_importance.pkl"))

    if drop_model is not None:
        joblib.dump(drop_model, os.path.join(model_dir, "drop_model.pkl"))

    if explainer is not None:
        joblib.dump(explainer,  os.path.join(model_dir, "shap_explainer.pkl"))

    metrics = {
        "auc": round(auc, 4),
        "auc_raw": round(auc_raw, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "best_threshold": round(best_thresh, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "model_type": "LightGBM" if HAS_LIGHTGBM else "XGBoost",
        "n_features": len(FEATURE_COLS),
        "drop_f1": round(drop_f1, 4),
        "drop_acc": round(drop_acc, 4),
    }
    joblib.dump(metrics, os.path.join(model_dir, "metrics.pkl"))
    joblib.dump(model_comparison, os.path.join(model_dir, "model_comparison.pkl"))
    joblib.dump(funnel_data, os.path.join(model_dir, "funnel_data.pkl"))
    joblib.dump(drop_rates, os.path.join(model_dir, "drop_rates.pkl"))

    # Save enriched feature dataframe
    feature_df_copy.to_csv(
        os.path.join(os.path.dirname(model_dir), "..", "data", "features_with_segments.csv"),
        index=False,
    )

    print(f"\n  [OK] All artifacts saved to {model_dir}/")

    return {
        "conv_model": conv_model,
        "cal_model": cal_model,
        "drop_model": drop_model,
        "scaler": scaler,
        "kmeans": kmeans,
        "metrics": metrics,
        "model_comparison": model_comparison,
        "seg_profiles": seg_profiles,
        "seg_label_map": seg_label_map,
        "feat_imp": feat_imp,
        "shap_importance": shap_importance,
        "shap_explainer": explainer,
        "funnel_data": funnel_data,
        "drop_rates": drop_rates,
        "feature_df": feature_df_copy,
        "X_test": X_test,
        "y_test": y_test,
        "shap_values_test": shap_values_test,
    }
