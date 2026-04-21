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


# ── Segment labels ─────────────────────────────────────────────────────────────
SEGMENT_LABELS_LIST = [
    "🟢 High-Value Converters",
    "🔵 Exploring Newcomers",
    "🟡 Hesitant Comparers",
    "🔴 Dormant Drop-offs",
]

# ── Drop-off model: leak-free feature set ──────────────────────────────────────
# EXCLUDED from FEATURE_COLS because they directly encode which stage was reached:
#   time_on_landing / fd_view / details / kyc  → non-zero means stage was reached
#   total_time_in_funnel  → sum of all stage times
#   kyc_attempts, kyc_avg_time  → KYC-stage specific
#   details_max_scroll    → Details-stage specific
#   mobile_kyc_friction   → device_mobile × kyc_attempts  (leaks KYC)
#   reentry_details_time  → has_reentry × time_on_details (leaks Details)
#   evening_details_depth → evening_ratio × details_max_scroll (leaks Details)
# Using full FEATURE_COLS gave drop-off F1 = 1.0000 — textbook data leakage.
_DROP_FEATURE_COLS = [
    "age", "city_tier", "income_num", "device_mobile",
    "referral_paid", "referral_partner",
    "funnel_attempts_feat", "has_reentry",
    "session_hour_mean", "evening_session_ratio", "weekday_ratio",
    "avg_scroll_depth",
    "n_devices_used", "device_switched",
    "avg_stage_gap_sec", "fastest_transition_sec",
    "tier3_mobile",
]

_SEG_FEATURES = [
    "total_events", "total_time_in_funnel", "avg_scroll_depth",
    "funnel_attempts_feat", "evening_session_ratio", "max_stage_idx",
    "time_on_details", "kyc_attempts", "device_mobile", "income_num",
]


def _assign_segment_labels(seg_profiles: list) -> dict:
    if len(seg_profiles) > len(SEGMENT_LABELS_LIST):
        raise ValueError(
            f"Got {len(seg_profiles)} segments but only "
            f"{len(SEGMENT_LABELS_LIST)} labels defined."
        )
    sorted_profiles = sorted(
        seg_profiles,
        key=lambda x: (-x["conversion_rate"], -x["avg_events"]),
    )
    return {p["segment"]: SEGMENT_LABELS_LIST[i] for i, p in enumerate(sorted_profiles)}


def train_and_save(feature_df: pd.DataFrame, events_df: pd.DataFrame,
                   model_dir: str = "models/saved") -> dict:
    os.makedirs(model_dir, exist_ok=True)

    X = feature_df[FEATURE_COLS].copy()
    y = feature_df[TARGET_COL].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    # ── MODEL 1: Conversion Predictor ─────────────────────────────────────────
    print("\n  Training conversion predictor...")

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    if HAS_LIGHTGBM:
        # class_weight='balanced' and scale_pos_weight conflict — use only scale_pos_weight
        conv_model = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
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

    cal_model = CalibratedClassifierCV(conv_model, cv=3, method="sigmoid")
    cal_model.fit(X_train, y_train)

    y_proba     = cal_model.predict_proba(X_test)[:, 1]
    y_proba_raw = conv_model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_proba)
    auc_raw     = roc_auc_score(y_test, y_proba_raw)

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_proba)
    f1s         = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_thresh = float(thresh_arr[f1s[:-1].argmax()])
    y_pred      = (y_proba >= best_thresh).astype(int)
    report      = classification_report(y_test, y_pred, output_dict=True)

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(conv_model, X, y, cv=cv, scoring="roc_auc")

    print(f"    AUC: {auc:.4f} (calibrated) | {auc_raw:.4f} (raw)")
    print(f"    F1:  {report['1']['f1-score']:.4f} | Threshold: {best_thresh:.3f}")
    print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Baseline
    print("\n  Training baseline (Logistic Regression)")
    scaler_lr = StandardScaler()
    lr_model  = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(scaler_lr.fit_transform(X_train), y_train)
    lr_proba  = lr_model.predict_proba(scaler_lr.transform(X_test))[:, 1]
    lr_auc    = roc_auc_score(y_test, lr_proba)
    lr_f1     = f1_score(y_test, (lr_proba >= 0.5).astype(int))
    print(f"    Baseline AUC: {lr_auc:.4f} | F1: {lr_f1:.4f}")

    primary_f1 = report["1"]["f1-score"]
    model_comparison = {
        "baseline": {"name": "Logistic Regression", "auc": round(lr_auc, 4), "f1": round(lr_f1, 4)},
        "primary":  {
            "name": "LightGBM" if HAS_LIGHTGBM else "XGBoost",
            "auc":  round(auc, 4),
            "f1":   round(primary_f1, 4),
        },
        "improvement_auc": round((auc - lr_auc) / max(lr_auc, 1e-9) * 100, 1),
        "improvement_f1":  round((primary_f1 - lr_f1) / max(lr_f1, 0.01) * 100, 1),
    }

    # ── MODEL 2: Drop-off Stage Predictor (leak-free) ─────────────────────────
    print("\n  Training drop-off stage predictor")
    print(f"    Using {len(_DROP_FEATURE_COLS)} leak-free features "
          f"(excluded {len(FEATURE_COLS) - len(_DROP_FEATURE_COLS)} stage-time columns)")

    drop_model = None
    drop_f1    = 0.0
    drop_acc   = 0.0

    non_converted = feature_df[feature_df[TARGET_COL] == 0].copy()
    if len(non_converted) > 50:
        available = [c for c in _DROP_FEATURE_COLS if c in non_converted.columns]
        missing   = set(_DROP_FEATURE_COLS) - set(available)
        if missing:
            print(f"    [WARN] Drop features missing: {missing}")

        X_drop = non_converted[available].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_drop = non_converted["max_stage_idx"]

        valid_stages = y_drop.value_counts().pipe(lambda s: s[s >= 10]).index
        mask   = y_drop.isin(valid_stages)
        X_drop = X_drop[mask]
        y_drop = y_drop[mask]

        if y_drop.nunique() >= 2:
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
            drop_model.fit(X_drop_train, y_drop_train)
            drop_pred = drop_model.predict(X_drop_test)
            drop_f1   = float(f1_score(y_drop_test, drop_pred, average="weighted"))
            drop_acc  = float(accuracy_score(y_drop_test, drop_pred))
            print(f"    Drop-off F1 (weighted): {drop_f1:.4f} | Accuracy: {drop_acc:.4f}")
        else:
            print("    [WARN] Not enough stage diversity for multiclass model")
    else:
        print("    [WARN] Not enough non-converted users for drop-off model")

    # ── MODEL 3: Behavioral Segmentation (KMeans) ─────────────────────────────
    print("\n  Training behavioral segmentation")

    missing_seg = [c for c in _SEG_FEATURES if c not in feature_df.columns]
    if missing_seg:
        raise KeyError(f"Segmentation features missing: {missing_seg}")

    X_seg        = feature_df[_SEG_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler       = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)

    kmeans          = KMeans(n_clusters=4, random_state=42, n_init=15)
    feature_df_copy = feature_df.copy()
    feature_df_copy["segment"] = kmeans.fit_predict(X_seg_scaled)

    seg_profiles = []
    for seg in range(4):
        s = feature_df_copy[feature_df_copy["segment"] == seg]
        seg_profiles.append({
            "segment":            int(seg),
            "size":               int(len(s)),
            "conversion_rate":    float(s["converted"].mean()) if len(s) else 0.0,
            "avg_events":         float(s["total_events"].mean()) if len(s) else 0.0,
            "avg_time_in_funnel": float(s["total_time_in_funnel"].mean()) if len(s) else 0.0,
            "avg_scroll_depth":   float(s["avg_scroll_depth"].mean()) if len(s) else 0.0,
            "avg_details_time":   float(s["time_on_details"].mean()) if len(s) else 0.0,
            "avg_kyc_attempts":   float(s["kyc_attempts"].mean()) if len(s) else 0.0,
            "mobile_pct":         float(s["device_mobile"].mean()) if len(s) else 0.0,
            "reentry_rate":       float(s["has_reentry"].mean()) if len(s) else 0.0,
            "avg_income":         float(s["income_num"].mean()) if len(s) else 0.0,
            "city_tier_dist":     s["city_tier"].value_counts(normalize=True).to_dict() if len(s) else {},
        })

    seg_label_map = _assign_segment_labels(seg_profiles)
    feature_df_copy["segment_name"] = feature_df_copy["segment"].map(seg_label_map)
    for p in seg_profiles:
        p["label"] = seg_label_map[p["segment"]]

    print(f"    4 segments created")
    for p in sorted(seg_profiles, key=lambda x: -x["conversion_rate"]):
        clean_label = p["label"].encode("ascii", "ignore").decode().strip()
        print(f"      {clean_label}: {p['size']} users, {p['conversion_rate']:.1%} conv")

    # ── SHAP Explainability ────────────────────────────────────────────────────
    print("\n  [4/4] Computing SHAP values...")

    shap_values_test = None
    explainer        = None
    if HAS_SHAP:
        try:
            explainer        = shap.TreeExplainer(conv_model)
            shap_values_test = explainer.shap_values(X_test)
            if isinstance(shap_values_test, list):
                shap_values_test = shap_values_test[1]
            print(f"    [OK] SHAP computed for {len(X_test)} test samples")
        except Exception as e:
            print(f"    [WARN] SHAP failed: {e}")
    else:
        print("    [WARN] SHAP not installed (pip install shap)")

    feat_imp = pd.Series(
        conv_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    shap_importance = (
        pd.Series(np.abs(shap_values_test).mean(axis=0), index=FEATURE_COLS)
        .sort_values(ascending=False)
        if shap_values_test is not None
        else feat_imp
    )

    # ── Funnel Analytics ───────────────────────────────────────────────────────
    # This block is at function scope — not inside any conditional branch above.
    funnel_counts = {}
    for stage in ["landing", "fd_view", "details", "kyc", "deposit"]:
        funnel_counts[stage] = int(
            events_df[events_df["stage"] == stage]["user_id"].nunique()
        )

    stage_labels = {
        "landing":  "Landing Page",
        "fd_view":  "FD Product View",
        "details":  "FD Details",
        "kyc":      "KYC Verification",
        "deposit":  "FD Deposited",
    }
    funnel_data = {
        "stages":     list(stage_labels.values()),
        "values":     [funnel_counts.get(s, 0) for s in stage_labels.keys()],
        "stage_keys": list(stage_labels.keys()),
    }

    stage_list = list(stage_labels.keys())
    drop_rates = {
        stage_list[i]: round(
            1 - funnel_counts.get(stage_list[i], 0)
              / max(funnel_counts.get(stage_list[i - 1], 1), 1),
            4,
        )
        for i in range(1, len(stage_list))
    }

    # ── Save all artifacts ─────────────────────────────────────────────────────
    print("\n  Saving artifacts...")

    def _dump(obj, name):
        joblib.dump(obj, os.path.join(model_dir, name))

    _dump(conv_model,       "conv_model.pkl")
    _dump(cal_model,        "cal_model.pkl")
    _dump(scaler,           "scaler.pkl")
    _dump(kmeans,           "kmeans.pkl")
    _dump(seg_profiles,     "seg_profiles.pkl")
    _dump(seg_label_map,    "seg_label_map.pkl")
    _dump(feat_imp,         "feat_imp.pkl")
    _dump(shap_importance,  "shap_importance.pkl")

    if drop_model is not None:
        _dump(drop_model, "drop_model.pkl")
    if explainer is not None:
        _dump(explainer,  "shap_explainer.pkl")

    metrics = {
        "auc":            round(auc, 4),
        "auc_raw":        round(auc_raw, 4),
        "precision":      round(report["1"]["precision"], 4),
        "recall":         round(report["1"]["recall"], 4),
        "f1":             round(primary_f1, 4),
        "best_threshold": round(best_thresh, 4),
        "cv_auc_mean":    round(float(cv_scores.mean()), 4),
        "cv_auc_std":     round(float(cv_scores.std()), 4),
        "test_size":      int(len(X_test)),
        "train_size":     int(len(X_train)),
        "model_type":     "LightGBM" if HAS_LIGHTGBM else "XGBoost",
        "n_features":     len(FEATURE_COLS),
        "drop_f1":        round(drop_f1, 4),
        "drop_acc":       round(drop_acc, 4),
    }
    _dump(metrics,          "metrics.pkl")
    _dump(model_comparison, "model_comparison.pkl")
    _dump(funnel_data,      "funnel_data.pkl")
    _dump(drop_rates,       "drop_rates.pkl")

    csv_path = os.path.normpath(
        os.path.join(model_dir, "..", "..", "data", "features_with_segments.csv")
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    feature_df_copy.to_csv(csv_path, index=False)

    print(f"\n  [OK] All artifacts saved to {model_dir}/")

    return {
        "conv_model":       conv_model,
        "cal_model":        cal_model,
        "drop_model":       drop_model,
        "scaler":           scaler,
        "kmeans":           kmeans,
        "metrics":          metrics,
        "model_comparison": model_comparison,
        "seg_profiles":     seg_profiles,
        "seg_label_map":    seg_label_map,
        "feat_imp":         feat_imp,
        "shap_importance":  shap_importance,
        "shap_explainer":   explainer,
        "funnel_data":      funnel_data,
        "drop_rates":       drop_rates,
        "feature_df":       feature_df_copy,
        "X_test":           X_test,
        "y_test":           y_test,
        "shap_values_test": shap_values_test,
    }