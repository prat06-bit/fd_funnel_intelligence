import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from landing_page import render_landing_page

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FD Funnel Intelligence · Decision Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN TOKENS & CSS ───────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700;800&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main { background: #060a14; }
.block-container { padding: 1.5rem 3rem; max-width: 1500px; }

/* ── cards ── */
.kpi-card {
    background: linear-gradient(145deg, #0f1629 0%, #0a0f1e 100%);
    border: 1px solid #1e2a45;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    transition: border-color .25s, transform .25s;
}
.kpi-card:hover { border-color: #00c2ff40; transform: translateY(-2px); }
.kpi-val  { font-family:'Sora',sans-serif; font-size:1.85rem; font-weight:800; letter-spacing:-1px; }
.kpi-label{ font-size:.68rem; color:#5a6a8a; text-transform:uppercase; letter-spacing:.12em; margin-top:4px; }

/* ── health score ring ── */
.health-ring {
    background: linear-gradient(145deg, #0f1629 0%, #0a0f1e 100%);
    border: 1px solid #1e2a45;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.health-score { font-size: 3.2rem; font-weight: 800; letter-spacing: -2px; }
.health-label { font-size: .72rem; color: #5a6a8a; text-transform: uppercase; letter-spacing: .14em; margin-top: 4px; }

/* ── section heading ── */
.sec-head {
    font-size: 1rem; font-weight: 600; color: #cdd9f5;
    border-left: 3px solid #00c2ff; padding-left: 12px;
    margin: 1.4rem 0 .8rem 0; letter-spacing: .02em;
}

/* ── risk/priority badges ── */
.badge-high   { background:#ff293010; color:#ff4d5a; border:1px solid #ff293040; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:600; }
.badge-medium { background:#f59e0b10; color:#fbbf24; border:1px solid #f59e0b40; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:600; }
.badge-low    { background:#22c55e10; color:#4ade80; border:1px solid #22c55e40; border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:600; }

/* ── insight box ── */
.insight-box {
    background: #0a0f1e; border: 1px solid #1e2a45;
    border-left: 3px solid #00c2ff;
    border-radius: 10px; padding: 18px 20px;
    color: #8899bb; line-height: 1.75; font-size: .88rem;
    white-space: pre-wrap;
}

/* ── intervention card ── */
.intervention-card {
    background: linear-gradient(145deg, #0d1420 0%, #0a0f1e 100%);
    border: 1px solid #1e2a45; border-left: 3px solid #a78bfa;
    border-radius: 10px; padding: 16px 18px; margin-bottom: 10px;
    transition: border-color .2s;
}
.intervention-card:hover { border-color: #a78bfa60; }
.intv-type { font-size: .65rem; color: #a78bfa; text-transform: uppercase; letter-spacing: .14em; font-weight: 600; }
.intv-action { font-size: .88rem; color: #cdd9f5; margin: 6px 0; line-height: 1.5; }
.intv-meta { font-size: .75rem; color: #5a6a8a; }

/* ── tab bar ── */
div[data-testid="stTabs"] > div > div > button {
    color:#5a6a8a !important; font-family:'Sora',sans-serif !important;
    font-size:.82rem !important; padding:8px 16px !important;
}
div[data-testid="stTabs"] > div > div > button[aria-selected="true"] {
    color:#00c2ff !important; border-bottom:2px solid #00c2ff !important;
}
div[data-testid="stTabs"] > div:first-child { border-bottom:1px solid #1e2a45 !important; }

/* ── sliders & inputs ── */
.stSlider > label, .stSelectbox > label, .stNumberInput > label { color:#5a6a8a !important; font-size:.8rem !important; }
[data-baseweb="select"] > div { background:#0f1629 !important; border-color:#1e2a45 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#00c2ff !important; }

/* ── delta widget ── */
.delta-positive { color: #4ade80; font-weight: 600; }
.delta-negative { color: #ff4d5a; font-weight: 600; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060a14; }
::-webkit-scrollbar-thumb { background: #1e2a45; border-radius: 4px; }

code { font-family:'DM Mono',monospace; color:#00c2ff; background:#0f1629; padding:1px 5px; border-radius:4px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora", color="#8899bb", size=12),
    margin=dict(l=20, r=20, t=30, b=20),
)
COLORS = {
    "cyan": "#00c2ff", "green": "#4ade80", "red": "#ff4d5a",
    "amber": "#fbbf24", "purple": "#a78bfa", "pink": "#f472b6",
    "slate": "#5a6a8a",
}
SEG_COLORS = {
    "🟢 High-Value Converters": COLORS["green"],
    "🔵 Exploring Newcomers":   COLORS["cyan"],
    "🟡 Hesitant Comparers":    COLORS["amber"],
    "🔴 Dormant Drop-offs":     COLORS["red"],
}


# ── LOAD / TRAIN ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all():
    from data.generate_data import generate_fd_data
    from features.pipeline import build_features, FEATURE_COLS
    from models.train_models import train_and_save

    model_dir = os.path.join(ROOT, "models", "saved")
    model_path = os.path.join(model_dir, "conv_model.pkl")

    if not os.path.exists(model_path):
        # First run: generate data, engineer features, train models
        data = generate_fd_data(2000, save_dir=os.path.join(ROOT, "data", "raw"))
        feature_df = build_features(data["users"], data["funnel_events"], data["fd_transactions"])
        results = train_and_save(feature_df, data["funnel_events"], model_dir=model_dir)
        return results, FEATURE_COLS

    # Load pre-trained artifacts
    results = {
        "conv_model":      joblib.load(os.path.join(model_dir, "conv_model.pkl")),
        "cal_model":       joblib.load(os.path.join(model_dir, "cal_model.pkl")),
        "scaler":          joblib.load(os.path.join(model_dir, "scaler.pkl")),
        "kmeans":          joblib.load(os.path.join(model_dir, "kmeans.pkl")),
        "metrics":         joblib.load(os.path.join(model_dir, "metrics.pkl")),
        "model_comparison":joblib.load(os.path.join(model_dir, "model_comparison.pkl")),
        "seg_profiles":    joblib.load(os.path.join(model_dir, "seg_profiles.pkl")),
        "seg_label_map":   joblib.load(os.path.join(model_dir, "seg_label_map.pkl")),
        "feat_imp":        joblib.load(os.path.join(model_dir, "feat_imp.pkl")),
        "shap_importance": joblib.load(os.path.join(model_dir, "shap_importance.pkl")),
        "funnel_data":     joblib.load(os.path.join(model_dir, "funnel_data.pkl")),
        "drop_rates":      joblib.load(os.path.join(model_dir, "drop_rates.pkl")),
    }

    # Load optional artifacts
    drop_model_path = os.path.join(model_dir, "drop_model.pkl")
    results["drop_model"] = joblib.load(drop_model_path) if os.path.exists(drop_model_path) else None

    explainer_path = os.path.join(model_dir, "shap_explainer.pkl")
    results["shap_explainer"] = joblib.load(explainer_path) if os.path.exists(explainer_path) else None

    # Load feature dataframe
    feat_csv = os.path.join(ROOT, "data", "features_with_segments.csv")
    if os.path.exists(feat_csv):
        results["feature_df"] = pd.read_csv(feat_csv)
    else:
        results["feature_df"] = pd.DataFrame()

    return results, FEATURE_COLS


with st.spinner("⚡ Initialising decision engine (first run trains models on 2,000 users)…"):
    from features.pipeline import FEATURE_COLS, FEATURE_LABELS
    results, FEAT_COLS = load_all()

model       = results["conv_model"]
cal_model   = results["cal_model"]
metrics     = results["metrics"]
comparison  = results["model_comparison"]
seg_profiles = results["seg_profiles"]
seg_label_map = results["seg_label_map"]
feat_imp    = results["feat_imp"]
shap_imp    = results["shap_importance"]
funnel_data = results["funnel_data"]
drop_rates  = results["drop_rates"]
df          = results.get("feature_df", pd.DataFrame())
drop_model  = results.get("drop_model")
shap_explainer = results.get("shap_explainer")


# ── Compute Funnel Health Score ────────────────────────────────────────────────
from engine.decision_engine import compute_funnel_health_score
health = compute_funnel_health_score(funnel_data, drop_rates, metrics)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="padding:20px 0 6px 0">
  <div style="font-size:.65rem;color:#00c2ff;text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px">
    Blostem · Track 03 · Data Analytics & Insights
  </div>
  <h1 style="color:#e8eeff;font-size:2rem;font-weight:800;margin:0;letter-spacing:-.03em">
    FD Funnel Intelligence Engine
  </h1>
  <p style="color:#5a6a8a;margin-top:8px;font-size:.85rem">
    Decision engine that predicts drop-off &nbsp;·&nbsp; explains why &nbsp;·&nbsp;
    recommends interventions &nbsp;·&nbsp; quantifies revenue impact
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── KPI ROW + HEALTH SCORE ────────────────────────────────────────────────────
conv_rate = health["conversion_rate"]
total_users = funnel_data["values"][0] if funnel_data["values"] else 0

# Health score color
h_score = health["overall_score"]
h_color = COLORS["green"] if h_score >= 65 else COLORS["amber"] if h_score >= 40 else COLORS["red"]

c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1, 1, 1, 1, 1])

kpis = [
    (f"{h_score:.0f}", "Funnel Health", h_color),
    (f"{total_users:,}", "Users Tracked", COLORS["cyan"]),
    (f"{conv_rate:.1f}%", "Conversion Rate", COLORS["green"] if conv_rate > 20 else COLORS["red"]),
    (f"{metrics['auc']:.3f}", f"Model AUC ({metrics.get('model_type', 'ML')})", COLORS["purple"]),
    (f"{metrics['f1']:.3f}", "Model F1 Score", COLORS["amber"]),
    (f"{health['worst_drop_rate']:.0f}%", f"Worst Drop ({health['worst_bottleneck'].replace('_',' ').title()})", COLORS["red"]),
]

for col, (val, label, color) in zip([c1, c2, c3, c4, c5, c6], kpis):
    with col:
        css = 'health-ring' if label == 'Funnel Health' else 'kpi-card'
        st.markdown(
            f'<div class="{css}"><div class="{"health-score" if label == "Funnel Health" else "kpi-val"}" style="color:{color}">{val}</div>'
            f'<div class="{"health-label" if label == "Funnel Health" else "kpi-label"}">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Funnel Analytics",
    "🎯  Prediction Engine",
    "⚡  What-If Simulator",
    "👥  Segments & Insights",
    "🏗️  Model & Architecture",
    "🚨  Priority Queue",
])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — FUNNEL ANALYTICS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    col_funnel, col_drops = st.columns([1.4, 1])

    with col_funnel:
        st.markdown('<div class="sec-head">Real-Time Funnel (Event-Driven)</div>', unsafe_allow_html=True)
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data["stages"],
            x=funnel_data["values"],
            textinfo="value+percent initial",
            marker=dict(color=[COLORS["cyan"], COLORS["purple"], COLORS["amber"], COLORS["pink"], COLORS["green"]]),
            connector=dict(line=dict(color="#1e2a45", width=1)),
        ))
        fig_funnel.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_drops:
        st.markdown('<div class="sec-head">Stage-wise Drop-off Rates</div>', unsafe_allow_html=True)
        drop_df = pd.DataFrame([
            {"Stage": k.replace("_", " ").title(), "Drop Rate": v * 100}
            for k, v in drop_rates.items()
        ])
        if len(drop_df) > 0:
            fig_drop = go.Figure(go.Bar(
                x=drop_df["Drop Rate"],
                y=drop_df["Stage"],
                orientation="h",
                marker=dict(
                    color=drop_df["Drop Rate"],
                    colorscale=[[0, COLORS["green"]], [0.5, COLORS["amber"]], [1, COLORS["red"]]],
                    showscale=False,
                ),
                text=drop_df["Drop Rate"].round(1).astype(str) + "%",
                textposition="outside",
            ))
            fig_drop.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig_drop, use_container_width=True)

    # ── Feature importance (SHAP-based) ────────────────────────────────────
    st.markdown('<div class="sec-head">Top Conversion Drivers (SHAP Feature Importance)</div>', unsafe_allow_html=True)

    col_fi1, col_fi2 = st.columns(2)

    with col_fi1:
        si_df = shap_imp.head(10).reset_index()
        si_df.columns = ["feature", "importance"]
        si_df["label"] = si_df["feature"].map(FEATURE_LABELS).fillna(si_df["feature"])
        fig_si = go.Figure(go.Bar(
            x=si_df["importance"],
            y=si_df["label"],
            orientation="h",
            marker_color=COLORS["cyan"],
            marker_line_width=0,
        ))
        fig_si.update_layout(**PLOTLY_LAYOUT, height=340,
                             title=dict(text="SHAP Importance (|mean|)", font=dict(size=13, color="#8899bb")))
        fig_si.update_yaxes(autorange="reversed", gridcolor="#1e2a45")
        st.plotly_chart(fig_si, use_container_width=True)

    with col_fi2:
        fi_df = feat_imp.head(10).reset_index()
        fi_df.columns = ["feature", "importance"]
        fi_df["label"] = fi_df["feature"].map(FEATURE_LABELS).fillna(fi_df["feature"])
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["label"],
            orientation="h",
            marker_color=COLORS["purple"],
            marker_line_width=0,
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, height=340,
                             title=dict(text="Tree Feature Importance (Gain)", font=dict(size=13, color="#8899bb")))
        fig_fi.update_yaxes(autorange="reversed", gridcolor="#1e2a45")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Drop-off by device and city tier ───────────────────────────────────
    if len(df) > 0:
        col_dev, col_tier = st.columns(2)

        with col_dev:
            st.markdown('<div class="sec-head">Conversion by Device Type</div>', unsafe_allow_html=True)
            if "device_type" in df.columns:
                dev_conv = df.groupby("device_type")["converted"].mean().reset_index()
                dev_conv.columns = ["Device", "Conv Rate"]
                dev_conv["Conv Rate"] *= 100
                fig_dev = go.Figure(go.Bar(
                    x=dev_conv["Device"].str.title(),
                    y=dev_conv["Conv Rate"],
                    marker_color=[COLORS["cyan"], COLORS["amber"], COLORS["purple"]],
                    text=dev_conv["Conv Rate"].round(1).astype(str) + "%",
                    textposition="outside",
                ))
                fig_dev.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig_dev, use_container_width=True)

        with col_tier:
            st.markdown('<div class="sec-head">Conversion by City Tier</div>', unsafe_allow_html=True)
            tier_conv = df.groupby("city_tier")["converted"].mean().reset_index()
            tier_conv.columns = ["Tier", "Conv Rate"]
            tier_conv["Conv Rate"] *= 100
            tier_conv["label"] = tier_conv["Tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"})
            fig_tier = go.Figure(go.Bar(
                x=tier_conv["label"],
                y=tier_conv["Conv Rate"],
                marker_color=[COLORS["green"], COLORS["amber"], COLORS["red"]],
                text=tier_conv["Conv Rate"].round(1).astype(str) + "%",
                textposition="outside",
            ))
            fig_tier.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig_tier, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — PREDICTION ENGINE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="sec-head">Individual User Prediction + Decision Engine</div>', unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        age            = st.slider("Age", 22, 70, 34)
        city_tier_val  = st.selectbox("City Tier", [1, 2, 3], index=1)
        income_bracket = st.selectbox("Income Bracket", ["<3L", "3-7L", "7-15L", ">15L"], index=1)

    with col_in2:
        device_type    = st.selectbox("Device", ["mobile", "desktop", "tablet"])
        referral_src   = st.selectbox("Referral Source", ["organic", "google_ads", "partner_referral", "social_media", "direct"])
        max_stage      = st.selectbox("Deepest Stage Reached", ["Landing", "FD View", "Details", "KYC"], index=2)

    with col_in3:
        time_details   = st.slider("Time on Details Page (sec)", 0, 300, 45)
        time_kyc       = st.slider("Time on KYC Page (sec)", 0, 300, 0)
        kyc_attempts   = st.slider("KYC Attempts", 0, 5, 1)
        funnel_attempts = st.slider("Funnel Re-entries", 1, 4, 1)

    predict_btn = st.button("⚡  Run Decision Engine", use_container_width=True, type="primary")

    if predict_btn:
        from features.pipeline import INCOME_MAP, STAGE_ORDER
        from engine.decision_engine import evaluate_user

        stage_map = {"Landing": 0, "FD View": 1, "Details": 2, "KYC": 3}
        max_stage_idx = stage_map.get(max_stage, 2)
        inc_num = INCOME_MAP.get(income_bracket, 2)

        # Build feature vector (match FEATURE_COLS order)
        input_dict = {
            "age": age, "city_tier": city_tier_val, "income_num": inc_num,
            "device_mobile": 1 if device_type == "mobile" else 0,
            "device_desktop": 1 if device_type == "desktop" else 0,
            "referral_paid": 1 if referral_src in ["google_ads", "social_media"] else 0,
            "referral_partner": 1 if referral_src == "partner_referral" else 0,
            "max_stage_idx": max_stage_idx,
            "total_events": max_stage_idx + 1 + funnel_attempts,
            "funnel_attempts_feat": funnel_attempts,
            "has_reentry": 1 if funnel_attempts > 1 else 0,
            "session_hour_mean": 15.0,
            "evening_session_ratio": 0.3,
            "weekday_ratio": 0.6,
            "time_on_landing": 20,
            "time_on_fd_view": 30,
            "time_on_details": time_details,
            "time_on_kyc": time_kyc,
            "total_time_in_funnel": 20 + 30 + time_details + time_kyc,
            "avg_scroll_depth": 55.0,
            "details_max_scroll": min(time_details / 2 + 20, 100),
            "kyc_attempts": kyc_attempts,
            "kyc_drop": 1 if max_stage_idx == 3 and kyc_attempts > 0 else 0,
            "n_devices_used": 1,
            "device_switched": 0,
            "avg_stage_gap_sec": 25.0,
            "fastest_transition_sec": 10.0,
            "mobile_kyc_friction": (1 if device_type == "mobile" else 0) * kyc_attempts,
            "high_income_engagement": (1 if inc_num >= 3 else 0) * (20 + 30 + time_details + time_kyc),
            "tier3_mobile": 1 if city_tier_val == 3 and device_type == "mobile" else 0,
            "reentry_details_time": (1 if funnel_attempts > 1 else 0) * time_details,
            "evening_details_depth": 0.3 * min(time_details / 2 + 20, 100),
        }
        from features.pipeline import build_features_from_input

        input_df = build_features_from_input(input_dict)
        raw_prob = float(cal_model.predict_proba(input_df)[0, 1])

        threshold = 0.067  # from training logs

        conv_prob = min(raw_prob / threshold , 1.0)

        # Predict drop-off stage
        predicted_drop = max_stage_idx
        if drop_model is not None:
            try:
                predicted_drop = int(drop_model.predict(input_df)[0])
            except Exception:
                pass

        # Run decision engine
        ml_predictions = {
            "conversion_probability": conv_prob,
            "predicted_drop_stage_idx": predicted_drop,
            "segment": "Interactive User",
            "top_risk_factors": [],
        }

        # Add user-friendly fields for LLM
        input_dict["user_id"] = "INTERACTIVE"
        input_dict["device_type"] = device_type
        input_dict["income_bracket"] = income_bracket
        input_dict["referral_source"] = referral_src

        plan = evaluate_user(input_dict, ml_predictions)

        # Risk display
        risk_color = COLORS["red"] if plan.risk_level == "high" else COLORS["amber"] if plan.risk_level == "medium" else COLORS["green"]
        risk_badge = f'<span class="badge-{plan.risk_level}">{plan.risk_level.upper()} RISK</span>'

        rc1, rc2 = st.columns([1, 2])

        with rc1:
            st.markdown(
                f"""
<div class="kpi-card" style="padding:28px;">
  <div style="font-size:.68rem;color:#5a6a8a;text-transform:uppercase;letter-spacing:.12em">Conversion Probability</div>
  <div style="font-size:3.2rem;font-weight:800;color:{risk_color};letter-spacing:-.03em">{conv_prob*100:.1f}%</div>
  <div style="margin-top:8px">{risk_badge}</div>
  <div style="margin-top:12px;font-size:.75rem;color:#5a6a8a">
    Predicted drop: <strong style="color:#cdd9f5">{plan.predicted_drop_stage}</strong><br>
    Revenue at risk: <strong style="color:{COLORS['amber']}">₹{plan.estimated_revenue_at_risk:,.0f}</strong>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conv_prob * 100,
                number=dict(suffix="%", font=dict(color=risk_color, size=24)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#1e2a45"),
                    bar=dict(color=risk_color),
                    bgcolor="#0f1629",
                    bordercolor="#1e2a45",
                    steps=[
                        dict(range=[0, 25], color="rgba(255, 41, 48, 0.15)"),
                        dict(range=[25, 50], color="rgba(245, 158, 11, 0.15)"),
                        dict(range=[50, 100], color="rgba(34, 197, 94, 0.15)"),
                    ],
                ),
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180,
                                margin=dict(l=20, r=20, t=10, b=0), font=dict(color="#8899bb"))
            st.plotly_chart(fig_g, use_container_width=True)

        with rc2:
            st.markdown('<div class="sec-head">🎯 Decision Engine — Recommended Interventions</div>', unsafe_allow_html=True)
            if plan.interventions:

                for intv in plan.interventions[:4]:
                    st.markdown(f"""
                <div class="intervention-card">
                    <div class="intv-type">{intv.intervention_type} · Priority: {intv.priority_score:.0f}</div>
                    <div class="intv-action">{intv.action}</div>
                    <div class="intv-meta">

                    +{intv.expected_lift_pct:.0f}% lift &nbsp;·&nbsp;
                    Gain: ₹{int(intv.expected_gain):,} &nbsp;·&nbsp;
                    Net Gain: ₹{int(intv.net_roi):,} &nbsp;·&nbsp;
                    Payback: {(intv.expected_gain / max(intv.cost_per_user, 1)):.1f}x

                    </div>
                </div>
                """, unsafe_allow_html=True)

                best_action = plan.interventions[0]
                new_prob = getattr(best_action, "expected_new_prob", conv_prob)
                new_prob = min(new_prob, 1.0)

                st.markdown('<div class="sec-head">📊 Impact Simulation</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)

                with c1:
                    st.metric("Current Conversion", f"{conv_prob*100:.1f}%")

                with c2:
                    st.metric("After Intervention", f"{new_prob*100:.1f}%")

            else:
                st.markdown(
                    '<div class="insight-box" style="color:#4ade80;text-align:center">✅ Low risk — no intervention needed</div>',
                    unsafe_allow_html=True
                )
        # ── SHAP Explanation ───────────────────────────────────────────────
        if shap_explainer is not None:
            st.markdown('<div class="sec-head">🔍 SHAP — Why This Prediction</div>', unsafe_allow_html=True)
            try:
                import shap
                sv = shap_explainer.shap_values(input_df)
                if isinstance(sv, list):
                    sv = sv[1]
                shap_series = pd.Series(sv[0], index=FEAT_COLS)
                top_positive = shap_series.nlargest(5)
                top_negative = shap_series.nsmallest(5)

                shap_display = pd.concat([top_positive, top_negative]).reset_index()
                shap_display.columns = ["feature", "shap_value"]
                shap_display["label"] = shap_display["feature"].map(FEATURE_LABELS).fillna(shap_display["feature"])
                shap_display = shap_display.sort_values("shap_value")

                fig_shap = go.Figure(go.Bar(
                    x=shap_display["shap_value"],
                    y=shap_display["label"],
                    orientation="h",
                    marker_color=[COLORS["green"] if v > 0 else COLORS["red"] for v in shap_display["shap_value"]],
                    marker_line_width=0,
                ))
                fig_shap.update_layout(
                    **PLOTLY_LAYOUT, height=320,
                    title=dict(text="SHAP Values: Green = helps conversion, Red = hurts conversion",
                               font=dict(size=11, color="#5a6a8a")),
                )
                fig_shap.update_yaxes(gridcolor="#1e2a45")
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="insight-box"> SHAP visualization error: {e}</div>', unsafe_allow_html=True)

        # ── LLM Insight ────────────────────────────────────────────────────
        st.markdown('<div class="sec-head"> AI Strategy Recommendation</div>', unsafe_allow_html=True)
        with st.spinner("NVIDIA Llama-3.3-70B analysing intervention plan…"):
            from utils.llm_insights import intervention_explanation
            insight = intervention_explanation(input_dict, plan.to_dict())
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # ── Retention Playbook ─────────────────────────────────────────────
        st.markdown("<hr style='border-color:#1e2a45;margin:2rem 0'>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head"> Recommended Retention Actions</div>', unsafe_allow_html=True)

        from utils.retention_playbook import generate_playbook, playbook_to_dataframe

        # Map input_dict fields to what the playbook engine expects
        customer_for_playbook = {
            "num_fds_booked": 0,
            "support_tickets": kyc_attempts * 0.8,  # proxy: KYC friction ≈ support friction
            "last_login_days": 0 if funnel_attempts > 1 else 90,
            "platform_sessions": funnel_attempts + 2,
            "income_bracket": income_bracket,
            "avg_fd_amount": 50_000 * (1 + (INCOME_MAP.get(income_bracket, 2) - 2) * 0.3),
        }

        playbook = generate_playbook(customer_for_playbook, 1 - conv_prob)
        pb_display = playbook_to_dataframe(playbook)

        if pb_display["actions"]:
            st.markdown(
                f'<p style="color:#5a6a8a;font-size:.85rem">'
                f'{pb_display["total_actions"]} actions recommended &nbsp;·&nbsp; '
                f'Expected impact: {pb_display["expected_combined_impact"]}'
                f'</p>',
                unsafe_allow_html=True,
            )

            for i, action in enumerate(pb_display["actions"], 1):
                with st.expander(f"{i}. {action['action']}", expanded=(i == 1)):
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.write(f"**Description:** {action['description']}")
                        st.write(f"**Timeline:** {action['timeline']}")
                        st.write(f"**Owner:** {action['owner']}")
                    with cols[1]:
                        st.write(f"**Success Metric**\n{action['success_metric']}")
                        st.write(f"**Impact**\n{action['expected_impact']}")
        else:
            st.markdown(
                '<p style="color:#5a6a8a">No immediate interventions needed. Continue monitoring.</p>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            '<div class="insight-box" style="color:#3a4a6a;text-align:center;padding:40px">'
            'Configure user attributes above and click <strong style="color:#00c2ff">Run Decision Engine</strong> '
            'to get conversion predictions, SHAP explanations, and prioritized intervention plans'
            '</div>',
            unsafe_allow_html=True,
        )



# TAB 3 — WHAT-IF SIMULATOR                      
with tab3:
    from engine.simulator import SCENARIOS, simulate_intervention, simulate_all_scenarios

    st.markdown('<div class="sec-head">What-If Intervention Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#5a6a8a;font-size:.82rem;margin-bottom:1rem">'
        'Select an intervention scenario to simulate its impact on the funnel. '
        'Uses counterfactual reasoning to estimate conversion lift, revenue impact, and ROI.'
        '</p>',
        unsafe_allow_html=True,
    )

    scenario_key = st.selectbox(
        "Select Intervention Scenario",
        list(SCENARIOS.keys()),
        format_func=lambda k: f"{SCENARIOS[k]['name']} — {SCENARIOS[k]['description'][:60]}…",
    )

    if st.button("🔮  Simulate Scenario", use_container_width=True, type="primary"):
        result = simulate_intervention(df, funnel_data, scenario_key)

        if "error" in result:
            st.error(result["error"])
        else:
            scenario = result["scenario"]

            # ── Impact KPIs ────────────────────────────────────────────────
            ik1, ik2, ik3, ik4 = st.columns(4)
            with ik1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val" style="color:{COLORS["cyan"]}">+{result["additional_conversions"]}</div>'
                    f'<div class="kpi-label">Additional Conversions</div></div>', unsafe_allow_html=True)
            with ik2:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val" style="color:{COLORS["green"]}">₹{result["additional_revenue"]:,.0f}</div>'
                    f'<div class="kpi-label">Revenue Impact</div></div>', unsafe_allow_html=True)
            with ik3:
                roi_str = f'{result["roi"]}×' if isinstance(result["roi"], (int, float)) else result["roi"]
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val" style="color:{COLORS["purple"]}">{roi_str}</div>'
                    f'<div class="kpi-label">ROI</div></div>', unsafe_allow_html=True)
            with ik4:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-val" style="color:{COLORS["amber"]}">{result["n_affected_users"]}</div>'
                    f'<div class="kpi-label">Users Affected ({result["affected_pct"]}%)</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Side-by-side funnel ────────────────────────────────────────
            st.markdown('<div class="sec-head">Baseline vs. Intervention — Funnel Comparison</div>', unsafe_allow_html=True)

            fc1, fc2 = st.columns(2)
            with fc1:
                fig_base = go.Figure(go.Funnel(
                    y=result["stages"], x=result["baseline_values"],
                    textinfo="value+percent initial",
                    marker=dict(color=[COLORS["slate"]] * len(result["stages"])),
                    connector=dict(line=dict(color="#1e2a45", width=1)),
                ))
                fig_base.update_layout(**PLOTLY_LAYOUT, height=300,
                                       title=dict(text=f"Baseline ({result['baseline_conversion_rate']}% conv.)",
                                                  font=dict(size=13, color="#8899bb")))
                st.plotly_chart(fig_base, use_container_width=True)

            with fc2:
                fig_intv = go.Figure(go.Funnel(
                    y=result["stages"], x=result["intervention_values"],
                    textinfo="value+percent initial",
                    marker=dict(color=[COLORS["cyan"], COLORS["purple"], COLORS["amber"], COLORS["pink"], COLORS["green"]]),
                    connector=dict(line=dict(color="#1e2a45", width=1)),
                ))
                fig_intv.update_layout(**PLOTLY_LAYOUT, height=300,
                                       title=dict(text=f"With Intervention ({result['intervention_conversion_rate']}% conv. · +{result['conversion_lift_pp']}pp)",
                                                  font=dict(size=13, color="#4ade80")))
                st.plotly_chart(fig_intv, use_container_width=True)

            # ── LLM narrative ──────────────────────────────────────────────
            st.markdown('<div class="sec-head">🤖 AI Business Case Summary</div>', unsafe_allow_html=True)
            with st.spinner("Generating what-if narrative…"):
                from utils.llm_insights import whatif_narrative
                narrative = whatif_narrative(result)
            st.markdown(f'<div class="insight-box">{narrative}</div>', unsafe_allow_html=True)

    # ── All scenarios comparison ───────────────────────────────────────────
    st.markdown("<hr style='border-color:#1e2a45;margin:2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">All Scenario Comparison</div>', unsafe_allow_html=True)

    if st.button("📊  Compare All Scenarios", use_container_width=True):
        scenarios_df = simulate_all_scenarios(df, funnel_data)
        if len(scenarios_df) > 0:
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No scenarios could be simulated with current data.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — SEGMENTS & AI INSIGHTS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="sec-head">Behavioral Segments (KMeans, k=4)</div>', unsafe_allow_html=True)

    if len(df) > 0 and "segment_name" in df.columns:
        sc1, sc2 = st.columns([3, 2])

        with sc1:
            sample_size = min(800, len(df))
            scatter_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            fig_sc = px.scatter(
                scatter_df,
                x="total_time_in_funnel",
                y="max_stage_idx",
                color="segment_name",
                color_discrete_map=SEG_COLORS,
                size="total_events",
                opacity=0.7,
                labels={
                    "total_time_in_funnel": "Total Funnel Time (seconds)",
                    "max_stage_idx": "Deepest Stage Reached",
                    "segment_name": "Segment",
                },
            )
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=380, legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10), bgcolor="rgba(0,0,0,0)",
            ))
            st.plotly_chart(fig_sc, use_container_width=True)

        with sc2:
            seg_counts = df["segment_name"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig_d = go.Figure(go.Pie(
                labels=seg_counts["segment"],
                values=seg_counts["count"],
                hole=0.55,
                marker=dict(colors=[SEG_COLORS.get(s, COLORS["cyan"]) for s in seg_counts["segment"]]),
                textinfo="percent",
                textfont=dict(size=12),
            ))
            fig_d.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=True,
                                legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_d, use_container_width=True)

        # Segment profile table
        st.markdown('<div class="sec-head">Segment Profiles</div>', unsafe_allow_html=True)
        rows = []
        for p in sorted(seg_profiles, key=lambda x: -x["conversion_rate"]):
            rows.append({
                "Segment": p["label"],
                "Users": f"{p['size']:,}",
                "Conv. Rate": f"{p['conversion_rate']*100:.1f}%",
                "Avg Events": f"{p['avg_events']:.1f}",
                "Avg Funnel Time": f"{p['avg_time_in_funnel']:.0f}s",
                "Avg Scroll": f"{p['avg_scroll_depth']:.0f}%",
                "KYC Attempts": f"{p['avg_kyc_attempts']:.2f}",
                "Mobile %": f"{p['mobile_pct']*100:.0f}%",
                "Re-entry %": f"{p['reentry_rate']*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Radar chart
        st.markdown('<div class="sec-head">Segment Comparison — Normalised Metrics</div>', unsafe_allow_html=True)
        cats = ["Conv. Rate", "Engagement", "Scroll Depth", "KYC Friction", "Re-entry"]
        fig_r = go.Figure()
        for p in seg_profiles:
            vals = [
                p["conversion_rate"] * 100 / 50,
                p["avg_events"] / 15,
                p["avg_scroll_depth"] / 100,
                p["avg_kyc_attempts"] / 3,
                p["reentry_rate"],
            ]
            vals += [vals[0]]
            cats_closed = cats + [cats[0]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats_closed, fill="toself",
                name=p["label"],
                line=dict(color=SEG_COLORS.get(p["label"], COLORS["cyan"]), width=2),
                opacity=0.6,
            ))
        fig_r.update_layout(
            **PLOTLY_LAYOUT, height=400,
            polar=dict(
                bgcolor="#0f1629",
                radialaxis=dict(visible=True, gridcolor="#1e2a45", linecolor="#1e2a45", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="#1e2a45", linecolor="#1e2a45"),
            ),
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # AI Segment Insight
    st.markdown("<hr style='border-color:#1e2a45;margin:2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">🤖 AI Segment Intelligence — Powered by NVIDIA Llama-3.3-70B</div>', unsafe_allow_html=True)

    selected_seg = st.selectbox(
        "Select a segment to analyse",
        options=[p["label"] for p in sorted(seg_profiles, key=lambda x: -x["conversion_rate"])],
    )

    if st.button("🤖  Generate Segment Strategy", use_container_width=True):
        profile = next(p for p in seg_profiles if p["label"] == selected_seg)
        with st.spinner(f"NVIDIA Llama analysing {selected_seg}…"):
            from utils.llm_insights import segment_insight
            text = segment_insight(profile)
        st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — MODEL & ARCHITECTURE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab5:
    # ── Model Performance ──────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Model Performance</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, (k, v) in zip(
        [mc1, mc2, mc3, mc4, mc5],
        [
            ("AUC-ROC", metrics["auc"]),
            ("F1 Score", metrics["f1"]),
            ("Precision", metrics["precision"]),
            ("Recall", metrics["recall"]),
            ("CV AUC", f"{metrics.get('cv_auc_mean', 0):.3f}±{metrics.get('cv_auc_std', 0):.3f}"),
        ],
    ):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-val" style="color:{COLORS["cyan"]}">{v}</div>'
                f'<div class="kpi-label">{k}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Model Comparison ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Model Comparison — Baseline vs. Primary</div>', unsafe_allow_html=True)

    comp_df = pd.DataFrame([
        {
            "Model": comparison["baseline"]["name"],
            "AUC-ROC": comparison["baseline"]["auc"],
            "F1 Score": comparison["baseline"]["f1"],
            "Type": "Baseline",
        },
        {
            "Model": comparison["primary"]["name"],
            "AUC-ROC": comparison["primary"]["auc"],
            "F1 Score": comparison["primary"]["f1"],
            "Type": "Primary (Calibrated)",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="insight-box" style="border-left-color:{COLORS["green"]}">'
        f'📈 <strong style="color:#4ade80">+{comparison["improvement_auc"]}% AUC improvement</strong> over baseline | '
        f'<strong style="color:#4ade80">+{comparison["improvement_f1"]}% F1 improvement</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Health Score Breakdown ─────────────────────────────────────────────
    st.markdown('<div class="sec-head">Funnel Health Score Breakdown</div>', unsafe_allow_html=True)

    hc1, hc2, hc3, hc4 = st.columns(4)
    health_components = [
        ("Conv. Efficiency", health["conversion_efficiency"], "40%", COLORS["green"]),
        ("Flow Smoothness", health["flow_smoothness"], "25%", COLORS["amber"]),
        ("Model Confidence", health["model_confidence"], "20%", COLORS["purple"]),
        ("Engagement Depth", health["engagement_depth"], "15%", COLORS["cyan"]),
    ]
    for col, (name, score, weight, color) in zip([hc1, hc2, hc3, hc4], health_components):
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-val" style="color:{color}">{score:.0f}</div>'
                f'<div class="kpi-label">{name} (wt: {weight})</div></div>',
                unsafe_allow_html=True,
            )

    # ── Architecture Card ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">System Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="insight-box">
<strong style="color:#cdd9f5">Architecture</strong><br><br>
• <strong>Conversion Model</strong>: {metrics.get('model_type', 'LightGBM')} ({metrics.get('n_features', 30)} features, calibrated with Platt scaling)<br>
• <strong>Drop-off Model</strong>: XGBoost multiclass (F1: {metrics.get('drop_f1', 0):.3f})<br>
• <strong>Segmentation</strong>: KMeans k=4 on StandardScaled behavioral features<br>
• <strong>Decision Engine</strong>: Hybrid ML + 8 business rules with ROI-based prioritization<br>
• <strong>Simulator</strong>: Counterfactual what-if engine with 6 intervention scenarios<br>
• <strong>Explainability</strong>: SHAP TreeExplainer for per-prediction attribution<br>
• <strong>Feature Engineering</strong>: 30+ features (behavioral, temporal, friction, interaction)<br>
• <strong>Validation</strong>: 5-fold stratified CV (AUC: {metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f})<br>
• <strong>Training set</strong>: {metrics['train_size']:,} records &nbsp;|&nbsp; <strong>Test set</strong>: {metrics['test_size']:,} records<br>
• <strong>LLM</strong>: NVIDIA Llama-3.3-70B (structured intervention generation)
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Executive Summary ──────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#1e2a45;margin:2rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">📋 Executive Portfolio Summary</div>', unsafe_allow_html=True)
    if st.button("📋  Generate Executive Summary", use_container_width=True):
        with st.spinner("Generating executive summary…"):
            from utils.llm_insights import portfolio_summary
            summary = portfolio_summary(metrics, health, len(df), comparison)
        st.markdown(f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 6 — INTERVENTION PRIORITY QUEUE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab6:
    from engine.decision_engine import batch_evaluate, build_priority_queue

    st.markdown('<div class="sec-head">🚨 Real-Time Intervention Priority Queue</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#5a6a8a;font-size:.82rem;margin-bottom:1rem">'
        'The decision engine has evaluated every user in the dataset. '
        'Below are the top 50 users ranked by intervention ROI — the ones where acting now '
        'generates the highest expected revenue recovery per rupee spent.'
        '</p>',
        unsafe_allow_html=True,
    )

    if len(df) > 0:
        pq_btn = st.button("🚨  Build Priority Queue", use_container_width=True, type="primary")

        if pq_btn:
            with st.spinner("⚡ Running decision engine on all users…"):
                # Run batch evaluation
                conv_proba = cal_model.predict_proba(df[FEAT_COLS].fillna(0))[:, 1]

                drop_stages = None
                if drop_model is not None:
                    try:
                        drop_stages = drop_model.predict(df[FEAT_COLS].fillna(0))
                    except Exception:
                        pass

                seg_names = df.get("segment_name", pd.Series(["Unknown"] * len(df)))
                if "segment_name" not in df.columns:
                    seg_names = pd.Series(["Unknown"] * len(df), index=df.index)

                plans = batch_evaluate(
                    df[FEAT_COLS].fillna(0),
                    conv_proba,
                    drop_stages,
                    seg_names,
                )

            queue_df = build_priority_queue(plans, top_n=50)

            # ── Summary KPIs ──────────────────────────────────────────────────────────
            high_risk = sum(1 for p in plans if p.risk_level == "high")
            med_risk  = sum(1 for p in plans if p.risk_level == "medium")
            total_rev_at_risk = sum(p.estimated_revenue_at_risk for p in plans)
            actionable = sum(1 for p in plans if p.interventions)

            pq1, pq2, pq3, pq4 = st.columns(4)
            for col, (val, label, color) in zip(
                [pq1, pq2, pq3, pq4],
                [
                    (f"{high_risk}",        "High-Risk Users",       COLORS["red"]),
                    (f"{med_risk}",          "Medium-Risk Users",     COLORS["amber"]),
                    (f"₹{total_rev_at_risk:,.0f}", "Total Revenue at Risk", COLORS["purple"]),
                    (f"{actionable}",        "Users with Actions",    COLORS["cyan"]),
                ],
            ):
                with col:
                    st.markdown(
                        f'<div class="kpi-card"><div class="kpi-val" style="color:{color}">{val}</div>'
                        f'<div class="kpi-label">{label}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Revenue-at-Risk Waterfall (top 20 users) ────────────────────────
            st.markdown('<div class="sec-head">💰 Revenue at Risk — Top 20 Users by Priority Score</div>', unsafe_allow_html=True)

            top_plans = sorted(
                [p for p in plans if p.interventions and p.risk_level in ("high", "medium")],
                key=lambda x: -(x.interventions[0].priority_score if x.interventions else 0),
            )[:20]

            if top_plans:
                wf_users  = [p.user_id for p in top_plans]
                wf_rev    = [p.estimated_revenue_at_risk for p in top_plans]
                wf_risk   = [p.risk_level for p in top_plans]
                wf_colors = [COLORS["red"] if r == "high" else COLORS["amber"] for r in wf_risk]

                fig_wf = go.Figure(go.Bar(
                    x=wf_users,
                    y=wf_rev,
                    marker_color=wf_colors,
                    text=[f"₹{v:,.0f}" for v in wf_rev],
                    textposition="outside",
                    textfont=dict(size=10),
                ))
                fig_wf.update_layout(
                    **PLOTLY_LAYOUT, height=320,
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="#1e2a45"),
                    showlegend=False,
                )
                st.plotly_chart(fig_wf, use_container_width=True)

            # ── Risk Distribution Pie ─────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            pq_c1, pq_c2 = st.columns([2, 1])

            with pq_c1:
                st.markdown('<div class="sec-head">📊 Ranked Intervention Queue (Top 50)</div>', unsafe_allow_html=True)
                if len(queue_df) > 0:
                    # Color-code risk level column
                    st.dataframe(
                        queue_df,
                        use_container_width=True,
                        column_config={
                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                            "conversion_prob": st.column_config.TextColumn("Conv. Prob", width="small"),
                            "priority_score": st.column_config.NumberColumn(
                                "Priority Score", format="%.0f", width="small"
                            ),
                            "expected_lift": st.column_config.TextColumn("Est. Lift", width="small"),
                            "revenue_at_risk": st.column_config.TextColumn("Rev. at Risk", width="medium"),
                            "top_action": st.column_config.TextColumn("Recommended Action", width="large"),
                        },
                    )
                else:
                    st.info("No high/medium risk users with interventions found.")

            with pq_c2:
                st.markdown('<div class="sec-head">Risk Distribution</div>', unsafe_allow_html=True)
                risk_counts = {"High": high_risk, "Medium": med_risk,
                               "Low": len(plans) - high_risk - med_risk}
                fig_risk = go.Figure(go.Pie(
                    labels=list(risk_counts.keys()),
                    values=list(risk_counts.values()),
                    hole=0.55,
                    marker=dict(colors=[COLORS["red"], COLORS["amber"], COLORS["green"]]),
                    textinfo="percent+label",
                    textfont=dict(size=11),
                ))
                fig_risk.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig_risk, use_container_width=True)

                # Intervention type breakdown
                st.markdown('<div class="sec-head">Intervention Types</div>', unsafe_allow_html=True)
                intv_counts: dict = {}
                for p in plans:
                    for iv in p.interventions[:1]:  # top intervention only
                        intv_counts[iv.intervention_type] = intv_counts.get(iv.intervention_type, 0) + 1
                if intv_counts:
                    fig_intv_types = go.Figure(go.Bar(
                        x=list(intv_counts.keys()),
                        y=list(intv_counts.values()),
                        marker_color=[COLORS["cyan"], COLORS["purple"], COLORS["amber"],
                                      COLORS["green"], COLORS["pink"]],
                    ))
                    fig_intv_types.update_layout(**PLOTLY_LAYOUT, height=230, showlegend=False)
                    st.plotly_chart(fig_intv_types, use_container_width=True)

        else:
            st.markdown(
                '<div class="insight-box" style="color:#3a4a6a;text-align:center;padding:40px">'
                'Click <strong style="color:#00c2ff">Build Priority Queue</strong> to run '
                'the decision engine across all users and generate the ranked intervention list'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠ No feature data available. Run setup.py first to generate and train.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#2a3a55;font-size:.72rem;margin-top:2.5rem;padding-top:1rem;"
    "border-top:1px solid #1e2a45'>"
    "FD Funnel Intelligence Engine &nbsp;·&nbsp; Blostem Hackathon 2026 &nbsp;·&nbsp; Track 03: Data Analytics & Insights"
    "</div>",
    unsafe_allow_html=True,
)