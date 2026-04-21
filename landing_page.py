"""
Landing page for FD Funnel Intelligence Engine.
Call render_landing_page() from app.py — returns True when user clicks Enter.
"""
from __future__ import annotations
import streamlit as st


_STATS = [
    ("0.9435", "AUC-ROC",        "Calibrated LightGBM"),
    ("0.7396", "F1 Score",       "Optimal threshold 0.347"),
    ("0.9412", "5-Fold CV AUC",  "±0.0106 std — stable"),
    ("21.1%",  "Funnel Conv.",   "2,000 users · 7,955 events"),
]

_MODULES = [
    ("📊", "Funnel Analytics",    "Real-time event-driven funnel with stage-by-stage drop-off rates and SHAP-ranked conversion drivers."),
    ("⚡", "Prediction Engine",   "Per-user conversion probability from calibrated LightGBM + 8 ROI-ranked business rule interventions."),
    ("🔮", "What-If Simulator",   "Counterfactual impact modelling across 6 intervention scenarios with revenue and ROI quantification."),
    ("👥", "Segment Intelligence","KMeans k=4 behavioural clustering + NVIDIA Llama-3.3-70B plain-English segment briefings."),
    ("🎯", "Priority Queue",      "Batch decision engine ranking every user by net intervention ROI — highest revenue recovery first."),
    ("🏗️", "Model Architecture",  "Full model card: LightGBM + XGBoost drop-off + SHAP explainability + Platt-scaled calibration."),
]

_DIFF = [
    ("Generic BI dashboards",  "Show what happened",            "We show what will happen and what to do"),
    ("Rule-based CRM systems", "Static segments, fixed rules",  "ML-driven segments + ROI-ranked dynamic rules"),
    ("Data science notebooks", "Offline, non-interactive",      "Live inference + real-time decision engine"),
    ("A/B testing platforms",  "Test after the fact",           "Counterfactual simulation before deployment"),
]

_STACK = [
    ("LightGBM", "Conversion model (400 estimators, Platt calibrated)"),
    ("XGBoost",  "Drop-off stage predictor (multiclass, weighted F1 0.7290)"),
    ("KMeans",   "4-cluster behavioural segmentation on 10 scaled features"),
    ("SHAP",     "TreeExplainer — per-prediction attribution, global importance"),
    ("NVIDIA Llama-3.3-70B", "Structured LLM insights via NVIDIA NIM API"),
    ("Streamlit + Plotly",   "Interactive dashboard, 6 tabs, real-time charts"),
]


def render_landing_page() -> bool:
    """Render the landing page. Returns True when the user clicks Enter App."""

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.main{background:#04080f;}
.block-container{padding:0!important;max-width:100%!important;}

/* ── hero ── */
.lp-hero{
    background:radial-gradient(ellipse 80% 60% at 50% 0%,#001a2e 0%,#04080f 70%);
    padding:80px 60px 60px;
    border-bottom:1px solid #0d2137;
    position:relative;overflow:hidden;
}
.lp-hero::before{
    content:'';position:absolute;inset:0;
    background:repeating-linear-gradient(0deg,transparent,transparent 59px,#0d213720 60px),
               repeating-linear-gradient(90deg,transparent,transparent 59px,#0d213720 60px);
    pointer-events:none;
}
.lp-tag{font-size:.65rem;color:#00c2ff;text-transform:uppercase;letter-spacing:.22em;margin-bottom:16px;}
.lp-title{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6vw,5.5rem);
    color:#e8f4ff;line-height:1;letter-spacing:.02em;margin:0 0 16px 0;}
.lp-title span{color:#00c2ff;}
.lp-sub{font-size:1.05rem;color:#6a85a5;max-width:640px;line-height:1.7;margin-bottom:32px;}

/* ── stat strip ── */
.stat-strip{display:flex;gap:40px;flex-wrap:wrap;margin-top:40px;}
.stat-item{border-left:2px solid #00c2ff;padding-left:16px;}
.stat-num{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;color:#e8f4ff;letter-spacing:.02em;}
.stat-lbl{font-size:.72rem;color:#4a6a8a;text-transform:uppercase;letter-spacing:.1em;}
.stat-sub{font-size:.68rem;color:#2a4a6a;margin-top:2px;}

/* ── sections ── */
.lp-section{padding:64px 60px;border-bottom:1px solid #0d2137;}
.lp-section-alt{background:#060c16;}
.sec-label{font-size:.62rem;color:#00c2ff;text-transform:uppercase;letter-spacing:.22em;margin-bottom:10px;}
.sec-h2{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;color:#e8f4ff;letter-spacing:.04em;margin:0 0 8px 0;}
.sec-desc{font-size:.92rem;color:#5a7a9a;line-height:1.7;max-width:680px;}

/* ── module cards ── */
.mod-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:32px;}
.mod-card{background:#060c16;border:1px solid #0d2137;border-radius:12px;padding:22px;
    transition:border-color .2s,transform .2s;}
.mod-card:hover{border-color:#00c2ff30;transform:translateY(-3px);}
.mod-icon{font-size:1.4rem;margin-bottom:10px;}
.mod-title{font-size:.85rem;font-weight:600;color:#cde0f5;margin-bottom:6px;}
.mod-desc{font-size:.78rem;color:#4a6a8a;line-height:1.6;}

/* ── stack pills ── */
.stack-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:24px;}
.stack-pill{background:#060c16;border:1px solid #0d2137;border-radius:8px;
    padding:12px 18px;min-width:220px;flex:1;}
.pill-name{font-family:'DM Mono',monospace;font-size:.78rem;color:#00c2ff;margin-bottom:4px;}
.pill-desc{font-size:.72rem;color:#3a5a7a;}

/* ── diff table ── */
.diff-table{width:100%;border-collapse:collapse;margin-top:24px;}
.diff-table th{font-size:.65rem;color:#2a4a6a;text-transform:uppercase;letter-spacing:.14em;
    padding:10px 16px;border-bottom:1px solid #0d2137;text-align:left;}
.diff-table td{padding:14px 16px;border-bottom:1px solid #06101a;font-size:.82rem;vertical-align:top;}
.diff-table tr:hover td{background:#060c1660;}
.td-comp{color:#4a6a8a;}
.td-them{color:#4a6a8a;}
.td-us{color:#4ade80;font-weight:500;}

/* ── score card ── */
.score-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:24px;}
.score-card{background:#060c16;border:1px solid #0d2137;border-radius:12px;padding:24px;text-align:center;}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:#00c2ff;letter-spacing:.04em;}
.score-label{font-size:.7rem;color:#4a6a8a;text-transform:uppercase;letter-spacing:.12em;margin-top:4px;}
.score-note{font-size:.68rem;color:#2a4a6a;margin-top:6px;font-family:'DM Mono',monospace;}

/* ── cta ── */
.lp-cta{padding:72px 60px;text-align:center;
    background:radial-gradient(ellipse 60% 80% at 50% 100%,#001a2e 0%,#04080f 70%);}
.cta-h{font-family:'Bebas Neue',sans-serif;font-size:3rem;color:#e8f4ff;letter-spacing:.04em;margin-bottom:12px;}
.cta-sub{font-size:.92rem;color:#4a6a8a;margin-bottom:36px;}

/* hide streamlit chrome on landing */
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-hero">
  <div class="lp-tag">Blostem AI Builder Hackathon 2026 &nbsp;·&nbsp; Track 03: Data Analytics & Insights</div>
  <h1 class="lp-title">FD Funnel<br><span>Intelligence</span><br>Engine</h1>
  <p class="lp-sub">
    A production-grade decision engine that predicts which users will abandon your Fixed Deposit
    booking funnel, explains exactly why, and tells your team precisely what to do — ranked by
    revenue impact per rupee spent.
  </p>
  <div class="stat-strip">
    <div class="stat-item">
      <div class="stat-num">0.9435</div>
      <div class="stat-lbl">AUC-ROC</div>
      <div class="stat-sub">Calibrated LightGBM</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">0.7396</div>
      <div class="stat-lbl">F1 Score</div>
      <div class="stat-sub">Threshold-optimised</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">21.1%</div>
      <div class="stat-lbl">Funnel Conversion</div>
      <div class="stat-sub">2,000 users · 7,955 events</div>
    </div>
    <div class="stat-item">
      <div class="stat-num">₹50K+</div>
      <div class="stat-lbl">Prize Pool</div>
      <div class="stat-sub">Built solo · 3 weeks</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── What It Does ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-section">
  <div class="sec-label">What It Does</div>
  <h2 class="sec-h2">Six systems. One pipeline.</h2>
  <p class="sec-desc">
    From raw funnel events to a ranked list of users to contact today — entirely automated.
  </p>
  <div class="mod-grid">
    <div class="mod-card"><div class="mod-icon">📊</div>
      <div class="mod-title">Funnel Analytics</div>
      <div class="mod-desc">Real-time event-driven funnel with stage-by-stage drop-off rates and SHAP-ranked conversion drivers.</div>
    </div>
    <div class="mod-card"><div class="mod-icon">⚡</div>
      <div class="mod-title">Prediction Engine</div>
      <div class="mod-desc">Per-user conversion probability from calibrated LightGBM + 8 ROI-ranked business rule interventions.</div>
    </div>
    <div class="mod-card"><div class="mod-icon">🔮</div>
      <div class="mod-title">What-If Simulator</div>
      <div class="mod-desc">Counterfactual impact across 6 intervention scenarios with revenue and ROI quantified before you ship.</div>
    </div>
    <div class="mod-card"><div class="mod-icon">👥</div>
      <div class="mod-title">Segment Intelligence</div>
      <div class="mod-desc">KMeans k=4 behavioural clustering + Llama-3.3-70B plain-English briefings per segment.</div>
    </div>
    <div class="mod-card"><div class="mod-icon">🎯</div>
      <div class="mod-title">Priority Queue</div>
      <div class="mod-desc">Batch decision engine ranking every user by net intervention ROI — highest revenue recovery first.</div>
    </div>
    <div class="mod-card"><div class="mod-icon">🏗️</div>
      <div class="mod-title">Model Architecture</div>
      <div class="mod-desc">Full model card with LightGBM + XGBoost + SHAP + Platt calibration + 5-fold CV validation.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Model Scores ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-section lp-section-alt">
  <div class="sec-label">Model Performance</div>
  <h2 class="sec-h2">Numbers that matter.</h2>
  <p class="sec-desc">
    Trained on 2,000 synthetic users modelled on real Blostem FD funnel patterns.
    5-fold stratified cross-validation confirms generalisation, not overfitting.
  </p>
  <div class="score-grid">
    <div class="score-card">
      <div class="score-num">0.9435</div>
      <div class="score-label">AUC-ROC (calibrated)</div>
      <div class="score-note">raw: 0.9377 · CV: 0.9412 ± 0.0106</div>
    </div>
    <div class="score-card">
      <div class="score-num">0.7432</div>
      <div class="score-label">F1 Score</div>
      <div class="score-note">threshold: 0.334 · scale_pos_weight balanced</div>
    </div>
    <div class="score-card">
      <div class="score-num">0.7290</div>
      <div class="score-label">Drop-off Stage F1</div>
      <div class="score-note">XGBoost multiclass · weighted average</div>
    </div>
    <div class="score-card">
      <div class="score-num">4</div>
      <div class="score-label">Behavioural Segments</div>
      <div class="score-note">63.3% → 0.0% conversion spread across clusters</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── How We Built It ───────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-section">
  <div class="sec-label">How We Built It</div>
  <h2 class="sec-h2">The full stack.</h2>
  <p class="sec-desc">
    Every component chosen for a reason. No copy-pasted templates —
    each module reviewed, debugged, and tested against edge cases.
  </p>
  <div class="stack-grid">
    <div class="stack-pill">
      <div class="pill-name">LightGBM</div>
      <div class="pill-desc">Conversion model — 400 estimators, Platt-calibrated, scale_pos_weight for class imbalance</div>
    </div>
    <div class="stack-pill">
      <div class="pill-name">XGBoost</div>
      <div class="pill-desc">Drop-off stage predictor — multiclass, weighted F1 0.7290 on held-out test set</div>
    </div>
    <div class="stack-pill">
      <div class="pill-name">SHAP TreeExplainer</div>
      <div class="pill-desc">Per-prediction attribution and global feature importance — not just "what" but "why"</div>
    </div>
    <div class="stack-pill">
      <div class="pill-name">KMeans k=4</div>
      <div class="pill-desc">Behavioural segmentation on 10 StandardScaled features — 4 distinct funnel personas</div>
    </div>
    <div class="stack-pill">
      <div class="pill-name">NVIDIA Llama-3.3-70B</div>
      <div class="pill-desc">Structured LLM insights via NIM API — segment briefs, intervention narratives, exec summaries</div>
    </div>
    <div class="stack-pill">
      <div class="pill-name">30+ Features</div>
      <div class="pill-desc">Behavioral · temporal · KYC friction · device switching · interaction terms — all pre-outcome</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Differentiation ───────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-section lp-section-alt">
  <div class="sec-label">Why This Is Different</div>
  <h2 class="sec-h2">Not another dashboard.</h2>
  <p class="sec-desc">
    Every analytics tool tells you what happened. This one tells you what will happen,
    which users to act on, and exactly what to say to them — in rupees.
  </p>
  <table class="diff-table">
    <tr>
      <th>Existing Tool</th>
      <th>What They Do</th>
      <th>What We Do Instead</th>
    </tr>
    <tr>
      <td class="td-comp">Generic BI dashboards (Metabase, Redash)</td>
      <td class="td-them">Show what happened last quarter</td>
      <td class="td-us">Predict who drops off today, before it happens</td>
    </tr>
    <tr>
      <td class="td-comp">Rule-based CRM (Salesforce, Zoho)</td>
      <td class="td-them">Static segments, manually defined rules</td>
      <td class="td-us">ML-driven segments + ROI-ranked dynamic rule engine</td>
    </tr>
    <tr>
      <td class="td-comp">Data science notebooks</td>
      <td class="td-them">Offline analysis, PDF reports, no live inference</td>
      <td class="td-us">Live single-user prediction with SHAP waterfall in &lt;1s</td>
    </tr>
    <tr>
      <td class="td-comp">A/B testing platforms</td>
      <td class="td-them">Test interventions after shipping them</td>
      <td class="td-us">Counterfactual simulation with revenue impact before you build</td>
    </tr>
    <tr>
      <td class="td-comp">Generic churn tools</td>
      <td class="td-them">Binary churn score, no action guidance</td>
      <td class="td-us">Ranked intervention queue — who to call, what to say, ROI per ₹ spent</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

    # ── Purpose ───────────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-section">
  <div class="sec-label">The Purpose</div>
  <h2 class="sec-h2">Built for Blostem's real problem.</h2>
  <p class="sec-desc" style="max-width:760px">
    Blostem's infrastructure connects 30+ fintech platforms to 10+ banks. Every FD booking
    that drops at KYC or Details is direct revenue lost — not just for Blostem, but for every
    partner platform on the network. This engine gives Blostem's product and growth teams a
    single interface to understand the funnel, predict failures, simulate fixes, and deploy
    targeted interventions — with the expected ₹ impact quantified before a single line of
    product code is written.
  </p>
  <div style="margin-top:28px;display:flex;gap:32px;flex-wrap:wrap;">
    <div style="border-left:2px solid #00c2ff;padding-left:16px;">
      <div style="font-size:.68rem;color:#4a6a8a;text-transform:uppercase;letter-spacing:.1em">Problem</div>
      <div style="color:#cde0f5;font-size:.88rem;margin-top:4px;">FD funnel drop-off is invisible until it's too late</div>
    </div>
    <div style="border-left:2px solid #4ade80;padding-left:16px;">
      <div style="font-size:.68rem;color:#4a6a8a;text-transform:uppercase;letter-spacing:.1em">Solution</div>
      <div style="color:#cde0f5;font-size:.88rem;margin-top:4px;">Predict, explain, intervene — all in one decision engine</div>
    </div>
    <div style="border-left:2px solid #a78bfa;padding-left:16px;">
      <div style="font-size:.68rem;color:#4a6a8a;text-transform:uppercase;letter-spacing:.1em">Output</div>
      <div style="color:#cde0f5;font-size:.88rem;margin-top:4px;">Ranked user list with ₹ ROI per intervention action</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-cta">
  <div class="cta-h">Ready to see it live?</div>
  <div class="cta-sub">
    AUC 0.9435 · F1 0.74 · 30+ features · 6 tabs · real-time inference
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        clicked = st.button(
            "⚡  Enter the Decision Engine →",
            use_container_width=True,
            type="primary",
        )
    st.markdown("<br>", unsafe_allow_html=True)
    return clicked