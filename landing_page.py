from __future__ import annotations
import streamlit as st

def render_landing_page() -> bool:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.main{background:#03070f;}
.block-container{padding:0!important;max-width:100%!important;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}

/* ── hero ── */
.hero{
    min-height:92vh;
    background:#03070f;
    position:relative;overflow:hidden;
    padding:80px 72px 64px;
    display:flex;flex-direction:column;justify-content:center;
}
.hero-bg{
    position:absolute;inset:0;
    background:
        radial-gradient(ellipse 55% 70% at 80% 20%, #00c2ff08 0%, transparent 60%),
        radial-gradient(ellipse 40% 50% at 20% 80%, #4ade8006 0%, transparent 60%),
        radial-gradient(ellipse 30% 40% at 60% 60%, #a78bfa05 0%, transparent 50%);
    pointer-events:none;
}
.hero-grid{
    position:absolute;inset:0;
    background-image:
        linear-gradient(#0d213720 1px, transparent 1px),
        linear-gradient(90deg, #0d213720 1px, transparent 1px);
    background-size:64px 64px;
    mask-image:radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
    pointer-events:none;
}
.hero-content{position:relative;z-index:1;}
.hero-eyebrow{
    display:inline-flex;align-items:center;gap:8px;
    background:#00c2ff10;border:1px solid #00c2ff25;
    border-radius:100px;padding:5px 14px 5px 8px;
    font-size:.65rem;color:#00c2ff;text-transform:uppercase;
    letter-spacing:.18em;margin-bottom:28px;width:fit-content;
}
.eyebrow-dot{width:6px;height:6px;border-radius:50%;background:#00c2ff;
    box-shadow:0 0 8px #00c2ff;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.hero-title{
    font-family:'Bebas Neue',sans-serif;
    font-size:clamp(3.6rem,6.5vw,6.2rem);
    line-height:.95;letter-spacing:.03em;
    color:#e8f4ff;margin:0 0 8px 0;
}
.hero-title .accent{
    background:linear-gradient(135deg,#00c2ff,#4ade80 60%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
}
.hero-sub{
    font-size:1.05rem;color:#4a6a8a;
    max-width:580px;line-height:1.75;margin:20px 0 40px;
    font-weight:300;
}

/* ── stat bar ── */
.stat-bar{
    display:flex;gap:0;
    border:1px solid #0d2137;border-radius:14px;
    overflow:hidden;background:#060c16;
    width:fit-content;margin-top:8px;
}
.stat-item{padding:20px 32px;border-right:1px solid #0d2137;position:relative;}
.stat-item:last-child{border-right:none;}
.stat-item::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--c),transparent);
    opacity:.6;
}
.stat-num{font-family:'Bebas Neue',sans-serif;font-size:2.2rem;
    color:#e8f4ff;letter-spacing:.04em;line-height:1;}
.stat-lbl{font-size:.62rem;color:#2a4a6a;text-transform:uppercase;
    letter-spacing:.14em;margin-top:4px;}
.stat-sub{font-family:'JetBrains Mono',monospace;font-size:.6rem;
    color:#1a3a5a;margin-top:3px;}

/* ── section ── */
.section{padding:72px 72px;border-top:1px solid #080f1a;}
.section-alt{background:#050b14;}
.s-label{
    font-family:'JetBrains Mono',monospace;
    font-size:.6rem;color:#00c2ff;text-transform:uppercase;
    letter-spacing:.22em;margin-bottom:10px;
}
.s-h2{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;
    color:#e8f4ff;letter-spacing:.04em;margin:0 0 6px;}
.s-desc{font-size:.9rem;color:#3a5a7a;line-height:1.75;max-width:620px;font-weight:300;}

/* ── module cards ── */
.mod-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
    background:#0d2137;border-radius:16px;overflow:hidden;margin-top:36px;}
.mod-card{
    background:#060c16;padding:28px 26px;
    transition:background .2s;cursor:default;position:relative;overflow:hidden;
}
.mod-card::after{
    content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--accent,#00c2ff),transparent);
    opacity:0;transition:opacity .3s;
}
.mod-card:hover{background:#080f1a;}
.mod-card:hover::after{opacity:1;}
.mod-num{font-family:'JetBrains Mono',monospace;font-size:.6rem;
    color:#0d2137;margin-bottom:14px;}
.mod-icon{font-size:1.6rem;margin-bottom:12px;display:block;}
.mod-title{font-size:.9rem;font-weight:600;color:#cde0f5;margin-bottom:8px;}
.mod-desc{font-size:.78rem;color:#2a4a6a;line-height:1.65;}

/* ── score grid ── */
.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:32px;}
.score-card{
    background:#060c16;border:1px solid #0d2137;
    border-radius:14px;padding:28px 20px;text-align:center;
    position:relative;overflow:hidden;
}
.score-card::before{
    content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);
    width:60%;height:1px;background:var(--c,#00c2ff);opacity:.4;
}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:2.8rem;
    letter-spacing:.04em;line-height:1;}
.score-label{font-size:.65rem;color:#2a4a6a;text-transform:uppercase;
    letter-spacing:.12em;margin-top:8px;}
.score-note{font-family:'JetBrains Mono',monospace;font-size:.6rem;
    color:#162a3a;margin-top:8px;}

/* ── stack ── */
.stack-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px;}
.stack-pill{
    background:#060c16;border:1px solid #0d2137;
    border-radius:10px;padding:18px 20px;
    border-left:3px solid var(--c,#00c2ff);
    transition:transform .2s,border-color .2s;
}
.stack-pill:hover{transform:translateX(4px);}
.pill-name{font-family:'JetBrains Mono',monospace;font-size:.75rem;
    color:var(--c,#00c2ff);margin-bottom:5px;font-weight:500;}
.pill-desc{font-size:.74rem;color:#2a4a6a;line-height:1.55;}

/* ── diff table ── */
.diff-wrap{margin-top:28px;border:1px solid #0d2137;border-radius:14px;overflow:hidden;}
.diff-table{width:100%;border-collapse:collapse;}
.diff-table th{
    font-family:'JetBrains Mono',monospace;
    font-size:.58rem;color:#1a3a5a;text-transform:uppercase;
    letter-spacing:.14em;padding:14px 20px;
    background:#040810;border-bottom:1px solid #0d2137;text-align:left;
}
.diff-table td{padding:16px 20px;border-bottom:1px solid #060c16;
    font-size:.8rem;vertical-align:top;}
.diff-table tr:last-child td{border-bottom:none;}
.diff-table tr:hover td{background:#040810;}
.td-comp{color:#1a3a5a;font-weight:500;}
.td-them{color:#1e3a52;}
.td-us{color:#4ade80;font-weight:500;}
.td-arrow{color:#0d2137;font-size:.7rem;padding:16px 8px!important;}

/* ── purpose strip ── */
.purpose-strip{
    display:grid;grid-template-columns:repeat(3,1fr);
    gap:1px;background:#0d2137;
    border-radius:14px;overflow:hidden;margin-top:32px;
}
.purpose-item{background:#060c16;padding:28px 24px;}
.purpose-kicker{font-family:'JetBrains Mono',monospace;
    font-size:.58rem;color:var(--c,#00c2ff);text-transform:uppercase;
    letter-spacing:.18em;margin-bottom:10px;}
.purpose-text{font-size:.84rem;color:#3a6a8a;line-height:1.65;}

/* ── cta ── */
.cta-section{
    padding:80px 72px;text-align:center;
    background:radial-gradient(ellipse 50% 80% at 50% 100%,#00152200 0%,#03070f 60%);
    border-top:1px solid #080f1a;position:relative;
}
.cta-section::before{
    content:'';position:absolute;left:50%;top:0;transform:translateX(-50%);
    width:1px;height:60px;
    background:linear-gradient(to bottom,#00c2ff,transparent);
}
.cta-h{font-family:'Bebas Neue',sans-serif;font-size:3.2rem;
    color:#e8f4ff;letter-spacing:.04em;margin-bottom:12px;}
.cta-sub{font-size:.88rem;color:#2a4a6a;margin-bottom:40px;
    font-family:'JetBrains Mono',monospace;}
</style>
""", unsafe_allow_html=True)

    #  HERO 
    st.markdown("""
<div class="hero">
  <div class="hero-bg"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">

    <div class="hero-eyebrow">
      <span class="eyebrow-dot"></span>
      Blostem AI Builder Hackathon 2026 &nbsp;·&nbsp; Track 03 &nbsp;·&nbsp; Solo Build
    </div>

    <h1 class="hero-title">
      FD Funnel<br>
      <span class="accent">Intelligence</span><br>
      Engine
    </h1>

    <p class="hero-sub">
      A production-grade ML decision engine that predicts Fixed Deposit funnel
      drop-offs, explains every prediction with SHAP, and delivers a ranked
      intervention queue — sorted by revenue recovered per ₹ spent.
    </p>

    <div class="stat-bar">
      <div class="stat-item" style="--c:#00c2ff">
        <div class="stat-num" style="color:#00c2ff">0.9425</div>
        <div class="stat-lbl">AUC-ROC</div>
        <div class="stat-sub">Calibrated LightGBM · CV: 0.9420±0.0100</div>
      </div>
      <div class="stat-item" style="--c:#4ade80">
        <div class="stat-num" style="color:#4ade80">0.7432</div>
        <div class="stat-lbl">F1 Score</div>
        <div class="stat-sub">Threshold-optimised · 0.334</div>
      </div>
      <div class="stat-item" style="--c:#a78bfa">
        <div class="stat-num" style="color:#a78bfa">0.7290</div>
        <div class="stat-lbl">Drop-off F1</div>
        <div class="stat-sub">17 leak-free features · XGBoost</div>
      </div>
      <div class="stat-item" style="--c:#fbbf24">
        <div class="stat-num" style="color:#fbbf24">21.1%</div>
        <div class="stat-lbl">Funnel Conv.</div>
        <div class="stat-sub">2,000 users · 7,955 events</div>
      </div>
    </div>

  </div>
</div>
""", unsafe_allow_html=True)

    #  WHAT IT DOES 
    st.markdown("""
<div class="section">
  <div class="s-label">What It Does</div>
  <h2 class="s-h2">Six systems. One pipeline.</h2>
  <p class="s-desc">From raw event logs to a ranked list of users to contact today — end to end, no manual steps.</p>

  <div class="mod-grid">
    <div class="mod-card" style="--accent:#00c2ff">
      <div class="mod-num">01</div>
      <span class="mod-icon"></span>
      <div class="mod-title">Funnel Analytics</div>
      <div class="mod-desc">Event-driven funnel with stage-by-stage drop-off rates, SHAP-ranked drivers, and device/tier breakdowns.</div>
    </div>
    <div class="mod-card" style="--accent:#a78bfa">
      <div class="mod-num">02</div>
      <span class="mod-icon"></span>
      <div class="mod-title">Prediction Engine</div>
      <div class="mod-desc">Per-user conversion probability from calibrated LightGBM + 8 business rules ranked by net ROI in ₹.</div>
    </div>
    <div class="mod-card" style="--accent:#fbbf24">
      <div class="mod-num">03</div>
      <span class="mod-icon"></span>
      <div class="mod-title">What-If Simulator</div>
      <div class="mod-desc">Counterfactual impact across 6 intervention scenarios with revenue and ROI before you ship a single feature.</div>
    </div>
    <div class="mod-card" style="--accent:#4ade80">
      <div class="mod-num">04</div>
      <span class="mod-icon"></span>
      <div class="mod-title">Segment Intelligence</div>
      <div class="mod-desc">KMeans k=4 behavioural clustering with 0–63% conversion spread + NVIDIA Llama-3.3-70B plain-English briefs.</div>
    </div>
    <div class="mod-card" style="--accent:#f472b6">
      <div class="mod-num">05</div>
      <span class="mod-icon"></span>
      <div class="mod-title">Priority Queue</div>
      <div class="mod-desc">Batch decision engine ranking every user by net intervention ROI — highest revenue recovery per ₹ spent first.</div>
    </div>
    <div class="mod-card" style="--accent:#00c2ff">
      <div class="mod-num">06</div>
      <span class="mod-icon"></span>
      <div class="mod-title">Model Architecture</div>
      <div class="mod-desc">Full model card: LightGBM + XGBoost drop-off + SHAP explainability + Platt calibration + 5-fold CV.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    #  MODEL SCORES 
    st.markdown("""
<div class="section section-alt">
  <div class="s-label">Live Training Results</div>
  <h2 class="s-h2">Numbers from the actual run.</h2>
  <p class="s-desc">
    Trained on 2,000 synthetic users modelled on real FD funnel behaviour.
    Drop-off F1 uses 17 leak-free features — no data leakage.
  </p>
  <div class="score-grid">
    <div class="score-card" style="--c:#00c2ff">
      <div class="score-num" style="color:#00c2ff">0.9425</div>
      <div class="score-label">Conversion AUC</div>
      <div class="score-note">calibrated · raw: 0.9392</div>
    </div>
    <div class="score-card" style="--c:#4ade80">
      <div class="score-num" style="color:#4ade80">0.7432</div>
      <div class="score-label">Conversion F1</div>
      <div class="score-note">CV AUC: 0.9420 ± 0.0100</div>
    </div>
    <div class="score-card" style="--c:#a78bfa">
      <div class="score-num" style="color:#a78bfa">0.7290</div>
      <div class="score-label">Drop-off Stage F1</div>
      <div class="score-note">XGBoost · acc: 0.7310</div>
    </div>
    <div class="score-card" style="--c:#fbbf24">
      <div class="score-num" style="color:#fbbf24">63%</div>
      <div class="score-label">Top Segment Conv.</div>
      <div class="score-note">High-Value Converters · 346 users</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    #  HOW WE BUILT IT 
    st.markdown("""
<div class="section">
  <div class="s-label">Technical Stack</div>
  <h2 class="s-h2">Built from scratch. No templates.</h2>
  <p class="s-desc">Every component chosen for a specific reason, every bug hunted down. 30+ features, 5 models, 6 tabs.</p>
  <div class="stack-grid">
    <div class="stack-pill" style="--c:#00c2ff">
      <div class="pill-name">LightGBM + Platt Calibration</div>
      <div class="pill-desc">400 estimators, scale_pos_weight for imbalance, sigmoid-calibrated probabilities</div>
    </div>
    <div class="stack-pill" style="--c:#a78bfa">
      <div class="pill-name">XGBoost Drop-off Model</div>
      <div class="pill-desc">17 leak-free features, multiclass, weighted F1 0.729 — no data leakage</div>
    </div>
    <div class="stack-pill" style="--c:#4ade80">
      <div class="pill-name">SHAP TreeExplainer</div>
      <div class="pill-desc">Per-prediction waterfall + global importance — explains why, not just what</div>
    </div>
    <div class="stack-pill" style="--c:#fbbf24">
      <div class="pill-name">KMeans k=4 Segmentation</div>
      <div class="pill-desc">10 StandardScaled behavioural features · 0% to 63% conversion spread</div>
    </div>
    <div class="stack-pill" style="--c:#f472b6">
      <div class="pill-name">NVIDIA Llama-3.3-70B</div>
      <div class="pill-desc">Structured insights via NIM API — segment briefs, intervention narratives, exec summaries</div>
    </div>
    <div class="stack-pill" style="--c:#00c2ff">
      <div class="pill-name">30+ Engineered Features</div>
      <div class="pill-desc">Behavioural · temporal · KYC friction · device switching · interaction terms</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    #  DIFFERENTIATION 
    st.markdown("""
<div class="section section-alt">
  <div class="s-label">Competitive Positioning</div>
  <h2 class="s-h2">Not another dashboard.</h2>
  <p class="s-desc">Every existing tool tells you what happened. This tells you what will happen, who to act on, and exactly what to do.</p>
  <div class="diff-wrap">
    <table class="diff-table">
      <tr>
        <th>Existing Tool</th>
        <th></th>
        <th>What They Do</th>
        <th></th>
        <th>What We Do Instead</th>
      </tr>
      <tr>
        <td class="td-comp">BI Dashboards (Metabase, Redash)</td>
        <td class="td-arrow">→</td>
        <td class="td-them">Show what happened last quarter</td>
        <td class="td-arrow">→</td>
        <td class="td-us">Predict who drops today, before it happens</td>
      </tr>
      <tr>
        <td class="td-comp">Rule-based CRM (Salesforce, Zoho)</td>
        <td class="td-arrow">→</td>
        <td class="td-them">Static segments, manually defined rules</td>
        <td class="td-arrow">→</td>
        <td class="td-us">ML-driven segments + ROI-ranked dynamic rule engine</td>
      </tr>
      <tr>
        <td class="td-comp">Data science notebooks</td>
        <td class="td-arrow">→</td>
        <td class="td-them">Offline, PDF reports, no live inference</td>
        <td class="td-arrow">→</td>
        <td class="td-us">Live single-user prediction + SHAP waterfall in &lt;1s</td>
      </tr>
      <tr>
        <td class="td-comp">A/B testing platforms</td>
        <td class="td-arrow">→</td>
        <td class="td-them">Test interventions after you build them</td>
        <td class="td-arrow">→</td>
        <td class="td-us">Counterfactual ₹ impact simulation before you ship</td>
      </tr>
      <tr>
        <td class="td-comp">Generic churn tools</td>
        <td class="td-arrow">→</td>
        <td class="td-them">Binary score, no action guidance</td>
        <td class="td-arrow">→</td>
        <td class="td-us">Ranked queue — who to call, what to say, ROI per ₹</td>
      </tr>
    </table>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="section">
  <div class="s-label">The Purpose</div>
  <h2 class="s-h2">Built for Blostem's real problem.</h2>
  <p class="s-desc" style="max-width:700px">
    Blostem connects 30+ fintech platforms to 10+ banks. Every FD booking that
    drops at KYC or Details is direct revenue lost — for Blostem and every partner on the network.
    This engine gives product and growth teams one interface: understand, predict, simulate, act.
  </p>
  <div class="purpose-strip">
    <div class="purpose-item" style="--c:#ff4d5a">
      <div class="purpose-kicker" style="color:#ff4d5a">The Problem</div>
      <div class="purpose-text">FD funnel drop-off is invisible until it's already too late. Teams react instead of preventing.</div>
    </div>
    <div class="purpose-item" style="--c:#00c2ff">
      <div class="purpose-kicker">The System</div>
      <div class="purpose-text">Predict drop-off probability, explain with SHAP, simulate interventions, rank by ROI — automated.</div>
    </div>
    <div class="purpose-item" style="--c:#4ade80">
      <div class="purpose-kicker" style="color:#4ade80">The Output</div>
      <div class="purpose-text">A ranked user list with exact intervention, expected ₹ gain, and net ROI per rupee spent.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    #  CTA 
    st.markdown("""
<div class="cta-section">
  <div class="cta-h">See It Live.</div>
  <div class="cta-sub">AUC 0.9425 · F1 0.74 · Drop-off F1 0.73 · 30 features · 6 tabs · real-time inference</div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        clicked = st.button(
            "  Enter the Decision Engine  →",
            use_container_width=True,
            type="primary",
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    return clicked