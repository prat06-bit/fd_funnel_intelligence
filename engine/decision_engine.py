import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Optional

STAGE_NAMES = {0: "Landing", 1: "FD View", 2: "Details", 3: "KYC", 4: "Deposit"}


@dataclass
class Intervention:
    """A single actionable intervention."""
    rule_id: str
    intervention_type: str          # nudge | offer | support | re-engage | simplify
    action: str                     # what to do
    channel: str                    # in_app | email | sms | push | call
    timing: str                     # immediate | next_session | scheduled
    expected_lift_pct: float        # estimated conversion improvement
    cost_per_user: float            # ₹ cost per intervention
    priority_score: float = 0.0     # computed ROI-based priority
    business_rule: str = ""         # which rule triggered this

    def to_dict(self):
        return asdict(self)


@dataclass
class InterventionPlan:
    """Complete decision output for a user."""
    user_id: str
    conversion_probability: float
    predicted_drop_stage: str
    segment: str
    risk_level: str                 # high | medium | low
    interventions: List[Intervention] = field(default_factory=list)
    estimated_revenue_at_risk: float = 0.0
    top_risk_factors: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["interventions"] = [i.to_dict() for i in self.interventions]
        return d


#  Business Rules 
BUSINESS_RULES = [
    {
        "rule_id": "KYC_FRICTION_MOBILE",
        "name": "Mobile KYC Friction",
        "condition": lambda u, p: p.get("predicted_drop_stage") == "KYC" and u.get("device_mobile", 0) == 1,
        "intervention": Intervention(
            rule_id="KYC_FRICTION_MOBILE",
            intervention_type="simplify",
            action="Switch to Aadhaar-OTP verification flow (skip document upload on mobile)",
            channel="in_app",
            timing="immediate",
            expected_lift_pct=22.0,
            cost_per_user=0.0,
            business_rule="Mobile users at KYC have 2.3× higher drop-off than desktop users",
        ),
    },
    {
        "rule_id": "KYC_RETRY",
        "name": "KYC Re-attempt Support",
        "condition": lambda u, p: u.get("kyc_attempts", 0) >= 2 and u.get("converted", 0) == 0,
        "intervention": Intervention(
            rule_id="KYC_RETRY",
            intervention_type="support",
            action="Trigger assisted KYC flow with pre-filled fields + support chat widget",
            channel="in_app",
            timing="immediate",
            expected_lift_pct=18.0,
            cost_per_user=2.0,
            business_rule="Users with 2+ KYC attempts are 3× more likely to convert with assisted flow",
        ),
    },
    {
        "rule_id": "RATE_COMPARISON_NUDGE",
        "name": "Rate-Sensitive Comparator",
        "condition": lambda u, p: u.get("time_on_details", 0) > 60 and p.get("conversion_probability", 1) < 0.45,
        "intervention": Intervention(
            rule_id="RATE_COMPARISON_NUDGE",
            intervention_type="nudge",
            action="Display personalized rate comparison card highlighting best-match FD by tenor + amount",
            channel="in_app",
            timing="next_session",
            expected_lift_pct=14.0,
            cost_per_user=0.0,
            business_rule="Users spending >60s on details are comparison-shopping; personalized highlight converts 14% more",
        ),
    },
    {
        "rule_id": "DORMANT_REENGAGEMENT",
        "name": "Dormant Re-engagement",
        "condition": lambda u, p: u.get("has_reentry", 0) == 0 and u.get("max_stage_idx", 0) >= 2 and p.get("conversion_probability", 1) < 0.35,
        "intervention": Intervention(
            rule_id="DORMANT_REENGAGEMENT",
            intervention_type="re-engage",
            action="Send personalized FD maturity calculator showing projected returns for their income bracket",
            channel="email",
            timing="scheduled",
            expected_lift_pct=9.0,
            cost_per_user=0.5,
            business_rule="Users who reached Details but never returned show 9% re-engagement with projected returns",
        ),
    },
    {
        "rule_id": "EVENING_HIGH_INTENT",
        "name": "Evening High-Intent Capture",
        "condition": lambda u, p: u.get("evening_session_ratio", 0) > 0.5 and u.get("max_stage_idx", 0) >= 3,
        "intervention": Intervention(
            rule_id="EVENING_HIGH_INTENT",
            intervention_type="offer",
            action="Present limited-time rate bonus (+0.1% for 48hrs) during evening session",
            channel="in_app",
            timing="immediate",
            expected_lift_pct=16.0,
            cost_per_user=0.0,
            business_rule="Evening sessions indicate deliberate intent; time-limited offers convert 16% more",
        ),
    },
    {
        "rule_id": "TIER3_TRUST_BUILD",
        "name": "Tier-3 Trust Building",
        "condition": lambda u, p: u.get("city_tier", 1) == 3 and p.get("predicted_drop_stage") in ["KYC", "Details"],
        "intervention": Intervention(
            rule_id="TIER3_TRUST_BUILD",
            intervention_type="nudge",
            action="Show RBI insurance badge + '₹5L DICGC guaranteed' trust banner with regional language toggle",
            channel="in_app",
            timing="immediate",
            expected_lift_pct=11.0,
            cost_per_user=0.0,
            business_rule="Tier-3 users have 18% lower trust scores; DICGC visibility lifts conversion by 11%",
        ),
    },
    {
        "rule_id": "PARTNER_REFERRAL_WARMUP",
        "name": "Partner Referral Fast-Track",
        "condition": lambda u, p: u.get("referral_partner", 0) == 1 and u.get("max_stage_idx", 0) <= 1,
        "intervention": Intervention(
            rule_id="PARTNER_REFERRAL_WARMUP",
            intervention_type="nudge",
            action="Auto-apply partner welcome rate (+0.15%) and skip landing page — direct to FD configurator",
            channel="in_app",
            timing="immediate",
            expected_lift_pct=20.0,
            cost_per_user=0.0,
            business_rule="Partner-referred users have pre-existing trust; reduce friction to capture high-intent traffic",
        ),
    },
    {
        "rule_id": "HIGH_AMOUNT_CONCIERGE",
        "name": "High-Value Concierge",
        "condition": lambda u, p: u.get("income_num", 0) >= 3 and p.get("conversion_probability", 1) < 0.40,
        "intervention": Intervention(
            rule_id="HIGH_AMOUNT_CONCIERGE",
            intervention_type="support",
            action="Assign relationship manager callback within 2 hours for personalized FD portfolio consultation",
            channel="call",
            timing="scheduled",
            expected_lift_pct=25.0,
            cost_per_user=50.0,
            business_rule="High-income users who hesitate respond 25% better to personalized RM outreach",
        ),
    },
]


def evaluate_user(user_features: dict, ml_predictions: dict,
                  avg_fd_amount: float = 50_000) -> InterventionPlan:
    conv_prob = ml_predictions.get("conversion_probability", 0.5)
    drop_stage_idx = ml_predictions.get("predicted_drop_stage_idx", 0)
    drop_stage = STAGE_NAMES.get(drop_stage_idx, "Landing")
    segment = ml_predictions.get("segment", "Unknown")

    # Risk level
    if conv_prob < 0.25:
        risk = "high"
    elif conv_prob < 0.50:
        risk = "medium"
    else:
        risk = "low"

    # Estimated revenue at risk (if they don't convert)
    income_num = user_features.get("income_num", 2)
    est_amount = avg_fd_amount * (1 + (income_num - 2) * 0.3)
    revenue_at_risk = est_amount * (1 - conv_prob)

    plan = InterventionPlan(
        user_id=user_features.get("user_id", "UNKNOWN"),
        conversion_probability=conv_prob,
        predicted_drop_stage=drop_stage,
        segment=segment,
        risk_level=risk,
        estimated_revenue_at_risk=revenue_at_risk,
        top_risk_factors=ml_predictions.get("top_risk_factors", []),
    )

    # Evaluate each rule
    preds_with_stage = {**ml_predictions, "predicted_drop_stage": drop_stage}
    for rule in BUSINESS_RULES:
        try:
            if rule["condition"](user_features, preds_with_stage):
                intervention = Intervention(
                    rule_id=rule["intervention"].rule_id,
                    intervention_type=rule["intervention"].intervention_type,
                    action=rule["intervention"].action,
                    channel=rule["intervention"].channel,
                    timing=rule["intervention"].timing,
                    expected_lift_pct=rule["intervention"].expected_lift_pct,
                    cost_per_user=rule["intervention"].cost_per_user,
                    business_rule=rule["intervention"].business_rule,
                )
                lift = intervention.expected_lift_pct / 100
                cost = intervention.cost_per_user

                roi_result = compute_intervention_roi(
                    conv_prob,
                    lift,
                    est_amount,
                    cost
                )

                intervention.priority_score = round(roi_result["roi"], 1)

                intervention.expected_new_prob = roi_result["new_prob"]
                intervention.expected_gain = roi_result["expected_gain"]
                intervention.net_roi = roi_result["roi"]   
                plan.interventions.append(intervention)
        except Exception:
            continue

    plan.interventions.sort(key=lambda x: -x.priority_score)

    return plan


def batch_evaluate(feature_df: pd.DataFrame, conv_proba: np.ndarray,
                   drop_stages: Optional[np.ndarray],
                   segment_names: pd.Series,
                   shap_top_factors: Optional[list] = None,
                   avg_fd_amount: float = 50_000) -> List[InterventionPlan]:
    plans = []
    for i, (_, row) in enumerate(feature_df.iterrows()):
        user_features = row.to_dict()
        ml_predictions = {
            "conversion_probability": float(conv_proba[i]) if i < len(conv_proba) else 0.5,
            "predicted_drop_stage_idx": int(drop_stages[i]) if drop_stages is not None and i < len(drop_stages) else 0,
            "segment": str(segment_names.iloc[i]) if i < len(segment_names) else "Unknown",
            "top_risk_factors": shap_top_factors[i] if shap_top_factors and i < len(shap_top_factors) else [],
        }
        plan = evaluate_user(user_features, ml_predictions, avg_fd_amount)
        plans.append(plan)

    return plans


def build_priority_queue(plans: List[InterventionPlan], top_n: int = 50) -> pd.DataFrame:
    rows = []
    for plan in plans:
        if plan.risk_level in ("high", "medium") and plan.interventions:
            top_intervention = plan.interventions[0]
            rows.append({
                "user_id": plan.user_id,
                "conversion_prob": f"{plan.conversion_probability:.1%}",
                "risk_level": plan.risk_level.upper(),
                "drop_stage": plan.predicted_drop_stage,
                "segment": plan.segment,
                "top_action": top_intervention.action[:80] + "…" if len(top_intervention.action) > 80 else top_intervention.action,
                "channel": top_intervention.channel,
                "expected_lift": f"+{top_intervention.expected_lift_pct:.0f}%",
                "cost": f"₹{top_intervention.cost_per_user:.0f}",
                "expected_gain": f"₹{top_intervention.expected_gain:,.0f}",
                "new_conversion": f"{top_intervention.expected_new_prob*100:.1f}%",
                "revenue_at_risk": f"₹{plan.estimated_revenue_at_risk:,.0f}",
                "roi_value": top_intervention.net_roi,   # numeric
                "roi": f"₹{top_intervention.net_roi:,.0f}",
            })

    queue_df = pd.DataFrame(rows).sort_values("roi_value", ascending=False)
    queue_df = queue_df.reset_index(drop=True)
    queue_df.index += 1
    queue_df.index.name = "Rank"

    return queue_df


def compute_funnel_health_score(funnel_data: dict, drop_rates: dict,
                                 metrics: dict) -> dict:
    values = funnel_data["values"]
    total = max(values[0], 1)
    converted = values[-1]

    # 1. Conversion efficiency (0-100)
    conv_rate = converted / total
    conv_score = min(conv_rate / 0.30 * 100, 100)  # 30% = perfect score

    # 2. Flow smoothness (0-100)
    max_drop = max(drop_rates.values()) if drop_rates else 0.5
    smoothness_score = max(0, (1 - max_drop / 0.60) * 100)  # 60% single-stage drop = 0

    # 3. Model confidence (0-100)
    auc = metrics.get("auc", 0.5)
    model_score = max(0, (auc - 0.5) / 0.4 * 100)  # AUC=0.9 → 100

    # 4. Engagement depth (0-100)
    stages_arr = funnel_data["values"]
    if len(stages_arr) >= 2:
        avg_reach = sum(stages_arr[1:]) / (len(stages_arr) - 1) / max(stages_arr[0], 1)
        depth_score = min(avg_reach / 0.50 * 100, 100)
    else:
        depth_score = 50

    overall = (
        conv_score * 0.40
        + smoothness_score * 0.25
        + model_score * 0.20
        + depth_score * 0.15
    )

    # Worst bottleneck
    worst_stage = max(drop_rates, key=drop_rates.get) if drop_rates else "unknown"
    worst_drop = drop_rates.get(worst_stage, 0)

    return {
        "overall_score": round(overall, 1),
        "conversion_efficiency": round(conv_score, 1),
        "flow_smoothness": round(smoothness_score, 1),
        "model_confidence": round(model_score, 1),
        "engagement_depth": round(depth_score, 1),
        "worst_bottleneck": worst_stage,
        "worst_drop_rate": round(worst_drop * 100, 1),
        "conversion_rate": round(conv_rate * 100, 1),
    }
def compute_intervention_roi(conv_prob, lift, user_value, cost):
    new_prob = min(conv_prob + lift, 1.0)

    expected_gain = (new_prob - conv_prob) * user_value
    net_roi = expected_gain - cost

    return {
        "new_prob": new_prob,
        "expected_gain": expected_gain,
        "net_roi": net_roi
    }