from __future__ import annotations
import os
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

#  Config 
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_NVIDIA_MODEL    = "meta/llama-3.3-70b-instruct"

class _MaxTokens:
    INTERVENTION  = 300
    SEGMENT       = 280
    PORTFOLIO     = 350
    WHATIF        = 200


#  API layer 
def _get_client() -> OpenAI | None:
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not key:
        return None
    return OpenAI(base_url=_NVIDIA_BASE_URL, api_key=key)


def _call_nvidia(prompt: str, max_tokens: int) -> str:
    client = _get_client()
    if client is None:
        return (
            " NVIDIA_API_KEY not set. "
            "Add it to your .env file or run: set NVIDIA_API_KEY=your-key"
        )
    try:
        completion = client.chat.completions.create(
            model      = _NVIDIA_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.6,
            top_p      = 0.9,
            max_tokens = max_tokens,
        )
        text = completion.choices[0].message.content
        return text.strip() if text else " Model returned an empty response."

    except APIConnectionError:
        return " Could not reach NVIDIA API. Check your network connection."
    except APITimeoutError:
        return " NVIDIA API request timed out. Try again."
    except APIError as e:
        return f" NVIDIA API error {e.status_code}: {e.message}"
    except Exception as e:
        return f" Unexpected error: {type(e).__name__}: {e}"


#  Prompt builders 
def _build_intervention_prompt(user_profile: dict, intervention_plan: dict) -> str:
    interventions = intervention_plan.get("interventions", [])[:3]
    interventions_text = "\n".join(
        f"  {i}. [{intv['intervention_type'].upper()}] {intv['action']}"
        f" (Channel: {intv['channel']}, Expected lift: +{intv['expected_lift_pct']}%)"
        for i, intv in enumerate(interventions, 1)
    )

    risk_level = (intervention_plan.get("risk_level") or "N/A").upper()

    return f"""You are a senior fintech growth strategist at Blostem, an FD platform backed by Zerodha.

The decision engine has analyzed a user and produced the following intervention plan:

User Profile:
- Age: {user_profile.get('age', 'N/A')} | City Tier: {user_profile.get('city_tier', 'N/A')} | Device: {user_profile.get('device_type', 'N/A')}
- Income: {user_profile.get('income_bracket', 'N/A')} | Referral: {user_profile.get('referral_source', 'N/A')}
- Deepest funnel stage reached: {intervention_plan.get('predicted_drop_stage', 'N/A')}
- Total funnel time: {user_profile.get('total_time_in_funnel', 0):.0f} seconds
- KYC attempts: {user_profile.get('kyc_attempts', 0)}

ML Predictions:
- Conversion probability: {intervention_plan.get('conversion_probability', 0) * 100:.1f}%
- Risk level: {risk_level}
- Segment: {intervention_plan.get('segment', 'N/A')}
- Revenue at risk: ₹{intervention_plan.get('estimated_revenue_at_risk', 0):,.0f}

Recommended Interventions:
{interventions_text}

Write exactly 3 sentences:
1. Why this specific user is at risk based on their behavioral pattern (not just demographics).
2. Which of the recommended interventions will have the highest impact and why.
3. The specific business metric this intervention will move and by how much.

Be data-driven, specific, and actionable. Use numbers."""


def _build_segment_prompt(profile: dict) -> str:
    return f"""You are a senior fintech analyst at Blostem, a Fixed Deposit platform.
Analyze this behavioral segment and produce a concise business briefing.

Segment: {profile.get('label', 'Unknown')}
- Users: {profile['size']:,}
- Conversion rate: {profile['conversion_rate'] * 100:.1f}%
- Avg funnel events: {profile['avg_events']:.1f}
- Avg time in funnel: {profile['avg_time_in_funnel']:.0f} seconds
- Avg scroll depth: {profile['avg_scroll_depth']:.1f}%
- Avg KYC attempts: {profile['avg_kyc_attempts']:.2f}
- Mobile %: {profile['mobile_pct'] * 100:.0f}%
- Re-entry rate: {profile['reentry_rate'] * 100:.0f}%

Write exactly 3 sentences:
1. Who these users are behaviorally — what defines their funnel journey pattern.
2. The key business risk or opportunity this segment represents in revenue terms.
3. One specific, actionable intervention Blostem should deploy in the next 2 weeks, with expected conversion lift.

Be specific. Use numbers. No bullet points."""


def _build_portfolio_prompt(
    metrics:          dict,
    health_score:     dict,
    n_users:          int,
    model_comparison: dict | None,
) -> str:
    comparison_block = ""
    if model_comparison:
        baseline = model_comparison.get("baseline") or {}
        primary  = model_comparison.get("primary") or {}
        comparison_block = (
            f"\nModel Comparison:"
            f"\n- Baseline ({baseline.get('name', 'N/A')}): AUC {baseline.get('auc', 'N/A')}"
            f"\n- Primary ({primary.get('name', 'N/A')}): AUC {primary.get('auc', 'N/A')}"
            f"\n- Improvement: +{model_comparison.get('improvement_auc', 'N/A')}% AUC"
        )

    cv_auc = metrics.get("cv_auc_mean", "N/A")
    cv_std = metrics.get("cv_auc_std", "N/A")

    return f"""You are the Head of Analytics at Blostem, an FD platform.
Write a 4-sentence executive summary for the leadership team.

Portfolio overview:
- Total users tracked: {n_users:,}
- Funnel conversion rate: {health_score.get('conversion_rate', 0)}%
- Funnel Health Score: {health_score.get('overall_score', 0)}/100
- Worst bottleneck: {health_score.get('worst_bottleneck', 'N/A')} stage ({health_score.get('worst_drop_rate', 0)}% drop-off)
- Model AUC: {metrics['auc']} | F1: {metrics['f1']} | CV AUC: {cv_auc} ± {cv_std}
{comparison_block}

Sentence 1: State of the funnel in plain terms — what percentage of revenue is being lost.
Sentence 2: What the ML models tell us about prediction confidence and where the bottleneck is.
Sentence 3: The single highest-ROI intervention to deploy this week.
Sentence 4: The expected revenue impact if we act on the decision engine's recommendations.

Executive tone. Numbers-first. No jargon."""


def _build_whatif_prompt(simulation_result: dict) -> str:
    scenario = simulation_result.get("scenario") or {}

    return f"""You are a growth strategist presenting a what-if analysis to stakeholders.

Scenario: {scenario.get('name', 'N/A')}
Description: {scenario.get('description', 'N/A')}

Results:
- Users affected: {simulation_result.get('n_affected_users', 0)} ({simulation_result.get('affected_pct', 0)}% of total)
- Additional conversions: +{simulation_result.get('additional_conversions', 0)}
- Revenue impact: ₹{simulation_result.get('additional_revenue', 0):,.0f}
- Implementation cost: ₹{simulation_result.get('total_cost', 0):,.0f}
- ROI: {simulation_result.get('roi', 'N/A')}×
- Conversion rate: {simulation_result.get('baseline_conversion_rate', 0)}% → {simulation_result.get('intervention_conversion_rate', 0)}%
- Implementation effort: {scenario.get('implementation_effort', 'N/A')}

Write exactly 2 sentences:
1. Quantify the business case: revenue, conversions, and ROI in concrete terms.
2. State the recommended next step with a specific timeline.

Be crisp and executive-ready."""


#  Public API 
def intervention_explanation(user_profile: dict, intervention_plan: dict) -> str:
    return _call_nvidia(
        _build_intervention_prompt(user_profile, intervention_plan),
        _MaxTokens.INTERVENTION,
    )


def segment_insight(profile: dict) -> str:
    return _call_nvidia(_build_segment_prompt(profile), _MaxTokens.SEGMENT)


def portfolio_summary(
    metrics:          dict,
    health_score:     dict,
    n_users:          int,
    model_comparison: dict | None = None,
) -> str:
    return _call_nvidia(
        _build_portfolio_prompt(metrics, health_score, n_users, model_comparison),
        _MaxTokens.PORTFOLIO,
    )


def whatif_narrative(simulation_result: dict) -> str:
    return _call_nvidia(_build_whatif_prompt(simulation_result), _MaxTokens.WHATIF)