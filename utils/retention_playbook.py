from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable


#  Constants 
class Threshold:
    HIGH_CHURN        = 0.60
    MEDIUM_CHURN      = 0.45
    LOW_CHURN         = 0.40
    FEW_FDS           = 2
    HIGH_TICKETS      = 2           
    DORMANT_DAYS      = 60
    LOW_SESSIONS      = 5
    LOW_AMOUNT        = 100_000
    MAX_ACTIONS       = 4


HIGH_INCOME_BRACKETS = {">15L", "7-15L"}


#  Data model 
@dataclass
class PlaybookAction:
    action:           str
    description:      str
    timeline:         str
    owner:            str
    success_metric:   str
    churn_reduction:  int        
    impact_label:     str        
    target_audience:  str
    priority:         int = 0    

    def to_dict(self) -> dict:
        return asdict(self)


#  Rule engine 
@dataclass
class Rule:
    name:      str
    condition: Callable[[dict, float], bool]
    build:     Callable[[dict, float], PlaybookAction]


def _rules() -> list[Rule]:
    return [
        Rule(
            name="fd_education",
            condition=lambda c, p: p > Threshold.HIGH_CHURN
                                   and c.get("num_fds_booked", 0) <= Threshold.FEW_FDS,
            build=lambda c, p: PlaybookAction(
                action          = " FD Education Series",
                description     = (
                    "Send 5-email series explaining FD basics, comparison with "
                    "savings accounts, and how interest compounding works."
                ),
                timeline        = "Next 7 days",
                owner           = "Email Marketing",
                success_metric  = "User books 1 FD within 30 days",
                churn_reduction = 12,
                impact_label    = "-12% relative churn risk",
                target_audience = "New, underexposed users",
                priority        = 2,
            ),
        ),
        Rule(
            name="support_intervention",
            condition=lambda c, p: c.get("support_tickets", 0) >= Threshold.HIGH_TICKETS
                                   and p > Threshold.LOW_CHURN,
            build=lambda c, p: PlaybookAction(
                action          = " Dedicated Support Intervention",
                description     = (
                    "Assign a support specialist. Proactive check-in call. "
                    "Resolve all open issues within 24 hours."
                ),
                timeline        = "Immediate",
                owner           = "Support Team",
                success_metric  = "Support satisfaction score ≥ 4 / 5",
                churn_reduction = 18,
                impact_label    = "-18% relative churn risk",
                target_audience = "High-friction users",
                priority        = 4,
            ),
        ),
        Rule(
            name="winback_campaign",
            condition=lambda c, p: c.get("last_login_days", 0) > Threshold.DORMANT_DAYS
                                   and p > Threshold.HIGH_CHURN,
            build=lambda c, p: PlaybookAction(
                action          = " Win-Back Campaign",
                description     = (
                    "Limited-time offer: +25 bps interest boost for 90 days. "
                    "Multi-channel: SMS + Email + In-app push."
                ),
                timeline        = "Next 3 days (time-sensitive)",
                owner           = "Product / Growth",
                success_metric  = "1 new FD booking within 14 days",
                churn_reduction = 20,
                impact_label    = "-20% relative churn risk",
                target_audience = "Dormant users (>60 days inactive)",
                priority        = 3,
            ),
        ),
        Rule(
            name="ux_nudge",
            condition=lambda c, p: c.get("platform_sessions", 0) < Threshold.LOW_SESSIONS
                                   and p > Threshold.MEDIUM_CHURN,
            build=lambda c, p: PlaybookAction(
                action          = "UX Simplification + Nudge",
                description     = (
                    "Send in-app tutorial (30-second video). Highlight fastest "
                    "booking path. Enable 1-click FD renewal."
                ),
                timeline        = "Next 3 days",
                owner           = "Product",
                success_metric  = "Sessions increase to 8+ within 30 days",
                churn_reduction = 10,
                impact_label    = "-10% relative churn risk",
                target_audience = "Low-engagement users",
                priority        = 1,
            ),
        ),
        Rule(
            name="premium_upgrade",
            condition=lambda c, p: c.get("income_bracket") in HIGH_INCOME_BRACKETS
                                   and c.get("avg_fd_amount", 0) < Threshold.LOW_AMOUNT,
            build=lambda c, p: PlaybookAction(
                action          = " Premium FD Offering",
                description     = (
                    "Personalised call from Relationship Manager. "
                    "Present premium FD tiers (higher rates for ₹5L+)."
                ),
                timeline        = "This week",
                owner           = "Relationship Management",
                success_metric  = "₹5L+ booking OR RM relationship initiated",
                churn_reduction = 0,        
                impact_label    = "+30% estimated lifetime value",
                target_audience = "High-net-worth, low-penetration",
                priority        = 2,
            ),
        ),
    ]


#  Public API 
def generate_playbook(
    customer:       dict,
    churn_prob:     float,
    model_features: dict | None = None,   
    max_actions:    int         = Threshold.MAX_ACTIONS,
) -> list[PlaybookAction]:
    fired = [
        rule.build(customer, churn_prob)
        for rule in _rules()
        if rule.condition(customer, churn_prob)
    ]
    ffired.sort(
    key=lambda a: (a.priority, a.churn_reduction),
    reverse=True
)
    return fired[:max_actions]


def playbook_summary(actions: list[PlaybookAction]) -> dict:
    if not actions:
        return {
            "actions":          [],
            "total_actions":    0,
            "combined_churn_reduction_pp": 0,
            "combined_impact_label":       "No actions triggered",
        }

    remaining = 1.0
    for a in actions:
        remaining *= 1 - (a.churn_reduction / 100)
    combined_pp = round((1 - remaining) * 100)

    return {
        "actions":                         [a.to_dict() for a in actions],
        "total_actions":                   len(actions),
        "combined_churn_reduction_pp":     combined_pp,
        "combined_impact_label":           f"-{combined_pp}% combined churn risk reduction",
    }