"""Retention playbook engine — generates actionable recommendations."""

def generate_playbook(customer: dict, churn_prob: float, model_features: dict = None) -> list:
    """Generate 3-5 specific, testable retention actions for a customer."""
    playbook = []
    
    # Rule 1: High churn + few FDs = onboarding/education needed
    if churn_prob > 0.60 and customer['num_fds_booked'] <= 2:
        playbook.append({
            "action": "📧 FD Education Series",
            "description": "Send 5-email series explaining FD basics, comparison with savings, interest rates",
            "timeline": "Next 7 days",
            "owner": "Email Marketing",
            "success_metric": "User books 1 FD within 30 days",
            "expected_impact": "-12% relative churn risk",
            "target_audience": "New, underexposed users"
        })
    
    # Rule 2: High support tickets = friction/problems
    if customer['support_tickets'] > 1.5 and churn_prob > 0.40:
        playbook.append({
            "action": "🎯 Dedicated Support Intervention",
            "description": "Assign support specialist. Proactive check-in. Resolve all open issues within 24hr.",
            "timeline": "Immediate",
            "owner": "Support Team",
            "success_metric": "Support satisfaction score ≥ 4/5",
            "expected_impact": "-18% relative churn risk",
            "target_audience": "High-friction users"
        })
    
    # Rule 3: Old account + no login = dormant/win-back
    if customer['last_login_days'] > 60 and churn_prob > 0.50:
        playbook.append({
            "action": "🎁 Win-Back Campaign",
            "description": "Limited-time: +25 bps interest boost for 90 days. SMS + Email + In-app.",
            "timeline": "Next 3 days (time-sensitive)",
            "owner": "Product/Growth",
            "success_metric": "1 new FD booking within 14 days",
            "expected_impact": "-20% relative churn risk",
            "target_audience": "Dormant users (>60 days inactive)"
        })
    
    # Rule 4: Low sessions + medium+ churn = UX friction
    if customer['platform_sessions'] < 5 and churn_prob > 0.45:
        playbook.append({
            "action": "📱 UX Simplification + Nudge",
            "description": "Send in-app tutorial (30 sec video). Highlight fastest path. 1-click FD.",
            "timeline": "Next 3 days",
            "owner": "Product",
            "success_metric": "Sessions increase to 8+ within 30 days",
            "expected_impact": "-10% relative churn risk",
            "target_audience": "Low-engagement users"
        })
    
    # Rule 5: High income + low FD amount = untapped opportunity
    if customer['income_bracket'] in [">15L", "7-15L"] and customer['avg_fd_amount'] < 100_000:
        playbook.append({
            "action": "💎 Premium FD Offering",
            "description": "Personalized call from RM. Premium FD options (higher rates for ₹5L+).",
            "timeline": "This week",
            "owner": "Relationship Management",
            "success_metric": "₹5L+ booking OR relationship initiated",
            "expected_impact": "+30% lifetime value",
            "target_audience": "High-net-worth, low-penetration"
        })
    
    return playbook[:4]  # Max 4 recommendations per customer

def playbook_to_dataframe(playbook: list) -> dict:
    """Convert playbook to display-friendly format."""
    if not playbook:
        return {"actions": []}
    
    return {
        "actions": playbook,
        "total_actions": len(playbook),
        "expected_combined_impact": f"-{sum(int(p['expected_impact'].split('%')[0].replace('-', '')) for p in playbook if '%' in p['expected_impact'])}% relative risk"
    }