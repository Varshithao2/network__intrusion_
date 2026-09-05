"""
risk_assessment.py
--------------------
Simple, transparent risk-scoring layer applied AFTER the ML model makes a
prediction. This is intentionally rule-based (not another ML model) so it
stays easy to explain: given a predicted attack category, look up an
analyst-defined severity level.

This mirrors how a Fortinet FortiGate / FortiAnalyzer would tag events with
a severity (Low/Medium/High/Critical) after signature or anomaly detection,
even though the underlying detection logic here is ML-based rather than
signature-based.
"""

RISK_MAP = {
    "Normal": "LOW",
    "Other": "MEDIUM",
    "Unknown": "MEDIUM",
    "Infiltration": "HIGH",
    "Port Scan": "HIGH",
    "DoS": "HIGH",
    "Brute Force": "HIGH",
    "Web Attack": "HIGH",
    "Botnet": "HIGH",
    "Attack": "HIGH",  # fallback for binary-only models
}

RECOMMENDED_ACTIONS = {
    "LOW": "No action required. Continue routine monitoring.",
    "MEDIUM": "Review the flow manually; monitor the source for repeated activity.",
    "HIGH": (
        "Investigate immediately. Consider isolating or blocking the source "
        "through your firewall/IPS according to your organization's incident "
        "response procedure."
    ),
}


def assess_risk(attack_category: str) -> str:
    """Map a predicted attack category (or binary label) to a risk level."""
    if attack_category is None:
        return "MEDIUM"
    return RISK_MAP.get(attack_category, "MEDIUM")


def recommended_action(risk_level: str) -> str:
    return RECOMMENDED_ACTIONS.get(risk_level, RECOMMENDED_ACTIONS["MEDIUM"])


def build_alert(attack_category: str, confidence: float) -> dict:
    """Construct a structured alert dict for display in the dashboard.

    NOTE: This system only produces an educational recommendation. It does
    NOT block, drop, or otherwise act on network traffic.
    """
    risk = assess_risk(attack_category)
    return {
        "attack_type": attack_category,
        "risk_level": risk,
        "confidence": round(float(confidence) * 100, 2),
        "recommended_action": recommended_action(risk),
    }
