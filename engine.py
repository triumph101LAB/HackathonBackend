"""
PDI Engine — Predictive Deterioration Index
Core clinical logic: trend analysis, risk scoring, projection
"""

from dataclasses import dataclass
from typing import Literal

# ── Adult ICU clinical thresholds ────────────────────────────────────────────
#
# Normal ranges (user-specified):
#   RR   : 12–18 /min
#   HR   : 60–100 bpm
#   SBP  : 90–120 mmHg
#   DBP  : 60–80 mmHg
#   Temp : 36.5–37.3 °C
#   SpO2 : 95–100 %
#
# Rule: ANYTHING outside the normal range is immediately CRITICAL.
# warn == crit on both high and low ends to collapse the two tiers into one.

THRESHOLDS = {
    # Heart Rate — normal 60–100 bpm
    "hr": {
        "label": "Heart Rate",  "unit": "bpm",
        "min": 20,  "max": 200,
        "warn": 100, "crit": 100,
        "low_warn": 60, "low_crit": 60,
        "invert": False,
        "normal_range": "60–100 bpm",
    },
    # Respiratory Rate — normal 12–18 /min
    "rr": {
        "label": "Resp Rate",   "unit": "/min",
        "min": 0,   "max": 60,
        "warn": 18, "crit": 18,
        "low_warn": 12, "low_crit": 12,
        "invert": False,
        "normal_range": "12–18 /min",
    },
    # SpO2 — normal 95–100 %
    "spo2": {
        "label": "SpO2",        "unit": "%",
        "min": 60,  "max": 100,
        "warn": 101, "crit": 101,
        "low_warn": 95, "low_crit": 95,
        "invert": True,
        "normal_range": "95–100 %",
    },
    # Temperature — normal 36.5–37.3 °C
    "temp": {
        "label": "Temperature", "unit": "°C",
        "min": 33,  "max": 42,
        "warn": 37.3, "crit": 37.3,
        "low_warn": 36.5, "low_crit": 36.5,
        "invert": False,
        "normal_range": "36.5–37.5 °C",
    },
    # Systolic BP — normal 90–120 mmHg
    "sbp": {
        "label": "Systolic BP", "unit": "mmHg",
        "min": 50,  "max": 220,
        "warn": 120, "crit": 120,
        "low_warn": 90, "low_crit": 90,
        "invert": True,
        "normal_range": "90–120 mmHg",
    },
    # Diastolic BP — normal 60–80 mmHg
    "dbp": {
        "label": "Diastolic BP","unit": "mmHg",
        "min": 30,  "max": 160,
        "warn": 80, "crit": 80,
        "low_warn": 60, "low_crit": 60,
        "invert": True,
        "normal_range": "60–80 mmHg",
    },
}

# PDI composite weights (must sum to 100)
PDI_WEIGHTS = {"spo2": 25, "hr": 20, "sbp": 18, "dbp": 12, "temp": 15, "rr": 10}

RiskLevel = Literal["ok", "warn", "crit"]


# ── Linear regression ─────────────────────────────────────────────────────────

def linear_regression(values: list[float]) -> dict:
    """
    Fit a line to a series of evenly-spaced readings.
    Returns slope, intercept, and R² goodness-of-fit.
    """
    n = len(values)
    if n < 2:
        return {"slope": 0.0, "intercept": float(values[0]) if values else 0.0, "r2": 0.0}

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in values)

    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = mean_y - slope * mean_x
    r2 = (ss_xy ** 2 / (ss_xx * ss_yy)) if (ss_xx * ss_yy) != 0 else 0.0

    return {"slope": slope, "intercept": intercept, "r2": r2}


def project_vital(readings: list[float], hours_ahead: float = 4.0) -> dict:
    """
    Project a vital sign forward by `hours_ahead`.
    Readings are assumed to be 2-hourly (adjust interval_hours if different).
    Returns slope per hour, projected value, and confidence (R²).
    """
    interval_hours = 2.0
    reg = linear_regression(readings)
    slope = reg["slope"]
    intercept = reg["intercept"]

    # Each index = one interval; project fractional steps forward
    steps_ahead = hours_ahead / interval_hours
    last_idx = len(readings) - 1
    projected = intercept + slope * (last_idx + steps_ahead)

    return {
        "slope_per_hour": round(slope / interval_hours, 2),
        "projected_value": round(projected, 2),
        "r2": round(reg["r2"], 3),
        "hours_ahead": hours_ahead,
    }


# ── Risk assessment ───────────────────────────────────────────────────────────

def assess_risk(vital_key: str, value: float) -> RiskLevel:
    """
    Classify a single value as ok / warn / crit.
    Checks BOTH high and low thresholds — returns the worst level found.
    """
    t = THRESHOLDS[vital_key]
    levels = []

    # High-end check (tachycardia, fever, hypertension, hyperoxia)
    if value >= t["crit"]:   levels.append("crit")
    elif value >= t["warn"]: levels.append("warn")
    else:                    levels.append("ok")

    # Low-end check (bradycardia, hypothermia, hypotension, hypoxia)
    low_crit = t.get("low_crit")
    low_warn = t.get("low_warn")
    if low_crit is not None and value <= low_crit:   levels.append("crit")
    elif low_warn is not None and value <= low_warn: levels.append("warn")
    else:                                            levels.append("ok")

    risk_order = {"ok": 0, "warn": 1, "crit": 2}
    return max(levels, key=lambda r: risk_order[r])


def assess_vital(vital_key: str, readings: list[float]) -> dict:
    """
    Full assessment for one vital sign:
    current reading, projection, risk at both points.
    """
    if not readings:
        return {}

    current = readings[-1]
    proj = project_vital(readings)
    current_risk = assess_risk(vital_key, current)
    projected_risk = assess_risk(vital_key, proj["projected_value"])

    # Worst of current vs projected
    risk_order = {"ok": 0, "warn": 1, "crit": 2}
    worst = max([current_risk, projected_risk], key=lambda r: risk_order[r])

    return {
        "vital": vital_key,
        "label": THRESHOLDS[vital_key]["label"],
        "unit": THRESHOLDS[vital_key]["unit"],
        "current_value": round(current, 2),
        "current_risk": current_risk,
        "projected_value": proj["projected_value"],
        "projected_risk": projected_risk,
        "slope_per_hour": proj["slope_per_hour"],
        "r2": proj["r2"],
        "worst_risk": worst,
    }


# ── PDI composite score ───────────────────────────────────────────────────────

def calc_pdi_score(vital_data: dict[str, list[float]], ai_weight: float = 0.0) -> dict:
    """
    Compute the PDI composite score (0–100).
    ai_weight: 0.0–1.0 from Gemini NLP layer — shifts score upward when
    high-risk clinical markers are detected in nursing notes.
    """
    numeric_score = 0.0
    breakdown = {}

    # Only include vitals that have readings; re-normalise weights so that
    # missing vitals (e.g. BP not yet measured) don't deflate the composite score.
    available_keys = [k for k in PDI_WEIGHTS if vital_data.get(k)]
    total_available_weight = sum(PDI_WEIGHTS[k] for k in available_keys) or 1

    for key, weight in PDI_WEIGHTS.items():
        readings = vital_data.get(key, [])
        if not readings:
            breakdown[key] = {
                "label": THRESHOLDS[key]["label"],
                "points": 0, "max": weight, "ratio": 0.0, "missing": True,
            }
            continue

        proj = project_vital(readings)
        projected = proj["projected_value"]
        t = THRESHOLDS[key]

        # Normalised weight so score stays on a 0–100 scale with partial data
        norm_weight = (weight / total_available_weight) * 100

        # High-end deviation (tachycardia, fever, hypertension, hyperoxia)
        high_ratio = max(0.0, min(1.0,
            (projected - t["warn"]) / max(t["crit"] - t["warn"], 1)))

        # Low-end deviation (bradycardia, hypothermia, hypotension, hypoxia)
        low_ratio = 0.0
        if t.get("low_warn") is not None:
            span = max(t["low_warn"] - t.get("low_crit", t["low_warn"] - 1), 1)
            low_ratio = max(0.0, min(1.0, (t["low_warn"] - projected) / span))

        ratio = max(high_ratio, low_ratio)
        points = ratio * norm_weight
        numeric_score += points
        breakdown[key] = {
            "label": t["label"],
            "points": round(points, 1),
            "max": weight,
            "ratio": round(ratio, 3),
        }

    # Apply AI weight shift: if nurse notes flag high-risk markers,
    # boost score by up to 20 points proportionally
    ai_boost = round(ai_weight * 20, 1)
    final_score = min(100, round(numeric_score + ai_boost))

    risk_level = "crit" if final_score >= 60 else "warn" if final_score >= 30 else "ok"

    return {
        "score": final_score,
        "numeric_score": round(numeric_score, 1),
        "ai_boost": ai_boost,
        "risk_level": risk_level,
        "breakdown": breakdown,
    }


# ── Alert generation ──────────────────────────────────────────────────────────

ACTIONS = {
    "crit": [
        "Draw blood cultures immediately",
        "Notify ICU Physician STAT",
        "Increase monitoring to every 15 minutes",
        "Prepare resuscitation equipment",
        "Consider IV access",
    ],
    "warn": [
        "Notify senior nurse",
        "Increase monitoring to every 30 minutes",
        "Review recent nursing notes",
        "Prepare documentation for escalation",
    ],
    "ok": [
        "Continue routine monitoring every 2 hours",
    ],
}

def generate_alert(vital_assessments: list[dict], pdi: dict) -> dict | None:
    """
    Generate a pre-emptive alert if any vital is projected to be critical,
    or if the PDI score is elevated.
    """
    critical_vitals = [v for v in vital_assessments if v.get("worst_risk") == "crit"]
    warn_vitals = [v for v in vital_assessments if v.get("worst_risk") == "warn"]

    # Alert level is driven by vital projections, not just the PDI score
    if critical_vitals:
        level = "crit"
    elif warn_vitals:
        level = "warn"
    else:
        level = pdi["risk_level"]

    if not critical_vitals and not warn_vitals and level == "ok":
        return None

    triggered_vitals = critical_vitals if critical_vitals else warn_vitals

    return {
        "level": level,
        "pdi_score": pdi["score"],
        "triggered_by": [
            {
                "vital": v["label"],
                "current": f"{v['current_value']} {v['unit']}",
                "projected": f"{v['projected_value']} {v['unit']} in 4h",
                "slope": f"{v['slope_per_hour']:+.1f} {v['unit']}/hr",
            }
            for v in triggered_vitals
        ],
        "actions": ACTIONS[level],
    }


# ── Full assessment (single entry point for the API) ─────────────────────────

def run_full_assessment(vital_data: dict[str, list[float]], ai_weight: float = 0.0) -> dict:
    """
    Run the complete PDI pipeline for all vitals.
    Returns everything the frontend needs in one response.
    """
    assessments = [assess_vital(k, vital_data[k]) for k in THRESHOLDS if vital_data.get(k)]
    pdi = calc_pdi_score(vital_data, ai_weight)
    alert = generate_alert(assessments, pdi)

    return {
        "vitals": assessments,
        "pdi": pdi,
        "alert": alert,
    }