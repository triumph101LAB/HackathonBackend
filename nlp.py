"""
PDI NLP Layer — Claude-powered clinical note generation + risk analysis
Uses UMLS (Unified Medical Language System) clinical terminology as the
knowledge base for structured nursing note generation from raw vitals.
"""

import json, os
import anthropic
from anthropic import RateLimitError, APIError
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── UMLS concept clusters used for clinical language grounding ────────────────
# These map vital sign deviations to standardised UMLS clinical descriptors
# so generated notes use consistent, medically accurate terminology.

UMLS_VITAL_CONCEPTS = {
    "hr": {
        "high": {
            "umls_concept": "C0039231",  # Tachycardia
            "descriptors": ["tachycardic", "elevated heart rate", "increased pulse rate"],
            "clinical_concern": "possible sympathetic activation, pain, haemorrhage, or early sepsis",
        },
        "low": {
            "umls_concept": "C0004610",  # Bradycardia
            "descriptors": ["bradycardic", "decreased heart rate", "slow pulse"],
            "clinical_concern": "possible vagal response, medication effect, or cardiac conduction issue",
        },
    },
    "rr": {
        "high": {
            "umls_concept": "C0231835",  # Tachypnoea
            "descriptors": ["tachypnoeic", "increased respiratory rate", "rapid breathing"],
            "clinical_concern": "possible respiratory compromise, metabolic acidosis, or pain",
        },
        "low": {
            "umls_concept": "C0700067",  # Bradypnoea
            "descriptors": ["bradypnoeic", "decreased respiratory rate", "slow respirations"],
            "clinical_concern": "possible sedation effect, CNS depression, or fatigue",
        },
    },
    "spo2": {
        "low": {
            "umls_concept": "C0242184",  # Hypoxaemia
            "descriptors": ["hypoxaemic", "decreased oxygen saturation", "desaturating"],
            "clinical_concern": "possible respiratory failure, airway compromise, or V/Q mismatch",
        },
    },
    "temp": {
        "high": {
            "umls_concept": "C0015967",  # Fever
            "descriptors": ["febrile", "pyrexial", "elevated temperature"],
            "clinical_concern": "possible infection, inflammatory process, or drug reaction",
        },
        "low": {
            "umls_concept": "C0020672",  # Hypothermia
            "descriptors": ["hypothermic", "low body temperature", "subnormal temperature"],
            "clinical_concern": "possible environmental exposure, septic shock, or metabolic disorder",
        },
    },
    "sbp": {
        "high": {
            "umls_concept": "C0020538",  # Hypertension
            "descriptors": ["hypertensive", "elevated systolic pressure"],
            "clinical_concern": "possible pain, anxiety, or hypertensive crisis",
        },
        "low": {
            "umls_concept": "C0020649",  # Hypotension
            "descriptors": ["hypotensive", "decreased systolic pressure"],
            "clinical_concern": "possible haemodynamic compromise, haemorrhage, or distributive shock",
        },
    },
    "dbp": {
        "high": {
            "umls_concept": "C0020538",
            "descriptors": ["elevated diastolic pressure"],
            "clinical_concern": "possible hypertensive state",
        },
        "low": {
            "umls_concept": "C0020649",
            "descriptors": ["decreased diastolic pressure", "widened pulse pressure"],
            "clinical_concern": "possible septic or distributive shock",
        },
    },
}

THRESHOLDS_NORMAL = {
    "hr":   {"low": 60,   "high": 100,  "unit": "bpm"},
    "rr":   {"low": 12,   "high": 18,   "unit": "/min"},
    "spo2": {"low": 95,   "high": 100,  "unit": "%"},
    "temp": {"low": 36.5, "high": 37.3, "unit": "°C"},
    "sbp":  {"low": 90,   "high": 120,  "unit": "mmHg"},
    "dbp":  {"low": 60,   "high": 80,   "unit": "mmHg"},
}


def _build_vital_context(vitals: dict, assessment: dict) -> str:
    """
    Build a structured clinical context string from current vitals and
    their assessment, grounded in UMLS terminology.
    """
    lines = []
    vital_assessments = {v["vital"]: v for v in assessment.get("vitals", [])}

    for key, thresh in THRESHOLDS_NORMAL.items():
        readings = vitals.get(key, [])
        if not readings:
            continue

        current = readings[-1]
        va      = vital_assessments.get(key, {})
        slope   = va.get("slope_per_hour", 0)
        proj    = va.get("projected_value", current)
        risk    = va.get("worst_risk", "ok")

        direction = None
        if current > thresh["high"]:
            direction = "high"
        elif current < thresh["low"]:
            direction = "low"

        umls = UMLS_CONCEPTS = UMLS_VITAL_CONCEPTS.get(key, {})
        concept = umls.get(direction, {}) if direction else {}
        descriptor = concept.get("descriptors", [None])[0]
        concern    = concept.get("clinical_concern", "")

        status = f"{descriptor} ({current} {thresh['unit']})" if descriptor else f"within normal range ({current} {thresh['unit']})"
        trend  = f"trending {'up' if slope > 0 else 'down'} at {abs(slope):.1f} {thresh['unit']}/hr, projected {proj} {thresh['unit']} in 4h"

        line = f"- {key.upper()}: {status}; {trend}"
        if concern and risk != "ok":
            line += f"; concern: {concern}"
        lines.append(line)

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert ICU nursing documentation assistant.
Your role is to generate structured, professional nursing observation notes using
standardised clinical terminology grounded in the Unified Medical Language System (UMLS).

You will receive a patient's current vital signs with trend analysis and produce:
1. A structured SOAP-style nursing note
2. A risk weight (0.0–1.0) based on the clinical picture
3. Flagged clinical concerns using UMLS-aligned terminology

IMPORTANT RULES:
- Use precise UMLS-aligned clinical language (e.g. "tachycardic" not "fast heartbeat")
- Focus on DETERIORATION risk — do NOT diagnose or use disease-specific terms like "sepsis"
- Note both current values AND the 4-hour trajectory
- Be concise — this is a bedside note, not a discharge summary
- Always note which vitals are within normal limits

You must respond ONLY with valid JSON — no preamble, no markdown fences.

Response structure:
{
  "generated_note": "Full SOAP nursing note as a string",
  "flagged_terms": ["UMLS-aligned clinical concern phrases"],
  "risk_weight": 0.0,
  "severity": "low | moderate | high",
  "reasoning": "One sentence clinical summary using UMLS terminology.",
  "recommended_actions": ["action 1", "action 2"]
}

Risk weight scale:
- 0.0–0.2: All vitals stable, no concerning trends
- 0.2–0.5: One or more vitals trending toward abnormal range
- 0.5–0.75: Multiple abnormal vitals with adverse trajectory
- 0.75–1.0: Imminent deterioration risk — immediate escalation required"""


def generate_nursing_note(vitals: dict, assessment: dict, patient_meta: dict) -> dict:
    """
    Generate a UMLS-grounded nursing note from patient vitals and assessment.
    Returns structured note with risk analysis.
    """
    if not vitals or not any(v for v in vitals.values()):
        return {
            "generated_note": "Insufficient vital signs data to generate note.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "No vital signs recorded.",
            "recommended_actions": [],
        }

    vital_context = _build_vital_context(vitals, assessment)
    pdi_score     = assessment.get("pdi", {}).get("score", "N/A")
    risk_level    = assessment.get("pdi", {}).get("risk_level", "ok")

    prompt = f"""Generate a nursing observation note for this ICU patient.

Patient: {patient_meta.get('name', 'Unknown')}, Age: {patient_meta.get('age', 'N/A')}, Weight: {patient_meta.get('weight', 'N/A')}
Ward: {patient_meta.get('ward', 'ICU')}, Bed: {patient_meta.get('bed', 'N/A')}
PDI Risk Score: {pdi_score}/100 ({risk_level.upper()})

Current Vital Signs with 4-hour Trend Analysis:
{vital_context}

Generate a professional SOAP-format nursing note using UMLS-aligned clinical terminology.
Focus on deterioration risk — note trends and trajectories, not just current values."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Handle different response structures
        if isinstance(response, str):
            raw = response.strip()
        elif hasattr(response, 'content') and response.content:
            raw = response.content[0].text.strip()
        else:
            raise ValueError(f"Unexpected response structure: {type(response)}")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        result["risk_weight"] = max(0.0, min(1.0, float(result.get("risk_weight", 0.0))))
        return result

    except RateLimitError:
        return {
            "generated_note": "Rate limit exceeded — please try again in a moment.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "API rate limit hit. Free tier has strict quotas.",
            "recommended_actions": ["Retry in 60 seconds"],
            "error": True,
            "rate_limited": True,
        }
    except APIError as e:
        return {
            "generated_note": "API error — service temporarily unavailable.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": f"API error: {str(e)}",
            "recommended_actions": ["Check API key and quota"],
            "error": True,
        }
    except json.JSONDecodeError:
        return {
            "generated_note": "Note generation failed — manual documentation required.",
            "flagged_terms": [],
            "risk_weight": 0.3,
            "severity": "moderate",
            "reasoning": "Unable to parse AI response. Clinical review recommended.",
            "recommended_actions": ["Manual clinical review required"],
            "parse_error": True,
        }
    except Exception as e:
        return {
            "generated_note": "",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": f"Note generation unavailable: {type(e).__name__}: {str(e)}",
            "recommended_actions": [],
            "error": True,
        }

# def generate_nursing_note(vitals_context: str) -> dict:
#     """
#     Ask Claude to write a nursing note from current vitals,
#     then assess it for risk markers in one shot.
#     """
#     prompt = f"""Given these neonatal patient vitals: {vitals_context}

# Write a concise nursing observation note (2-3 sentences) in clinical language,
# then analyse it for risk markers.

# Respond ONLY with valid JSON:
# {{
#   "generated_note": "the nursing note text",
#   "flagged_terms": [],
#   "risk_weight": 0.0,
#   "severity": "low | moderate | high",
#   "reasoning": "one sentence",
#   "recommended_actions": []
# }}"""

#     try:
#         response = client.messages.create(
#             model="claude-sonnet-4-5",
#             max_tokens=512,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         raw = response.content[0].text.strip()
#         if raw.startswith("```"):
#             raw = raw.split("```")[1]
#             if raw.startswith("json"):
#                 raw = raw[4:]
#         result = json.loads(raw)
#         result["risk_weight"] = max(0.0, min(1.0, float(result.get("risk_weight", 0.0))))
#         return result
#     except Exception as e:
#         return {
#             "generated_note": "Note generation failed — check backend.",
#             "flagged_terms": [], "risk_weight": 0.0,
#             "severity": "low", "reasoning": str(e),
#             "recommended_actions": [], "error": True,
#         }

def analyze_nursing_note(note: str) -> dict:
    """
    Analyze a manually entered nursing note for clinical risk markers.
    Uses UMLS-grounded terminology for flag detection.
    """
    if not note or not note.strip():
        return {
            "generated_note": "",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "No note provided.",
            "recommended_actions": [],
        }

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyse this nursing observation note for deterioration risk using UMLS clinical terminology:\n\n\"{note.strip()}\""
            }],
        )

        # Handle different response structures
        if isinstance(response, str):
            raw = response.strip()
        elif hasattr(response, 'content') and response.content:
            raw = response.content[0].text.strip()
        else:
            raise ValueError(f"Unexpected response structure: {type(response)}")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        result["risk_weight"] = max(0.0, min(1.0, float(result.get("risk_weight", 0.0))))
        return result

    except RateLimitError:
        return {
            "generated_note": "Rate limit exceeded — please try again.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "API rate limit hit. Free tier has strict quotas.",
            "recommended_actions": ["Retry in 60 seconds"],
            "error": True,
            "rate_limited": True,
        }
    except APIError as e:
        return {
            "generated_note": "API error — service temporarily unavailable.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": f"API error: {str(e)}",
            "recommended_actions": ["Check API key and quota"],
            "error": True,
        }
    except json.JSONDecodeError:
        return {
            "generated_note": "Analysis failed — unable to parse response.",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": "Unable to parse AI response.",
            "recommended_actions": ["Manual review required"],
            "error": True,
        }
    except Exception as e:
        return {
            "generated_note": "",
            "flagged_terms": [],
            "risk_weight": 0.0,
            "severity": "low",
            "reasoning": f"Analysis unavailable: {type(e).__name__}: {str(e)}",
            "recommended_actions": [],
            "error": True,
        }