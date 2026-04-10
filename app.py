"""
PDI Flask API — Nurse authentication, pre-seeded patients, UMLS note generation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv(override=True)
from engine import run_full_assessment, THRESHOLDS
from nlp import generate_nursing_note, analyze_nursing_note


from db import (
    ping, seed_nurses, seed_patients,
    authenticate_nurse, get_nurse,
    get_patient, update_patient_meta,
    append_vitals, append_note,
    list_patients_summary,
    create_session, resolve_session, delete_session,
)
from datetime import datetime

app = Flask(__name__)
CORS(app, supports_credentials=True)


# ── Seed data — nurses and patients loaded at startup ─────────────────────────

SEED_NURSE_LIST = [
    {"name": "Nurse Amaka Obi",    "username": "amaka",   "password": "nurse123", "role": "nurse"},
    {"name": "Nurse Tunde Adeyemi","username": "tunde",   "password": "nurse123", "role": "nurse"},
    {"name": "Nurse Chioma Eze",   "username": "chioma",  "password": "nurse123", "role": "nurse"},
    {"name": "Admin",              "username": "admin",   "password": "admin123", "role": "admin"},
]

SEED_PATIENT_LIST = [
    {"patient_id": "ICU-001", "name": "Adaeze Okonkwo",  "age": "45", "weight": "68kg", "ward": "ICU Bay A", "bed": "Bed 1", "diagnosis": "Post-operative monitoring"},
    {"patient_id": "ICU-002", "name": "Emeka Nwosu",     "age": "62", "weight": "82kg", "ward": "ICU Bay A", "bed": "Bed 2", "diagnosis": "Respiratory failure"},
    {"patient_id": "ICU-003", "name": "Fatima Bello",    "age": "38", "weight": "61kg", "ward": "ICU Bay A", "bed": "Bed 3", "diagnosis": "Haemodynamic instability"},
    {"patient_id": "ICU-004", "name": "Kolade Martins",  "age": "55", "weight": "90kg", "ward": "ICU Bay B", "bed": "Bed 1", "diagnosis": "Cardiac monitoring"},
    {"patient_id": "ICU-005", "name": "Ngozi Adeleke",   "age": "71", "weight": "58kg", "ward": "ICU Bay B", "bed": "Bed 2", "diagnosis": "Post-resuscitation care"},
    {"patient_id": "ICU-006", "name": "Seun Dada",       "age": "29", "weight": "75kg", "ward": "ICU Bay B", "bed": "Bed 3", "diagnosis": "Trauma observation"},
]

with app.app_context():
    seed_nurses(SEED_NURSE_LIST)
    seed_patients(SEED_PATIENT_LIST)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _auth(req):
    """Resolve Bearer token → nurse_id. Returns (nurse_id, None) or (None, error_response)."""
    header = req.headers.get("Authorization", "")
    token  = header.removeprefix("Bearer ").strip()
    nid    = resolve_session(token)
    if not nid:
        return None, (jsonify({"error": "Unauthorized — please log in"}), 401)
    return nid, None


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "PDI Backend", "version": "3.0.0", "mongodb": ping()})


# ── Nurse auth ────────────────────────────────────────────────────────────────

@app.route("/auth/login", methods=["POST"])
def login():
    """
    Nurse login by password only — username is inferred from the app context.
    Body: { "username": "amaka", "password": "nurse123" }
    Returns: { "token": "...", "nurse": { name, role, nurse_id } }
    """
    data     = request.json or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    nurse = authenticate_nurse(username, password)
    if not nurse:
        return jsonify({"error": "Incorrect password — please try again"}), 401

    token = create_session(nurse["nurse_id"])
    return jsonify({
        "token": token,
        "nurse": {
            "nurse_id": nurse["nurse_id"],
            "name":     nurse["name"],
            "role":     nurse["role"],
            "username": nurse["username"],
        }
    })


@app.route("/auth/logout", methods=["POST"])
def logout():
    header = request.headers.get("Authorization", "")
    token  = header.removeprefix("Bearer ").strip()
    if token:
        delete_session(token)
    return jsonify({"message": "Logged out"})


@app.route("/auth/me", methods=["GET"])
def me():
    """Validate token and return nurse info."""
    nurse_id, err = _auth(request)
    if err:
        return err
    nurse = get_nurse(nurse_id)
    if not nurse:
        return jsonify({"error": "Nurse not found"}), 404
    return jsonify({"nurse": nurse})


# ── Ward census — all patients ────────────────────────────────────────────────

@app.route("/patients", methods=["GET"])
def list_patients():
    """
    Return all pre-seeded patients with their latest risk assessment.
    Used by the Ward Census screen — nurses click a card to open a patient.
    """
    nurse_id, err = _auth(request)
    if err:
        return err

    patients = list_patients_summary()
    result = []
    for p in patients:
        assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
        result.append({
            "patient_id":    p["patient_id"],
            "name":          p["name"],
            "age":           p.get("age", ""),
            "ward":          p.get("ward", ""),
            "bed":           p.get("bed", ""),
            "diagnosis":     p.get("diagnosis", ""),
            "admitted":      p.get("admitted", ""),
            "latest_vitals": {k: v[-1] for k, v in p["vitals"].items() if v},
            "has_vitals":    any(v for v in p["vitals"].values()),
            "pdi_score":     assessment["pdi"]["score"],
            "risk_level":    assessment["pdi"]["risk_level"],
            "alert":         assessment["alert"],
        })
    # Sort by risk level — critical first
    order = {"crit": 0, "warn": 1, "ok": 2}
    result.sort(key=lambda x: order.get(x["risk_level"], 3))
    return jsonify(result)


# ── Individual patient ────────────────────────────────────────────────────────

@app.route("/patients/<patient_id>", methods=["GET"])
def get_patient_data(patient_id):
    """Full patient record + current assessment for the selected patient."""
    nurse_id, err = _auth(request)
    if err:
        return err

    p = get_patient(patient_id)
    if not p:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
    return jsonify({
        "patient":    {
            "patient_id": p["patient_id"], "name": p["name"],
            "age": p.get("age",""), "weight": p.get("weight",""),
            "ward": p.get("ward",""), "bed": p.get("bed",""),
            "diagnosis": p.get("diagnosis",""), "admitted": p.get("admitted",""),
        },
        "vitals":     p["vitals"],
        "times":      p["times"],
        "notes":      p.get("notes", []),
        "ai_weight":  p.get("ai_weight", 0.0),
        "assessment": assessment,
    })


@app.route("/patients/<patient_id>/meta", methods=["PATCH"])
def update_meta(patient_id):
    nurse_id, err = _auth(request)
    if err:
        return err
    update_patient_meta(patient_id, request.json or {})
    return jsonify({"message": "Updated"})


# ── Vitals logging ────────────────────────────────────────────────────────────

@app.route("/patients/<patient_id>/vitals", methods=["POST"])
def log_vitals(patient_id):
    """
    Log a new set of vital signs for the specified patient.
    Body: { "time": "16:00", "hr": 95, "rr": 20, "spo2": 93, "temp": 37.8, "sbp": 128, "dbp": 84 }
    """
    nurse_id, err = _auth(request)
    if err:
        return err

    p = get_patient(patient_id)
    if not p:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    data   = request.json or {}
    errors = []
    vitals = {}

    for key in ["hr", "rr", "spo2", "temp"]:
        val = data.get(key)
        if val is None:
            errors.append(f"Missing: {key}")
            continue
        try:
            vitals[key] = float(val)
        except (TypeError, ValueError):
            errors.append(f"Invalid value for {key}: {val}")

    for key in ["sbp", "dbp"]:
        val = data.get(key)
        if val is not None:
            try:
                vitals[key] = float(val)
            except (TypeError, ValueError):
                errors.append(f"Invalid value for {key}: {val}")

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    time_label = data.get("time") or datetime.now().strftime("%H:%M")
    append_vitals(patient_id, vitals, time_label)

    p = get_patient(patient_id)
    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))
    return jsonify({"message": "Vitals logged", "time": time_label, "assessment": assessment})


# ── UMLS nursing note generation ──────────────────────────────────────────────

@app.route("/patients/<patient_id>/generate-note", methods=["POST"])
def generate_note(patient_id):
    """
    Auto-generate a UMLS-grounded nursing note from the patient's current vitals.
    No body required — uses stored vitals data.
    Optionally accepts: { "additional_observations": "..." } for supplementary context.
    """
    nurse_id, err = _auth(request)
    if err:
        return err

    p = get_patient(patient_id)
    if not p:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    if not any(v for v in p["vitals"].values()):
        return jsonify({"error": "No vitals recorded yet — log vitals before generating a note"}), 400

    extra = (request.json or {}).get("additional_observations", "")
    assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))

    patient_meta = {
        "name":    p["name"],
        "age":     p.get("age",""),
        "weight":  p.get("weight",""),
        "ward":    p.get("ward",""),
        "bed":     p.get("bed",""),
        "diagnosis": p.get("diagnosis",""),
    }

    result = generate_nursing_note(p["vitals"], assessment, patient_meta)

    # Append extra observations into the note if provided
    if extra and result.get("generated_note"):
        result["generated_note"] += f"\n\nAdditional nurse observations: {extra}"

    append_note(patient_id, result.get("generated_note", ""), result)
    p = get_patient(patient_id)
    updated_assessment = run_full_assessment(p["vitals"], p.get("ai_weight", 0.0))

    return jsonify({
        "note":               result,
        "updated_assessment": updated_assessment,
    })


@app.route("/patients/<patient_id>/assess", methods=["GET"])
def assess(patient_id):
    nurse_id, err = _auth(request)
    if err:
        return err
    p = get_patient(patient_id)
    if not p:
        return jsonify({"error": "Not found"}), 404
    return jsonify(run_full_assessment(p["vitals"], p.get("ai_weight", 0.0)))


@app.route("/thresholds", methods=["GET"])
def get_thresholds():
    return jsonify(THRESHOLDS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)