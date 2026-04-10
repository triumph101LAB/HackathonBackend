"""
PDI Database Layer — MongoDB Atlas
"""

import os, uuid, hashlib, hmac, time
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME   = os.environ.get("MONGO_DB", "pdi")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set — rename _env to .env or check your environment variables")

_client = None
_db     = None


def get_db():
    global _client, _db
    if _db is None:
        if not MONGO_URI:
            raise RuntimeError("MONGO_URI is not set — check your .env file")

        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=15000,   # 15s — Atlas needs time on cold start
            connectTimeoutMS=15000,
            socketTimeoutMS=30000,
            retryWrites=True,
        )

        # Force a real connection immediately so we catch errors early
        _client.admin.command("ping")
        print(f"[db] Connected to MongoDB Atlas — database: {DB_NAME}")

        _db = _client[DB_NAME]
        _ensure_indexes(_db)
    return _db


def ping() -> bool:
    try:
        get_db().command("ping")
        return True
    except Exception as e:
        print(f"[db] ping failed: {e}")
        return False


def _ensure_indexes(db):
    db.nurses.create_index([("username_lower", ASCENDING)], unique=True)
    db.patients.create_index([("patient_id", ASCENDING)], unique=True)
    db.sessions.create_index([("token", ASCENDING)], unique=True)
    db.sessions.create_index([("nurse_id", ASCENDING)])
    print("[db] Indexes ensured")


# ── Password hashing ──────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        return hmac.compare_digest(
            hmac.new(salt.encode(), password.encode(), hashlib.sha256).hexdigest(), h
        )
    except Exception:
        return False


# ── Nurse management ──────────────────────────────────────────────────────────

def seed_nurses(nurses: list[dict]):
    """Seed nurse accounts. Safe to call every startup — skips existing."""
    db = get_db()
    seeded = 0
    for n in nurses:
        try:
            db.nurses.insert_one({
                "nurse_id":       f"N-{str(uuid.uuid4())[:6].upper()}",
                "name":           n["name"],
                "username":       n["username"],
                "username_lower": n["username"].lower(),
                "password_hash":  _hash_password(n["password"]),
                "role":           n.get("role", "nurse"),
                "created_at":     datetime.now(timezone.utc).isoformat(),
            })
            seeded += 1
        except DuplicateKeyError:
            pass
    if seeded:
        print(f"[db] Seeded {seeded} new nurses")
    else:
        print(f"[db] Nurses already seeded ({db.nurses.count_documents({})} in DB)")


def authenticate_nurse(username: str, password: str) -> dict | None:
    db = get_db()
    doc = db.nurses.find_one({"username_lower": username.strip().lower()}, {"_id": 0})
    if not doc:
        return None
    if not _verify_password(password, doc["password_hash"]):
        return None
    return {k: v for k, v in doc.items() if k != "password_hash"}


def get_nurse(nurse_id: str) -> dict | None:
    db = get_db()
    return db.nurses.find_one({"nurse_id": nurse_id}, {"_id": 0, "password_hash": 0})


# ── Patient management ────────────────────────────────────────────────────────

def seed_patients(patients: list[dict]):
    """Seed patient records. Safe to call every startup — skips existing."""
    db = get_db()
    seeded = 0
    for p in patients:
        try:
            db.patients.insert_one({
                "patient_id":  p["patient_id"],
                "name":        p["name"],
                "age":         p.get("age", ""),
                "weight":      p.get("weight", ""),
                "ward":        p.get("ward", "ICU Bay A"),
                "bed":         p.get("bed", ""),
                "diagnosis":   p.get("diagnosis", "Under observation"),
                "admitted":    p.get("admitted", datetime.now(timezone.utc).isoformat()),
                "vitals":      {"hr":[], "rr":[], "spo2":[], "temp":[], "sbp":[], "dbp":[]},
                "times":       [],
                "notes":       [],
                "ai_weight":   0.0,
                "created_at":  datetime.now(timezone.utc).isoformat(),
            })
            seeded += 1
        except DuplicateKeyError:
            pass
    if seeded:
        print(f"[db] Seeded {seeded} new patients")
    else:
        print(f"[db] Patients already seeded ({db.patients.count_documents({})} in DB)")


def get_patient(patient_id: str) -> dict | None:
    db = get_db()
    return db.patients.find_one({"patient_id": patient_id}, {"_id": 0})


def update_patient_meta(patient_id: str, fields: dict):
    allowed = {"age", "weight", "bed", "ward", "diagnosis"}
    safe = {k: v for k, v in fields.items() if k in allowed}
    if safe:
        get_db().patients.update_one({"patient_id": patient_id}, {"$set": safe})


def append_vitals(patient_id: str, vitals: dict, time_label: str) -> bool:
    push_ops = {f"vitals.{k}": v for k, v in vitals.items()}
    push_ops["times"] = time_label
    result = get_db().patients.update_one({"patient_id": patient_id}, {"$push": push_ops})
    return result.modified_count > 0


def append_note(patient_id: str, note_text: str, analysis: dict):
    entry = {
        "time":     datetime.now(timezone.utc).strftime("%H:%M"),
        "date":     datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "text":     note_text,
        "analysis": analysis,
    }
    get_db().patients.update_one(
        {"patient_id": patient_id},
        {"$push": {"notes": entry}, "$set": {"ai_weight": analysis.get("risk_weight", 0.0)}}
    )


def list_patients_summary() -> list:
    db = get_db()
    return list(db.patients.find(
        {},
        {"_id": 0, "patient_id": 1, "name": 1, "age": 1, "ward": 1, "bed": 1,
         "diagnosis": 1, "admitted": 1, "vitals": 1, "times": 1, "ai_weight": 1}
    ))


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(nurse_id: str) -> str:
    token = str(uuid.uuid4())
    get_db().sessions.insert_one({
        "token":      token,
        "nurse_id":   nurse_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return token


def resolve_session(token: str) -> str | None:
    if not token:
        return None
    doc = get_db().sessions.find_one({"token": token}, {"_id": 0, "nurse_id": 1})
    return doc["nurse_id"] if doc else None


def delete_session(token: str):
    get_db().sessions.delete_one({"token": token})