"""Diagnosis history: session-level retention plus optional SQLite persistence.

Persistence is OFF unless PLANTVERSE_PERSIST_HISTORY=1, because the app has no
user accounts: a shared database on a public deployment would expose one
person's records to everyone. Only metadata is stored - never the image itself
(just a SHA-256 hash as an optional reference).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(os.getenv("PLANTVERSE_DB_PATH", str(Path(__file__).parent / "data" / "plantverse.db")))
FIELDS = ("id", "created_at", "plant_label", "plant_score", "plant_status", "scientific_name",
          "disease_label", "disease_name", "disease_score", "disease_status", "health_status",
          "recommendations", "image_sha256")


def persistence_enabled() -> bool:
    return os.getenv("PLANTVERSE_PERSIST_HISTORY", "0").strip().lower() in {"1", "true", "yes"}


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS diagnoses (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            plant_label TEXT, plant_score REAL, plant_status TEXT, scientific_name TEXT,
            disease_label TEXT, disease_name TEXT, disease_score REAL, disease_status TEXT,
            health_status TEXT, recommendations TEXT, image_sha256 TEXT)"""
    )
    return conn


def build_record(plant: Optional[Dict[str, Any]], disease: Optional[Dict[str, Any]],
                 image_bytes: Optional[bytes] = None, scientific_name: Optional[str] = None) -> Dict[str, Any]:
    """Create a history record from the two analysis results (either may be missing/failed)."""
    plant, disease = plant or {}, disease or {}
    ok = ("success", "uncertain")
    return {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plant_label": plant.get("label") if plant.get("status") in ok else None,
        "plant_score": plant.get("score") if plant.get("status") in ok else None,
        "plant_status": plant.get("status", "not_run"),
        "scientific_name": scientific_name,
        "disease_label": disease.get("disease_name") if disease.get("status") in ok else None,
        "disease_name": disease.get("disease_name") if disease.get("status") in ok else None,
        "disease_score": disease.get("confidence") if disease.get("status") in ok else None,
        "disease_status": disease.get("status", "not_run"),
        "health_status": disease.get("health_status") if disease.get("status") in ok else None,
        "recommendations": list(disease.get("recommendations") or []),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest() if image_bytes else None,
    }


def save_record(record: Dict[str, Any], path: Optional[Path] = None) -> bool:
    """Persist a record. Returns True only if it was really written to disk."""
    if path is None and not persistence_enabled():
        return False
    try:
        with _connect(path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO diagnoses ({','.join(FIELDS)}) VALUES ({','.join('?' * len(FIELDS))})",
                [json.dumps(record[f]) if f == "recommendations" else record.get(f) for f in FIELDS],
            )
        return True
    except (sqlite3.Error, OSError, KeyError):
        return False


def list_records(limit: int = 100, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    if path is None and not persistence_enabled():
        return []
    try:
        with _connect(path) as conn:
            rows = conn.execute("SELECT * FROM diagnoses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    except (sqlite3.Error, OSError):
        return []
    out = []
    for row in rows:
        rec = dict(row)
        try:
            rec["recommendations"] = json.loads(rec.get("recommendations") or "[]")
        except ValueError:
            rec["recommendations"] = []
        out.append(rec)
    return out


def get_record(record_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    return next((r for r in list_records(1000, path) if r["id"] == record_id), None)


def delete_record(record_id: str, path: Optional[Path] = None) -> bool:
    if path is None and not persistence_enabled():
        return False
    try:
        with _connect(path) as conn:
            return conn.execute("DELETE FROM diagnoses WHERE id = ?", (record_id,)).rowcount > 0
    except (sqlite3.Error, OSError):
        return False


def export_report(record: Dict[str, Any], fmt: str = "txt") -> str:
    """Render a record as JSON or a plain-text report."""
    if fmt == "json":
        return json.dumps({k: record.get(k) for k in FIELDS}, indent=2)
    pct = lambda v: f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
    lines = [
        "PlantVerse Analysis Report", "=" * 26,
        f"ID: {record.get('id')}", f"Date (UTC): {record.get('created_at')}", "",
        f"Plant: {record.get('plant_label') or 'not identified'} "
        f"(status: {record.get('plant_status')}, confidence: {pct(record.get('plant_score'))})",
        f"Scientific name: {record.get('scientific_name') or 'not available'}",
        f"Health: {record.get('disease_name') or 'not determined'} "
        f"(status: {record.get('disease_status')}, confidence: {pct(record.get('disease_score'))})",
        "", "Suggested next steps:",
    ]
    lines += [f"  - {r}" for r in record.get("recommendations") or []] or ["  (none)"]
    lines += ["", "NOTE: This is an automated AI estimate, not a definitive diagnosis.",
              "Confirm with a local agricultural extension service or plant pathologist."]
    return "\n".join(lines)


def to_session_entry(record: Dict[str, Any], image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """Add the display keys used by the Plant Library history tab."""
    entry = dict(record)
    score = record.get("disease_score")
    entry.update(
        plant_name=record.get("plant_label") or "Unidentified plant",
        disease_name=record.get("disease_name") or "Health not determined",
        health_status=record.get("health_status") or "Unknown",
        confidence=score or 0.0,
        confidence_pct=round((score or 0.0) * 100, 1),
        pathogen="N/A",
        created_at=str(record.get("created_at", ""))[:16].replace("T", " ") + " UTC",
        image_bytes=image_bytes,
        is_sighting=False,
    )
    return entry
