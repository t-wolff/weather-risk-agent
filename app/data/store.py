"""All database access. Moving to another database should only touch this file."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import db_path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())
    return conn


def replace_nri(conn, rows: list[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM nri_hazards")
        conn.executemany(
            "INSERT INTO nri_hazards VALUES (:hub_id, :nri_field, :annual_frequency, :loss_ratio_rating)",
            rows,
        )


def nri_for_hub(conn, hub_id: str) -> dict[str, dict]:
    rows = conn.execute("SELECT * FROM nri_hazards WHERE hub_id = ?", (hub_id,))
    return {
        r["nri_field"]: {
            "annual_frequency": r["annual_frequency"],
            "loss_ratio_rating": r["loss_ratio_rating"],
        }
        for r in rows
    }


def cached_year(conn, hub_id: str, year: int) -> dict | None:
    row = conn.execute(
        "SELECT * FROM weather_cache WHERE hub_id = ? AND year = ?", (hub_id, year)
    ).fetchone()
    if not row:
        return None
    return {
        "days": json.loads(row["days_json"]),
        "complete": bool(row["complete"]),
        "fetched_at": row["fetched_at"],
    }


def save_year(conn, hub_id: str, year: int, days: list[dict], complete: bool) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO weather_cache VALUES (?, ?, ?, ?, ?)",
            (hub_id, year, json.dumps(days), int(complete), now()),
        )


def replace_declarations(conn, hub_id: str, rows: list[dict]) -> None:
    with conn:
        conn.execute("DELETE FROM declarations WHERE hub_id = ?", (hub_id,))
        conn.executemany(
            "INSERT OR IGNORE INTO declarations VALUES "
            "(:hub_id, :disaster_number, :declaration_type, :incident_type, :title, :declaration_date)",
            rows,
        )


def hurricane_declarations(conn, hub_id: str, start_year: int, end_year: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM declarations WHERE hub_id = ? AND incident_type = 'Hurricane' "
        "AND substr(declaration_date, 1, 4) BETWEEN ? AND ? ORDER BY declaration_date",
        (hub_id, str(start_year), str(end_year)),
    )
    return [dict(r) for r in rows]


def save_snapshot(conn, score: dict, as_of: str, method_version: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO score_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            (score["hub_id"], as_of, method_version, score["edd"], score["tier"], json.dumps(score)),
        )


def latest_snapshots(conn, method_version: str) -> list[dict]:
    """Newest score per hub, for one scoring version."""
    rows = conn.execute(
        """SELECT s.breakdown_json, s.as_of FROM score_snapshots s
           JOIN (SELECT hub_id, MAX(as_of) AS as_of FROM score_snapshots
                 WHERE method_version = ? GROUP BY hub_id) m
             ON s.hub_id = m.hub_id AND s.as_of = m.as_of
           WHERE s.method_version = ?""",
        (method_version, method_version),
    )
    return [{**json.loads(r["breakdown_json"]), "as_of": r["as_of"]} for r in rows]


def previous_snapshot(conn, hub_id: str, as_of: str, method_version: str) -> dict | None:
    row = conn.execute(
        "SELECT edd, tier, as_of FROM score_snapshots "
        "WHERE hub_id = ? AND as_of < ? AND method_version = ? ORDER BY as_of DESC LIMIT 1",
        (hub_id, as_of, method_version),
    ).fetchone()
    return dict(row) if row else None


def add_alert(conn, alert: dict) -> bool:
    """Returns False when this hub already has an alert for that date."""
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO alerts (hub_id, as_of, old_edd, new_edd, old_tier, new_tier, "
            "reason, delivery_status, created_at) VALUES (:hub_id, :as_of, :old_edd, :new_edd, "
            ":old_tier, :new_tier, :reason, :delivery_status, :created_at)",
            {**alert, "created_at": now()},
        )
    return cursor.rowcount > 0


def set_alert_delivery(conn, hub_id: str, as_of: str, status: str) -> None:
    with conn:
        conn.execute(
            "UPDATE alerts SET delivery_status = ? WHERE hub_id = ? AND as_of = ?",
            (status, hub_id, as_of),
        )


def list_alerts(conn, limit: int = 100) -> list[dict]:
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]


def client_exists(conn, client_hash: str) -> bool:
    row = conn.execute("SELECT 1 FROM clients WHERE client_hash = ?", (client_hash,))
    return row.fetchone() is not None


def create_client(conn, client_hash: str) -> bool:
    """Returns False when the ID is already taken."""
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO clients VALUES (?, ?)", (client_hash, now())
        )
    return cursor.rowcount > 0


def create_conversation(conn, client_hash: str) -> str:
    conversation_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            "INSERT INTO conversations VALUES (?, ?, ?)", (conversation_id, client_hash, now())
        )
    return conversation_id


def owns_conversation(conn, client_hash: str, conversation_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ? AND client_hash = ?",
        (conversation_id, client_hash),
    )
    return row.fetchone() is not None


def list_conversations(conn, client_hash: str) -> list[dict]:
    rows = conn.execute(
        """SELECT c.id, c.created_at,
                  (SELECT user_message FROM turns t WHERE t.conversation_id = c.id
                   ORDER BY t.id LIMIT 1) AS first_message
           FROM conversations c WHERE c.client_hash = ? ORDER BY c.created_at DESC""",
        (client_hash,),
    )
    return [dict(r) for r in rows]


def conversation_turns(conn, client_hash: str, conversation_id: str) -> list[dict]:
    """Empty unless this client owns the conversation."""
    rows = conn.execute(
        """SELECT t.* FROM turns t JOIN conversations c ON c.id = t.conversation_id
           WHERE t.conversation_id = ? AND c.client_hash = ? ORDER BY t.id""",
        (conversation_id, client_hash),
    )
    return [
        {
            "user_message": r["user_message"],
            "answer": json.loads(r["answer_json"]),
            "trace": json.loads(r["trace_json"]),
            "llm_messages": json.loads(r["llm_messages_json"]),
            "flag": r["flag"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def add_turn(conn, turn: dict) -> None:
    with conn:
        conn.execute(
            "INSERT INTO turns (conversation_id, user_message, answer_json, trace_json, "
            "llm_messages_json, flag, flag_reason, flag_source, ip_hash, created_at) VALUES "
            "(:conversation_id, :user_message, :answer_json, :trace_json, :llm_messages_json, "
            ":flag, :flag_reason, :flag_source, :ip_hash, :created_at)",
            {
                **turn,
                "answer_json": json.dumps(turn["answer"]),
                "trace_json": json.dumps(turn["trace"]),
                "llm_messages_json": json.dumps(turn["llm_messages"]),
                "created_at": now(),
            },
        )


def flagged_turns(conn, flag: str | None = None, limit: int = 200) -> list[dict]:
    query = (
        "SELECT conversation_id, user_message, flag, flag_reason, flag_source, ip_hash, created_at "
        "FROM turns WHERE flag != 'none'"
    )
    params: list = []
    if flag:
        query += " AND flag = ?"
        params.append(flag)
    rows = conn.execute(query + " ORDER BY id DESC LIMIT ?", (*params, limit))
    return [dict(r) for r in rows]
