import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "invoice_agent.db"

_db_lock = threading.RLock()


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_json(value):
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def get_db():
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _db_lock:
        conn = get_db()

        conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                status TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                UNIQUE(principal, message_id)
            );

            CREATE TABLE IF NOT EXISTS proposals (
                task_id TEXT NOT NULL,
                package_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                proposal_hash TEXT NOT NULL,

                PRIMARY KEY(task_id, package_id),
                UNIQUE(task_id, action_id)
            );

            CREATE TABLE IF NOT EXISTS package_cache (
                package_hash TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,

                PRIMARY KEY(principal, message_id)
            );
            """
        )

        conn.commit()
        conn.close()


def create_task(
    principal,
    message_id,
    message_hash,
    batch_id,
    context_id,
    task_json,
    created_at,
):
    task_id = "task-" + uuid.uuid4().hex

    with _db_lock:
        conn = get_db()

        conn.execute(
            """
            INSERT INTO tasks (
                task_id,
                context_id,
                principal,
                message_id,
                message_hash,
                batch_id,
                status,
                task_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                context_id,
                principal,
                message_id,
                message_hash,
                batch_id,
                "TASK_STATE_INPUT_REQUIRED",
                canonical_json(task_json),
                created_at,
                created_at,
            ),
        )

        conn.execute(
            """
            INSERT INTO messages (
                principal,
                message_id,
                message_hash,
                task_id
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                principal,
                message_id,
                message_hash,
                task_id,
            ),
        )

        conn.commit()
        conn.close()

    return task_id


def get_message(principal, message_id):
    with _db_lock:
        conn = get_db()

        row = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE principal = ?
              AND message_id = ?
            """,
            (principal, message_id),
        ).fetchone()

        conn.close()

    return dict(row) if row else None


def get_task(task_id):
    with _db_lock:
        conn = get_db()

        row = conn.execute(
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()

        conn.close()

    if not row:
        return None

    result = dict(row)
    result["task"] = json.loads(result.pop("task_json"))
    return result


def get_task_for_principal(task_id, principal):
    with _db_lock:
        conn = get_db()

        row = conn.execute(
            """
            SELECT *
            FROM tasks
            WHERE task_id = ?
              AND principal = ?
            """,
            (task_id, principal),
        ).fetchone()

        conn.close()

    if not row:
        return None

    result = dict(row)
    result["task"] = json.loads(result.pop("task_json"))
    return result


def list_tasks(principal):
    with _db_lock:
        conn = get_db()

        rows = conn.execute(
            """
            SELECT task_json
            FROM tasks
            WHERE principal = ?
            ORDER BY created_at ASC
            """,
            (principal,),
        ).fetchall()

        conn.close()

    return [
        json.loads(row["task_json"])
        for row in rows
    ]


def update_task(task_id, status, task_json, updated_at):
    with _db_lock:
        conn = get_db()

        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = ?,
                task_json = ?,
                updated_at = ?
            WHERE task_id = ?
              AND status NOT IN (
                  'TASK_STATE_COMPLETED',
                  'TASK_STATE_CANCELED'
              )
            """,
            (
                status,
                canonical_json(task_json),
                updated_at,
                task_id,
            ),
        )

        conn.commit()
        changed = cursor.rowcount == 1
        conn.close()

    return changed


def insert_proposal(
    task_id,
    package_id,
    action_id,
    proposal,
):
    proposal_hash = sha256_json(proposal)

    with _db_lock:
        conn = get_db()

        conn.execute(
            """
            INSERT OR IGNORE INTO proposals (
                task_id,
                package_id,
                action_id,
                proposal_json,
                proposal_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                task_id,
                package_id,
                action_id,
                canonical_json(proposal),
                proposal_hash,
            ),
        )

        conn.commit()
        conn.close()

    return proposal_hash


def get_proposals(task_id):
    with _db_lock:
        conn = get_db()

        rows = conn.execute(
            """
            SELECT *
            FROM proposals
            WHERE task_id = ?
            ORDER BY rowid
            """,
            (task_id,),
        ).fetchall()

        conn.close()

    return [
        {
            "package_id": row["package_id"],
            "action_id": row["action_id"],
            "proposal": json.loads(row["proposal_json"]),
            "proposal_hash": row["proposal_hash"],
        }
        for row in rows
    ]


def get_cached_decision(package_hash):
    with _db_lock:
        conn = get_db()

        row = conn.execute(
            """
            SELECT decision_json
            FROM package_cache
            WHERE package_hash = ?
            """,
            (package_hash,),
        ).fetchone()

        conn.close()

    if not row:
        return None

    return json.loads(row["decision_json"])


def save_cached_decision(package_hash, decision):
    with _db_lock:
        conn = get_db()

        conn.execute(
            """
            INSERT OR IGNORE INTO package_cache (
                package_hash,
                decision_json
            )
            VALUES (?, ?)
            """,
            (
                package_hash,
                canonical_json(decision),
            ),
        )

        conn.commit()
        conn.close()
