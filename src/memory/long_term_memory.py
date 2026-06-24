import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


MEMORY_DB_PATH = os.getenv("LONG_TERM_MEMORY_DB_PATH", "long_term_memory.db")


def get_memory_db_path() -> str:
    """
    Return long-term memory SQLite database path.
    """

    return MEMORY_DB_PATH


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    """
    Add a column if it does not exist.
    """

    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_columns = {row[1] for row in rows}

    if column_name not in existing_columns:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def init_long_term_memory_table() -> None:
    """
    Initialize long-term memory table.

    Compatible with old table schema.
    """

    db_path = Path(MEMORY_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Compatible migration for old database.
        _ensure_column(
            conn=conn,
            table_name="long_term_memories",
            column_name="source_text",
            column_type="TEXT",
        )

        # Use unique index instead of UNIQUE in CREATE TABLE,
        # because old tables will not be recreated automatically.
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_long_term_memory_user_content
            ON long_term_memories(user_id, content)
            """
        )

        conn.commit()


def create_memory(
    user_id: str,
    content: str,
    memory_type: str = "note",
    source_text: str | None = None,
) -> dict[str, Any] | None:
    """
    Create a long-term memory item.

    If the same memory already exists for the user, return the existing item.
    """

    init_long_term_memory_table()

    user_id = user_id.strip() or "default"
    memory_type = memory_type.strip() or "note"
    content = content.strip()

    if not content:
        return None

    now = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                """
                INSERT INTO long_term_memories (
                    user_id,
                    memory_type,
                    content,
                    source_text,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, memory_type, content, source_text, now, now),
            )
            conn.commit()

            memory_id = cursor.lastrowid

            return {
                "id": memory_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "content": content,
                "source_text": source_text,
                "created_at": now,
                "updated_at": now,
            }

        except sqlite3.IntegrityError:
            row = conn.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_type,
                    content,
                    source_text,
                    created_at,
                    updated_at
                FROM long_term_memories
                WHERE user_id = ? AND content = ?
                """,
                (user_id, content),
            ).fetchone()

            return dict(row) if row else None


def list_memories(user_id: str = "default") -> list[dict[str, Any]]:
    """
    List long-term memories for a user.
    """

    init_long_term_memory_table()

    user_id = user_id.strip() or "default"

    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                memory_type,
                content,
                source_text,
                created_at,
                updated_at
            FROM long_term_memories
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_memory(memory_id: int, user_id: str = "default") -> bool:
    """
    Delete a memory item.
    """

    init_long_term_memory_table()

    user_id = user_id.strip() or "default"

    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        cursor = conn.execute(
            """
            DELETE FROM long_term_memories
            WHERE id = ? AND user_id = ?
            """,
            (memory_id, user_id),
        )
        conn.commit()

        return cursor.rowcount > 0


def format_memories_for_prompt(
    user_id: str = "default",
    limit: int = 8,
) -> str:
    """
    Format long-term memories as prompt context.
    """

    memories = list_memories(user_id=user_id)

    if not memories:
        return ""

    selected_memories = memories[:limit]

    lines = []

    for index, memory in enumerate(selected_memories, start=1):
        memory_type = memory.get("memory_type", "memory")
        content = memory.get("content", "")

        lines.append(f"{index}. [{memory_type}] {content}")

    return "\n".join(lines)


def get_memory_stats(user_id: str = "default") -> dict[str, Any]:
    """
    Return long-term memory statistics.
    """

    init_long_term_memory_table()

    memories = list_memories(user_id=user_id)

    return {
        "db_path": MEMORY_DB_PATH,
        "user_id": user_id,
        "memory_count": len(memories),
    }
