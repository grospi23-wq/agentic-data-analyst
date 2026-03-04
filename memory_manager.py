"""
memory_manager.py
-----------------
Async SQLite persistence layer for the analyst memory store.

All reads and writes are keyed by schema_hash, so files with the same
structural signature (sheet + column names) share a memory family,
regardless of filename or row-level changes across quarterly refreshes.

Database layout (single file: analyst_memory.db)
─────────────────────────────────────────────────
  TABLE join_records      — one row per validated join config
  TABLE quality_warnings  — one row per unique finding type/column pair
  TABLE run_log           — lightweight counter + last_run_at per hash
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, List, Optional

import aiosqlite

from memory_schema import AnalystMemoryStore, JoinMemoryRecord, QualityWarning

logger = logging.getLogger(__name__)

# Default path — override by passing db_path to MemoryManager()
DEFAULT_DB_PATH = Path("analyst_memory.db")

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS join_records (
    schema_hash             TEXT    NOT NULL,
    left_sheet              TEXT    NOT NULL,
    right_sheet             TEXT    NOT NULL,
    on_column               TEXT    NOT NULL,
    join_type               TEXT    NOT NULL,
    result_name             TEXT    NOT NULL,
    row_count_after_join    INTEGER NOT NULL DEFAULT 0,
    critic_score            REAL    NOT NULL DEFAULT 0.0,
    created_at              TEXT    NOT NULL,
    last_seen_at            TEXT    NOT NULL,
    PRIMARY KEY (schema_hash, left_sheet, right_sheet, on_column, join_type)
);

CREATE TABLE IF NOT EXISTS quality_warnings (
    schema_hash                  TEXT    NOT NULL,
    finding_type                 TEXT    NOT NULL,
    affected_column_or_insight   TEXT    NOT NULL,
    reason                       TEXT    NOT NULL,
    severity                     TEXT    NOT NULL,
    times_seen                   INTEGER NOT NULL DEFAULT 1,
    first_seen_at                TEXT    NOT NULL,
    last_seen_at                 TEXT    NOT NULL,
    PRIMARY KEY (schema_hash, finding_type, affected_column_or_insight)
);

CREATE TABLE IF NOT EXISTS run_log (
    schema_hash  TEXT    PRIMARY KEY,
    total_runs   INTEGER NOT NULL DEFAULT 0,
    last_run_at  TEXT    NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Helper: ISO-8601 UTC timestamps
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Thin async wrapper around aiosqlite.

    Usage (typical mission flow)
    ────────────────────────────
    manager = MemoryManager()
    await manager.init()

    # Load context before the CDO planning phase
    store = await manager.load(schema_hash)

    # ... run agents ...

    # Persist successful joins + critic warnings at mission end
    await manager.persist(store, new_joins, new_warnings, critic_approved=True)

    await manager.close()
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the connection and run DDL (idempotent)."""
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        logger.debug("MemoryManager initialised — db=%s", self._db_path)

    async def close(self) -> None:
        """Close the underlying connection gracefully."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator["MemoryManager"]:
        """Async context manager that handles init/close automatically."""
        await self.init()
        try:
            yield self
        finally:
            await self.close()

    # ------------------------------------------------------------------
    # Internal guard
    # ------------------------------------------------------------------

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryManager not initialised — call await manager.init() first.")
        return self._conn

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    async def load(self, schema_hash: str) -> AnalystMemoryStore:
        """
        Load the full memory store for a schema family.
        Returns an empty store (not None) if this is the first run.
        """
        conn = self._require_conn()
        store = AnalystMemoryStore(schema_hash=schema_hash)

        # --- join_records ---
        async with conn.execute(
            "SELECT * FROM join_records WHERE schema_hash = ?", (schema_hash,)
        ) as cur:
            async for row in cur:
                store.successful_joins.append(
                    JoinMemoryRecord(
                        schema_hash=row["schema_hash"],
                        left_sheet=row["left_sheet"],
                        right_sheet=row["right_sheet"],
                        on_column=row["on_column"],
                        join_type=row["join_type"],
                        result_name=row["result_name"],
                        row_count_after_join=row["row_count_after_join"],
                        critic_score=row["critic_score"],
                        created_at=_parse_dt(row["created_at"]),
                        last_seen_at=_parse_dt(row["last_seen_at"]),
                    )
                )

        # --- quality_warnings ---
        async with conn.execute(
            "SELECT * FROM quality_warnings WHERE schema_hash = ?", (schema_hash,)
        ) as cur:
            async for row in cur:
                store.quality_warnings.append(
                    QualityWarning(
                        schema_hash=row["schema_hash"],
                        finding_type=row["finding_type"],
                        affected_column_or_insight=row["affected_column_or_insight"],
                        reason=row["reason"],
                        severity=row["severity"],
                        times_seen=row["times_seen"],
                        first_seen_at=_parse_dt(row["first_seen_at"]),
                        last_seen_at=_parse_dt(row["last_seen_at"]),
                    )
                )

        # --- run_log ---
        async with conn.execute(
            "SELECT total_runs, last_run_at FROM run_log WHERE schema_hash = ?",
            (schema_hash,),
        ) as cur:
            row = await cur.fetchone()
            if row:
                store.total_runs = row["total_runs"]
                store.last_run_at = _parse_dt(row["last_run_at"])

        logger.info(
            "Memory loaded — hash=%s joins=%d warnings=%d runs=%d",
            schema_hash,
            len(store.successful_joins),
            len(store.quality_warnings),
            store.total_runs,
        )
        return store

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    async def persist(
        self,
        store: AnalystMemoryStore,
        new_joins: List[JoinMemoryRecord],
        new_warnings: List[QualityWarning],
        *,
        critic_approved: bool,
    ) -> None:
        """
        Persist new data back to SQLite.
        Manually commits the transaction to avoid thread restart collisions.
        """
        conn = self._require_conn()
        now = _now_iso()

        # --- upsert join_records ---
        for j in new_joins:
            await conn.execute(
                """
                INSERT INTO join_records
                    (schema_hash, left_sheet, right_sheet, on_column,
                     join_type, result_name, row_count_after_join,
                     critic_score, created_at, last_seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(schema_hash, left_sheet, right_sheet, on_column, join_type)
                DO UPDATE SET
                    row_count_after_join = excluded.row_count_after_join,
                    critic_score         = excluded.critic_score,
                    last_seen_at         = excluded.last_seen_at
                """,
                (
                    store.schema_hash,
                    j.left_sheet,
                    j.right_sheet,
                    j.on_column,
                    j.join_type,
                    j.result_name,
                    j.row_count_after_join,
                    j.critic_score,
                    j.created_at.isoformat(),
                    now,
                ),
            )

        # --- upsert quality_warnings ---
        warnings_to_persist = [
            w for w in new_warnings
            if w.severity == "warn" or (w.severity == "block" and critic_approved)
        ]
        for w in warnings_to_persist:
            await conn.execute(
                """
                INSERT INTO quality_warnings
                    (schema_hash, finding_type, affected_column_or_insight,
                     reason, severity, times_seen, first_seen_at, last_seen_at)
                VALUES (?,?,?,?,?,1,?,?)
                ON CONFLICT(schema_hash, finding_type, affected_column_or_insight)
                DO UPDATE SET
                    times_seen   = times_seen + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    store.schema_hash,
                    w.finding_type,
                    w.affected_column_or_insight,
                    w.reason,
                    w.severity,
                    now,
                    now,
                ),
            )

        # --- increment run_log ---
        await conn.execute(
            """
            INSERT INTO run_log (schema_hash, total_runs, last_run_at)
            VALUES (?, 1, ?)
            ON CONFLICT(schema_hash)
            DO UPDATE SET
                total_runs  = total_runs + 1,
                last_run_at = excluded.last_run_at
            """,
            (store.schema_hash, now),
        )

        # Explicitly commit the transaction
        await conn.commit()

        logger.info(
            "Memory persisted - hash=%s new_joins=%d new_warnings=%d approved=%s",
            store.schema_hash,
            len(new_joins),
            len(warnings_to_persist),
            critic_approved,
        )

    # ------------------------------------------------------------------
    # HOUSEKEEPING
    # ------------------------------------------------------------------

    async def prune_stale_joins(
        self, schema_hash: str, older_than_days: int = 365
    ) -> int:
        """
        Remove join records not seen in `older_than_days` days.
        Returns the number of rows deleted.
        """
        conn = self._require_conn()
        cutoff = datetime.now(timezone.utc).replace(
            year=datetime.now(timezone.utc).year - (older_than_days // 365)
        ).isoformat()

        cur = await conn.execute(
            "DELETE FROM join_records WHERE schema_hash = ? AND last_seen_at < ?",
            (schema_hash, cutoff),
        )
        deleted = cur.rowcount
        
        # Explicitly commit the transaction
        await conn.commit()

        logger.info("Pruned %d stale join records for hash=%s", deleted, schema_hash)
        return deleted
