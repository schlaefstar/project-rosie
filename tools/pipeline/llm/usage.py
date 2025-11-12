"""Utilities for tracking LLM usage and costs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping


@dataclass
class UsageRecord:
    """Structured record persisted by the usage ledger."""

    provider: str
    model: str
    event_id: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_row(self) -> tuple[Any, ...]:
        return (
            self.timestamp.isoformat(),
            self.provider,
            self.model,
            self.event_id,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cost_usd,
            json.dumps(self.metadata, default=str),
        )


class UsageLedger:
    """Persists usage records to a local SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_records (
                    recorded_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    event_id TEXT,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    metadata TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_records_timestamp ON usage_records(recorded_at)"
            )

    def record(self, record: UsageRecord) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO usage_records (
                    recorded_at,
                    provider,
                    model,
                    event_id,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cost_usd,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record.as_row(),
            )

    def summarize_daily_costs(self) -> list[MutableMapping[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                """
                SELECT substr(recorded_at, 1, 10) AS day,
                       provider,
                       model,
                       SUM(cost_usd) AS total_cost,
                       SUM(total_tokens) AS total_tokens
                FROM usage_records
                GROUP BY day, provider, model
                ORDER BY day DESC, provider, model
                """
            )
            return [dict(row) for row in cursor.fetchall()]


__all__ = ["UsageLedger", "UsageRecord"]


