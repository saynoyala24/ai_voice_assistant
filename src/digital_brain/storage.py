from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    location TEXT NOT NULL,
    perceived_objects TEXT NOT NULL,
    perceived_people TEXT NOT NULL,
    active_goal TEXT NOT NULL,
    action_chosen TEXT NOT NULL,
    predicted_outcome TEXT NOT NULL,
    actual_outcome TEXT NOT NULL,
    reward REAL NOT NULL,
    prediction_error REAL NOT NULL,
    emotional_state TEXT NOT NULL,
    importance REAL NOT NULL,
    compressed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS semantic_facts (
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (subject, relation, object)
);
CREATE TABLE IF NOT EXISTS emotional_associations (
    stimulus TEXT PRIMARY KEY,
    emotion TEXT NOT NULL,
    intensity REAL NOT NULL,
    valence REAL NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS procedural_skills (
    skill_name TEXT PRIMARY KEY,
    mastery REAL NOT NULL,
    practice_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS identity_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS world_beliefs (
    belief TEXT PRIMARY KEY,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    summary TEXT NOT NULL,
    improvement_goal TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    vector TEXT NOT NULL,
    metadata TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_json(value: str) -> Any:
    return json.loads(value)
