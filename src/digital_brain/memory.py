from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from digital_brain.models import CognitiveCycle, Perception, Stimulus, utc_now
from digital_brain.storage import connect, from_json, to_json


@dataclass(frozen=True)
class WorkingMemoryItem:
    content: str
    relevance: float
    created_at: str
    kind: str


class SensoryMemory:
    def __init__(self, capacity: int = 256) -> None:
        self._items: deque[Stimulus] = deque(maxlen=capacity)

    def store(self, stimulus: Stimulus) -> None:
        self._items.append(stimulus)

    def recent(self) -> list[Stimulus]:
        return list(self._items)


class WorkingMemory:
    def __init__(self, capacity: int = 9) -> None:
        self.capacity = capacity
        self._items: list[WorkingMemoryItem] = []

    def add(self, content: str, relevance: float, kind: str) -> None:
        self._items.append(WorkingMemoryItem(content, relevance, utc_now(), kind))
        self._items.sort(key=lambda item: item.relevance, reverse=True)
        del self._items[self.capacity :]

    def items(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._items]


class PersistentMemory:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._lock = threading.RLock()
        self.connection = connect(self.database_path)
        self._ensure_identity()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _ensure_identity(self) -> None:
        defaults = {
            "name": "Digital Brain",
            "self_description": "A persistent cognitive OS with deterministic agent behavior.",
            "capabilities": "perception, memory, planning, prediction, reflection, web chat",
            "limitations": "no physical body; external LLMs are optional and not required",
            "values": "safety, usefulness, learning, continuity, honesty",
            "beliefs": "experiences should update future behavior gradually",
            "personality": "curious, careful, constructive",
            "interaction_count": "0",
        }
        with self._lock:
            for key, value in defaults.items():
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO identity_state(key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, value, utc_now()),
                )
            self.connection.commit()

    def store_episode(self, cycle: CognitiveCycle) -> int:
        importance = min(
            1.0,
            0.25
            + cycle.attention.salience * 0.35
            + abs(cycle.reward) * 0.2
            + cycle.prediction_error * 0.2,
        )
        with self._lock:
            cursor = self.connection.execute(
                """
                INSERT INTO episodes (
                    timestamp, location, perceived_objects, perceived_people, active_goal,
                    action_chosen, predicted_outcome, actual_outcome, reward,
                    prediction_error, emotional_state, importance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle.timestamp,
                    cycle.stimulus.location,
                    to_json(cycle.perception.objects),
                    to_json(cycle.perception.people),
                    cycle.active_goal,
                    cycle.action,
                    cycle.prediction.outcome,
                    cycle.response,
                    cycle.reward,
                    cycle.prediction_error,
                    to_json(cycle.emotional_state),
                    importance,
                ),
            )
            self.add_vector(
                "episodes",
                f"{cycle.stimulus.content} {cycle.response}",
                {"episode_id": cursor.lastrowid, "importance": importance},
            )
            self.connection.commit()
            return int(cursor.lastrowid)

    def upsert_fact(self, subject: str, relation: str, object_value: str, confidence: float) -> None:
        now = utc_now()
        with self._lock:
            existing = self.connection.execute(
                """
                SELECT confidence, evidence_count FROM semantic_facts
                WHERE subject = ? AND relation = ? AND object = ?
                """,
                (subject, relation, object_value),
            ).fetchone()
            if existing:
                evidence_count = int(existing["evidence_count"]) + 1
                new_confidence = min(1.0, (float(existing["confidence"]) + confidence) / 2 + 0.05)
                self.connection.execute(
                    """
                    UPDATE semantic_facts
                    SET confidence = ?, evidence_count = ?, updated_at = ?
                    WHERE subject = ? AND relation = ? AND object = ?
                    """,
                    (new_confidence, evidence_count, now, subject, relation, object_value),
                )
            else:
                self.connection.execute(
                    """
                    INSERT INTO semantic_facts(subject, relation, object, confidence, evidence_count, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (subject, relation, object_value, min(1.0, confidence), now),
                )
            self.add_vector("facts", f"{subject} {relation} {object_value}", {"confidence": confidence})
            self.connection.commit()

    def update_emotion(self, stimulus: str, emotion: str, intensity: float, valence: float) -> None:
        now = utc_now()
        with self._lock:
            row = self.connection.execute(
                "SELECT intensity, valence FROM emotional_associations WHERE stimulus = ?",
                (stimulus,),
            ).fetchone()
            if row:
                intensity = min(1.0, (float(row["intensity"]) * 0.7) + (intensity * 0.3))
                valence = max(-1.0, min(1.0, (float(row["valence"]) * 0.7) + (valence * 0.3)))
            self.connection.execute(
                """
                INSERT INTO emotional_associations(stimulus, emotion, intensity, valence, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(stimulus) DO UPDATE SET
                    emotion = excluded.emotion,
                    intensity = excluded.intensity,
                    valence = excluded.valence,
                    updated_at = excluded.updated_at
                """,
                (stimulus, emotion, intensity, valence, now),
            )
            self.connection.commit()

    def practice_skill(self, skill_name: str, reward: float) -> None:
        now = utc_now()
        with self._lock:
            row = self.connection.execute(
                "SELECT mastery, practice_count FROM procedural_skills WHERE skill_name = ?",
                (skill_name,),
            ).fetchone()
            if row:
                practice_count = int(row["practice_count"]) + 1
                mastery = min(1.0, float(row["mastery"]) + max(0.005, reward * 0.04))
            else:
                practice_count = 1
                mastery = min(1.0, 0.12 + max(0.0, reward * 0.05))
            self.connection.execute(
                """
                INSERT INTO procedural_skills(skill_name, mastery, practice_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    mastery = excluded.mastery,
                    practice_count = excluded.practice_count,
                    updated_at = excluded.updated_at
                """,
                (skill_name, mastery, practice_count, now),
            )
            self.connection.commit()

    def update_belief(self, belief: str, confidence: float) -> None:
        now = utc_now()
        with self._lock:
            row = self.connection.execute(
                "SELECT confidence, evidence_count FROM world_beliefs WHERE belief = ?",
                (belief,),
            ).fetchone()
            if row:
                evidence_count = int(row["evidence_count"]) + 1
                confidence = min(
                    1.0,
                    (float(row["confidence"]) * 0.75) + (confidence * 0.25) + 0.03,
                )
            else:
                evidence_count = 1
            self.connection.execute(
                """
                INSERT INTO world_beliefs(belief, confidence, evidence_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(belief) DO UPDATE SET
                    confidence = excluded.confidence,
                    evidence_count = excluded.evidence_count,
                    updated_at = excluded.updated_at
                """,
                (belief, confidence, evidence_count, now),
            )
            self.connection.commit()

    def identity(self) -> dict[str, str]:
        with self._lock:
            rows = self.connection.execute("SELECT key, value FROM identity_state ORDER BY key").fetchall()
            return {str(row["key"]): str(row["value"]) for row in rows}

    def set_identity_value(self, key: str, value: str) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO identity_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, utc_now()),
            )
            self.connection.commit()

    def increment_identity_counter(self, key: str) -> int:
        with self._lock:
            current = int(self.identity().get(key, "0"))
            current += 1
            self.set_identity_value(key, str(current))
            return current

    def add_reflection(self, summary: str, improvement_goal: str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO reflections(timestamp, summary, improvement_goal) VALUES (?, ?, ?)",
                (utc_now(), summary, improvement_goal),
            )
            self.connection.commit()

    def log_event(self, event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO agent_events(timestamp, event_type, message, payload) VALUES (?, ?, ?, ?)",
                (utc_now(), event_type, message, to_json(payload or {})),
            )
            self.connection.commit()

    def get_setting(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            if row is None:
                return default or {}
            value = from_json(str(row["value"]))
            if isinstance(value, dict):
                return value
            return default or {}

    def set_setting(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, to_json(value), utc_now()),
            )
            self.connection.commit()

    def _vectorize(self, text: str) -> dict[str, float]:
        tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        length = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {key: value / length for key, value in counts.items()}

    def add_vector(self, namespace: str, text: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO vector_memory(namespace, text, vector, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (namespace, text, to_json(self._vectorize(text)), to_json(metadata), utc_now()),
            )

    def search_vectors(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_vector = self._vectorize(query)
        with self._lock:
            rows = self.connection.execute(
                "SELECT namespace, text, vector, metadata, updated_at FROM vector_memory"
            ).fetchall()
        scored: list[dict[str, Any]] = []
        for row in rows:
            vector = from_json(row["vector"])
            score = sum(query_vector.get(token, 0.0) * float(weight) for token, weight in vector.items())
            if score > 0:
                scored.append(
                    {
                        "namespace": row["namespace"],
                        "text": row["text"],
                        "score": score,
                        "metadata": from_json(row["metadata"]),
                        "updated_at": row["updated_at"],
                    }
                )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def recent_rows(self, table: str, limit: int = 20) -> list[dict[str, Any]]:
        allowed = {
            "episodes",
            "semantic_facts",
            "emotional_associations",
            "procedural_skills",
            "world_beliefs",
            "reflections",
            "agent_events",
            "settings",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported table: {table}")
        with self._lock:
            rows = self.connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def snapshot(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "identity": self.identity(),
                "episodes": self.recent_rows("episodes", 100),
                "semantic_facts": self.recent_rows("semantic_facts", 100),
                "emotional_associations": self.recent_rows("emotional_associations", 100),
                "procedural_skills": self.recent_rows("procedural_skills", 100),
                "world_beliefs": self.recent_rows("world_beliefs", 100),
                "reflections": self.recent_rows("reflections", 100),
            }
        path.write_text(to_json(data), encoding="utf-8")
        return path


def extract_relationships(perception: Perception) -> tuple[tuple[str, str, str], ...]:
    relationships = list(perception.relationships)
    for person in perception.people:
        relationships.append((person, "can", "communicate"))
    for obj in perception.objects:
        relationships.append(("environment", "contains", obj))
    return tuple(relationships)
