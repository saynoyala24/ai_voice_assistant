from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class StimulusKind(StrEnum):
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    SYSTEM = "system"


@dataclass(frozen=True)
class Stimulus:
    kind: StimulusKind
    content: str
    source: str = "user"
    location: str = "web"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Perception:
    text: str
    objects: tuple[str, ...]
    people: tuple[str, ...]
    relationships: tuple[tuple[str, str, str], ...]
    motion: tuple[str, ...]
    emotion_estimate: str
    uncertainty: float


@dataclass(frozen=True)
class AttentionResult:
    salience: float
    drivers: dict[str, float]
    admitted_to_working_memory: bool


@dataclass(frozen=True)
class Prediction:
    outcome: str
    probability: float
    confidence: float


@dataclass(frozen=True)
class DebateVoiceScore:
    voice: str
    action: str
    score: float
    reason: str


@dataclass(frozen=True)
class PlanStep:
    action: str
    expected_reward: float
    risk: float
    uncertainty: float
    goal_satisfaction: float


@dataclass(frozen=True)
class CognitiveCycle:
    stimulus: Stimulus
    perception: Perception
    attention: AttentionResult
    active_goal: str
    prediction: Prediction
    debate: tuple[DebateVoiceScore, ...]
    plan: tuple[PlanStep, ...]
    action: str
    response: str
    reward: float
    prediction_error: float
    emotional_state: dict[str, float]
    identity_delta: dict[str, str]
    timestamp: str = field(default_factory=utc_now)


@dataclass
class Hormones:
    curiosity: float = 0.55
    stress: float = 0.12
    reward: float = 0.25
    confidence: float = 0.45
    attachment: float = 0.2
    energy: float = 0.88

    def clamp(self) -> None:
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.stress = max(0.0, min(1.0, self.stress))
        self.reward = max(0.0, min(1.0, self.reward))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.attachment = max(0.0, min(1.0, self.attachment))
        self.energy = max(0.0, min(1.0, self.energy))

    def as_dict(self) -> dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "stress": self.stress,
            "reward": self.reward,
            "confidence": self.confidence,
            "attachment": self.attachment,
            "energy": self.energy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hormones:
        values = cls(
            curiosity=float(data.get("curiosity", cls.curiosity)),
            stress=float(data.get("stress", cls.stress)),
            reward=float(data.get("reward", cls.reward)),
            confidence=float(data.get("confidence", cls.confidence)),
            attachment=float(data.get("attachment", cls.attachment)),
            energy=float(data.get("energy", cls.energy)),
        )
        values.clamp()
        return values
