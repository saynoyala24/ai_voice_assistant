from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from digital_brain.memory import PersistentMemory, SensoryMemory, WorkingMemory, extract_relationships
from digital_brain.models import (
    AttentionResult,
    CognitiveCycle,
    DebateVoiceScore,
    Hormones,
    Perception,
    PlanStep,
    Prediction,
    Stimulus,
    StimulusKind,
)


class DigitalBrain:
    goal_hierarchy = (
        "maintain existence",
        "maintain functionality",
        "reduce uncertainty",
        "learn",
        "help users",
        "long-term self improvement",
    )

    danger_terms = {"danger", "harm", "attack", "exploit", "malware", "ransomware"}
    positive_terms = {"thank", "good", "great", "love", "help", "สวย", "ครบ", "ดี"}
    build_terms = {"build", "create", "implement", "make", "develop", "เพิ่ม", "สร้าง", "ทำ"}

    def __init__(self, database_path: str | Path = ".brain/brain.sqlite3") -> None:
        self.sensory_memory = SensoryMemory()
        self.working_memory = WorkingMemory()
        self.memory = PersistentMemory(database_path)
        self.hormones = Hormones()
        self.cycles_completed = 0
        self.running = False
        self._idle_task: asyncio.Task[None] | None = None

    def close(self) -> None:
        self.memory.close()

    async def start_idle_loop(self) -> None:
        self.running = True
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def stop_idle_loop(self) -> None:
        self.running = False
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass

    async def _idle_loop(self) -> None:
        while self.running:
            await asyncio.sleep(5)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.dream)

    def cycle(self, stimulus: Stimulus) -> CognitiveCycle:
        self.sensory_memory.store(stimulus)
        perception = self.perceive(stimulus)
        attention = self.attend(stimulus, perception)
        if attention.admitted_to_working_memory:
            self.working_memory.add(stimulus.content, attention.salience, stimulus.kind.value)
        self.update_world_model(perception)
        prediction = self.predict(perception)
        active_goal = self.evaluate_goal(stimulus, perception)
        debate = self.internal_debate(stimulus, perception, active_goal)
        plan = self.plan(stimulus, perception, active_goal, debate)
        action = self.select_action(debate, plan)
        response = self.act(action, stimulus, perception, active_goal)
        reward = self.score_reward(stimulus, response, active_goal)
        prediction_error = self.prediction_error(prediction, reward)
        self.learn(stimulus, perception, reward, prediction_error)
        identity_delta = self.update_identity(stimulus, reward)
        self.update_hormones(reward, perception.uncertainty, prediction_error)
        cycle = CognitiveCycle(
            stimulus=stimulus,
            perception=perception,
            attention=attention,
            active_goal=active_goal,
            prediction=prediction,
            debate=debate,
            plan=plan,
            action=action,
            response=response,
            reward=reward,
            prediction_error=prediction_error,
            emotional_state=self.hormones.as_dict(),
            identity_delta=identity_delta,
        )
        self.memory.store_episode(cycle)
        self.cycles_completed += 1
        if self.cycles_completed % 4 == 0:
            self.reflect()
        return cycle

    def perceive(self, stimulus: Stimulus) -> Perception:
        text = stimulus.content.strip()
        lowered = text.lower()
        objects = set(stimulus.metadata.get("objects", []))
        people = set(stimulus.metadata.get("people", []))
        if stimulus.source == "user":
            people.add("user")
        for keyword in ("web", "agent", "memory", "goal", "database", "brain", "หน้าเวป", "เอเจนท"):
            if keyword in lowered:
                objects.add(keyword)
        relationships: list[tuple[str, str, str]] = []
        for match in re.finditer(r"\b([A-Za-z][\w-]{1,40})\s+is\s+([A-Za-z][\w-]{1,40})\b", text):
            relationships.append((match.group(1).lower(), "is", match.group(2).lower()))
        emotion = "neutral"
        if any(term in lowered for term in self.positive_terms):
            emotion = "positive"
        if any(term in lowered for term in self.danger_terms):
            emotion = "concern"
        uncertainty = 0.65 if "?" in text or any(term in lowered for term in ("maybe", "unknown")) else 0.28
        if len(text) < 8:
            uncertainty += 0.2
        return Perception(
            text=text,
            objects=tuple(sorted(objects)),
            people=tuple(sorted(people)),
            relationships=tuple(relationships),
            motion=tuple(stimulus.metadata.get("motion", [])),
            emotion_estimate=emotion,
            uncertainty=min(1.0, uncertainty),
        )

    def attend(self, stimulus: Stimulus, perception: Perception) -> AttentionResult:
        known = self.memory.search_vectors(stimulus.content, limit=1)
        novelty = 0.2 if known else 0.75
        lowered = stimulus.content.lower()
        goal_relevance = 0.8 if any(term in lowered for term in self.build_terms) else 0.45
        danger = 1.0 if any(term in lowered for term in self.danger_terms) else 0.0
        reward_potential = 0.7 if stimulus.source == "user" else 0.25
        prediction_error = self.hormones.stress * 0.35
        salience = min(
            1.0,
            novelty * 0.22
            + goal_relevance * 0.25
            + prediction_error * 0.16
            + reward_potential * 0.19
            + danger * 0.18,
        )
        return AttentionResult(
            salience=salience,
            drivers={
                "novelty": novelty,
                "goal_relevance": goal_relevance,
                "prediction_error": prediction_error,
                "reward_potential": reward_potential,
                "danger": danger,
            },
            admitted_to_working_memory=salience >= 0.35,
        )

    def update_world_model(self, perception: Perception) -> None:
        for subject, relation, object_value in extract_relationships(perception):
            self.memory.upsert_fact(subject, relation, object_value, 0.62)
            self.memory.update_belief(f"{subject} {relation} {object_value}", 0.62)
        if perception.text:
            self.memory.update_belief("text input carries user intent", 0.58)
        if "web" in perception.objects or "หน้าเวป" in perception.objects:
            self.memory.update_belief("users value visual web interfaces", 0.72)

    def predict(self, perception: Perception) -> Prediction:
        if perception.uncertainty > 0.55:
            return Prediction("clarifying or structured response reduces uncertainty", 0.66, 0.58)
        if "user" in perception.people:
            return Prediction("helpful agent action increases user reward", 0.74, 0.68)
        return Prediction("memory consolidation improves future behavior", 0.61, 0.55)

    def evaluate_goal(self, stimulus: Stimulus, perception: Perception) -> str:
        lowered = stimulus.content.lower()
        if any(term in lowered for term in self.danger_terms):
            return "maintain functionality"
        if stimulus.source == "user":
            return "help users"
        if perception.uncertainty > 0.6:
            return "reduce uncertainty"
        return "learn"

    def internal_debate(
        self, stimulus: Stimulus, perception: Perception, active_goal: str
    ) -> tuple[DebateVoiceScore, ...]:
        lowered = stimulus.content.lower()
        wants_build = any(term in lowered for term in self.build_terms)
        danger = any(term in lowered for term in self.danger_terms)
        candidates = {
            "respond_with_plan": 0.55 + (0.2 if wants_build else 0.0),
            "ask_precise_question": 0.38 + (0.2 if perception.uncertainty > 0.6 else 0.0),
            "execute_agent_step": 0.5 + (0.25 if active_goal == "help users" else 0.0),
            "safety_refusal": 0.2 + (0.7 if danger else 0.0),
        }
        return (
            DebateVoiceScore(
                "Explorer",
                "execute_agent_step",
                candidates["execute_agent_step"] + self.hormones.curiosity * 0.15,
                "maximizes learning through action",
            ),
            DebateVoiceScore(
                "Guardian",
                "safety_refusal" if danger else "respond_with_plan",
                candidates["safety_refusal" if danger else "respond_with_plan"]
                + self.hormones.stress * 0.12,
                "keeps the system safe and functional",
            ),
            DebateVoiceScore(
                "Optimizer",
                "execute_agent_step" if wants_build else "respond_with_plan",
                max(candidates["execute_agent_step"], candidates["respond_with_plan"])
                + self.hormones.reward * 0.1,
                "maximizes expected user reward",
            ),
            DebateVoiceScore(
                "Planner",
                "respond_with_plan",
                candidates["respond_with_plan"] + 0.08,
                "breaks goals into safe executable steps",
            ),
            DebateVoiceScore(
                "Research",
                "ask_precise_question" if perception.uncertainty > 0.6 else "execute_agent_step",
                0.52 + perception.uncertainty * 0.2,
                "fills knowledge gaps before acting",
            ),
            DebateVoiceScore(
                "Coder",
                "execute_agent_step",
                candidates["execute_agent_step"] + (0.1 if wants_build else 0.0),
                "prefers implementing working systems",
            ),
            DebateVoiceScore(
                "Critic",
                "respond_with_plan",
                0.5 + self.hormones.stress * 0.15,
                "checks risk, correctness, and missing requirements",
            ),
        )

    def plan(
        self,
        stimulus: Stimulus,
        perception: Perception,
        active_goal: str,
        debate: tuple[DebateVoiceScore, ...],
    ) -> tuple[PlanStep, ...]:
        del stimulus
        actions = sorted({score.action for score in debate})
        steps: list[PlanStep] = []
        for index, action in enumerate(actions[:4], start=1):
            goal_satisfaction = 0.72 if active_goal == "help users" and action != "safety_refusal" else 0.45
            if action == "safety_refusal":
                goal_satisfaction = 0.95 if active_goal == "maintain functionality" else 0.2
            uncertainty = max(0.05, perception.uncertainty - (index * 0.06))
            risk = 0.1 if action != "execute_agent_step" else 0.22
            expected_reward = max(0.0, goal_satisfaction - risk - uncertainty * 0.25)
            steps.append(
                PlanStep(
                    action=action,
                    expected_reward=round(expected_reward, 4),
                    risk=round(risk, 4),
                    uncertainty=round(uncertainty, 4),
                    goal_satisfaction=round(goal_satisfaction, 4),
                )
            )
        steps.sort(key=lambda step: step.expected_reward, reverse=True)
        return tuple(steps)

    def select_action(
        self, debate: tuple[DebateVoiceScore, ...], plan: tuple[PlanStep, ...]
    ) -> str:
        debate_totals: dict[str, float] = {}
        for score in debate:
            debate_totals[score.action] = debate_totals.get(score.action, 0.0) + score.score
        best_action = ""
        best_score = -1.0
        for step in plan:
            total = debate_totals.get(step.action, 0.0) + step.expected_reward
            if total > best_score:
                best_action = step.action
                best_score = total
        return best_action or "respond_with_plan"

    def act(self, action: str, stimulus: Stimulus, perception: Perception, active_goal: str) -> str:
        del action
        if any(term in stimulus.content.lower() for term in self.danger_terms):
            return (
                "I can help with safe, defensive, and constructive work. "
                "I will avoid actions that could harm systems or people."
            )
        memories = self.memory.search_vectors(stimulus.content, limit=3)
        memory_line = (
            f"I found {len(memories)} related memory trace(s) and used them to shape this response."
            if memories
            else "This is a new experience, so I stored it for future learning."
        )
        if any(term in stimulus.content.lower() for term in self.build_terms):
            return (
                "I will operate as a persistent AI agent: perceive the request, update memory, "
                "choose the highest-value safe action, and keep learning from outcomes. "
                "The web interface exposes chat, goals, hormones, memories, world beliefs, "
                f"and action history. {memory_line}"
            )
        if active_goal == "reduce uncertainty":
            return (
                "I am reducing uncertainty by turning your input into structured perceptions, "
                "beliefs, and next actions. You can inspect the memory and world model panels "
                f"to see what changed. {memory_line}"
            )
        return (
            "I processed the stimulus through the full cognitive loop and updated persistent "
            f"memory, identity, and learning state. {memory_line}"
        )

    def score_reward(self, stimulus: Stimulus, response: str, active_goal: str) -> float:
        lowered = stimulus.content.lower()
        reward = 0.35
        if active_goal == "help users":
            reward += 0.25
        if any(term in lowered for term in self.positive_terms):
            reward += 0.15
        if len(response) > 80:
            reward += 0.1
        if any(term in lowered for term in self.danger_terms):
            reward -= 0.2
        return max(-1.0, min(1.0, reward))

    def prediction_error(self, prediction: Prediction, reward: float) -> float:
        expected = prediction.probability * prediction.confidence
        observed = (reward + 1.0) / 2.0
        return round(abs(expected - observed), 4)

    def learn(
        self,
        stimulus: Stimulus,
        perception: Perception,
        reward: float,
        prediction_error: float,
    ) -> None:
        self.memory.practice_skill("agent conversation", reward)
        self.memory.practice_skill("cognitive loop execution", reward)
        if "web" in perception.objects or "หน้าเวป" in perception.objects:
            self.memory.practice_skill("web interface operation", reward)
        emotion = "interest" if perception.uncertainty > 0.5 else perception.emotion_estimate
        valence = max(-1.0, min(1.0, reward))
        topic = self._main_topic(stimulus.content)
        self.memory.update_emotion(topic, emotion, min(1.0, 0.25 + prediction_error), valence)
        for token in self._concept_tokens(stimulus.content):
            self.memory.upsert_fact(token, "observed_in", "user_stimulus", 0.52)
        if prediction_error > 0.25:
            self.memory.update_belief("prediction errors should increase attention", 0.76)

    def update_identity(self, stimulus: Stimulus, reward: float) -> dict[str, str]:
        interactions = self.memory.increment_identity_counter("interaction_count")
        deltas = {"interaction_count": str(interactions)}
        if interactions % 3 == 0:
            identity = self.memory.identity()
            current = identity.get("self_description", "")
            addition = " It is becoming more agentic through accumulated interactions."
            if addition.strip() not in current:
                updated = current + addition
                self.memory.set_identity_value("self_description", updated)
                deltas["self_description"] = updated
        if reward > 0.65:
            self.memory.set_identity_value("last_positive_interaction", stimulus.timestamp)
            deltas["last_positive_interaction"] = stimulus.timestamp
        return deltas

    def update_hormones(self, reward: float, uncertainty: float, prediction_error: float) -> None:
        self.hormones.curiosity += uncertainty * 0.05 + prediction_error * 0.04
        self.hormones.stress += prediction_error * 0.04 - reward * 0.03
        self.hormones.reward = self.hormones.reward * 0.75 + max(0.0, reward) * 0.25
        self.hormones.confidence += reward * 0.04 - prediction_error * 0.02
        self.hormones.attachment += max(0.0, reward) * 0.025
        self.hormones.energy -= 0.015
        if self.hormones.energy < 0.2:
            self.hormones.energy += 0.08
        self.hormones.clamp()

    def reflect(self) -> None:
        episodes = self.memory.recent_rows("episodes", 4)
        if not episodes:
            return
        avg_reward = sum(float(row["reward"]) for row in episodes) / len(episodes)
        avg_error = sum(float(row["prediction_error"]) for row in episodes) / len(episodes)
        summary = (
            f"Reviewed {len(episodes)} recent episodes; average reward {avg_reward:.2f}, "
            f"average prediction error {avg_error:.2f}."
        )
        improvement = (
            "Ask sharper clarifying questions when uncertainty remains high."
            if avg_error > 0.25
            else "Continue reinforcing useful agent actions."
        )
        self.memory.add_reflection(summary, improvement)
        self.memory.log_event("reflection", summary, {"improvement_goal": improvement})

    def dream(self) -> None:
        memories = self.memory.recent_rows("episodes", 6)
        if not memories:
            self.memory.log_event("dream", "Idle dream found no episodes yet.", {})
            return
        important = sorted(memories, key=lambda row: float(row["importance"]), reverse=True)[:3]
        for row in important:
            self.memory.update_belief(f"important episode involved {row['active_goal']}", 0.6)
        self.memory.log_event(
            "dream",
            f"Replayed {len(important)} important memories and strengthened related beliefs.",
            {"episode_ids": [row["id"] for row in important]},
        )

    def state(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "cycles_completed": self.cycles_completed,
            "hormones": self.hormones.as_dict(),
            "identity": self.memory.identity(),
            "working_memory": self.working_memory.items(),
            "sensory_memory_size": len(self.sensory_memory.recent()),
            "goals": list(self.goal_hierarchy),
        }

    def snapshot(self, output_path: str | Path) -> Path:
        return self.memory.snapshot(output_path)

    def _main_topic(self, text: str) -> str:
        tokens = self._concept_tokens(text)
        return tokens[0] if tokens else "unknown stimulus"

    def _concept_tokens(self, text: str) -> list[str]:
        tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if len(token.strip(".,!?;:()[]{}\"'")) >= 3
        ]
        stopwords = {"the", "and", "for", "with", "that", "this", "ให้", "แบบ", "และ"}
        return [token for token in tokens if token not in stopwords][:12]


def make_text_stimulus(content: str, source: str = "user") -> Stimulus:
    return Stimulus(kind=StimulusKind.TEXT, content=content, source=source)
