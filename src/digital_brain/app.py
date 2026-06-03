from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from digital_brain.action_system import ActionRequest, LocalActionController
from digital_brain.engine import DigitalBrain, make_text_stimulus
from digital_brain.models import Stimulus, StimulusKind
from digital_brain.perception_system import ComputerPerception


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    source: str = "user"


class StimulusRequest(BaseModel):
    kind: StimulusKind = StimulusKind.TEXT
    content: str = Field(min_length=1, max_length=4000)
    source: str = "user"
    location: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionPayload(BaseModel):
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    approval_token: str | None = None


def create_app(
    database_path: str | Path = ".brain/brain.sqlite3",
    safe_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Digital Brain Cognitive OS", version="0.1.0")
    brain = DigitalBrain(database_path)
    perception = ComputerPerception()
    actions = LocalActionController(brain.memory, safe_root or Path.cwd())
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await brain.stop_idle_loop()
        brain.close()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "state": brain.state()}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        cycle = brain.cycle(make_text_stimulus(request.message, request.source))
        return {"reply": cycle.response, "cycle": _cycle_payload(cycle), "state": brain.state()}

    @app.post("/api/stimuli")
    def stimuli(request: StimulusRequest) -> dict[str, Any]:
        cycle = brain.cycle(
            Stimulus(
                kind=request.kind,
                content=request.content,
                source=request.source,
                location=request.location,
                metadata=request.metadata,
            )
        )
        return {"cycle": _cycle_payload(cycle), "state": brain.state()}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return brain.state()

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        observation = perception.observe_system()
        brain.memory.log_event("perception", "Observed operating system state.", observation.__dict__)
        return observation.__dict__

    @app.get("/api/files")
    def files(path: str = ".") -> dict[str, Any]:
        return {"items": perception.list_files(path)}

    @app.post("/api/actions")
    def execute_action(payload: ActionPayload) -> dict[str, Any]:
        result = actions.execute(
            ActionRequest(
                action_type=payload.action_type,
                parameters=payload.parameters,
                approval_token=payload.approval_token,
            )
        )
        return result.__dict__

    @app.get("/api/memory/{table_name}")
    def memory(table_name: str, limit: int = 30) -> dict[str, Any]:
        return {"items": brain.memory.recent_rows(table_name, max(1, min(100, limit)))}

    @app.get("/api/search")
    def search(q: str, limit: int = 5) -> dict[str, Any]:
        return {"items": brain.memory.search_vectors(q, max(1, min(20, limit)))}

    @app.post("/api/loop/start")
    async def start_loop() -> dict[str, Any]:
        await brain.start_idle_loop()
        return brain.state()

    @app.post("/api/loop/stop")
    async def stop_loop() -> dict[str, Any]:
        await brain.stop_idle_loop()
        return brain.state()

    @app.post("/api/dream")
    def dream() -> dict[str, Any]:
        brain.dream()
        return {"state": brain.state(), "events": brain.memory.recent_rows("agent_events", 10)}

    @app.post("/api/reflect")
    def reflect() -> dict[str, Any]:
        brain.reflect()
        return {"state": brain.state(), "reflections": brain.memory.recent_rows("reflections", 10)}

    @app.post("/api/snapshot")
    def snapshot() -> dict[str, Any]:
        path = brain.snapshot(".brain/snapshot.json")
        return {"path": str(path)}

    return app


def _cycle_payload(cycle: Any) -> dict[str, Any]:
    return {
        "stimulus": {
            "kind": cycle.stimulus.kind.value,
            "content": cycle.stimulus.content,
            "source": cycle.stimulus.source,
            "location": cycle.stimulus.location,
            "metadata": cycle.stimulus.metadata,
            "timestamp": cycle.stimulus.timestamp,
        },
        "perception": cycle.perception.__dict__,
        "attention": {
            "salience": cycle.attention.salience,
            "drivers": cycle.attention.drivers,
            "admitted_to_working_memory": cycle.attention.admitted_to_working_memory,
        },
        "active_goal": cycle.active_goal,
        "prediction": cycle.prediction.__dict__,
        "debate": [score.__dict__ for score in cycle.debate],
        "plan": [step.__dict__ for step in cycle.plan],
        "action": cycle.action,
        "response": cycle.response,
        "reward": cycle.reward,
        "prediction_error": cycle.prediction_error,
        "emotional_state": cycle.emotional_state,
        "identity_delta": cycle.identity_delta,
        "timestamp": cycle.timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=".brain/brain.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(create_app(args.database), host=args.host, port=args.port)


app = create_app()


if __name__ == "__main__":
    main()
