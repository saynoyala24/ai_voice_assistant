from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from digital_brain.action_system import ActionRequest, LocalActionController
from digital_brain.engine import DigitalBrain, make_text_stimulus
from digital_brain.llm import LLMConfig, OptionalLLM
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


class LLMConfigPayload(BaseModel):
    provider: str = "off"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = ""
    system_prompt: str = LLMConfig.system_prompt


@dataclass
class AppRuntime:
    brain: DigitalBrain
    perception: ComputerPerception
    actions: LocalActionController
    llm: OptionalLLM


def create_app(
    database_path: str | Path = ".brain/brain.sqlite3",
    safe_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Digital Brain Cognitive OS", version="0.1.0")
    runtime: AppRuntime | None = None
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def get_runtime() -> AppRuntime:
        nonlocal runtime
        if runtime is None:
            brain = DigitalBrain(database_path)
            runtime = AppRuntime(
                brain=brain,
                perception=ComputerPerception(),
                actions=LocalActionController(brain.memory, safe_root or Path.cwd()),
                llm=OptionalLLM(),
            )
        return runtime

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if runtime is not None:
            await runtime.brain.stop_idle_loop()
            runtime.brain.close()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "state": get_runtime().brain.state()}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        current = get_runtime()
        brain = current.brain
        cycle = brain.cycle(make_text_stimulus(request.message, request.source))
        config = LLMConfig.from_dict(brain.memory.get_setting("llm", LLMConfig().as_dict()))
        memory_context = brain.memory.search_vectors(request.message, 4)
        llm_result = current.llm.complete(config, request.message, cycle.response, memory_context)
        if config.provider != "off":
            brain.memory.log_event(
                "llm",
                "Generated language-layer response." if llm_result.ok else "LLM fallback used.",
                llm_result.__dict__,
            )
        return {
            "reply": llm_result.text,
            "cycle": _cycle_payload(cycle),
            "state": brain.state(),
            "llm": llm_result.__dict__,
        }

    @app.post("/api/stimuli")
    def stimuli(request: StimulusRequest) -> dict[str, Any]:
        brain = get_runtime().brain
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
        return get_runtime().brain.state()

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        current = get_runtime()
        observation = current.perception.observe_system()
        current.brain.memory.log_event("perception", "Observed operating system state.", observation.__dict__)
        return observation.__dict__

    @app.get("/api/files")
    def files(path: str = ".") -> dict[str, Any]:
        return {"items": get_runtime().perception.list_files(path)}

    @app.post("/api/actions")
    def execute_action(payload: ActionPayload) -> dict[str, Any]:
        result = get_runtime().actions.execute(
            ActionRequest(
                action_type=payload.action_type,
                parameters=payload.parameters,
                approval_token=payload.approval_token,
            )
        )
        return result.__dict__

    @app.get("/api/memory/{table_name}")
    def memory(table_name: str, limit: int = 30) -> dict[str, Any]:
        return {"items": get_runtime().brain.memory.recent_rows(table_name, max(1, min(100, limit)))}

    @app.get("/api/search")
    def search(q: str, limit: int = 5) -> dict[str, Any]:
        return {"items": get_runtime().brain.memory.search_vectors(q, max(1, min(20, limit)))}

    @app.get("/api/llm/config")
    def get_llm_config() -> dict[str, Any]:
        config = LLMConfig.from_dict(
            get_runtime().brain.memory.get_setting("llm", LLMConfig().as_dict())
        )
        return config.as_dict()

    @app.post("/api/llm/config")
    def set_llm_config(payload: LLMConfigPayload) -> dict[str, Any]:
        config = LLMConfig.from_dict(payload.model_dump())
        get_runtime().brain.memory.set_setting("llm", config.as_dict())
        return config.as_dict()

    @app.get("/api/llm/models")
    def llm_models() -> dict[str, Any]:
        current = get_runtime()
        config = LLMConfig.from_dict(current.brain.memory.get_setting("llm", LLMConfig().as_dict()))
        return {"items": current.llm.list_models(config)}

    @app.post("/api/loop/start")
    async def start_loop() -> dict[str, Any]:
        brain = get_runtime().brain
        await brain.start_idle_loop()
        return brain.state()

    @app.post("/api/loop/stop")
    async def stop_loop() -> dict[str, Any]:
        brain = get_runtime().brain
        await brain.stop_idle_loop()
        return brain.state()

    @app.post("/api/dream")
    def dream() -> dict[str, Any]:
        brain = get_runtime().brain
        brain.dream()
        return {"state": brain.state(), "events": brain.memory.recent_rows("agent_events", 10)}

    @app.post("/api/reflect")
    def reflect() -> dict[str, Any]:
        brain = get_runtime().brain
        brain.reflect()
        return {"state": brain.state(), "reflections": brain.memory.recent_rows("reflections", 10)}

    @app.post("/api/snapshot")
    def snapshot() -> dict[str, Any]:
        brain = get_runtime().brain
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


if __name__ == "__main__":
    main()
