# Digital Brain Cognitive OS

A persistent cognitive architecture that runs as a real web-based AI agent without requiring an LLM.

The system implements a continuous loop:

```text
Stimulus → Perception → Attention → Working Memory → World Model Update
→ Prediction → Goal Evaluation → Internal Debate → Planning → Action Selection
→ Environment Feedback → Memory Storage → Learning → Reflection → Identity Update
```

## Features

- FastAPI backend with a polished browser UI
- Deterministic agent chat that works without external APIs
- Persistent SQLite memory:
  - episodic memory
  - semantic facts
  - emotional associations
  - procedural skills
  - identity state
  - world model beliefs
  - vector-like semantic retrieval
- Sensory memory ring buffer and bounded working memory
- Hormone-style cognitive variables: curiosity, stress, reward, confidence, attachment, energy
- Goal hierarchy and deterministic internal debate voices
- Planning engine with fixed rollout simulation
- Idle dream/reflection cycle
- JSON snapshots for portable state inspection
- Optional LLM language layer:
  - off by default
  - Ollama local models
  - OpenAI-compatible endpoints
  - model/provider selectable from the web UI

## Setup

The project targets Python 3.13, but also runs on Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the web app

```bash
digital-brain-web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The chat works without any LLM. If you want more natural language like ChatGPT, open the
AI Model panel and choose:

- `Off` for deterministic cognition only
- `Ollama local` for local models such as `llama3.1`, `mistral`, or `qwen2.5`
- `OpenAI-compatible endpoint` for a self-hosted compatible server

No proprietary API is required. Always check the license of the model you choose; the app does
not bundle copyrighted model weights.

## Run the optional Windows desktop GUI

Install the Windows extras, then launch the PySide6 control panel:

```bash
python -m pip install -e ".[windows]"
digital-brain-desktop --database .brain/brain.sqlite3
```

## Run bounded cognitive cycles from the CLI

```bash
digital-brain --database .brain/brain.sqlite3 --cycles 3 --stimulus "Build a useful memory system"
```

## Test

```bash
python -m pytest
```

## Project layout

```text
src/digital_brain/
  app.py          FastAPI app and APIs
  engine.py       Core cognitive loop
  memory.py       Sensory, working, and persistent memory
  models.py       Domain dataclasses
  storage.py      SQLite schema and persistence helpers
  static/         Web UI
tests/            Unit and API tests
```