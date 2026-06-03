from fastapi.testclient import TestClient

from digital_brain.app import create_app


def test_chat_api_returns_cycle_and_persists(tmp_path):
    client = TestClient(create_app(tmp_path / "brain.sqlite3"))
    response = client.post("/api/chat", json={"message": "สร้าง AI agent ที่มีหน้าเว็บสวยงาม"})
    assert response.status_code == 200
    data = response.json()
    assert data["reply"]
    assert data["cycle"]["active_goal"] == "help users"

    memory = client.get("/api/memory/episodes")
    assert memory.status_code == 200
    assert memory.json()["items"]


def test_system_and_safe_action_api(tmp_path):
    sample = tmp_path / "README.md"
    sample.write_text("sample file", encoding="utf-8")
    client = TestClient(create_app(tmp_path / "brain.sqlite3", safe_root=tmp_path))
    system = client.get("/api/system")
    assert system.status_code == 200
    assert "operating_system" in system.json()

    action = client.post("/api/actions", json={"action_type": "write_file", "parameters": {"path": "x.txt", "content": "hi"}})
    assert action.status_code == 200
    assert action.json()["requires_approval"] is True

    readme = client.post("/api/actions", json={"action_type": "read_file", "parameters": {"path": "README.md"}})
    assert readme.status_code == 200
    assert readme.json()["ok"] is True
