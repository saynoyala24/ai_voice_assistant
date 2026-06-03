from digital_brain.engine import DigitalBrain, make_text_stimulus


def test_cycle_persists_memory_and_updates_state(tmp_path):
    brain = DigitalBrain(tmp_path / "brain.sqlite3")
    try:
        cycle = brain.cycle(make_text_stimulus("Build a beautiful web agent with memory"))
        assert cycle.active_goal == "help users"
        assert cycle.response
        assert brain.state()["cycles_completed"] == 1
        assert brain.memory.recent_rows("episodes", 5)
        assert brain.memory.recent_rows("semantic_facts", 5)
        assert brain.memory.recent_rows("procedural_skills", 5)
    finally:
        brain.close()


def test_learning_changes_future_retrieval(tmp_path):
    brain = DigitalBrain(tmp_path / "brain.sqlite3")
    try:
        brain.cycle(make_text_stimulus("The agent should remember Windows control goals"))
        matches = brain.memory.search_vectors("Windows control", 3)
        assert matches
        assert matches[0]["score"] > 0
    finally:
        brain.close()


def test_reflection_and_dream_create_events(tmp_path):
    brain = DigitalBrain(tmp_path / "brain.sqlite3")
    try:
        for index in range(4):
            brain.cycle(make_text_stimulus(f"Learn from interaction {index}"))
        brain.dream()
        assert brain.memory.recent_rows("reflections", 3)
        assert brain.memory.recent_rows("agent_events", 3)
    finally:
        brain.close()
