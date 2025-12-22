"""Tests for SessionMetadata helpers."""

from molx_core.memory.recorder import SessionMetadata


def test_session_metadata_from_json_string() -> None:
    """JSON strings returned from Postgres should deserialize cleanly."""
    payload = '{"latest": {"query": "ping"}, "turns": [], "reports": []}'
    metadata = SessionMetadata.from_dict(payload)
    assert metadata.latest == {"query": "ping"}
    assert metadata.turns == []
