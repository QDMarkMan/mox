"""Tests for ManagedSession message restoration."""

from molx_core.memory import SessionData
from molx_server.session import ManagedSession


def test_managed_session_parses_string_messages() -> None:
    raw_payload = '[{"role": "user", "content": "hi"}]'
    session_data = SessionData(session_id="sess", messages=raw_payload)
    managed = ManagedSession(session_data)

    assert managed.session_data.messages == [{"role": "user", "content": "hi"}]
    assert len(managed.chat_session.get_history()) == 1
