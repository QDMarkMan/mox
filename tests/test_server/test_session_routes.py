"""Tests for session metadata serialization."""

from molx_core.memory.recorder import ReportRecord, SessionMetadata, TurnRecord
from molx_server.session_utils import metadata_to_response


def test_metadata_to_response_serializes_turns() -> None:
    """Helper converts metadata into response model with reports and turns."""
    metadata = SessionMetadata()
    metadata.add_turn(
        TurnRecord(
            turn_id="turn-1",
            query="Analyze series",
            response="✅ Completed",
            report=ReportRecord(report_path="reports/sar.html", summary="Summary"),
        )
    )

    response = metadata_to_response("sess-42", metadata)

    assert response.session_id == "sess-42"
    assert len(response.turns) == 1
    assert response.turns[0].query == "Analyze series"
    assert response.reports[0].report_path == "reports/sar.html"
