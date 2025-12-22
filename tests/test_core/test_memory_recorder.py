"""Tests for shared session memory recorder."""

from types import SimpleNamespace

from molx_core.memory import SessionData
from molx_core.memory.recorder import (
    FileRecord,
    ReportRecord,
    SessionMetadata,
    SessionRecorder,
    TurnRecord,
)


def _build_state_with_results() -> dict:
    """Helper to construct a representative agent state payload."""
    return {
        "intent": SimpleNamespace(value="sar_analysis"),
        "intent_reasoning": "User wants SAR analysis",
        "intent_confidence": 0.95,
        "plan_reasoning": "Plan data cleaning → SAR → report",
        "tasks": {
            "clean": {
                "id": "clean",
                "type": "data_cleaner",
                "description": "Clean uploaded dataset",
                "status": "done",
                "depends_on": [],
            },
            "report": {
                "id": "report",
                "type": "reporter",
                "description": "Generate SAR report",
                "status": "done",
                "depends_on": ["clean"],
            },
        },
        "results": {
            "clean": {
                "compounds": [{"compound_id": "CPD-1"}],
                "cleaning_stats": {"cleaned_count": 1},
                "output_files": {"csv": "clean.csv"},
            },
            "report": {
                "report_path": "reports/sar.html",
                "report_intent": "full_report",
                "sar_analysis": {"summary": "All good"},
            },
        },
        "reflection": {"success": True, "summary": "complete"},
    }


def test_session_recorder_persists_turn_metadata() -> None:
    """Recording a turn stores latest snapshot, structured data, and report info."""
    session = SessionData(session_id="sess-123")
    recorder = SessionRecorder(session)

    state = _build_state_with_results()
    recorder.record_turn(query="Analyze aspirin", response="✅ Done", state=state)

    metadata = session.metadata
    assert metadata.latest["query"] == "Analyze aspirin"
    assert metadata.latest["response"].startswith("✅")

    structured = metadata.structured_data
    assert structured["clean"]["compound_count"] == 1
    assert structured["report"]["report_path"] == "reports/sar.html"

    assert metadata.reports
    assert metadata.reports[-1].report_path == "reports/sar.html"


def test_session_metadata_round_trip() -> None:
    """SessionMetadata serializes and deserializes consistently."""
    metadata = SessionMetadata()
    metadata.add_turn(
        TurnRecord(
            turn_id="turn-1",
            query="Hello",
            response="World",
            report=ReportRecord(report_path="reports/sar.html", summary="Summary"),
        )
    )

    dumped = metadata.to_dict()
    restored = SessionMetadata.from_dict(dumped)

    assert len(restored.turns) == 1
    assert restored.turns[0].query == "Hello"
    assert restored.reports[0].report_path == "reports/sar.html"


def test_metadata_tracks_uploaded_files(tmp_path) -> None:
    """Adding uploaded files should be reflected in metadata serialization."""
    metadata = SessionMetadata()
    upload_path = tmp_path / "dataset.csv"
    upload_path.write_text("smiles,activity\nCCO,1")

    record = FileRecord(
        file_id="file-1",
        file_name="dataset.csv",
        file_path=str(upload_path),
        content_type="text/csv",
        size_bytes=upload_path.stat().st_size,
    )
    metadata.add_uploaded_file(record)

    dumped = metadata.to_dict()
    assert dumped["uploaded_files"][0]["file_name"] == "dataset.csv"

    restored = SessionMetadata.from_dict(dumped)
    assert restored.uploaded_files[0].file_path == str(upload_path)


def test_recorder_captures_artifacts_from_results(tmp_path) -> None:
    """Record turn should emit artifact metadata when reports/output files exist."""
    session = SessionData(session_id="sess-artifacts")
    recorder = SessionRecorder(session)

    report_path = tmp_path / "sar.html"
    report_path.write_text("<html></html>")

    csv_path = tmp_path / "clean.csv"
    csv_path.write_text("smiles,activity\nCCO,1")

    state = {
        "tasks": {
            "clean": {"id": "clean", "type": "data_cleaner", "description": ""}
        },
        "results": {
            "clean": {
                "report_path": str(report_path),
                "output_files": {"csv": str(csv_path)},
            }
        },
    }

    recorder.record_turn(query="Run", response="done", state=state)
    metadata = session.metadata
    assert metadata.artifacts
    paths = {artifact.file_path for artifact in metadata.artifacts}
    assert str(report_path) in paths
    assert str(csv_path) in paths
