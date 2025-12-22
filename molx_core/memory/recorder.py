"""Session metadata schema and recorder utilities.

Defines structured metadata for each agent session and helpers for
recording turn-by-turn information that can be persisted by any
ConversationStore implementation.
"""

from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, TYPE_CHECKING, Union
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover
    from molx_core.memory.store import SessionData


MAX_TURNS_STORED = 50
MAX_REPORTS_STORED = 20
MAX_FILE_RECORDS_STORED = 50


def _utc_iso() -> str:
    return datetime.utcnow().isoformat()


def _safe_preview(text: Optional[str], limit: int = 240) -> Optional[str]:
    if text is None:
        return None
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[:limit].rstrip()}…"


@dataclass
class ReportRecord:
    """Metadata for a generated report."""

    report_path: str
    summary: Optional[str] = None
    preview: Optional[str] = None
    created_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_path": self.report_path,
            "summary": self.summary,
            "preview": self.preview,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReportRecord":
        return cls(
            report_path=data.get("report_path", ""),
            summary=data.get("summary"),
            preview=data.get("preview"),
            created_at=data.get("created_at", _utc_iso()),
        )


@dataclass
class TurnRecord:
    """Structured representation of a single agent turn."""

    turn_id: str
    query: str
    response: str
    intent: Optional[str] = None
    intent_reasoning: Optional[str] = None
    intent_confidence: Optional[float] = None
    plan_reasoning: Optional[str] = None
    tasks: list[dict[str, Any]] = field(default_factory=list)
    reflection: dict[str, Any] = field(default_factory=dict)
    structured_data: dict[str, Any] = field(default_factory=dict)
    report: Optional[ReportRecord] = None
    artifacts: list[FileRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "intent": self.intent,
            "intent_reasoning": self.intent_reasoning,
            "intent_confidence": self.intent_confidence,
            "plan_reasoning": self.plan_reasoning,
            "tasks": self.tasks,
            "reflection": self.reflection,
            "structured_data": self.structured_data,
            "created_at": self.created_at,
        }
        if self.report:
            data["report"] = self.report.to_dict()
        if self.artifacts:
            data["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TurnRecord":
        report = data.get("report")
        artifacts = data.get("artifacts", [])
        return cls(
            turn_id=data.get("turn_id", str(uuid4())),
            query=data.get("query", ""),
            response=data.get("response", ""),
            intent=data.get("intent"),
            intent_reasoning=data.get("intent_reasoning"),
            intent_confidence=data.get("intent_confidence"),
            plan_reasoning=data.get("plan_reasoning"),
            tasks=list(data.get("tasks", [])),
            reflection=dict(data.get("reflection", {})),
            structured_data=dict(data.get("structured_data", {})),
            report=ReportRecord.from_dict(report) if isinstance(report, Mapping) else None,
            artifacts=[FileRecord.from_dict(artifact) for artifact in artifacts if isinstance(artifact, Mapping)],
            created_at=data.get("created_at", _utc_iso()),
        )


@dataclass
class SessionMetadata:
    """Structured session metadata stored alongside conversations."""

    turns: list[TurnRecord] = field(default_factory=list)
    latest: dict[str, Any] = field(default_factory=dict)
    structured_data: dict[str, Any] = field(default_factory=dict)
    reports: list[ReportRecord] = field(default_factory=list)
    uploaded_files: list[FileRecord] = field(default_factory=list)
    artifacts: list[FileRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "latest": self.latest,
            "structured_data": self.structured_data,
            "reports": [report.to_dict() for report in self.reports],
            "uploaded_files": [record.to_dict() for record in self.uploaded_files],
            "artifacts": [record.to_dict() for record in self.artifacts],
        }

    @classmethod
    def from_dict(cls, data: Optional[Union[Mapping[str, Any], str]]) -> "SessionMetadata":
        if not data:
            return cls()

        payload: Optional[Mapping[str, Any]]
        if isinstance(data, Mapping):
            payload = data
        elif isinstance(data, str):
            try:
                parsed = json.loads(data)
                payload = parsed if isinstance(parsed, Mapping) else None
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None

        if not payload:
            return cls()

        turns_data = payload.get("turns", [])
        reports_data = payload.get("reports", [])
        uploaded = payload.get("uploaded_files", [])
        artifacts = payload.get("artifacts", [])
        metadata = cls(
            turns=[TurnRecord.from_dict(turn) for turn in turns_data],
            latest=dict(payload.get("latest", {})),
            structured_data=dict(payload.get("structured_data", {})),
            reports=[ReportRecord.from_dict(report) for report in reports_data],
            uploaded_files=[FileRecord.from_dict(item) for item in uploaded],
            artifacts=[FileRecord.from_dict(item) for item in artifacts],
        )
        metadata._truncate()
        return metadata

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.latest = turn.to_dict()

        if turn.report:
            self.reports.append(turn.report)

        if turn.artifacts:
            self.add_artifacts(turn.artifacts)

        if turn.structured_data:
            self._merge_structured(turn.structured_data)

        self._truncate()

    def add_uploaded_file(self, record: FileRecord) -> None:
        self._upsert(self.uploaded_files, record)
        self._truncate()

    def add_artifacts(self, records: list[FileRecord]) -> None:
        for record in records:
            self._upsert(self.artifacts, record)
        self._truncate()

    def find_file(self, file_id: str) -> Optional[FileRecord]:
        for bucket in (self.uploaded_files, self.artifacts):
            for record in bucket:
                if record.file_id == file_id:
                    return record
        return None

    def _truncate(self) -> None:
        if len(self.turns) > MAX_TURNS_STORED:
            self.turns = self.turns[-MAX_TURNS_STORED:]
        if len(self.reports) > MAX_REPORTS_STORED:
            self.reports = self.reports[-MAX_REPORTS_STORED:]
        if len(self.uploaded_files) > MAX_FILE_RECORDS_STORED:
            self.uploaded_files = self.uploaded_files[-MAX_FILE_RECORDS_STORED:]
        if len(self.artifacts) > MAX_FILE_RECORDS_STORED:
            self.artifacts = self.artifacts[-MAX_FILE_RECORDS_STORED:]

    def _upsert(self, bucket: list[FileRecord], record: FileRecord) -> None:
        bucket[:] = [existing for existing in bucket if existing.file_id != record.file_id]
        bucket.append(record)

    def _merge_structured(self, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            self.structured_data[key] = value


class SessionRecorder:
    """Utility for recording session metadata from agent state dictionaries."""

    def __init__(self, session: "SessionData") -> None:
        from molx_core.memory.store import SessionData as SessionDataType

        if not isinstance(session, SessionDataType):  # pragma: no cover - runtime guard
            raise TypeError("session must be a SessionData instance")
        self._session = session

    @property
    def metadata(self) -> SessionMetadata:
        metadata = self._session.metadata
        if not isinstance(metadata, SessionMetadata):
            metadata = SessionMetadata.from_dict(metadata)
            self._session.metadata = metadata
        return metadata

    def record_turn(
        self,
        *,
        query: str,
        response: str,
        state: Optional[Mapping[str, Any]] = None,
    ) -> TurnRecord:
        """Record a single agent turn using the provided state."""
        intent = None
        intent_reasoning = None
        intent_confidence = None
        plan_reasoning = None
        tasks_summary: list[dict[str, Any]] = []
        reflection: dict[str, Any] = {}
        structured_data: dict[str, Any] = {}
        report_record: Optional[ReportRecord] = None
        artifacts: list[FileRecord] = []

        if state:
            intent_obj = state.get("intent")
            intent = getattr(intent_obj, "value", intent_obj)
            intent_reasoning = state.get("intent_reasoning")
            intent_confidence = state.get("intent_confidence")
            plan_reasoning = state.get("plan_reasoning")
            tasks_summary = _summarize_tasks(state.get("tasks"))
            reflection = dict(state.get("reflection", {}))
            structured_data = _summarize_results(state.get("results"))
            report_record = _extract_report(state, response)
            artifacts = _extract_artifacts(state)

        turn = TurnRecord(
            turn_id=str(uuid4()),
            query=query,
            response=response,
            intent=intent,
            intent_reasoning=intent_reasoning,
            intent_confidence=intent_confidence,
            plan_reasoning=plan_reasoning,
            tasks=tasks_summary,
            reflection=reflection,
            structured_data=structured_data,
            report=report_record,
            artifacts=artifacts,
        )

        self.metadata.add_turn(turn)
        return turn


def _summarize_tasks(tasks: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    summary: list[dict[str, Any]] = []
    for task_id, task in list(tasks.items())[:10]:
        if not isinstance(task, Mapping):
            continue
        summary.append(
            {
                "id": task_id,
                "type": task.get("type"),
                "status": task.get("status"),
                "description": _safe_preview(str(task.get("description", "")), 160),
                "depends_on": task.get("depends_on", []),
            }
        )
    return summary


def _summarize_results(results: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    summary: dict[str, Any] = {}
    for task_id, value in results.items():
        summary[task_id] = _summarize_result_value(value)
    return summary


def _summarize_result_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        summary: dict[str, Any] = {}
        for key in (
            "error",
            "report_path",
            "report_intent",
            "analysis_mode",
            "target_position",
            "target_molecules",
            "activity_columns",
            "cleaning_stats",
            "output_files",
        ):
            if key in value:
                summary[key] = value[key]

        if "compounds" in value:
            compounds = value.get("compounds", [])
            summary["compound_count"] = len(compounds) if isinstance(compounds, list) else 0

        if "sar_analysis" in value:
            sar = value.get("sar_analysis", {})
            if isinstance(sar, Mapping):
                summary["sar_analysis"] = {
                    "positions": list(sar.get("position_insights", {}))[:5],
                    "summary": _safe_preview(str(sar.get("summary", "")), 200),
                }

        if "ocat_pairs" in value:
            pairs = value.get("ocat_pairs", [])
            summary["ocat_pair_count"] = len(pairs) if isinstance(pairs, list) else 0

        if summary:
            return summary
        return {"keys": list(value.keys())[:10]}

    if isinstance(value, list):
        return {"items": len(value)}

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _extract_report(state: Mapping[str, Any], response: str) -> Optional[ReportRecord]:
    results = state.get("results")
    if not isinstance(results, Mapping):
        return None

    for result in results.values():
        if isinstance(result, Mapping) and result.get("report_path"):
            return ReportRecord(
                report_path=str(result["report_path"]),
                summary=_safe_preview(response, 320),
                preview=_safe_preview(_find_report_preview(result), 320),
            )
    return None


def _find_report_preview(result: Mapping[str, Any]) -> Optional[str]:
    sar_analysis = result.get("sar_analysis")
    if isinstance(sar_analysis, Mapping):
        summary = sar_analysis.get("summary") or sar_analysis.get("optimization_suggestions")
        if isinstance(summary, list):
            return "; ".join(summary)
        if isinstance(summary, str):
            return summary
    return None


def _extract_artifacts(state: Mapping[str, Any]) -> list[FileRecord]:
    results = state.get("results")
    if not isinstance(results, Mapping):
        return []

    artifacts: list[FileRecord] = []
    for task_id, result in results.items():
        if not isinstance(result, Mapping):
            continue

        report_path = result.get("report_path")
        if report_path:
            artifacts.append(
                _build_file_record(
                    path=str(report_path),
                    description=f"{task_id} report",
                )
            )

        output_files = result.get("output_files")
        if isinstance(output_files, Mapping):
            for format_name, path in output_files.items():
                if not isinstance(path, str):
                    continue
                artifacts.append(
                    _build_file_record(
                        path=path,
                        description=f"{task_id} ({format_name})",
                    )
                )

    return artifacts


def _build_file_record(*, path: str, description: Optional[str] = None) -> FileRecord:
    file_path = os.fspath(path)
    file_name = os.path.basename(file_path)
    content_type = mimetypes.guess_type(file_name)[0]
    size: Optional[int]
    try:
        stat = os.stat(file_path)
        size = stat.st_size
    except OSError:
        size = None

    return FileRecord(
        file_id=str(uuid4()),
        file_name=file_name or os.path.basename(file_path),
        file_path=file_path,
        content_type=content_type,
        size_bytes=size,
        description=description,
    )
@dataclass
class FileRecord:
    """Metadata for user uploaded files and generated artifacts."""

    file_id: str
    file_name: str
    file_path: str
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    description: Optional[str] = None
    created_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileRecord":
        return cls(
            file_id=str(data.get("file_id", uuid4())),
            file_name=str(data.get("file_name", "")),
            file_path=str(data.get("file_path", "")),
            content_type=data.get("content_type"),
            size_bytes=data.get("size_bytes"),
            description=data.get("description"),
            created_at=data.get("created_at", _utc_iso()),
        )
