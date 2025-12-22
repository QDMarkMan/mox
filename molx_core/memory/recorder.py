"""Session metadata schema and recorder utilities.

Defines structured metadata for each agent session and helpers for
recording turn-by-turn information that can be persisted by any
ConversationStore implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover
    from molx_core.memory.store import SessionData


MAX_TURNS_STORED = 50
MAX_REPORTS_STORED = 20


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
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TurnRecord":
        report = data.get("report")
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
            created_at=data.get("created_at", _utc_iso()),
        )


@dataclass
class SessionMetadata:
    """Structured session metadata stored alongside conversations."""

    turns: list[TurnRecord] = field(default_factory=list)
    latest: dict[str, Any] = field(default_factory=dict)
    structured_data: dict[str, Any] = field(default_factory=dict)
    reports: list[ReportRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "latest": self.latest,
            "structured_data": self.structured_data,
            "reports": [report.to_dict() for report in self.reports],
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "SessionMetadata":
        if not data:
            return cls()
        turns_data = data.get("turns", []) if isinstance(data, Mapping) else []
        reports_data = data.get("reports", []) if isinstance(data, Mapping) else []
        metadata = cls(
            turns=[TurnRecord.from_dict(turn) for turn in turns_data],
            latest=dict(data.get("latest", {})),
            structured_data=dict(data.get("structured_data", {})),
            reports=[ReportRecord.from_dict(report) for report in reports_data],
        )
        metadata._truncate()
        return metadata

    def add_turn(self, turn: TurnRecord) -> None:
        self.turns.append(turn)
        self.latest = turn.to_dict()

        if turn.report:
            self.reports.append(turn.report)

        if turn.structured_data:
            self._merge_structured(turn.structured_data)

        self._truncate()

    def _truncate(self) -> None:
        if len(self.turns) > MAX_TURNS_STORED:
            self.turns = self.turns[-MAX_TURNS_STORED:]
        if len(self.reports) > MAX_REPORTS_STORED:
            self.reports = self.reports[-MAX_REPORTS_STORED:]

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
