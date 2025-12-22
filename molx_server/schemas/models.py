"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class AgentStatus(str, Enum):
    """Agent execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStatus(str, Enum):
    """Session status."""

    ACTIVE = "active"
    EXPIRED = "expired"
    CLOSED = "closed"


# =============================================================================
# Agent Request/Response Models
# =============================================================================


class AgentRequest(BaseModel):
    """Request model for agent invocation."""

    query: str = Field(..., description="User query to process", min_length=1)
    session_id: Optional[str] = Field(
        None, description="Session ID for multi-turn conversations"
    )
    config: Optional[dict[str, Any]] = Field(
        None, description="Optional agent configuration overrides"
    )
    stream: bool = Field(False, description="Enable streaming response")


class AgentResponse(BaseModel):
    """Response model for agent invocation."""

    result: str = Field(..., description="Agent response content")
    structured_data: Optional[dict[str, Any]] = Field(
        None, description="Structured data from analysis"
    )
    session_id: str = Field(..., description="Session ID for this conversation")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Response metadata (timing, tokens, etc.)"
    )
    status: AgentStatus = Field(
        default=AgentStatus.COMPLETED, description="Execution status"
    )


class AgentErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type/class name")
    detail: Optional[str] = Field(None, description="Detailed error information")
    session_id: Optional[str] = Field(None, description="Session ID if available")


# =============================================================================
# SAR Analysis Models
# =============================================================================


class SARAnalysisRequest(BaseModel):
    """Request model for SAR analysis."""

    query: str = Field(..., description="SAR analysis query", min_length=1)
    compounds: Optional[list[dict[str, Any]]] = Field(
        None, description="Optional list of compound data"
    )
    session_id: Optional[str] = Field(None, description="Session ID")
    config: Optional[dict[str, Any]] = Field(None, description="Analysis configuration")


class SARAnalysisResponse(BaseModel):
    """Response model for SAR analysis."""

    report: str = Field(..., description="SAR analysis report (markdown)")
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Structured SAR data"
    )
    session_id: str = Field(..., description="Session ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")


# =============================================================================
# Session Models
# =============================================================================


class SessionInfo(BaseModel):
    """Session information model."""

    session_id: str = Field(..., description="Unique session identifier")
    status: SessionStatus = Field(..., description="Session status")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    message_count: int = Field(0, description="Number of messages in session")
    preview: Optional[str] = Field(
        None, description="Latest user query preview for display"
    )


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: list[SessionInfo] = Field(
        default_factory=list, description="Active sessions sorted by activity"
    )


class SessionCreateResponse(BaseModel):
    """Response for session creation."""

    session_id: str = Field(..., description="Created session ID")
    created_at: datetime = Field(..., description="Creation timestamp")


class SessionHistoryResponse(BaseModel):
    """Response for session history."""

    session_id: str = Field(..., description="Session ID")
    messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Conversation history"
    )


class SessionReportInfo(BaseModel):
    """Report metadata for a session."""

    report_path: str = Field(..., description="Path to generated report")
    summary: Optional[str] = Field(None, description="High-level summary")
    preview: Optional[str] = Field(None, description="Preview snippet")
    created_at: datetime = Field(..., description="Report creation time")


class SessionTurnRecord(BaseModel):
    """Structured turn data stored in session metadata."""

    turn_id: str = Field(..., description="Unique turn identifier")
    query: str = Field(..., description="User query content")
    response: str = Field(..., description="Agent response")
    intent: Optional[str] = Field(None, description="Detected intent value")
    intent_reasoning: Optional[str] = Field(None, description="Intent reasoning text")
    intent_confidence: Optional[float] = Field(None, description="Intent confidence score")
    plan_reasoning: Optional[str] = Field(None, description="Planner reasoning")
    tasks: list[dict[str, Any]] = Field(
        default_factory=list, description="Summary of tasks executed"
    )
    reflection: dict[str, Any] = Field(
        default_factory=dict, description="Reflection output"
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Structured results captured"
    )
    report: Optional[SessionReportInfo] = Field(
        None, description="Report information if generated"
    )
    created_at: datetime = Field(..., description="Turn timestamp")


class SessionMetadataResponse(BaseModel):
    """Response for session metadata retrieval."""

    session_id: str = Field(..., description="Session identifier")
    latest: dict[str, Any] = Field(
        default_factory=dict, description="Latest turn snapshot"
    )
    structured_data: dict[str, Any] = Field(
        default_factory=dict, description="Structured data index"
    )
    reports: list[SessionReportInfo] = Field(
        default_factory=list, description="Report metadata entries"
    )
    turns: list[SessionTurnRecord] = Field(
        default_factory=list, description="Recorded turns"
    )


# =============================================================================
# Health Check Models
# =============================================================================


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus = Field(..., description="Overall health status")
    version: str = Field(..., description="Server version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    components: dict[str, HealthStatus] = Field(
        default_factory=dict, description="Component health statuses"
    )


class ReadyResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether the server is ready")
    checks: dict[str, bool] = Field(
        default_factory=dict, description="Individual readiness checks"
    )


# =============================================================================
# Batch Processing Models
# =============================================================================


class BatchRequest(BaseModel):
    """Request for batch processing."""

    queries: list[str] = Field(
        ..., description="List of queries to process", min_length=1, max_length=100
    )
    session_id: Optional[str] = Field(None, description="Shared session ID")
    config: Optional[dict[str, Any]] = Field(None, description="Shared configuration")


class BatchItemResult(BaseModel):
    """Result for a single batch item."""

    index: int = Field(..., description="Original query index")
    query: str = Field(..., description="Original query")
    result: Optional[str] = Field(None, description="Result if successful")
    error: Optional[str] = Field(None, description="Error if failed")
    status: AgentStatus = Field(..., description="Item status")


class BatchResponse(BaseModel):
    """Response for batch processing."""

    results: list[BatchItemResult] = Field(..., description="Individual results")
    total: int = Field(..., description="Total number of queries")
    successful: int = Field(..., description="Number of successful queries")
    failed: int = Field(..., description="Number of failed queries")
    session_id: str = Field(..., description="Session ID")
