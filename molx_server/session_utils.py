"""Utilities for converting session metadata into API responses."""

from molx_agent.memory import SessionMetadata
from molx_server.schemas.models import SessionMetadataResponse


def metadata_to_response(
    session_id: str, metadata: SessionMetadata
) -> SessionMetadataResponse:
    """Convert SessionMetadata into the API response model."""
    return SessionMetadataResponse(
        session_id=session_id,
        latest=metadata.latest,
        structured_data=metadata.structured_data,
        reports=[report.to_dict() for report in metadata.reports],
        turns=[turn.to_dict() for turn in metadata.turns],
    )
