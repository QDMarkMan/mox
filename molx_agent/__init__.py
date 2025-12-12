# type: ignore[attr-defined]
"""Drug design agent"""

from importlib import metadata


def get_version() -> str:
    try:
        return metadata.version(__name__)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
