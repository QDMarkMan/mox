"""Load available chemistry tools."""

# Import only the tools that are compatible with current langchain version
from molx_agent.tools.rdkit import FuncGroups, MolSimilarity, SMILES2Weight

__all__ = [
    "MolSimilarity",
    "SMILES2Weight",
    "FuncGroups",
]

# Try to import additional tools, skip if not compatible
try:
    from molx_agent.tools.converters import Query2CAS, Query2SMILES, SMILES2Name

    __all__.extend(["Query2CAS", "Query2SMILES", "SMILES2Name"])
except ImportError:
    pass

try:
    from molx_agent.tools.safety import (
        ControlChemCheck,
        ExplosiveCheck,
        SafetySummary,
        SimilarControlChemCheck,
    )

    __all__.extend([
        "ExplosiveCheck",
        "ControlChemCheck",
        "SimilarControlChemCheck",
        "SafetySummary",
    ])
except ImportError:
    pass
