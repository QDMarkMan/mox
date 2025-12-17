"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com 
*  @Date [2025-12-16 18:35:19].
*  @Description Chemical molecule converters.
*  @Todo: Need test
**************************************************************************
"""


from typing import Any, Optional

from langchain_core.tools import BaseTool

from molx_agent.tools.utils import (
    is_multiple_smiles,
    is_smiles,
    pubchem_query2smiles,
    query2cas,
    smiles2name,
)


class Query2SMILES(BaseTool):
    """Convert molecule name to SMILES."""

    name: str = "Name2SMILES"
    description: str = "Input a molecule name, returns SMILES."
    url: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"

    def _run(self, query: str) -> str:
        if is_smiles(query) and is_multiple_smiles(query):
            return "Multiple SMILES strings detected, input one molecule at a time."
        try:
            smi = pubchem_query2smiles(query, self.url)
            return smi
        except Exception as e:
            return str(e)


class Query2CAS(BaseTool):
    """Convert molecule to CAS number."""

    name: str = "Mol2CAS"
    description: str = "Input molecule (name or SMILES), returns CAS number."
    url_cid: Optional[str] = None
    url_data: Optional[str] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url_cid = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{}/{}/cids/JSON"
        )
        self.url_data = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{}/JSON"
        )

    def _run(self, query: str) -> str:
        try:
            smiles = query if is_smiles(query) else None
            try:
                cas = query2cas(query, self.url_cid, self.url_data)
            except ValueError as e:
                return str(e)
            return cas
        except ValueError:
            return "CAS number not found"


class SMILES2Name(BaseTool):
    """Convert SMILES to molecule name."""

    name: str = "SMILES2Name"
    description: str = "Input SMILES, returns molecule name."

    def _run(self, query: str) -> str:
        try:
            if not is_smiles(query):
                try:
                    query = pubchem_query2smiles(query, None)
                except Exception:
                    raise ValueError("Invalid molecule input, no Pubchem entry")
            name = smiles2name(query)
            return name
        except Exception as e:
            return "Error: " + str(e)

