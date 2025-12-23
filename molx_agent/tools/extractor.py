"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com 
*  @Date [2025-12-17 11:21:20].
*  @Description Extractor tool.
**************************************************************************
"""

import json
import logging
import re
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def identify_columns_with_llm(columns: list[str], llm: Any) -> dict:
    """Identify special columns using LLM.
    
    Args:
        columns: List of column names.
        llm: LLM instance.
        
    Returns:
        Dictionary with identified column names.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    import json
    
    prompt = f"""
    Given the following CSV/Excel column names: {columns}
    
    Identify the column names for:
    1. SMILES (molecular structure). Look for 'smiles', 'structure', 'mol', etc.
    2. Compound ID (name/number). Look for 'id', 'name', 'number', 'code', etc.
    3. Activity Data. Look for 'IC50', 'EC50', 'Ki', 'activity', 'inhibition', etc. There can be MULTIPLE activity columns.
    
    Return ONLY a JSON object with this structure:
    {{
        "smiles_col": "column_name_or_null",
        "id_col": "column_name_or_null",
        "activity_cols": ["col1", "col2"] 
    }}
    If a column is not found, set it to null (or empty list for activity_cols).
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a data processing assistant. Output valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        result = json.loads(content)
        return result
    except Exception as e:
        logger.error(f"LLM column identification failed: {e}")
        return {}


def _calculate_activity_stats(activities: list) -> dict:
    """Calculate activity statistics."""
    valid = [a for a in activities if a is not None]
    if not valid:
        return {"valid_count": 0}

    return {
        "valid_count": len(valid),
        "min": round(min(valid), 4),
        "max": round(max(valid), 4),
        "mean": round(sum(valid) / len(valid), 4),
    }


def _extract_from_dataframe(df, file_path: str, llm: Any = None) -> dict:
    """Extract molecular data from pandas DataFrame.

    Args:
        df: Pandas DataFrame.
        file_path: Source file path.
        llm: Optional LLM for intelligent column identification.

    Returns:
        Extracted data dictionary.
    """
    # Default strategies
    smiles_keywords = ['smiles', 'smi', 'structure', 'canonical_smiles', 'mol']
    activity_keywords = ['ic50', 'ic90', 'ec50', 'ki', 'kd', 'activity', 'potency']
    id_keywords = ['compound', 'id', 'name', 'number', 'cpd', 'mol_id']

    smiles_col = None
    id_col = None
    activity_cols = []

    # 1. Try LLM if available
    if llm:
        try:
            llm_result = identify_columns_with_llm(list(df.columns), llm)
            if llm_result.get("smiles_col"):
                smiles_col = llm_result["smiles_col"]
            if llm_result.get("id_col"):
                id_col = llm_result["id_col"]
            if llm_result.get("activity_cols"):
                activity_cols = llm_result["activity_cols"]
        except Exception as e:
            logger.warning(f"LLM identification failed, falling back to keywords: {e}")

    # 2. Fallback to keywords if not found
    if not smiles_col:
        for col in df.columns:
            if any(kw in col.lower() for kw in smiles_keywords):
                smiles_col = col
                break
    
    if not id_col:
        for col in df.columns:
            if any(kw in col.lower() for kw in id_keywords):
                id_col = col
                break
                
    if not activity_cols:
        for col in df.columns:
            if any(kw in col.lower() for kw in activity_keywords):
                activity_cols.append(col)
    
    # Ensure columns exist in DF (LLM might hallucinate)
    if smiles_col and smiles_col not in df.columns: smiles_col = None
    if id_col and id_col not in df.columns: id_col = None
    activity_cols = [c for c in activity_cols if c in df.columns]

    compounds = []
    for idx, row in df.iterrows():
        # Get SMILES
        smiles = row.get(smiles_col) if smiles_col else None
        if smiles is None or (isinstance(smiles, float) and smiles != smiles):
            continue
        smiles = str(smiles).strip()
        if not smiles:
            continue

        # Get activities
        activities = {}
        primary_activity = None
        
        for col in activity_cols:
            val = row.get(col)
            if val is not None and val == val:
                try:
                    # Handle ">200" etc.
                    if isinstance(val, str):
                        val_clean = val.replace('>', '').replace('<', '').strip()
                        float_val = float(val_clean)
                    else:
                        float_val = float(val)
                    
                    activities[col] = float_val
                    if primary_activity is None:
                        primary_activity = float_val
                except (ValueError, TypeError):
                    # Keep original string if cannot parse
                    activities[col] = val

        # Get compound ID
        compound_id = row.get(id_col) if id_col else f"cpd_{idx}"
        if compound_id is None or (isinstance(compound_id, float) and compound_id != compound_id):
            compound_id = f"cpd_{idx}"

        compound = {
            "compound_id": str(compound_id),
            "smiles": smiles,
            "activity": primary_activity, # Primary activity for backward compatibility
            "activities": activities,     # All activities
        }

        # Add all other columns as properties
        other_props = {}
        for col in df.columns:
            if col != smiles_col and col != id_col and col not in activity_cols:
                val = row.get(col)
                if val is not None and val == val:
                    other_props[col] = val
        if other_props:
            compound["properties"] = other_props

        compounds.append(compound)

    return {
        "compounds": compounds,
        "source_file": file_path,
        "file_type": file_path.split('.')[-1].lower(),
        "total_molecules": len(compounds),
        "columns": {
            "smiles": smiles_col,
            "id": id_col,
            "activity_cols": activity_cols,
        },
        "activity_columns": activity_cols,  # Explicit list for downstream multi-activity support
        "activity_stats": _calculate_activity_stats([c.get("activity") for c in compounds]),
    }


# =============================================================================
# Tools
# =============================================================================

class ExtractFromCSVInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file")

class ExtractFromCSVTool(BaseTool):
    name: str = "extract_from_csv"
    description: str = "Extract molecular data from a CSV file."
    args_schema: Type[BaseModel] = ExtractFromCSVInput
    llm: Optional[Any] = None

    def __init__(self, llm: Optional[Any] = None):
        super().__init__()
        self.llm = llm

    def _run(self, file_path: str) -> dict:
        import pandas as pd
        df = pd.read_csv(file_path)
        return _extract_from_dataframe(df, file_path, self.llm)


class ExtractFromExcelInput(BaseModel):
    file_path: str = Field(description="Path to the Excel file")

class ExtractFromExcelTool(BaseTool):
    name: str = "extract_from_excel"
    description: str = "Extract molecular data from an Excel file."
    args_schema: Type[BaseModel] = ExtractFromExcelInput
    llm: Optional[Any] = None

    def __init__(self, llm: Optional[Any] = None):
        super().__init__()
        self.llm = llm

    def _run(self, file_path: str) -> dict:
        import pandas as pd
        df = pd.read_excel(file_path)
        return _extract_from_dataframe(df, file_path, self.llm)


class ExtractFromSDFInput(BaseModel):
    file_path: str = Field(description="Path to the SDF file")

class ExtractFromSDFTool(BaseTool):
    name: str = "extract_from_sdf"
    description: str = "Extract molecular data from an SDF file."
    args_schema: Type[BaseModel] = ExtractFromSDFInput

    def _run(self, file_path: str) -> dict:
        from rdkit import Chem

        compounds = []
        activity_cols = []
        activity_keywords = ['activity', 'ic50', 'ic90', 'ec50', 'ki', 'kd', 'potency', 'inhibition']
        suppl = Chem.SDMolSupplier(file_path)

        for mol in suppl:
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol)
            props = mol.GetPropsAsDict()

            # Find all activity properties and build activities dict
            activities = {}
            primary_activity = None
            
            for key in props:
                key_lower = key.lower()
                if any(kw in key_lower for kw in activity_keywords):
                    try:
                        val = props[key]
                        # Handle string values like ">200"
                        if isinstance(val, str):
                            val_clean = val.replace('>', '').replace('<', '').strip()
                            float_val = float(val_clean)
                        else:
                            float_val = float(val)
                        
                        activities[key] = float_val
                        if primary_activity is None:
                            primary_activity = float_val
                        
                        # Track activity columns
                        if key not in activity_cols:
                            activity_cols.append(key)
                    except (ValueError, TypeError):
                        # Keep original string if cannot parse
                        activities[key] = props[key]

            compound = {
                "smiles": smiles,
                "activity": primary_activity,
                "activities": activities,
                "properties": props,
            }

            # Extract compound name/ID
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            if name:
                compound["compound_id"] = name

            compounds.append(compound)

        return {
            "compounds": compounds,
            "source_file": file_path,
            "file_type": "sdf",
            "total_molecules": len(compounds),
            "activity_columns": activity_cols,
        }

