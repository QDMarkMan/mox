"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-16].
*  @Description DataCleaner Agent - Extract, parse and clean molecular data.
**************************************************************************
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "output")


# =============================================================================
# Data Extraction Utilities
# =============================================================================

def detect_input_type(content: str) -> str:
    """Detect if input is JSON data, file path, or raw text.

    Args:
        content: User input content.

    Returns:
        One of: 'json', 'file', 'text'
    """
    content = content.strip()

    # Check if it's a file path
    file_patterns = [
        r'^(/[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb))$',
        r'^([A-Za-z]:\\[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb))$',
        r'(/[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb))',
    ]
    for pattern in file_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return 'file'

    # Check if it's JSON
    if content.startswith('{') or content.startswith('['):
        try:
            json.loads(content)
            return 'json'
        except json.JSONDecodeError:
            pass

    return 'text'


def extract_file_path(text: str) -> Optional[str]:
    """Extract file path from text."""
    patterns = [
        r'(/[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb))',
        r'([A-Za-z]:\\[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def read_csv_file(file_path: str) -> dict:
    """Read molecular data from CSV file.

    Args:
        file_path: Path to CSV file.

    Returns:
        Extracted data dictionary.
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    return _extract_from_dataframe(df, file_path)


def read_excel_file(file_path: str) -> dict:
    """Read molecular data from Excel file.

    Args:
        file_path: Path to Excel file.

    Returns:
        Extracted data dictionary.
    """
    import pandas as pd

    df = pd.read_excel(file_path)
    return _extract_from_dataframe(df, file_path)


def read_sdf_file(file_path: str) -> dict:
    """Read molecular data from SDF file.

    Args:
        file_path: Path to SDF file.

    Returns:
        Extracted data dictionary.
    """
    from rdkit import Chem

    compounds = []
    suppl = Chem.SDMolSupplier(file_path)

    for mol in suppl:
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol)
        props = mol.GetPropsAsDict()

        # Find activity property
        activity = None
        for key in props:
            key_lower = key.lower()
            if any(kw in key_lower for kw in ['activity', 'ic50', 'ic90', 'ec50', 'ki']):
                try:
                    activity = float(props[key])
                except (ValueError, TypeError):
                    pass
                break

        compound = {
            "smiles": smiles,
            "activity": activity,
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
    }


def _extract_from_dataframe(df, file_path: str) -> dict:
    """Extract molecular data from pandas DataFrame.

    Args:
        df: Pandas DataFrame.
        file_path: Source file path.

    Returns:
        Extracted data dictionary.
    """
    # Column mapping strategies
    smiles_keywords = ['smiles', 'smi', 'structure', 'canonical_smiles', 'mol']
    activity_keywords = ['ic50', 'ic90', 'ec50', 'ki', 'kd', 'activity', 'potency']
    id_keywords = ['compound', 'id', 'name', 'number', 'cpd', 'mol_id']

    # Find columns
    smiles_col = None
    activity_col = None
    id_col = None

    for col in df.columns:
        col_lower = col.lower()
        if smiles_col is None and any(kw in col_lower for kw in smiles_keywords):
            smiles_col = col
        if activity_col is None and any(kw in col_lower for kw in activity_keywords):
            activity_col = col
        if id_col is None and any(kw in col_lower for kw in id_keywords):
            id_col = col

    compounds = []
    for idx, row in df.iterrows():
        # Get SMILES
        smiles = row.get(smiles_col) if smiles_col else None
        if smiles is None or (isinstance(smiles, float) and smiles != smiles):  # NaN check
            continue
        smiles = str(smiles).strip()
        if not smiles:
            continue

        # Get activity  
        activity = None
        if activity_col:
            act_val = row.get(activity_col)
            if act_val is not None and act_val == act_val:  # NaN check
                try:
                    activity = float(act_val)
                except (ValueError, TypeError):
                    pass

        # Get compound ID
        compound_id = row.get(id_col) if id_col else f"cpd_{idx}"
        if compound_id is None or (isinstance(compound_id, float) and compound_id != compound_id):
            compound_id = f"cpd_{idx}"

        compound = {
            "compound_id": str(compound_id),
            "smiles": smiles,
            "activity": activity,
        }

        # Add all other columns as properties
        other_props = {}
        for col in df.columns:
            if col not in [smiles_col, activity_col, id_col]:
                val = row.get(col)
                if val is not None and val == val:  # NaN check
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
            "activity": activity_col,
            "id": id_col,
        },
        "activity_stats": _calculate_activity_stats([c.get("activity") for c in compounds]),
    }


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


def parse_json_data(json_str: str) -> dict:
    """Parse JSON data input.

    Args:
        json_str: JSON string.

    Returns:
        Parsed data dictionary.
    """
    data = json.loads(json_str)

    if isinstance(data, list):
        # List of compounds
        compounds = []
        for item in data:
            if isinstance(item, dict):
                smiles = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50") or item.get("IC90")
                if smiles:
                    compounds.append({
                        "smiles": smiles,
                        "activity": activity,
                        "compound_id": item.get("compound_id", item.get("id", "")),
                    })
        return {
            "compounds": compounds,
            "source": "json_input",
            "total_molecules": len(compounds),
        }

    elif isinstance(data, dict):
        if "compounds" in data:
            return data
        elif "smiles" in data:
            return {
                "compounds": [data],
                "source": "json_input",
                "total_molecules": 1,
            }

    return {"raw_data": data, "source": "json_input"}


# =============================================================================
# Data Cleaning Utilities
# =============================================================================

def clean_compound_data(compounds: list[dict]) -> list[dict]:
    """Clean and validate compound data.

    Args:
        compounds: List of compound dictionaries.

    Returns:
        Cleaned compounds list.
    """
    from rdkit import Chem

    cleaned = []
    for cpd in compounds:
        smiles = cpd.get("smiles", "")
        if not smiles:
            continue

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES skipped: {smiles[:50]}")
            continue

        # Canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        cleaned_cpd = {
            "smiles": canonical_smiles,
            "original_smiles": smiles if smiles != canonical_smiles else None,
            "activity": cpd.get("activity"),
            "compound_id": cpd.get("compound_id", ""),
        }

        # Copy other properties
        if "properties" in cpd:
            cleaned_cpd["properties"] = cpd["properties"]

        cleaned.append(cleaned_cpd)

    return cleaned


def save_cleaned_data(data: dict, task_id: str) -> dict:
    """Save cleaned data to output files.

    Args:
        data: Cleaned data dictionary.
        task_id: Task identifier.

    Returns:
        Dictionary with output file paths.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"cleaned_{task_id}_{timestamp}"

    output_files = {}

    # Save as JSON
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    output_files["json"] = json_path

    # Save compounds as CSV
    compounds = data.get("compounds", [])
    if compounds:
        import pandas as pd

        # Flatten for CSV
        rows = []
        for cpd in compounds:
            row = {
                "compound_id": cpd.get("compound_id", ""),
                "smiles": cpd.get("smiles", ""),
                "activity": cpd.get("activity"),
            }
            # Add properties
            props = cpd.get("properties", {})
            row.update(props)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
        df.to_csv(csv_path, index=False)
        output_files["csv"] = csv_path

    return output_files


# =============================================================================
# DataCleaner Agent
# =============================================================================

class DataCleanerAgent(BaseAgent):
    """DataCleaner Agent - Extract, parse and clean molecular data.

    Responsibilities:
    1. Detect input type (JSON data, file path, or raw text)
    2. Read different file formats (CSV, Excel, SDF) and extract SMILES + activity
    3. Clean and validate compound data for Drug Design downstream use
    """

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Extracts and cleans molecular data from various sources",
        )

    def run(self, state: AgentState) -> AgentState:
        """Execute data cleaning task.

        Args:
            state: Current agent state with task info.

        Returns:
            Updated state with cleaned data.
        """
        from rich.console import Console

        console = Console()

        tid = state.get("current_task_id")
        if not tid:
            return state

        task = state.get("tasks", {}).get(tid)
        if not task:
            return state

        console.print(f"[cyan]🧹 DataCleaner: Processing task {tid}...[/]")

        try:
            # Get input content
            description = task.get("description", "")
            inputs = task.get("inputs", {})
            user_query = state.get("user_query", "")

            # Try multiple sources for file path or data
            # Priority: explicit inputs > task description > user_query
            content = inputs.get("data") or inputs.get("file_path") or ""
            
            # If no explicit input, try to find file path in description or user_query
            if not content or detect_input_type(content) == 'text':
                # Check if description contains a file path
                file_in_desc = extract_file_path(description)
                if file_in_desc and os.path.exists(file_in_desc):
                    content = file_in_desc
                else:
                    # Check if user_query contains a file path
                    file_in_query = extract_file_path(user_query)
                    if file_in_query and os.path.exists(file_in_query):
                        content = file_in_query
                    else:
                        # Fall back to description or user_query
                        content = description or user_query

            # Step 1: Detect input type
            input_type = detect_input_type(content)
            console.print(f"[dim]   Input type: {input_type}[/]")

            # Step 2: Extract data based on type
            if input_type == 'json':
                console.print("[dim]   Parsing JSON data...[/]")
                data = parse_json_data(content)

            elif input_type == 'file':
                file_path = extract_file_path(content)
                if not file_path or not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                console.print(f"[dim]   Reading file: {file_path}[/]")

                ext = file_path.lower().split('.')[-1]
                if ext == 'csv':
                    data = read_csv_file(file_path)
                elif ext in ['xlsx', 'xls']:
                    data = read_excel_file(file_path)
                elif ext == 'sdf':
                    data = read_sdf_file(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")

            else:
                # Try to extract file path from text
                file_path = extract_file_path(content)
                if file_path and os.path.exists(file_path):
                    console.print(f"[dim]   Found file in text: {file_path}[/]")
                    ext = file_path.lower().split('.')[-1]
                    if ext == 'csv':
                        data = read_csv_file(file_path)
                    elif ext in ['xlsx', 'xls']:
                        data = read_excel_file(file_path)
                    elif ext == 'sdf':
                        data = read_sdf_file(file_path)
                    else:
                        raise ValueError(f"Unsupported file type: {ext}")
                else:
                    raise ValueError("No valid data source found in input")

            # Step 3: Clean compound data
            compounds = data.get("compounds", [])
            if compounds:
                console.print(f"[dim]   Cleaning {len(compounds)} compounds...[/]")
                cleaned_compounds = clean_compound_data(compounds)
                data["compounds"] = cleaned_compounds
                data["cleaning_stats"] = {
                    "original_count": len(compounds),
                    "cleaned_count": len(cleaned_compounds),
                    "removed": len(compounds) - len(cleaned_compounds),
                }

            # Step 4: Save output files
            output_files = save_cleaned_data(data, tid)
            data["output_files"] = output_files

            # Update state
            state["results"][tid] = data
            state["tasks"][tid]["status"] = "done"

            # Print summary
            n_compounds = len(data.get("compounds", []))
            n_with_activity = len([c for c in data.get("compounds", []) if c.get("activity") is not None])

            console.print(f"[green]✓ DataCleaner: Extracted {n_compounds} compounds[/]")
            console.print(f"[dim]   With activity: {n_with_activity}[/]")
            for fmt, path in output_files.items():
                console.print(f"[dim]   {fmt.upper()}: {path}[/]")

        except Exception as e:
            console.print(f"[red]✗ DataCleaner error: {e}[/]")
            logger.error(f"DataCleaner error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"

        return state
