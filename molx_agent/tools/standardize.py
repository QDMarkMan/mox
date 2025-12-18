"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com 
*  @Date [2025-12-17].
*  @Description Standardize & Normalize tools for molecular data.
**************************************************************************
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "output")


class StandardizeMolecule(BaseTool):
    """Standardize and normalize molecular structures."""

    name: str = "StandardizeMolecule"
    description: str = (
        "Standardize a SMILES string: remove salts, neutralize charges, "
        "canonicalize tautomers. Input: SMILES string."
    )

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Error: Invalid SMILES '{smiles}'"

            # 1. Remove fragments (keep largest)
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # 2. Remove salts
            remover = rdMolStandardize.FragmentRemover()
            mol = remover.remove(mol)

            # 3. Normalize
            normalizer = rdMolStandardize.Normalizer()
            mol = normalizer.normalize(mol)

            # 4. Canonicalize tautomer
            enumerator = rdMolStandardize.TautomerEnumerator()
            mol = enumerator.Canonicalize(mol)

            std_smiles = Chem.MolToSmiles(mol, canonical=True)

            return json.dumps({
                "original": smiles,
                "standardized": std_smiles,
                "atoms": mol.GetNumAtoms(),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            })
        except Exception as e:
            return f"Error: {e}"


class BatchStandardize(BaseTool):
    """Batch standardize multiple molecules."""

    name: str = "BatchStandardize"
    description: str = (
        "Standardize multiple SMILES (comma or newline separated). "
        "Returns list of standardized SMILES."
    )

    def _run(self, smiles_list: str) -> str:
        try:
            # Parse input
            smiles_items = [s.strip() for s in smiles_list.replace(",", "\n").split("\n") if s.strip()]

            results = []
            standardizer = StandardizeMolecule()

            for smi in smiles_items:
                result = json.loads(standardizer._run(smi))
                if "standardized" in result:
                    results.append({
                        "original": smi,
                        "standardized": result["standardized"],
                        "valid": True,
                    })
                else:
                    results.append({"original": smi, "valid": False, "error": result})

            valid_count = sum(1 for r in results if r.get("valid"))
            return json.dumps({
                "total": len(results),
                "valid": valid_count,
                "invalid": len(results) - valid_count,
                "results": results,
            })
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# Compound Data Cleaning Tools
# =============================================================================

class CleanCompoundDataInput(BaseModel):
    """Input for CleanCompoundDataTool."""
    compounds: list[dict] = Field(description="List of compound dictionaries with smiles, activity, etc.")


class CleanCompoundDataTool(BaseTool):
    """Clean and validate compound data - canonicalize SMILES, filter invalid."""

    name: str = "clean_compound_data"
    description: str = (
        "Clean and validate compound data. Canonicalizes SMILES, "
        "removes invalid molecules, preserves activities and properties."
    )
    args_schema: type[BaseModel] = CleanCompoundDataInput

    def _run(self, compounds: list[dict]) -> dict:
        """Clean compound data."""
        cleaned = []
        for cpd in compounds:
            smiles = cpd.get("smiles", "")
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES skipped: {smiles[:50]}")
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            cleaned_cpd = {
                "smiles": canonical_smiles,
                "original_smiles": smiles if smiles != canonical_smiles else None,
                "activity": cpd.get("activity"),
                "compound_id": cpd.get("compound_id", ""),
            }
            
            if "activities" in cpd:
                cleaned_cpd["activities"] = cpd["activities"]

            if "properties" in cpd:
                cleaned_cpd["properties"] = cpd["properties"]

            cleaned.append(cleaned_cpd)

        return {
            "compounds": cleaned,
            "original_count": len(compounds),
            "cleaned_count": len(cleaned),
            "removed": len(compounds) - len(cleaned),
        }


class SaveCleanedDataInput(BaseModel):
    """Input for SaveCleanedDataTool."""
    data: dict = Field(description="Data dictionary with compounds to save")
    task_id: str = Field(default="data_cleaner", description="Task identifier for filename")


class SaveCleanedDataTool(BaseTool):
    """Save cleaned compound data to JSON and CSV files."""

    name: str = "save_cleaned_data"
    description: str = (
        "Save cleaned compound data to output files (JSON and CSV). "
        "Returns paths to saved files."
    )
    args_schema: type[BaseModel] = SaveCleanedDataInput

    def _run(self, data: dict, task_id: str = "data_cleaner") -> dict:
        """Save data to files."""
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

            rows = []
            for cpd in compounds:
                row = {
                    "compound_id": cpd.get("compound_id", ""),
                    "smiles": cpd.get("smiles", ""),
                    "activity": cpd.get("activity"),
                }
                props = cpd.get("properties", {})
                row.update(props)
                rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
            df.to_csv(csv_path, index=False)
            output_files["csv"] = csv_path

        return output_files

