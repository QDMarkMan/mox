"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com 
*  @Date [2025-12-15 14:38:16].
*  @Description Standardize & Normalize.
**************************************************************************
"""

from typing import Optional
from langchain_core.tools import BaseTool
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import json

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
