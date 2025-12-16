"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-15].
*  @Description SAR core tools for structure-activity relationship analysis.
**************************************************************************
"""

import json
import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors, rdRGroupDecomposition
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

logger = logging.getLogger(__name__)


# =============================================================================
# Scaffold Analysis & Clustering
# =============================================================================


class ExtractScaffold(BaseTool):
    """Extract Murcko scaffold from a molecule."""

    name: str = "ExtractScaffold"
    description: str = (
        "Extract Murcko scaffold (core ring structure) from a SMILES. "
        "Returns both generic and specific scaffolds."
    )

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Error: Invalid SMILES '{smiles}'"

            # Get Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)

            # Get generic scaffold (all atoms -> C, all bonds -> single)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            generic_smi = Chem.MolToSmiles(generic)

            return json.dumps({
                "original": smiles,
                "scaffold": scaffold_smi,
                "generic_scaffold": generic_smi,
                "num_rings": scaffold.GetRingInfo().NumRings(),
            })
        except Exception as e:
            return f"Error: {e}"


class ClusterMolecules(BaseTool):
    """Cluster molecules by structural similarity."""

    name: str = "ClusterMolecules"
    description: str = (
        "Cluster molecules by Tanimoto similarity. "
        "Input: comma-separated SMILES. Optional threshold (default 0.7)."
    )
    threshold: float = 0.7

    def _run(self, smiles_input: str) -> str:
        try:
            # Parse input
            parts = smiles_input.split("|")
            smiles_str = parts[0]
            threshold = float(parts[1]) if len(parts) > 1 else self.threshold

            smiles_list = [s.strip() for s in smiles_str.replace(",", "\n").split("\n") if s.strip()]

            if len(smiles_list) < 2:
                return "Error: Need at least 2 molecules to cluster"

            # Generate fingerprints
            mols = []
            fps = []
            valid_smiles = []

            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                    valid_smiles.append(smi)

            if len(fps) < 2:
                return "Error: Less than 2 valid molecules"

            # Calculate distance matrix
            n = len(fps)
            dists = []
            for i in range(1, n):
                for j in range(i):
                    dists.append(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))

            # Butina clustering
            clusters = Butina.ClusterData(dists, n, 1 - threshold, isDistData=True)

            # Format results
            cluster_results = []
            for i, cluster in enumerate(clusters):
                cluster_smiles = [valid_smiles[idx] for idx in cluster]
                cluster_results.append({
                    "cluster_id": i,
                    "size": len(cluster),
                    "members": cluster_smiles,
                    "centroid": cluster_smiles[0],  # First is centroid
                })

            return json.dumps({
                "total_molecules": len(valid_smiles),
                "num_clusters": len(clusters),
                "threshold": threshold,
                "clusters": cluster_results,
            })
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# R-Group Analysis
# =============================================================================

class DefineRSites(BaseTool):
    """Define R-sites on a core scaffold."""

    name: str = "DefineRSites"
    description: str = (
        "Analyze a scaffold SMILES and identify possible R-group attachment sites. "
        "Returns positions where R-groups can be attached."
    )

    def _run(self, scaffold_smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(scaffold_smiles)
            if mol is None:
                return f"Error: Invalid SMILES '{scaffold_smiles}'"

            # Find atoms that could be R-group attachment points
            r_sites = []

            for atom in mol.GetAtoms():
                # Check if atom has available valence for attachment
                explicit_valence = atom.GetTotalValence()
                default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())

                # For carbon atoms in rings, check for substitution points
                if atom.GetAtomicNum() == 6:  # Carbon
                    # Ring carbons with hydrogen
                    if atom.IsInRing() and atom.GetTotalNumHs() > 0:
                        r_sites.append({
                            "atom_idx": atom.GetIdx(),
                            "atom_symbol": atom.GetSymbol(),
                            "in_ring": True,
                            "num_hydrogens": atom.GetTotalNumHs(),
                            "neighbors": [n.GetSymbol() for n in atom.GetNeighbors()],
                        })

            # Create SMILES with R-group labels if possible
            core_with_r = scaffold_smiles
            for i, site in enumerate(r_sites[:4]):  # Max 4 R-groups
                site["r_label"] = f"R{i + 1}"

            return json.dumps({
                "scaffold": scaffold_smiles,
                "num_r_sites": len(r_sites),
                "r_sites": r_sites[:10],  # Limit output
                "suggestion": "Use RGroupDecomposition tool with core:[*:1]...[*:n] format",
            })
        except Exception as e:
            return f"Error: {e}"


class RGroupDecomposition(BaseTool):
    """Perform R-group decomposition on a set of molecules."""

    name: str = "RGroupDecomposition"
    description: str = (
        "Decompose molecules into core + R-groups. "
        "Input format: 'core_smiles|mol1,mol2,mol3' where core has [*:1], [*:2] etc."
    )

    def _run(self, input_str: str) -> str:
        try:
            parts = input_str.split("|")
            if len(parts) != 2:
                return "Error: Format should be 'core_smiles|mol1,mol2,mol3'"

            core_smiles = parts[0].strip()
            mol_smiles = [s.strip() for s in parts[1].split(",") if s.strip()]

            # Parse core
            core = Chem.MolFromSmiles(core_smiles)
            if core is None:
                return f"Error: Invalid core SMILES '{core_smiles}'"

            # Parse molecules
            mols = []
            valid_smiles = []
            for smi in mol_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)
                    valid_smiles.append(smi)

            if not mols:
                return "Error: No valid molecules to decompose"

            # Perform R-group decomposition
            rgd = rdRGroupDecomposition.RGroupDecomposition(core)

            results = []
            for i, mol in enumerate(mols):
                match = rgd.Add(mol)
                if match >= 0:
                    results.append({
                        "molecule": valid_smiles[i],
                        "matched": True,
                    })
                else:
                    results.append({
                        "molecule": valid_smiles[i],
                        "matched": False,
                    })

            # Get decomposition results
            rgd.Process()
            rgroups = rgd.GetRGroupsAsColumns()

            # Format R-group data
            rgroup_data = {}
            for key, values in rgroups.items():
                rgroup_data[key] = [Chem.MolToSmiles(v) if v else None for v in values]

            matched_count = sum(1 for r in results if r["matched"])

            return json.dumps({
                "core": core_smiles,
                "total_molecules": len(mol_smiles),
                "matched": matched_count,
                "unmatched": len(mol_smiles) - matched_count,
                "r_groups": rgroup_data,
                "decomposition": results,
            })
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# MMP (Matched Molecular Pair) Analysis
# =============================================================================

class FindMCS(BaseTool):
    """Find Maximum Common Substructure between molecules."""

    name: str = "FindMCS"
    description: str = (
        "Find the maximum common substructure (MCS) between molecules. "
        "Input: comma-separated SMILES of 2+ molecules."
    )

    def _run(self, smiles_input: str) -> str:
        try:
            smiles_list = [s.strip() for s in smiles_input.split(",") if s.strip()]

            if len(smiles_list) < 2:
                return "Error: Need at least 2 molecules"

            mols = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)

            if len(mols) < 2:
                return "Error: Less than 2 valid molecules"

            # Find MCS
            mcs_result = rdFMCS.FindMCS(
                mols,
                timeout=10,
                matchValences=True,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
            )

            mcs_smarts = mcs_result.smartsString
            mcs_mol = Chem.MolFromSmarts(mcs_smarts)

            return json.dumps({
                "input_molecules": smiles_list,
                "mcs_smarts": mcs_smarts,
                "mcs_atoms": mcs_result.numAtoms,
                "mcs_bonds": mcs_result.numBonds,
                "canceled": mcs_result.canceled,
            })
        except Exception as e:
            return f"Error: {e}"


class AnalyzeMMP(BaseTool):
    """Analyze matched molecular pairs for SAR."""

    name: str = "AnalyzeMMP"
    description: str = (
        "Analyze two molecules as a matched molecular pair. "
        "Input: 'smiles1|smiles2|activity1|activity2' to compare structures and activities."
    )

    def _run(self, input_str: str) -> str:
        try:
            parts = input_str.split("|")
            if len(parts) < 2:
                return "Error: Need at least 2 SMILES separated by |"

            smi1, smi2 = parts[0].strip(), parts[1].strip()
            act1 = float(parts[2]) if len(parts) > 2 and parts[2].strip() else None
            act2 = float(parts[3]) if len(parts) > 3 and parts[3].strip() else None

            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)

            if not mol1 or not mol2:
                return "Error: Invalid SMILES"

            # Calculate fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            # Find MCS
            mcs = rdFMCS.FindMCS([mol1, mol2], timeout=5)
            mcs_smarts = mcs.smartsString

            # Calculate property differences
            props = {
                "MW": (Descriptors.MolWt(mol1), Descriptors.MolWt(mol2)),
                "LogP": (Descriptors.MolLogP(mol1), Descriptors.MolLogP(mol2)),
                "HBD": (Descriptors.NumHDonors(mol1), Descriptors.NumHDonors(mol2)),
                "HBA": (Descriptors.NumHAcceptors(mol1), Descriptors.NumHAcceptors(mol2)),
                "TPSA": (Descriptors.TPSA(mol1), Descriptors.TPSA(mol2)),
                "RotBonds": (Descriptors.NumRotatableBonds(mol1), Descriptors.NumRotatableBonds(mol2)),
            }

            prop_diff = {k: round(v[1] - v[0], 2) for k, v in props.items()}

            result = {
                "mol1": smi1,
                "mol2": smi2,
                "similarity": round(similarity, 4),
                "mcs_smarts": mcs_smarts,
                "mcs_atoms": mcs.numAtoms,
                "property_changes": prop_diff,
            }

            if act1 is not None and act2 is not None:
                result["activity_change"] = round(act2 - act1, 4)
                result["fold_change"] = round(act2 / act1, 4) if act1 != 0 else None

            return json.dumps(result)
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# Property Calculation & Validation
# =============================================================================

class CalculateProperties(BaseTool):
    """Calculate molecular properties for SAR analysis."""

    name: str = "CalculateProperties"
    description: str = (
        "Calculate key molecular properties (MW, LogP, HBD, HBA, TPSA, etc.). "
        "Input: SMILES string."
    )

    def _run(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Error: Invalid SMILES '{smiles}'"

            props = {
                "smiles": smiles,
                "MW": round(Descriptors.MolWt(mol), 2),
                "LogP": round(Descriptors.MolLogP(mol), 2),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "TPSA": round(Descriptors.TPSA(mol), 2),
                "RotatableBonds": Descriptors.NumRotatableBonds(mol),
                "Rings": rdMolDescriptors.CalcNumRings(mol),
                "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "HeavyAtoms": mol.GetNumHeavyAtoms(),
                "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
            }

            # Lipinski RO5 check
            ro5_violations = 0
            if props["MW"] > 500:
                ro5_violations += 1
            if props["LogP"] > 5:
                ro5_violations += 1
            if props["HBD"] > 5:
                ro5_violations += 1
            if props["HBA"] > 10:
                ro5_violations += 1

            props["RO5_violations"] = ro5_violations
            props["RO5_compliant"] = ro5_violations == 0

            return json.dumps(props)
        except Exception as e:
            return f"Error: {e}"


class ValidateSARData(BaseTool):
    """Validate SAR data for completeness and consistency."""

    name: str = "ValidateSARData"
    description: str = (
        "Validate SAR data quality. "
        "Input: JSON array of {smiles, activity} objects."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)

            if not isinstance(data, list):
                return "Error: Input should be a JSON array"

            results = {
                "total": len(data),
                "valid_smiles": 0,
                "invalid_smiles": [],
                "has_activity": 0,
                "missing_activity": 0,
                "activity_range": {"min": None, "max": None},
                "duplicates": [],
            }

            seen_smiles = {}
            activities = []

            for i, item in enumerate(data):
                smi = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50") or item.get("IC90")

                # Check SMILES validity
                mol = Chem.MolFromSmiles(smi) if smi else None
                if mol:
                    results["valid_smiles"] += 1
                    canonical = Chem.MolToSmiles(mol)

                    # Check duplicates
                    if canonical in seen_smiles:
                        results["duplicates"].append({
                            "smiles": smi,
                            "indices": [seen_smiles[canonical], i],
                        })
                    else:
                        seen_smiles[canonical] = i
                else:
                    results["invalid_smiles"].append({"index": i, "smiles": smi})

                # Check activity
                if activity is not None:
                    results["has_activity"] += 1
                    try:
                        act_val = float(activity)
                        activities.append(act_val)
                    except (ValueError, TypeError):
                        pass
                else:
                    results["missing_activity"] += 1

            # Calculate activity range
            if activities:
                results["activity_range"] = {
                    "min": round(min(activities), 4),
                    "max": round(max(activities), 4),
                    "mean": round(sum(activities) / len(activities), 4),
                }

            # Quality score
            quality_score = (
                (results["valid_smiles"] / results["total"]) * 0.5
                + (results["has_activity"] / results["total"]) * 0.3
                + (1 - len(results["duplicates"]) / max(results["total"], 1)) * 0.2
            )
            results["quality_score"] = round(quality_score, 2)

            return json.dumps(results)
        except json.JSONDecodeError:
            return "Error: Invalid JSON input"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# Export all tools
# =============================================================================

SAR_TOOLS = [
    ExtractScaffold,
    ClusterMolecules,
    DefineRSites,
    RGroupDecomposition,
    FindMCS,
    AnalyzeMMP,
    CalculateProperties,
    ValidateSARData,
]


def get_sar_tools() -> list[BaseTool]:
    """Get all SAR analysis tools."""
    return [tool() for tool in SAR_TOOLS]
