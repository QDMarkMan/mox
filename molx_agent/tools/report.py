"""
**************************************************************************
*  @Copyright [2025] Tongfu.E.
*  @Email etongfu@outlook.com.
*  @Date [2025-12-15].
*  @Description SAR Analysis Tools - pure analysis logic, returns JSON.
**************************************************************************
"""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


# =============================================================================
# R-Group Analysis
# =============================================================================

class AnalyzeRGroupTable(BaseTool):
    """Analyze R-group substitution patterns and activity."""

    name: str = "AnalyzeRGroupTable"
    description: str = (
        "Analyze R-group table data to identify SAR trends. "
        "Input: JSON array with {compound_id, r_groups: {R1, R2...}, activity}."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            if not isinstance(data, list) or len(data) == 0:
                return json.dumps({"error": "Input should be a non-empty JSON array"})

            r_group_stats = {}
            activities = []

            for item in data:
                activity = item.get("activity") or item.get("IC50") or item.get("IC90")
                r_groups = item.get("r_groups", {})

                if activity is not None:
                    try:
                        act_val = float(activity)
                        activities.append(act_val)
                        for r_name, r_value in r_groups.items():
                            if r_name not in r_group_stats:
                                r_group_stats[r_name] = {}
                            if r_value not in r_group_stats[r_name]:
                                r_group_stats[r_name][r_value] = []
                            r_group_stats[r_name][r_value].append(act_val)
                    except (ValueError, TypeError):
                        pass

            # Calculate statistics
            r_group_analysis = {}
            for r_name, values in r_group_stats.items():
                r_group_analysis[r_name] = {}
                for r_value, acts in values.items():
                    r_group_analysis[r_name][r_value] = {
                        "count": len(acts),
                        "mean_activity": round(sum(acts) / len(acts), 2),
                        "min_activity": round(min(acts), 2),
                        "max_activity": round(max(acts), 2),
                    }

            # Recommendations
            recommendations = []
            for r_name, values in r_group_analysis.items():
                sorted_values = sorted(values.items(), key=lambda x: x[1]["mean_activity"])
                if sorted_values:
                    best = sorted_values[0]
                    worst = sorted_values[-1]
                    recommendations.append({
                        "position": r_name,
                        "best_substituent": best[0],
                        "best_activity": best[1]["mean_activity"],
                        "worst_substituent": worst[0],
                        "worst_activity": worst[1]["mean_activity"],
                    })

            return json.dumps({
                "total_compounds": len(data),
                "activity_range": {
                    "min": round(min(activities), 2) if activities else None,
                    "max": round(max(activities), 2) if activities else None,
                },
                "r_group_analysis": r_group_analysis,
                "recommendations": recommendations,
            })
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Activity Cliffs
# =============================================================================

class IdentifyActivityCliffs(BaseTool):
    """Identify activity cliffs - similar structures with large activity differences."""

    name: str = "IdentifyActivityCliffs"
    description: str = (
        "Find activity cliffs (similar molecules with >10x activity difference). "
        "Input: JSON array with {smiles, activity}."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            if not isinstance(data, list) or len(data) < 2:
                return json.dumps({"error": "Need at least 2 compounds"})

            mols = []
            for item in data:
                smi = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50") or item.get("IC90")

                if smi and activity is not None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        try:
                            mols.append({
                                "smiles": smi,
                                "activity": float(activity),
                                "fp": AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                            })
                        except (ValueError, TypeError):
                            pass

            # Find cliffs
            cliffs = []
            for i in range(len(mols)):
                for j in range(i + 1, len(mols)):
                    sim = DataStructs.TanimotoSimilarity(mols[i]["fp"], mols[j]["fp"])
                    if sim > 0.7:
                        a1, a2 = mols[i]["activity"], mols[j]["activity"]
                        fold_change = max(a1 / a2, a2 / a1) if min(a1, a2) > 0 else 0
                        if fold_change >= 10:
                            cliffs.append({
                                "mol1": mols[i]["smiles"],
                                "mol2": mols[j]["smiles"],
                                "similarity": round(sim, 3),
                                "activity1": a1,
                                "activity2": a2,
                                "fold_change": round(fold_change, 1),
                            })

            cliffs.sort(key=lambda x: x["fold_change"], reverse=True)

            return json.dumps({
                "total_compounds": len(mols),
                "activity_cliffs_found": len(cliffs),
                "cliffs": cliffs[:20],
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Scaffold SAR
# =============================================================================

class GenerateScaffoldSAR(BaseTool):
    """Generate scaffold-level SAR analysis."""

    name: str = "GenerateScaffoldSAR"
    description: str = (
        "Analyze scaffold SAR: is scaffold essential? Can it be replaced? "
        "Input: JSON with {core_scaffold, analogs: [{smiles, activity}]}."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            core_scaffold = data.get("core_scaffold", "")
            analogs = data.get("analogs", [])

            if not core_scaffold:
                return json.dumps({"error": "core_scaffold is required"})

            core_mol = Chem.MolFromSmiles(core_scaffold)
            if not core_mol:
                return json.dumps({"error": "Invalid core scaffold SMILES"})

            scaffold_matches = []
            scaffold_mismatches = []

            for item in analogs:
                smi = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50")

                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        try:
                            analog_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                            core_generic = Chem.MolToSmiles(
                                MurckoScaffold.MakeScaffoldGeneric(
                                    MurckoScaffold.GetScaffoldForMol(core_mol)
                                )
                            )
                            analog_generic = Chem.MolToSmiles(
                                MurckoScaffold.MakeScaffoldGeneric(analog_scaffold)
                            )

                            match_info = {
                                "smiles": smi,
                                "scaffold": Chem.MolToSmiles(analog_scaffold),
                                "activity": activity,
                                "scaffold_match": core_generic == analog_generic,
                            }

                            if core_generic == analog_generic:
                                scaffold_matches.append(match_info)
                            else:
                                scaffold_mismatches.append(match_info)
                        except Exception:
                            pass

            # Generate insights
            insights = {
                "core_scaffold": core_scaffold,
                "scaffold_essential": len(scaffold_mismatches) == 0 or (
                    len(scaffold_matches) > 0 and
                    all(m.get("activity", float("inf")) > 1000
                        for m in scaffold_mismatches if m.get("activity"))
                ),
                "total_analogs": len(analogs),
                "scaffold_conserved": len(scaffold_matches),
                "scaffold_varied": len(scaffold_mismatches),
            }

            # Activity comparison
            match_acts = [m["activity"] for m in scaffold_matches if m.get("activity") is not None]
            mismatch_acts = [m["activity"] for m in scaffold_mismatches if m.get("activity") is not None]

            if match_acts and mismatch_acts:
                avg_match = sum(match_acts) / len(match_acts)
                avg_mismatch = sum(mismatch_acts) / len(mismatch_acts)
                insights["scaffold_hopping_viable"] = avg_mismatch < avg_match * 10
                insights["avg_activity_conserved"] = round(avg_match, 2)
                insights["avg_activity_varied"] = round(avg_mismatch, 2)

            # Conclusion
            if insights.get("scaffold_essential"):
                insights["conclusion"] = "该骨架为活性核心，不可大幅更换。建议保守扩展或保持骨架不变。"
            elif insights.get("scaffold_hopping_viable"):
                insights["conclusion"] = "存在骨架跳跃(scaffold hopping)的可能性，部分骨架变体保持了较好活性。"
            else:
                insights["conclusion"] = "骨架更改导致活性显著下降。建议保持核心骨架结构。"

            return json.dumps(insights)
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Functional Group SAR
# =============================================================================

class GenerateFunctionalGroupSAR(BaseTool):
    """Analyze functional group contributions to activity."""

    name: str = "GenerateFunctionalGroupSAR"
    description: str = (
        "Analyze how functional groups affect activity. "
        "Input: JSON array with {smiles, activity}."
    )

    FG_SMARTS: dict = {
        "hydroxyl": "[OX2H]",
        "methoxy": "[OX2][CH3]",
        "amino": "[NX3;H2,H1;!$(NC=O)]",
        "amide": "[NX3][CX3](=[OX1])",
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "ester": "[CX3](=O)[OX2H0]",
        "ketone": "[CX3](=O)[#6]",
        "aldehyde": "[CX3H1](=O)",
        "halogen": "[F,Cl,Br,I]",
        "nitro": "[NX3+](=O)[O-]",
        "cyano": "[CX2]#[NX1]",
        "sulfone": "[SX4](=[OX1])(=[OX1])",
        "aromatic": "a",
    }

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            if not isinstance(data, list):
                return json.dumps({"error": "Input should be a JSON array"})

            fg_effects = {fg: {"with": [], "without": []} for fg in self.FG_SMARTS}

            for item in data:
                smi = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50")

                if smi and activity is not None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        try:
                            act_val = float(activity)
                            for fg_name, smarts in self.FG_SMARTS.items():
                                pattern = Chem.MolFromSmarts(smarts)
                                if pattern and mol.HasSubstructMatch(pattern):
                                    fg_effects[fg_name]["with"].append(act_val)
                                else:
                                    fg_effects[fg_name]["without"].append(act_val)
                        except (ValueError, TypeError):
                            pass

            # Calculate SAR conclusions
            sar_table = []
            for fg_name, effects in fg_effects.items():
                if effects["with"] and effects["without"]:
                    avg_with = sum(effects["with"]) / len(effects["with"])
                    avg_without = sum(effects["without"]) / len(effects["without"])
                    ratio = avg_with / avg_without if avg_without > 0 else 1

                    if ratio < 0.3:
                        effect, effect_cn = "essential", "必需"
                    elif ratio < 0.7:
                        effect, effect_cn = "beneficial", "有利"
                    elif ratio < 1.5:
                        effect, effect_cn = "tolerated", "中性"
                    else:
                        effect, effect_cn = "detrimental", "有害"

                    sar_table.append({
                        "functional_group": fg_name,
                        "count_with": len(effects["with"]),
                        "count_without": len(effects["without"]),
                        "avg_activity_with": round(avg_with, 2),
                        "avg_activity_without": round(avg_without, 2),
                        "effect": effect,
                        "effect_cn": effect_cn,
                        "fold_change": round(ratio, 2),
                    })

            effect_order = {"essential": 0, "beneficial": 1, "detrimental": 2, "tolerated": 3}
            sar_table.sort(key=lambda x: effect_order.get(x["effect"], 4))

            return json.dumps({
                "total_compounds": len(data),
                "functional_group_sar": sar_table,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Positional SAR
# =============================================================================

class GeneratePositionalSAR(BaseTool):
    """Analyze positional SAR - effects of substitution at different positions."""

    name: str = "GeneratePositionalSAR"
    description: str = (
        "Analyze SAR by substitution position. "
        "Input: JSON with {compounds: [{r_groups: {R1, R2...}, activity}]}."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            compounds = data.get("compounds", data if isinstance(data, list) else [])

            position_analysis = {}

            for item in compounds:
                r_groups = item.get("r_groups", {})
                activity = item.get("activity") or item.get("IC50") or item.get("IC90")

                if activity is not None:
                    try:
                        act_val = float(activity)
                        for pos, substituent in r_groups.items():
                            if pos not in position_analysis:
                                position_analysis[pos] = {"substituents": {}, "size_tolerance": []}

                            if substituent not in position_analysis[pos]["substituents"]:
                                position_analysis[pos]["substituents"][substituent] = []
                            position_analysis[pos]["substituents"][substituent].append(act_val)

                            size = len(str(substituent).replace("-", ""))
                            position_analysis[pos]["size_tolerance"].append({
                                "size": size,
                                "activity": act_val,
                            })
                    except (ValueError, TypeError):
                        pass

            # Generate insights
            positional_insights = []
            for pos, analysis in position_analysis.items():
                best_sub, best_activity = None, float("inf")
                for sub, activities in analysis["substituents"].items():
                    avg = sum(activities) / len(activities)
                    if avg < best_activity:
                        best_activity = avg
                        best_sub = sub

                size_data = analysis["size_tolerance"]
                if size_data:
                    small = [d for d in size_data if d["size"] <= 2]
                    large = [d for d in size_data if d["size"] > 4]
                    small_avg = sum(d["activity"] for d in small) / len(small) if small else None
                    large_avg = sum(d["activity"] for d in large) / len(large) if large else None

                    if small_avg and large_avg:
                        if large_avg < small_avg * 0.5:
                            size_pref = "大基团有利 (疏水pocket)"
                        elif small_avg < large_avg * 0.5:
                            size_pref = "仅容小基团 (steric clash)"
                        else:
                            size_pref = "尺寸无明显偏好"
                    else:
                        size_pref = "数据不足"

                    positional_insights.append({
                        "position": pos,
                        "best_substituent": best_sub,
                        "best_activity": round(best_activity, 2),
                        "num_substituents_tested": len(analysis["substituents"]),
                        "size_preference": size_pref,
                    })

            return json.dumps({
                "positions_analyzed": len(position_analysis),
                "positional_sar": positional_insights,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Conformational SAR
# =============================================================================

class GenerateConformationalSAR(BaseTool):
    """Analyze conformational and steric SAR factors."""

    name: str = "GenerateConformationalSAR"
    description: str = (
        "Analyze 3D/steric SAR factors: planarity, rigidity. "
        "Input: JSON array with {smiles, activity}."
    )

    def _run(self, data_json: str) -> str:
        try:
            data = json.loads(data_json)
            if not isinstance(data, list):
                return json.dumps({"error": "Input should be a JSON array"})

            planar_activities = []
            nonplanar_activities = []
            rigid_activities = []
            flexible_activities = []

            for item in data:
                smi = item.get("smiles") or item.get("SMILES", "")
                activity = item.get("activity") or item.get("IC50")

                if smi and activity is not None:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        try:
                            act_val = float(activity)
                            num_rotatable = Descriptors.NumRotatableBonds(mol)
                            num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
                            fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

                            is_planar = num_aromatic_rings >= 2 and fsp3 < 0.3
                            if is_planar:
                                planar_activities.append(act_val)
                            else:
                                nonplanar_activities.append(act_val)

                            is_rigid = num_rotatable <= 3
                            if is_rigid:
                                rigid_activities.append(act_val)
                            else:
                                flexible_activities.append(act_val)
                        except (ValueError, TypeError):
                            pass

            insights = {}

            if planar_activities and nonplanar_activities:
                avg_planar = sum(planar_activities) / len(planar_activities)
                avg_nonplanar = sum(nonplanar_activities) / len(nonplanar_activities)
                insights["planarity"] = {
                    "planar_count": len(planar_activities),
                    "nonplanar_count": len(nonplanar_activities),
                    "avg_planar_activity": round(avg_planar, 2),
                    "avg_nonplanar_activity": round(avg_nonplanar, 2),
                    "preference": "平面构象有利" if avg_planar < avg_nonplanar else "非平面构象可接受",
                }

            if rigid_activities and flexible_activities:
                avg_rigid = sum(rigid_activities) / len(rigid_activities)
                avg_flexible = sum(flexible_activities) / len(flexible_activities)
                insights["rigidity"] = {
                    "rigid_count": len(rigid_activities),
                    "flexible_count": len(flexible_activities),
                    "avg_rigid_activity": round(avg_rigid, 2),
                    "avg_flexible_activity": round(avg_flexible, 2),
                    "preference": "刚性构象有利" if avg_rigid < avg_flexible else "柔性可容忍",
                }

            conclusions = []
            if insights.get("planarity", {}).get("preference") == "平面构象有利":
                conclusions.append("分子需保持平面性，可能与π-π堆积或平面结合位点相关")
            if insights.get("rigidity", {}).get("preference") == "刚性构象有利":
                conclusions.append("构象锁定可提升活性，建议引入环化或刚性连接")

            insights["conclusions"] = conclusions if conclusions else ["构象因素影响不显著"]

            return json.dumps(insights)
        except Exception as e:
            return json.dumps({"error": str(e)})


# =============================================================================
# Export
# =============================================================================

REPORT_TOOLS = [
    AnalyzeRGroupTable,
    IdentifyActivityCliffs,
    GenerateScaffoldSAR,
    GenerateFunctionalGroupSAR,
    GeneratePositionalSAR,
    GenerateConformationalSAR,
]


def get_report_tools() -> list[BaseTool]:
    """Get all SAR analysis tools."""
    return [tool() for tool in REPORT_TOOLS]
