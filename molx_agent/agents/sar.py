"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-16].
*  @Description SAR Agent - R-group decomposition and SAR analysis decisions.
*  
*  This agent makes key SAR decisions:
*  1. Scaffold selection strategy (MCS vs Murcko vs custom)
*  2. R-group decomposition approach
*  3. OCAT (one-change-at-a-time) analysis
*  4. AI-powered analysis mode selection (Single-Point Scan detection)
*  
**************************************************************************
"""

import logging
from enum import Enum
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import invoke_llm

# Import tools from tools/sar.py
from molx_agent.tools.sar import (
    find_mcs_scaffold,
    find_common_murcko_scaffold,
    decompose_r_groups,
    simple_r_group_assignment,
    identify_ocat_series,
)

logger = logging.getLogger(__name__)


class SARAnalysisMode(Enum):
    """SAR analysis mode types determined by data characteristics."""
    
    FULL_SAR = "full_sar"                    # Standard multi-position analysis
    SINGLE_POINT_SCAN = "single_point_scan"  # Focus on one varying position
    SCAFFOLD_HOPPING = "scaffold_hopping"    # Different scaffolds comparison


class SARAgent(BaseAgent):
    """SAR Agent - handles R-group decomposition and SAR analysis decisions.
    
    This agent is responsible for:
    1. Deciding which scaffold selection strategy to use
    2. Performing R-group decomposition
    3. Running OCAT (one-change-at-a-time) analysis
    
    The actual computation is delegated to tools in molx_agent/tools/sar.py
    """

    def __init__(self) -> None:
        super().__init__(
            name="sar",
            description="Performs R-group decomposition and SAR analysis",
        )

    def run(self, state: AgentState) -> AgentState:
        """Execute SAR analysis task.
        
        Decision flow:
        1. Check for user-specified scaffold -> use custom strategy
        2. If 5+ compounds, try MCS scaffold 
        3. Fallback to Murcko scaffold
        4. Perform R-group decomposition
        5. Run OCAT analysis on decomposed compounds
        
        Args:
            state: Current agent state with compound data.
        
        Returns:
            Updated state with SAR analysis results.
        """
        from rich.console import Console

        console = Console()
        
        tid = state.get("current_task_id")
        if not tid:
            return state
        
        task = state.get("tasks", {}).get(tid)
        if not task:
            return state
        
        console.print(f"[cyan]🔬 SAR Agent: Processing task {tid}...[/]")
        
        try:
            # 1. Collect compound data from previous results
            compounds = self._collect_compounds(state)
            
            if not compounds:
                console.print("[yellow]⚠ No compound data found for SAR analysis[/]")
                state["results"][tid] = {"error": "No compound data available"}
                state["tasks"][tid]["status"] = "done"
                return state
            
            console.print(f"[dim]   Found {len(compounds)} compounds[/]")
            
            # 2. SELECT SCAFFOLD STRATEGY (key decision point)
            console.print("[dim]   Selecting scaffold strategy...[/]")
            scaffold_result = self._select_scaffold(compounds, task)
            
            console.print(f"[dim]   Strategy: {scaffold_result['strategy']}[/]")
            if scaffold_result.get("scaffold"):
                console.print(f"[dim]   Scaffold: {scaffold_result['scaffold'][:50]}...[/]")
            
            # 3. PERFORM R-GROUP DECOMPOSITION
            console.print("[dim]   Decomposing R-groups...[/]")
            decomposed = self._decompose_compounds(compounds, scaffold_result)
            
            console.print(f"[dim]   Decomposed {len(decomposed)} compounds[/]")
            
            # 4. AI DECISION: Determine optimal analysis mode
            console.print("[dim]   🧠 Analyzing data patterns for optimal analysis mode...[/]")
            mode_decision = self._decide_analysis_mode(decomposed, ocat_pairs=None)
            analysis_mode = mode_decision["mode"]
            
            console.print(f"[dim]   Analysis mode: {analysis_mode.value}[/]")
            if mode_decision.get("dominant_position"):
                console.print(f"[dim]   Dominant position: {mode_decision['dominant_position']}[/]")
            
            # 5. RUN OCAT ANALYSIS
            console.print("[dim]   Running OCAT analysis...[/]")
            ocat_pairs = identify_ocat_series(decomposed)
            
            console.print(f"[dim]   Found {len(ocat_pairs)} OCAT pairs[/]")
            
            # 6. Generate AI insights if Single-Point mode
            ai_insights = None
            if analysis_mode == SARAnalysisMode.SINGLE_POINT_SCAN and ocat_pairs:
                console.print("[dim]   🧠 Generating AI insights for Single-Point Scan...[/]")
                ai_insights = self._generate_single_point_insights(
                    decomposed, ocat_pairs, mode_decision.get("dominant_position")
                )
            
            # 7. Store results
            sar_result = {
                "scaffold_strategy": scaffold_result["strategy"],
                "scaffold": scaffold_result.get("scaffold"),
                "decomposed_compounds": decomposed,
                "ocat_pairs": ocat_pairs[:20],  # Top 20 most significant
                "total_ocat_pairs": len(ocat_pairs),
                "compounds": compounds,  # Pass through for reporter
                # AI decision results
                "analysis_mode": analysis_mode.value,
                "mode_decision": mode_decision,
                "ai_insights": ai_insights,
            }
            
            state["results"][tid] = sar_result
            state["tasks"][tid]["status"] = "done"
            
            console.print(f"[green]✓ SAR Agent: Completed analysis ({analysis_mode.value})[/]")
            
        except Exception as e:
            console.print(f"[red]✗ SAR Agent error: {e}[/]")
            logger.error(f"SAR Agent error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"
        
        return state

    def _collect_compounds(self, state: AgentState) -> list[dict]:
        """Collect compound data from previous task results."""
        compounds = []
        results = state.get("results", {})
        
        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue
            
            if "error" in result:
                continue
            
            # From DataCleaner
            if "compounds" in result:
                for cpd in result["compounds"]:
                    if isinstance(cpd, dict) and cpd.get("smiles"):
                        compounds.append(cpd)
        
        return compounds

    def _select_scaffold(self, compounds: list[dict], task: dict) -> dict:
        """Select scaffold strategy based on data characteristics.
        
        DECISION LOGIC:
        1. If user provided a scaffold in inputs -> use "custom" strategy
        2. If 5+ compounds -> try MCS (Maximum Common Substructure)
        3. If MCS fails or <5 compounds -> use Murcko scaffold
        4. If all else fails -> no scaffold (return empty r_groups)
        
        Args:
            compounds: List of compound dicts.
            task: Current task definition.
        
        Returns:
            Dict with strategy and scaffold.
        """
        inputs = task.get("inputs", {})
        
        # Check for user-specified scaffold (highest priority)
        custom_scaffold = inputs.get("scaffold") or inputs.get("core")
        if custom_scaffold:
            logger.info(f"Using custom scaffold: {custom_scaffold}")
            return {
                "strategy": "custom",
                "scaffold": custom_scaffold,
            }
        
        smiles_list = [c.get("smiles") for c in compounds if c.get("smiles")]
        
        if len(smiles_list) < 2:
            return {"strategy": "none", "scaffold": None}
        
        # Try MCS first for 5+ compounds (more reliable with larger datasets)
        if len(smiles_list) >= 5:
            mcs_scaffold = find_mcs_scaffold(smiles_list)
            if mcs_scaffold:
                logger.info(f"Using MCS scaffold: {mcs_scaffold[:50]}...")
                return {
                    "strategy": "mcs",
                    "scaffold": mcs_scaffold,
                }
        
        # Fall back to Murcko scaffold
        murcko_scaffold = find_common_murcko_scaffold(smiles_list)
        if murcko_scaffold:
            logger.info(f"Using Murcko scaffold: {murcko_scaffold}")
            return {
                "strategy": "murcko",
                "scaffold": murcko_scaffold,
            }
        
        return {"strategy": "none", "scaffold": None}

    def _decompose_compounds(
        self, compounds: list[dict], scaffold_result: dict
    ) -> list[dict]:
        """Decompose compounds based on selected scaffold.
        
        Uses RDKit RGroupDecomposition, falls back to simple
        assignment if that fails. Preserves original compound data.
        """
        scaffold = scaffold_result.get("scaffold")
        
        if not scaffold:
            # No scaffold: return compounds with empty r_groups
            return [
                {
                    "compound_id": c.get("compound_id", f"Cpd-{i}"),
                    "name": c.get("name") or c.get("Name") or c.get("compound_id", f"Cpd-{i}"),
                    "smiles": c.get("smiles"),
                    "activity": c.get("activity"),
                    "activities": c.get("activities", {}),  # Multi-activity support
                    "r_groups": {},
                }
                for i, c in enumerate(compounds)
            ]
        
        smiles_list = [c.get("smiles") for c in compounds]
        activities = [c.get("activity") for c in compounds]
        compound_ids = [c.get("compound_id", f"Cpd-{i}") for i, c in enumerate(compounds)]
        names = [c.get("name") or c.get("Name") or c.get("compound_id", f"Cpd-{i}") for i, c in enumerate(compounds)]
        
        # Try RGroupDecomposition first
        decomposed = decompose_r_groups(
            smiles_list, scaffold, activities, compound_ids
        )
        
        # Add name and activities dict fields to decomposed results
        if decomposed:
            for i, dec in enumerate(decomposed):
                # Find the original compound by smiles
                for j, cpd in enumerate(compounds):
                    if cpd.get("smiles") == dec.get("smiles"):
                        dec["name"] = cpd.get("name") or cpd.get("Name") or dec.get("compound_id", "")
                        # Copy activities dict for multi-activity table support
                        if cpd.get("activities"):
                            dec["activities"] = cpd["activities"]
                        break
            return decomposed
        
        # Fallback to simple assignment
        fallback = simple_r_group_assignment(
            smiles_list, scaffold, activities, compound_ids
        )
        
        # Add name and activities fields to fallback results
        for i, dec in enumerate(fallback):
            for j, cpd in enumerate(compounds):
                if cpd.get("smiles") == dec.get("smiles"):
                    dec["name"] = cpd.get("name") or cpd.get("Name") or dec.get("compound_id", "")
                    # Copy activities dict for multi-activity table support
                    if cpd.get("activities"):
                        dec["activities"] = cpd["activities"]
                    break
        
        return fallback

    def _decide_analysis_mode(
        self, decomposed: list[dict], ocat_pairs: list[dict] = None
    ) -> dict:
        """AI Decision: Determine optimal SAR analysis mode based on data patterns.
        
        Analyzes R-group variation patterns to decide:
        1. SINGLE_POINT_SCAN: One position varies, others constant (ideal for qSAR)
        2. FULL_SAR: Multiple positions vary (standard analysis)
        3. SCAFFOLD_HOPPING: Different scaffolds present
        
        Args:
            decomposed: List of decomposed compound dictionaries.
            ocat_pairs: Optional pre-computed OCAT pairs.
        
        Returns:
            Dict with mode, dominant_position (if single-point), and variation_stats.
        """
        if not decomposed or len(decomposed) < 3:
            return {
                "mode": SARAnalysisMode.FULL_SAR,
                "reason": "Insufficient data for mode detection",
            }
        
        # Analyze R-group variation patterns
        variation_stats = self._analyze_variation_patterns(decomposed)
        
        if not variation_stats["positions"]:
            return {
                "mode": SARAnalysisMode.FULL_SAR,
                "reason": "No R-group positions detected",
                "variation_stats": variation_stats,
            }
        
        # Decision criteria for Single-Point Scan:
        # 1. One position has significantly more variation than others
        # 2. That position has ≥3 unique substituents
        # 3. Other positions have ≤2 unique values each
        
        positions = variation_stats["positions"]
        total_variance = sum(p["unique_count"] for p in positions.values())
        
        dominant_position = None
        dominant_ratio = 0
        
        for pos, stats in positions.items():
            if stats["unique_count"] >= 3:
                ratio = stats["unique_count"] / total_variance if total_variance > 0 else 0
                if ratio > dominant_ratio:
                    dominant_ratio = ratio
                    dominant_position = pos
        
        # Check if one position dominates (>60% of total variation)
        # AND other positions have minimal variation
        other_positions_stable = all(
            positions[p]["unique_count"] <= 2 
            for p in positions 
            if p != dominant_position
        )
        
        if dominant_position and dominant_ratio >= 0.5 and other_positions_stable:
            return {
                "mode": SARAnalysisMode.SINGLE_POINT_SCAN,
                "dominant_position": dominant_position,
                "dominant_ratio": round(dominant_ratio, 2),
                "variation_stats": variation_stats,
                "reason": f"Position {dominant_position} dominates variation ({dominant_ratio:.0%})",
            }
        
        return {
            "mode": SARAnalysisMode.FULL_SAR,
            "variation_stats": variation_stats,
            "reason": "Multiple positions show significant variation",
        }

    def _analyze_variation_patterns(self, decomposed: list[dict]) -> dict:
        """Analyze R-group variation patterns across compounds.
        
        Args:
            decomposed: List of decomposed compound dictionaries.
        
        Returns:
            Dict with per-position statistics.
        """
        positions = {}
        
        for cpd in decomposed:
            r_groups = cpd.get("r_groups", {})
            for pos, value in r_groups.items():
                if pos not in positions:
                    positions[pos] = {"values": set(), "count": 0}
                positions[pos]["values"].add(value)
                positions[pos]["count"] += 1
        
        # Calculate statistics
        result = {"positions": {}, "total_compounds": len(decomposed)}
        
        for pos, data in positions.items():
            result["positions"][pos] = {
                "unique_count": len(data["values"]),
                "coverage": data["count"],
                "values": list(data["values"])[:10],  # Limit for display
            }
        
        return result

    def _generate_single_point_insights(
        self,
        decomposed: list[dict],
        ocat_pairs: list[dict],
        dominant_position: str,
    ) -> dict:
        """Generate AI-powered insights for Single-Point Scan analysis.
        
        Uses LLM to analyze SAR trends at the dominant position.
        
        Args:
            decomposed: Decomposed compounds.
            ocat_pairs: OCAT pairs.
            dominant_position: The R-group position to focus on.
        
        Returns:
            Dict with AI-generated SAR insights.
        """
        # Filter OCAT pairs to dominant position
        position_pairs = [
            p for p in ocat_pairs 
            if p.get("varying_position") == dominant_position
        ]
        
        if not position_pairs:
            return {"summary": "No OCAT pairs found at dominant position"}
        
        # Build data summary for LLM
        substituents = {}
        for cpd in decomposed:
            r_groups = cpd.get("r_groups", {})
            sub = r_groups.get(dominant_position)
            activity = cpd.get("activity")
            if sub and activity:
                if sub not in substituents:
                    substituents[sub] = []
                substituents[sub].append(activity)
        
        # Calculate average activity per substituent
        sub_summary = []
        for sub, acts in substituents.items():
            avg = sum(acts) / len(acts)
            sub_summary.append(f"  - {sub}: avg activity = {avg:.2f} (n={len(acts)})")
        
        # Prepare prompt
        prompt = f"""Analyze the following Single-Point SAR data for position {dominant_position}:

Substituents and Activities:
{chr(10).join(sub_summary[:15])}

Top Activity Changes (OCAT pairs):
"""
        for pair in position_pairs[:5]:
            prompt += f"  - {pair['substituent1']} → {pair['substituent2']}: "
            prompt += f"activity {pair['activity1']:.2f} → {pair['activity2']:.2f} "
            prompt += f"(fold change: {pair.get('fold_change', 'N/A')}x)\n"

        prompt += """
Provide a brief SAR analysis in JSON format:
{
    "key_findings": ["finding1", "finding2"],
    "best_substituent": "...",
    "worst_substituent": "...",
    "recommendation": "..."
}"""

        try:
            result = invoke_llm(
                "You are a medicinal chemistry SAR expert. Analyze the data and provide insights.",
                prompt,
                parse_json=True,
            )
            return {
                "position": dominant_position,
                "substituent_count": len(substituents),
                "ocat_pair_count": len(position_pairs),
                "llm_insights": result,
            }
        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return {
                "position": dominant_position,
                "substituent_count": len(substituents),
                "error": str(e),
            }

