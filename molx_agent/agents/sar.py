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
*  
**************************************************************************
"""

import logging
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState

# Import tools from tools/sar.py
from molx_agent.tools.sar import (
    find_mcs_scaffold,
    find_common_murcko_scaffold,
    decompose_r_groups,
    simple_r_group_assignment,
    identify_ocat_series,
)

logger = logging.getLogger(__name__)


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
            
            # 4. RUN OCAT ANALYSIS
            console.print("[dim]   Running OCAT analysis...[/]")
            ocat_pairs = identify_ocat_series(decomposed)
            
            console.print(f"[dim]   Found {len(ocat_pairs)} OCAT pairs[/]")
            
            # 5. Store results
            sar_result = {
                "scaffold_strategy": scaffold_result["strategy"],
                "scaffold": scaffold_result.get("scaffold"),
                "decomposed_compounds": decomposed,
                "ocat_pairs": ocat_pairs[:20],  # Top 20 most significant
                "total_ocat_pairs": len(ocat_pairs),
                "compounds": compounds,  # Pass through for reporter
            }
            
            state["results"][tid] = sar_result
            state["tasks"][tid]["status"] = "done"
            
            console.print(f"[green]✓ SAR Agent: Completed analysis[/]")
            
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

