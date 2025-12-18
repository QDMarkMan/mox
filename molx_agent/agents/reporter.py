"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description ReporterAgent - Orchestrates SAR analyses using report tools.
**************************************************************************
"""

import json
import logging
from datetime import datetime

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.tools.report import (
    RunFullSARAnalysisTool,
    GenerateHTMLReportTool,
    BuildReportSummaryTool,
)

logger = logging.getLogger(__name__)


class ReporterAgent(BaseAgent):
    """Reporter agent that orchestrates SAR analyses and generates HTML reports.
    
    Uses report tools for:
    - RunFullSARAnalysisTool: Execute all SAR analyses
    - GenerateHTMLReportTool: Generate HTML report
    - BuildReportSummaryTool: Create text summary
    """

    def __init__(self) -> None:
        super().__init__(
            name="reporter",
            description="Runs SAR analyses and generates HTML reports using tools",
        )
        
        # Initialize tools
        self.analysis_tool = RunFullSARAnalysisTool()
        self.report_tool = GenerateHTMLReportTool()
        self.summary_tool = BuildReportSummaryTool()

    def run(self, state: AgentState) -> AgentState:
        """Run SAR analyses and generate report using tools.

        Args:
            state: Current agent state with cleaned data.

        Returns:
            Updated state with final report.
        """
        from rich.console import Console

        console = Console()
        console.print("[cyan]📝 Reporter: Generating SAR report...[/]")

        results = state.get("results", {})
        tid = state.get("current_task_id", "reporter")

        try:
            # Step 1: Get compound data
            sar_agent_results = self._get_sar_agent_results(results)
            
            if sar_agent_results:
                console.print("[dim]   Using SAR Agent analysis results...[/]")
                compound_data = sar_agent_results.get("compounds", [])
                # Merge with SAR agent results for more complete data
                sar_results = self._merge_sar_results(sar_agent_results)
            else:
                compound_data = self._collect_compounds(results)
                
                if not compound_data:
                    console.print("[yellow]⚠ No compound data found[/]")
                    state["final_answer"] = "No compound data available for SAR analysis."
                    return state
                
                console.print(f"[dim]   Found {len(compound_data)} compounds[/]")
                console.print("[dim]   Running SAR analyses via tool...[/]")
                
                # Step 2: Run analysis using tool
                sar_results = self.analysis_tool.invoke({"compounds": compound_data})

            # Step 3: Generate HTML report using tool
            console.print("[dim]   Generating HTML report...[/]")
            report_result = self.report_tool.invoke({
                "sar_results": sar_results,
                "title": "SAR Analysis Report",
            })
            report_path = report_result["report_path"]

            # Step 4: Build summary using tool
            summary = self.summary_tool.invoke({
                "sar_results": sar_results,
                "report_path": report_path,
            })

            # Update state
            state["final_answer"] = summary
            
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {
                "sar_analysis": sar_results,
                "report_path": report_path,
            }

            console.print(f"[green]✓ Reporter: Saved to {report_path}[/]")

        except Exception as e:
            console.print(f"[red]✗ Reporter error: {e}[/]")
            logger.error(f"Reporter error: {e}")
            state["final_answer"] = f"Error generating report: {e}"
            state["error"] = str(e)

        return state

    def _get_sar_agent_results(self, results: dict) -> dict | None:
        """Get results from SAR Agent if available."""
        for task_id, result in results.items():
            if isinstance(result, dict) and "decomposed_compounds" in result:
                return result
        return None

    def _merge_sar_results(self, sar_agent_results: dict) -> dict:
        """Merge SAR Agent results with additional analyses."""
        compounds = sar_agent_results.get("compounds", [])
        decomposed = sar_agent_results.get("decomposed_compounds", [])
        
        # Run full analysis on compounds
        sar_results = self.analysis_tool.invoke({"compounds": compounds})
        
        # Merge with SAR agent specific results
        sar_results["scaffold_strategy"] = sar_agent_results.get("scaffold_strategy")
        sar_results["scaffold"] = sar_agent_results.get("scaffold")
        sar_results["r_group_analysis"] = {
            "decomposed_compounds": decomposed,
            "ocat_pairs": sar_agent_results.get("ocat_pairs", []),
        }
        
        return sar_results

    def _collect_compounds(self, results: dict) -> list[dict]:
        """Collect compound data from task results."""
        compounds = []

        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue
            if "error" in result:
                continue

            # Format 1: compounds as list
            if "compounds" in result and isinstance(result["compounds"], list):
                for i, item in enumerate(result["compounds"]):
                    if isinstance(item, dict):
                        smi = item.get("smiles") or item.get("SMILES", "")
                        if smi:
                            compounds.append({
                                "smiles": smi,
                                "activity": item.get("activity"),
                                "compound_id": item.get("compound_id", f"Cpd-{i+1}"),
                                "name": item.get("name") or item.get("Name"),
                            })

        # Fallback: load from files
        if not compounds:
            compounds = self._try_load_from_files(results)

        return compounds

    def _try_load_from_files(self, results: dict) -> list[dict]:
        """Try to load compounds from saved output files."""
        import os

        compounds = []

        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue

            output_files = result.get("output_files", {})
            json_path = output_files.get("json")

            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if isinstance(data, dict) and "compounds" in data:
                        for item in data["compounds"]:
                            smi = item.get("smiles", "")
                            if smi:
                                compounds.append({
                                    "smiles": smi,
                                    "activity": item.get("activity"),
                                    "compound_id": item.get("compound_id", ""),
                                })
                except Exception as e:
                    logger.warning(f"Failed to load {json_path}: {e}")

        return compounds
