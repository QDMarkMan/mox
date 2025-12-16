"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-15].
*  @Description Reporter agent - orchestrates SAR analyses and generates reports.
**************************************************************************
"""

import json
import logging
from datetime import datetime

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState

logger = logging.getLogger(__name__)


class ReporterAgent(BaseAgent):
    """Reporter agent that orchestrates SAR analyses and generates HTML reports."""

    def __init__(self) -> None:
        super().__init__(
            name="reporter",
            description="Runs SAR analyses and generates HTML reports",
        )

    def run(self, state: AgentState) -> AgentState:
        """Run all SAR analyses and generate report.

        Args:
            state: Current agent state with cleaned data.

        Returns:
            Updated state with final report.
        """
        from rich.console import Console

        console = Console()
        console.print("[cyan]📝 Reporter: Generating SAR report...[/]")

        results = state.get("results", {})

        try:
            # 1. Collect compound data
            compound_data = self._collect_compounds(results)

            if not compound_data:
                console.print("[yellow]⚠ No compound data found[/]")
                state["final_answer"] = "No compound data available for SAR analysis."
                return state

            console.print(f"[dim]   Found {len(compound_data)} compounds[/]")

            # 2. Run SAR analyses
            console.print("[dim]   Running SAR analyses...[/]")
            sar_results = self._run_analyses(compound_data)

            # 3. Generate HTML report
            console.print("[dim]   Generating HTML report...[/]")
            report_path = self._generate_report(sar_results)

            # 4. Update state
            state["final_answer"] = self._build_summary(sar_results, report_path)
            state["results"]["sar_analysis"] = sar_results
            state["results"]["report_path"] = report_path

            console.print(f"[green]✓ Reporter: Saved to {report_path}[/]")

        except Exception as e:
            console.print(f"[red]✗ Reporter error: {e}[/]")
            logger.error(f"Reporter error: {e}")
            state["final_answer"] = f"Error generating report: {e}"
            state["error"] = str(e)

        return state

    def _collect_compounds(self, results: dict) -> list[dict]:
        """Collect compound data from task results."""
        compounds = []

        for task_id, result in results.items():
            # Skip non-dict results
            if not isinstance(result, dict):
                continue

            # Skip error results
            if "error" in result:
                continue

            # Format 1: compounds as list of dicts (new DataCleaner format)
            if "compounds" in result and isinstance(result["compounds"], list):
                for item in result["compounds"]:
                    if isinstance(item, dict):
                        smi = item.get("smiles") or item.get("SMILES", "")
                        act = item.get("activity")
                        if smi and act is not None:
                            compounds.append({"smiles": smi, "activity": act})
                    elif isinstance(item, str):
                        # Old format: list of SMILES strings
                        activities = result.get("activities", [])
                        idx = result["compounds"].index(item)
                        act = activities[idx] if idx < len(activities) else None
                        if item and act is not None:
                            compounds.append({"smiles": item, "activity": act})

            # Format 2: raw_data list
            elif "raw_data" in result:
                for item in result["raw_data"]:
                    if isinstance(item, dict):
                        smi = item.get("smiles") or item.get("SMILES", "")
                        act = item.get("activity")
                        if smi and act is not None:
                            compounds.append({"smiles": smi, "activity": act})

        # If still no compounds found, try to read from output files
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

                    if isinstance(data, list):
                        for item in data:
                            smi = item.get("smiles") or item.get("SMILES", "")
                            act = item.get("activity") or item.get("IC50") or item.get("IC90")
                            if smi and act is not None:
                                compounds.append({"smiles": smi, "activity": act})
                    elif isinstance(data, dict):
                        smiles_list = data.get("compounds", [])
                        activities = data.get("activities", [])
                        for i, smi in enumerate(smiles_list):
                            act = activities[i] if i < len(activities) else None
                            if smi and act is not None:
                                compounds.append({"smiles": smi, "activity": act})
                except Exception as e:
                    logger.warning(f"Failed to load {json_path}: {e}")

        return compounds

    def _run_analyses(self, compound_data: list[dict]) -> dict:
        """Run all SAR analysis tools."""
        from molx_agent.tools.report import (
            AnalyzeRGroupTable,
            GenerateFunctionalGroupSAR,
            IdentifyActivityCliffs,
            GenerateConformationalSAR,
        )

        sar_results = {
            "total_compounds": len(compound_data),
            "generated_at": datetime.now().isoformat(),
        }

        data_json = json.dumps(compound_data)

        # R-group Analysis
        try:
            result = AnalyzeRGroupTable()._run(data_json)
            sar_results["r_group_analysis"] = json.loads(result)
        except Exception as e:
            logger.error(f"R-group analysis error: {e}")
            sar_results["r_group_analysis"] = {"error": str(e)}

        # Functional Group SAR
        try:
            result = GenerateFunctionalGroupSAR()._run(data_json)
            sar_results["functional_group_sar"] = json.loads(result)
        except Exception as e:
            logger.error(f"FG SAR error: {e}")
            sar_results["functional_group_sar"] = {"error": str(e)}

        # Activity Cliffs
        try:
            result = IdentifyActivityCliffs()._run(data_json)
            sar_results["activity_cliffs"] = json.loads(result)
        except Exception as e:
            logger.error(f"Activity cliffs error: {e}")
            sar_results["activity_cliffs"] = {"error": str(e)}

        # Conformational SAR
        try:
            result = GenerateConformationalSAR()._run(data_json)
            sar_results["conformational_sar"] = json.loads(result)
        except Exception as e:
            logger.error(f"Conformational SAR error: {e}")
            sar_results["conformational_sar"] = {"error": str(e)}

        return sar_results

    def _generate_report(self, sar_results: dict) -> str:
        """Generate HTML report using html_builder."""
        from molx_agent.tools.html_builder import build_sar_html_report, save_html_report

        html = build_sar_html_report(sar_results, "SAR Analysis Report")
        return save_html_report(html)

    def _build_summary(self, sar_results: dict, report_path: str) -> str:
        """Build text summary."""
        summary = "# SAR Analysis Report\n\n"
        summary += f"**Total Compounds:** {sar_results.get('total_compounds', 0)}\n\n"

        summary += "## Key Findings\n\n"

        # Functional groups
        fg = sar_results.get("functional_group_sar", {})
        if fg.get("functional_group_sar"):
            essential = [f["functional_group"] for f in fg["functional_group_sar"] if f.get("effect") == "essential"]
            if essential:
                summary += f"- **必需官能团:** {', '.join(essential)}\n"

        # Activity cliffs
        cliffs = sar_results.get("activity_cliffs", {})
        if cliffs.get("activity_cliffs_found"):
            summary += f"- **活性悬崖:** {cliffs['activity_cliffs_found']} 对\n"

        # Conformational
        conf = sar_results.get("conformational_sar", {})
        if conf.get("conclusions"):
            for c in conf["conclusions"]:
                summary += f"- {c}\n"

        summary += f"\n## Report\n📄 **HTML Report:** `{report_path}`\n"

        return summary
