"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description ReporterAgent - Orchestrates SAR analyses using report tools.
*               Supports intent-based report generation modes.
**************************************************************************
"""

import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.tools import get_registry

logger = logging.getLogger(__name__)


class ReportIntent(Enum):
    """Report generation intent types."""
    
    FULL_REPORT = "full_report"          # Default full SAR report
    SINGLE_SITE = "single_site"          # Focus on one R-group position
    MOLECULE_SUBSET = "molecule_subset"  # Analyze specific molecules
    CUSTOM = "custom"                    # Custom analysis


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
        
        # Get tools from registry
        registry = get_registry()
        self.analysis_tool = registry.get_tool_by_name("run_full_sar_analysis")
        self.report_tool = registry.get_tool_by_name("generate_html_report")
        self.summary_tool = registry.get_tool_by_name("build_report_summary")

    def run(self, state: AgentState) -> AgentState:
        """Run SAR analyses and generate report based on user intent.

        Supports three analysis modes:
        1. FULL_REPORT (default) - Complete SAR analysis
        2. SINGLE_SITE - Focus on one R-group position
        3. MOLECULE_SUBSET - Analyze specific molecules

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
        task = state.get("tasks", {}).get(tid, {})
        user_query = state.get("user_query", "")

        try:
            # Step 1: Parse report intent from task inputs or user query
            intent, intent_params = self._parse_report_intent(task, user_query)
            console.print(f"[dim]   Report intent: {intent.value}[/]")
            
            if intent_params:
                console.print(f"[dim]   Intent params: {intent_params}[/]")

            # Step 2: Collect compound data
            sar_agent_results = self._get_sar_agent_results(results)
            activity_cols = self._get_activity_columns(results)
            
            if activity_cols:
                console.print(f"[dim]   Activity columns: {activity_cols}[/]")
            
            if sar_agent_results:
                console.print("[dim]   Using SAR Agent analysis results...[/]")
                compound_data = sar_agent_results.get("compounds", [])
            else:
                compound_data = self._collect_compounds(results)
                
            if not compound_data:
                console.print("[yellow]⚠ No compound data found[/]")
                state["final_answer"] = "No compound data available for SAR analysis."
                return state
            
            console.print(f"[dim]   Found {len(compound_data)} compounds[/]")
            
            # Step 3: Route to appropriate analysis based on intent
            if intent == ReportIntent.SINGLE_SITE:
                target_position = intent_params.get("target_position", "R1")
                console.print(f"[dim]   Running single-site analysis for {target_position}...[/]")
                sar_results = self._run_single_site_analysis(
                    compound_data, sar_agent_results, target_position, activity_cols
                )
                report_title = f"SAR Analysis Report - {target_position} Position"
                
            elif intent == ReportIntent.MOLECULE_SUBSET:
                target_molecules = intent_params.get("target_molecules", [])
                console.print(f"[dim]   Running subset analysis for {len(target_molecules)} molecules...[/]")
                sar_results = self._run_molecule_subset_analysis(
                    compound_data, sar_agent_results, target_molecules, activity_cols
                )
                report_title = f"SAR Analysis Report - {len(target_molecules)} Selected Molecules"
                
            else:
                # Default: FULL_REPORT
                console.print("[dim]   Running full SAR analysis...[/]")
                if sar_agent_results:
                    sar_results = self._merge_sar_results(sar_agent_results, activity_cols)
                else:
                    sar_results = self.analysis_tool.invoke({
                        "compounds": compound_data,
                        "activity_columns": activity_cols if activity_cols else None,
                    })
                    if activity_cols:
                        sar_results["activity_columns"] = activity_cols
                report_title = "SAR Analysis Report"

            # Step 4: Generate HTML report
            console.print("[dim]   Generating HTML report...[/]")
            report_result = self.report_tool.invoke({
                "sar_results": sar_results,
                "title": report_title,
            })
            report_path = report_result["report_path"]

            # Step 5: Build summary
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
                "report_intent": intent.value,
            }

            console.print(f"[green]✓ Reporter: Saved to {report_path}[/]")

        except Exception as e:
            console.print(f"[red]✗ Reporter error: {e}[/]")
            logger.error(f"Reporter error: {e}")
            state["final_answer"] = f"Error generating report: {e}"
            state["error"] = str(e)

        return state

    def _parse_report_intent(
        self, task: dict, user_query: str
    ) -> tuple[ReportIntent, dict]:
        """Parse report generation intent from task inputs or user query.
        
        Detects:
        1. Single-site analysis: keywords like "R1位点", "单一位点", "只看R2"
        2. Molecule subset: specific compound IDs, names, or "只分析这几个"
        
        Args:
            task: Current task definition with inputs.
            user_query: Original user query string.
        
        Returns:
            Tuple of (ReportIntent, params dict).
        """
        inputs = task.get("inputs", {})
        params = {}
        
        # Check for explicit intent in task inputs (from Planner)
        explicit_intent = inputs.get("report_intent")
        if explicit_intent:
            if explicit_intent == "single_site":
                params["target_position"] = inputs.get("target_position", "R1")
                return ReportIntent.SINGLE_SITE, params
            elif explicit_intent == "molecule_subset":
                params["target_molecules"] = inputs.get("target_molecules", [])
                return ReportIntent.MOLECULE_SUBSET, params
        
        # Parse from user query
        query_lower = user_query.lower()
        
        # Check for single-site analysis patterns
        single_site_patterns = [
            r"只看\s*(R\d+)",            # 只看R1
            r"(R\d+)\s*位点",            # R1位点
            r"单一位点.*?(R\d+)",        # 单一位点 R1
            r"(R\d+)\s*位置",            # R1位置
            r"分析\s*(R\d+)",            # 分析R1
            r"(R\d+)\s*的\s*SAR",        # R1的SAR
            r"single.?site.*?(R\d+)",    # single-site R1
            r"position\s*(R\d+)",        # position R1
        ]
        
        for pattern in single_site_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["target_position"] = match.group(1).upper()
                return ReportIntent.SINGLE_SITE, params
        
        # Check for molecule subset patterns
        subset_patterns = [
            r"只分析\s*(.+?)(?:的|$)",                    # 只分析 XXX
            r"分析这几个.*?[:：]?\s*(.+)",               # 分析这几个分子: ...
            r"比较\s*(.+和.+?)(?:的|$)",              # 比较 X 和 Y 的
            r"(?:Cpd-?\d+[,，\s]*)+",                    # Cpd-1, Cpd-2, ...
        ]
        
        for pattern in subset_patterns:
            match = re.search(pattern, user_query)
            if match:
                # For comparison pattern, use full match to get all molecule IDs
                text_to_parse = match.group(0)
                if match.lastindex and match.lastindex >= 1:
                    text_to_parse = match.group(1)
                # Extract molecule identifiers
                molecules = self._extract_molecule_ids(text_to_parse)
                if molecules:
                    params["target_molecules"] = molecules
                    return ReportIntent.MOLECULE_SUBSET, params
        
        # Default to full report
        return ReportIntent.FULL_REPORT, {}

    def _extract_molecule_ids(self, text: str) -> list[str]:
        """Extract molecule identifiers from text.
        
        Supports formats:
        - Cpd-1, Cpd-2, Cpd-3
        - compound1, compound2
        - 分子1, 分子2
        """
        ids = []
        
        # Match Cpd-N pattern
        cpd_matches = re.findall(r"Cpd-?\d+", text, re.IGNORECASE)
        ids.extend([m.replace("Cpd", "Cpd-").replace("Cpd--", "Cpd-") for m in cpd_matches])
        
        # Match compound N or 化合物N pattern
        compound_matches = re.findall(r"(?:compound|化合物)\s*(\d+)", text, re.IGNORECASE)
        ids.extend([f"Cpd-{m}" for m in compound_matches])
        
        # Match 分子N pattern
        molecule_matches = re.findall(r"分子\s*(\d+)", text)
        ids.extend([f"Cpd-{m}" for m in molecule_matches])
        
        return list(set(ids))

    def _run_single_site_analysis(
        self,
        compounds: list[dict],
        sar_agent_results: Optional[dict],
        target_position: str,
        activity_cols: list[str],
    ) -> dict:
        """Run SAR analysis focused on a single R-group position.
        
        Args:
            compounds: List of compound dictionaries.
            sar_agent_results: Results from SAR Agent if available.
            target_position: R-group position to focus on (e.g., "R1").
            activity_cols: Activity column names.
        
        Returns:
            SAR analysis results focused on single position.
        """
        # Run full analysis first
        if sar_agent_results:
            sar_results = self._merge_sar_results(sar_agent_results, activity_cols)
        else:
            sar_results = self.analysis_tool.invoke({
                "compounds": compounds,
                "activity_columns": activity_cols if activity_cols else None,
            })
        
        # Filter OCAT pairs to only those varying at target position
        if "r_group_analysis" in sar_results:
            ocat_pairs = sar_results["r_group_analysis"].get("ocat_pairs", [])
            filtered_pairs = [
                p for p in ocat_pairs 
                if p.get("varying_position") == target_position
            ]
            sar_results["r_group_analysis"]["ocat_pairs"] = filtered_pairs
            sar_results["r_group_analysis"]["filtered_by_position"] = target_position
        
        # Add metadata
        sar_results["analysis_mode"] = "single_site"
        sar_results["target_position"] = target_position
        
        if activity_cols:
            sar_results["activity_columns"] = activity_cols
        
        return sar_results

    def _run_molecule_subset_analysis(
        self,
        compounds: list[dict],
        sar_agent_results: Optional[dict],
        target_molecules: list[str],
        activity_cols: list[str],
    ) -> dict:
        """Run SAR analysis on a subset of molecules.
        
        Args:
            compounds: Full list of compound dictionaries.
            sar_agent_results: Results from SAR Agent if available.
            target_molecules: List of molecule IDs/names to analyze.
            activity_cols: Activity column names.
        
        Returns:
            SAR analysis results for the subset.
        """
        # Normalize target molecule IDs
        target_set = set(m.lower().replace("-", "") for m in target_molecules)
        
        # Filter compounds
        filtered_compounds = []
        for cpd in compounds:
            cpd_id = (cpd.get("compound_id") or "").lower().replace("-", "")
            cpd_name = (cpd.get("name") or "").lower().replace("-", "")
            
            if cpd_id in target_set or cpd_name in target_set:
                filtered_compounds.append(cpd)
        
        if not filtered_compounds:
            # Fallback: try partial matching
            for cpd in compounds:
                cpd_id = cpd.get("compound_id", "")
                cpd_name = cpd.get("name", "")
                for target in target_molecules:
                    if target.lower() in cpd_id.lower() or target.lower() in cpd_name.lower():
                        filtered_compounds.append(cpd)
                        break
        
        logger.info(f"Filtered to {len(filtered_compounds)} compounds from {len(compounds)}")
        
        # Run analysis on filtered compounds
        sar_results = self.analysis_tool.invoke({
            "compounds": filtered_compounds,
            "activity_columns": activity_cols if activity_cols else None,
        })
        
        # Add metadata
        sar_results["analysis_mode"] = "molecule_subset"
        sar_results["target_molecules"] = target_molecules
        sar_results["original_count"] = len(compounds)
        
        if activity_cols:
            sar_results["activity_columns"] = activity_cols
        
        return sar_results

    def _get_sar_agent_results(self, results: dict) -> dict | None:
        """Get results from SAR Agent if available."""
        for task_id, result in results.items():
            if isinstance(result, dict) and "decomposed_compounds" in result:
                return result
        return None

    def _get_activity_columns(self, results: dict) -> list[str]:
        """Extract activity column names from data extraction results."""
        for task_id, result in results.items():
            if isinstance(result, dict):
                cols = result.get("activity_columns", [])
                if cols:
                    return cols
        return []

    def _merge_sar_results(self, sar_agent_results: dict, activity_columns: list[str] = None) -> dict:
        """Merge SAR Agent results with additional analyses."""
        compounds = sar_agent_results.get("compounds", [])
        decomposed = sar_agent_results.get("decomposed_compounds", [])
        
        # Run full analysis on compounds with multi-activity support
        sar_results = self.analysis_tool.invoke({
            "compounds": compounds,
            "activity_columns": activity_columns if activity_columns else None,
        })
        
        # Preserve activity_columns in results for HTML generation
        if activity_columns:
            sar_results["activity_columns"] = activity_columns
        
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
                                "activities": item.get("activities", {}),  # Multi-activity support
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
                                    "activities": item.get("activities", {}),  # Multi-activity support
                                    "compound_id": item.get("compound_id", ""),
                                    "name": item.get("name") or item.get("Name"),
                                })
                except Exception as e:
                    logger.warning(f"Failed to load {json_path}: {e}")

        return compounds

