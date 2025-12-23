"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-20].
*  @Description SAR Agent - Structure-Activity Relationship Analysis
*  @Version 1.0
*
*  Functionality:
*  - Scaffold Selection: Auto-select MCS/Murcko/Custom scaffold strategy
*  - R-Group Decomposition: Extract substituents at each position
*  - OCAT Analysis: Identify one-change-at-a-time molecular pairs
*  - Mode Detection: Auto-detect Single-Point Scan vs Full SAR mode
*  - LLM Insights: Generate AI-powered SAR trend analysis
*
*  Architecture: Prompt + Rules Pattern
*  - PROMPTS: LLM prompt templates
*  - RULES: Configurable decision rules (SinglePointScanRule, ScaffoldSelectionRule)
*  - SARAgentConfig: Feature toggles and thresholds
*
**************************************************************************
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import invoke_llm

from molx_agent.tools.sar import (
    find_mcs_scaffold,
    find_common_murcko_scaffold,
    decompose_r_groups,
    simple_r_group_assignment,
    identify_ocat_series,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class SARAnalysisMode(Enum):
    """SAR analysis modes."""
    FULL_SAR = "full_sar"
    SINGLE_POINT_SCAN = "single_point_scan"
    SCAFFOLD_HOPPING = "scaffold_hopping"
    ACTIVITY_CLIFF = "activity_cliff"


# ============================================================================
# Prompts
# ============================================================================

PROMPTS = {
    "system": """You are a medicinal chemistry SAR expert.
Analyze molecular data and provide insights on substituent effects and activity trends.
Always respond in valid JSON format when requested.""",

    "single_point_analysis": """Analyze Single-Point SAR data for position {position}:

Substituents and Activities:
{substituent_summary}

Top Activity Changes:
{ocat_summary}

Provide JSON:
{{"key_findings": [], "best_substituent": "", "worst_substituent": "", "recommendation": ""}}""",

    "full_sar_analysis": """Analyze multi-position SAR data:

Positions: {positions}
Compounds: {n_compounds}

Variation:
{variation_summary}

OCAT Pairs:
{ocat_summary}

Provide JSON:
{{"key_findings": [], "position_insights": {{}}, "optimization_suggestions": [], "summary": ""}}""",

    "scaffold_recommendation": """Recommend scaffold strategy for {n_compounds} compounds (diversity: {diversity}).
Options: MCS, Murcko, Custom.
Provide JSON: {{"strategy": "", "confidence": 0.0, "reasoning": ""}}""",
}


# ============================================================================
# Rules
# ============================================================================

@dataclass
class Rule:
    """Base decision rule."""
    name: str
    description: str
    enabled: bool = True
    
    def evaluate(self, context: dict) -> dict:
        raise NotImplementedError


@dataclass
class SinglePointScanRule(Rule):
    """Detect Single-Point Scan suitability."""
    name: str = "single_point_scan"
    description: str = "Detect single-position SAR suitability"
    min_compounds: int = 3
    min_substituents: int = 3
    max_other_variation: int = 2
    dominant_ratio: float = 0.5
    
    def evaluate(self, context: dict) -> dict:
        positions = context.get("variation_stats", {}).get("positions", {})
        if not positions:
            return {"satisfied": False, "reason": "No R-group positions"}
        
        total = sum(p["unique_count"] for p in positions.values())
        best_pos, best_ratio = None, 0
        
        for pos, stats in positions.items():
            if stats["unique_count"] >= self.min_substituents:
                ratio = stats["unique_count"] / total if total else 0
                if ratio > best_ratio:
                    best_ratio, best_pos = ratio, pos
        
        others_stable = all(
            positions[p]["unique_count"] <= self.max_other_variation
            for p in positions if p != best_pos
        )
        
        if best_pos and best_ratio >= self.dominant_ratio and others_stable:
            return {
                "satisfied": True,
                "mode": SARAnalysisMode.SINGLE_POINT_SCAN,
                "dominant_position": best_pos,
                "dominant_ratio": round(best_ratio, 2),
                "reason": f"{best_pos} dominates ({best_ratio:.0%})",
            }
        return {"satisfied": False, "reason": "Multiple positions vary"}


@dataclass
class ScaffoldSelectionRule(Rule):
    """Select scaffold extraction strategy."""
    name: str = "scaffold_selection"
    description: str = "Select scaffold method"
    min_for_mcs: int = 5
    min_for_any: int = 2
    
    def evaluate(self, context: dict) -> dict:
        n = context.get("n_compounds", 0)
        custom = context.get("custom_scaffold")
        
        if custom:
            return {"strategy": "custom", "scaffold": custom}
        if n < self.min_for_any:
            return {"strategy": "none", "scaffold": None}
        if n >= self.min_for_mcs:
            return {"strategy": "mcs", "try_mcs_first": True}
        return {"strategy": "murcko", "try_mcs_first": False}


RULES = {
    "single_point_scan": SinglePointScanRule(),
    "scaffold_selection": ScaffoldSelectionRule(),
}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SARAgentConfig:
    """SAR Agent configuration."""
    enable_mode_detection: bool = True
    enable_llm_insights: bool = True
    max_ocat_results: int = 20
    max_display_values: int = 10
    default_mode: SARAnalysisMode = field(default=SARAnalysisMode.FULL_SAR)


# ============================================================================
# SAR Agent
# ============================================================================

class SARAgent(BaseAgent):
    """SAR Agent for R-group decomposition and analysis."""

    def __init__(self, config: SARAgentConfig = None) -> None:
        super().__init__(name="sar", description="SAR analysis agent")
        self.config = config or SARAgentConfig()
        self.rules = RULES.copy()
        self.prompts = PROMPTS.copy()

    # --- Extension API ---
    
    def add_rule(self, name: str, rule: Rule) -> None:
        """Add/replace a decision rule."""
        self.rules[name] = rule
    
    def set_prompt(self, name: str, template: str) -> None:
        """Add/replace a prompt template."""
        self.prompts[name] = template
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """Get formatted prompt."""
        template = self.prompts.get(name, "")
        return template.format(**kwargs) if kwargs else template

    # --- Main Entry ---

    def run(self, state: AgentState) -> AgentState:
        """Execute SAR analysis."""
        from rich.console import Console
        console = Console()
        
        tid = state.get("current_task_id")
        task = state.get("tasks", {}).get(tid) if tid else None
        if not task:
            return state
        
        console.print(f"[cyan]🔬 SAR Agent: Processing task {tid}...[/]")
        
        try:
            # Collect compounds
            compounds = self._collect_compounds(state)
            if not compounds:
                state["results"][tid] = {"error": "No compound data"}
                state["tasks"][tid]["status"] = "done"
                return state
            console.print(f"[dim]   Found {len(compounds)} compounds[/]")
            
            # Scaffold selection
            scaffold_result = self._apply_scaffold_rule(compounds, task)
            console.print(f"[dim]   Scaffold: {scaffold_result['strategy']}[/]")
            
            # R-group decomposition
            decomposed = self._decompose_compounds(compounds, scaffold_result)
            console.print(f"[dim]   Decomposed {len(decomposed)} compounds[/]")
            
            # Mode detection
            mode_result = {"mode": self.config.default_mode, "reason": "Default"}
            if self.config.enable_mode_detection:
                mode_result = self._apply_mode_detection_rule(decomposed)
                console.print(f"[dim]   Mode: {mode_result['mode'].value}[/]")
            
            # OCAT analysis
            ocat_pairs = identify_ocat_series(decomposed)
            console.print(f"[dim]   OCAT pairs: {len(ocat_pairs)}[/]")
            
            # LLM insights
            ai_insights = None
            if self.config.enable_llm_insights and ocat_pairs:
                ai_insights = self._generate_insights(
                    mode_result["mode"], decomposed, ocat_pairs, mode_result
                )
            
            # Store results
            state["results"][tid] = {
                "scaffold_strategy": scaffold_result["strategy"],
                "scaffold": scaffold_result.get("scaffold"),
                "decomposed_compounds": decomposed,
                "ocat_pairs": ocat_pairs[:self.config.max_ocat_results],
                "total_ocat_pairs": len(ocat_pairs),
                "compounds": compounds,
                "analysis_mode": mode_result["mode"].value,
                "mode_decision": self._serialize(mode_result),
                "ai_insights": ai_insights,
            }
            state["tasks"][tid]["status"] = "done"
            console.print(f"[green]✓ Completed ({mode_result['mode'].value})[/]")
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/]")
            logger.error(f"SAR Agent error: {e}")
            state["results"][tid] = {"error": str(e)}
            state["tasks"][tid]["status"] = "done"
        
        return state

    # --- Rule Methods ---

    def _apply_scaffold_rule(self, compounds: list[dict], task: dict) -> dict:
        """Apply scaffold selection rule."""
        rule = self.rules["scaffold_selection"]
        context = {
            "n_compounds": len(compounds),
            "custom_scaffold": task.get("inputs", {}).get("scaffold") 
                              or task.get("inputs", {}).get("core"),
        }
        result = rule.evaluate(context)
        
        if result["strategy"] in ("custom", "none"):
            return result
        
        smiles_list = [c["smiles"] for c in compounds if c.get("smiles")]
        
        if result.get("try_mcs_first"):
            scaffold = find_mcs_scaffold(smiles_list)
            if scaffold:
                return {"strategy": "mcs", "scaffold": scaffold}
        
        scaffold = find_common_murcko_scaffold(smiles_list)
        return {"strategy": "murcko", "scaffold": scaffold} if scaffold else {"strategy": "none", "scaffold": None}

    def _apply_mode_detection_rule(self, decomposed: list[dict]) -> dict:
        """Apply mode detection rule."""
        rule = self.rules["single_point_scan"]
        
        if len(decomposed) < rule.min_compounds:
            return {"mode": self.config.default_mode, "reason": "Insufficient data"}
        
        variation_stats = self._analyze_variation_patterns(decomposed)
        result = rule.evaluate({"variation_stats": variation_stats})
        
        if result["satisfied"]:
            return {
                "mode": result["mode"],
                "dominant_position": result["dominant_position"],
                "dominant_ratio": result["dominant_ratio"],
                "variation_stats": variation_stats,
                "reason": result["reason"],
            }
        return {"mode": SARAnalysisMode.FULL_SAR, "variation_stats": variation_stats, "reason": result["reason"]}

    # --- Insight Generation ---

    def _generate_insights(self, mode: SARAnalysisMode, decomposed: list[dict], 
                          ocat_pairs: list[dict], mode_result: dict) -> dict:
        """Generate LLM insights."""
        if mode == SARAnalysisMode.SINGLE_POINT_SCAN:
            return self._generate_single_point_insights(
                decomposed, ocat_pairs, mode_result.get("dominant_position")
            )
        return None

    def _generate_single_point_insights(self, decomposed: list[dict], 
                                        ocat_pairs: list[dict], position: str) -> dict:
        """Generate Single-Point insights."""
        pairs = [p for p in ocat_pairs if p.get("varying_position") == position]
        if not pairs:
            return {"summary": "No OCAT pairs at position"}
        
        # Build summaries
        substituents = {}
        for cpd in decomposed:
            sub = cpd.get("r_groups", {}).get(position)
            act = cpd.get("activity")
            if sub and act:
                substituents.setdefault(sub, []).append(act)
        
        sub_lines = [f"  - {s}: avg={sum(a)/len(a):.2f}" for s, a in substituents.items()]
        ocat_lines = [
            f"  - {p['substituent1']}→{p['substituent2']}: {p['activity1']:.2f}→{p['activity2']:.2f}"
            for p in pairs[:5]
        ]
        
        prompt = self.get_prompt(
            "single_point_analysis",
            position=position,
            substituent_summary="\n".join(sub_lines[:15]),
            ocat_summary="\n".join(ocat_lines),
        )
        
        try:
            result = invoke_llm(self.prompts["system"], prompt, parse_json=True)
            return {"position": position, "substituent_count": len(substituents), "llm_insights": result}
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return {"position": position, "error": str(e)}

    # --- Helpers ---

    def _collect_compounds(self, state: AgentState) -> list[dict]:
        """Collect compounds from previous results."""
        compounds = []
        for result in state.get("results", {}).values():
            if isinstance(result, dict) and "compounds" in result and "error" not in result:
                compounds.extend(c for c in result["compounds"] if isinstance(c, dict) and c.get("smiles"))
        return compounds

    def _decompose_compounds(self, compounds: list[dict], scaffold_result: dict) -> list[dict]:
        """Decompose compounds into R-groups."""
        scaffold = scaffold_result.get("scaffold")
        
        if not scaffold:
            return [{
                "compound_id": c.get("compound_id", f"Cpd-{i}"),
                "name": c.get("name") or c.get("Name") or f"Cpd-{i}",
                "smiles": c.get("smiles"),
                "activity": c.get("activity"),
                "activities": c.get("activities", {}),
                "r_groups": {},
            } for i, c in enumerate(compounds)]
        
        smiles = [c.get("smiles") for c in compounds]
        activities = [c.get("activity") for c in compounds]
        ids = [c.get("compound_id", f"Cpd-{i}") for i, c in enumerate(compounds)]
        
        # Build lookup dict by compound_id for efficient matching
        # Use compound_id instead of SMILES since SMILES may differ after canonicalization
        compound_lookup = {c.get("compound_id", f"Cpd-{i}"): c for i, c in enumerate(compounds)}
        
        decomposed = decompose_r_groups(smiles, scaffold, activities, ids)
        if not decomposed:
            decomposed = simple_r_group_assignment(smiles, scaffold, activities, ids)
        
        # Merge activity data from original compounds using compound_id matching
        for dec in decomposed:
            cpd_id = dec.get("compound_id")
            cpd = compound_lookup.get(cpd_id)
            if cpd:
                dec["name"] = cpd.get("name") or cpd.get("Name") or cpd_id
                if cpd.get("activities"):
                    dec["activities"] = cpd["activities"]
        return decomposed

    def _analyze_variation_patterns(self, decomposed: list[dict]) -> dict:
        """Analyze R-group variation patterns."""
        positions = {}
        for cpd in decomposed:
            for pos, val in cpd.get("r_groups", {}).items():
                positions.setdefault(pos, {"values": set(), "count": 0})
                positions[pos]["values"].add(val)
                positions[pos]["count"] += 1
        
        return {
            "positions": {
                pos: {"unique_count": len(d["values"]), "coverage": d["count"],
                      "values": list(d["values"])[:self.config.max_display_values]}
                for pos, d in positions.items()
            },
            "total_compounds": len(decomposed),
        }

    def _serialize(self, obj: dict) -> dict:
        """Serialize Enum values for JSON."""
        return {
            k: v.value if isinstance(v, Enum) else self._serialize(v) if isinstance(v, dict) else v
            for k, v in obj.items()
        }
