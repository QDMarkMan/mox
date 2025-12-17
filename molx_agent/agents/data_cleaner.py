"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description DataCleanerAgent - AI-driven molecular data extraction.
*               Uses LangGraph to decide which extractor tool to use.
**************************************************************************
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import get_llm
from molx_agent.tools.extractor import (
    ExtractFromCSVTool,
    ExtractFromExcelTool,
    ExtractFromSDFTool,
)

logger = logging.getLogger(__name__)
console = Console()

OUTPUT_DIR = os.path.join(os.getcwd(), "output")


# =============================================================================
# Data Cleaning Utilities
# =============================================================================

def clean_compound_data(compounds: list[dict]) -> list[dict]:
    """Clean and validate compound data."""
    from rdkit import Chem

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

    return cleaned


def save_cleaned_data(data: dict, task_id: str) -> dict:
    """Save cleaned data to output files."""
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


# =============================================================================
# System Prompt for AI-driven extraction
# =============================================================================

DATA_CLEANER_PROMPT = """You are a molecular data extraction assistant.

Your task is to extract and clean molecular data from files or text input.

## Available Tools:
1. `extract_from_csv` - Extract data from CSV files
2. `extract_from_excel` - Extract data from Excel files (.xlsx, .xls)
3. `extract_from_sdf` - Extract data from SDF/MOL files

## Instructions:
1. Analyze the input to determine the file type
2. Call the appropriate extraction tool with the file path
3. The tool will return extracted compound data

## File Type Detection:
- `.csv` → use extract_from_csv
- `.xlsx`, `.xls` → use extract_from_excel  
- `.sdf`, `.mol`, `.sd` → use extract_from_sdf

If you receive a file path, extract it and call the appropriate tool.
"""


class DataCleanerAgent(BaseAgent):
    """AI-driven molecular data extraction agent.
    
    Uses LangGraph with extractor tools to intelligently
    decide how to extract data from different file formats.
    """

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Extracts and cleans molecular data using AI",
        )
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize tools
        self.tools = [
            ExtractFromCSVTool(llm=self.llm),
            ExtractFromExcelTool(llm=self.llm),
            ExtractFromSDFTool(),
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build graph
        self._build_graph()

    def _build_graph(self):
        """Build LangGraph for tool calling."""
        from langgraph.graph.message import add_messages
        from typing import Annotated
        from typing_extensions import TypedDict
        
        class ExtractorState(TypedDict):
            messages: Annotated[list, add_messages]
            extracted_data: Optional[dict]
        
        workflow = StateGraph(ExtractorState)
        
        def call_model(state: ExtractorState) -> dict:
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def process_tool_result(state: ExtractorState) -> dict:
            # Get last tool message
            messages = state["messages"]
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'tool':
                    # Parse tool result
                    try:
                        if isinstance(msg.content, str):
                            data = json.loads(msg.content)
                        else:
                            data = msg.content
                        return {"extracted_data": data}
                    except:
                        pass
            return {"extracted_data": None}
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("process", process_tool_result)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "process")
        workflow.add_edge("process", END)
        
        self._graph = workflow.compile()

    def _extract_file_path(self, text: str) -> Optional[str]:
        """Extract file path from text."""
        patterns = [
            r'(/[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb|sd))',
            r'([A-Za-z]:\\[^\s]+\.(?:csv|xlsx|xls|sdf|mol2|pdb|sd))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def run(self, state: AgentState) -> AgentState:
        """Execute data extraction using AI.
        
        Args:
            state: Agent state with task info.
            
        Returns:
            Updated state with extracted data.
        """
        console.print("\n[bold cyan]🧹 DataCleaner: Processing with AI...[/]")

        tid = state.get("current_task_id", "data_extraction")
        task = state.get("tasks", {}).get(tid, {})
        
        # Get input content
        description = task.get("description", "")
        inputs = task.get("inputs", {})
        user_query = state.get("user_query", "")
        
        # Find file path
        content = inputs.get("data") or inputs.get("file_path") or ""
        if not content:
            content = self._extract_file_path(description) or self._extract_file_path(user_query) or ""
        
        file_path = self._extract_file_path(content) if content else None
        
        if not file_path:
            console.print("[red]✗ No file path found[/]")
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {"error": "No file path found"}
            return state
        
        if not os.path.exists(file_path):
            console.print(f"[red]✗ File not found: {file_path}[/]")
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {"error": f"File not found: {file_path}"}
            return state
        
        console.print(f"   [dim]File: {file_path}[/]")
        
        try:
            # Use LangGraph to extract
            prompt = f"Extract molecular data from this file: {file_path}"
            
            result = self._graph.invoke({
                "messages": [
                    SystemMessage(content=DATA_CLEANER_PROMPT),
                    HumanMessage(content=prompt),
                ],
                "extracted_data": None,
            })
            
            data = result.get("extracted_data", {})
            
            if not data:
                # Fallback: direct extraction based on extension
                console.print("   [dim]Using direct extraction fallback...[/]")
                ext = file_path.lower().split('.')[-1]
                
                if ext == 'csv':
                    data = self.tools[0].invoke(file_path)
                elif ext in ['xlsx', 'xls']:
                    data = self.tools[1].invoke(file_path)
                elif ext in ['sdf', 'sd', 'mol']:
                    data = self.tools[2].invoke(file_path)
            
            # Clean compounds
            compounds = data.get("compounds", [])
            if compounds:
                console.print(f"   [dim]Cleaning {len(compounds)} compounds...[/]")
                cleaned_compounds = clean_compound_data(compounds)
                data["compounds"] = cleaned_compounds
                data["cleaning_stats"] = {
                    "original_count": len(compounds),
                    "cleaned_count": len(cleaned_compounds),
                    "removed": len(compounds) - len(cleaned_compounds),
                }
            
            # Save output
            output_files = save_cleaned_data(data, tid)
            data["output_files"] = output_files
            
            # Update state
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = data
            
            console.print(f"[green]✓ DataCleaner: Extracted {len(data.get('compounds', []))} compounds[/]")
            for fmt, path in output_files.items():
                console.print(f"   [dim]{fmt.upper()}: {path}[/]")
                
        except Exception as e:
            console.print(f"[red]✗ DataCleaner error: {e}[/]")
            logger.error(f"DataCleaner error: {e}")
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {"error": str(e)}

        return state
