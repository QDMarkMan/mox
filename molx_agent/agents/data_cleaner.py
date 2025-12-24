"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description DataCleanerAgent - LLM-assisted molecular data extraction.
*               Uses LLM for smart decisions but with deterministic workflow.
**************************************************************************
"""

import logging
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import get_llm
from molx_agent.agents.modules.tools import get_registry

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# System Prompt for data source detection
# =============================================================================

DATA_SOURCE_PROMPT = """Analyze the user input and identify the data source.

Return a JSON object with:
{
  "source_type": "file" | "inline_csv" | "unknown",
  "file_path": "path if source_type is file, else null",
  "file_type": "csv" | "excel" | "sdf" | null,
  "csv_content": "CSV content if inline, else null"
}

Rules:
1. Look for file paths ending with .csv, .xlsx, .xls, .sdf, .mol
2. Keep relative paths as-is (e.g., ./tests/data/file.csv)
3. If CSV data is in code blocks or raw format, extract it
4. Return only the JSON, no explanation"""


class DataCleanerAgent(BaseAgent):
    """LLM-assisted molecular data extraction and standardization agent.
    
    Uses a hybrid approach:
    - LLM for smart data source detection
    - Deterministic workflow for tool execution
    
    This avoids ReAct loops while still using AI for intelligence.
    """

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Extracts and standardizes molecular data using AI-assisted decisions",
        )
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Get tools from registry
        registry = get_registry()
        registry.inject_llm(self.llm)
        
        # Get specific tools by name
        self.resolve_path_tool = registry.get_tool_by_name("resolve_file_path")
        self.extract_csv_tool = registry.get_tool_by_name("extract_from_csv")
        self.extract_excel_tool = registry.get_tool_by_name("extract_from_excel")
        self.extract_sdf_tool = registry.get_tool_by_name("extract_from_sdf")
        self.parse_inline_csv_tool = registry.get_tool_by_name("parse_inline_csv")
        self.clean_data_tool = registry.get_tool_by_name("clean_compound_data")
        self.save_data_tool = registry.get_tool_by_name("save_cleaned_data")
        
        logger.info("DataCleanerAgent initialized with LLM-assisted workflow")

    def _detect_data_source(self, text: str) -> Dict[str, Any]:
        """Detect data source from text using fast regex (LLM as backup).
        
        Returns dict with source_type, file_path, file_type, csv_content.
        """
        # Try fast regex first
        result = self._fallback_detect_source(text)
        
        if result["source_type"] != "unknown":
            logger.debug(f"Regex detected source: {result}")
            return result
        
        # Only use LLM if regex failed
        try:
            response = self.llm.invoke([
                SystemMessage(content=DATA_SOURCE_PROMPT),
                HumanMessage(content=f"Input:\n{text}")
            ])
            
            content = response.content.strip()
            
            # Clean markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            logger.debug(f"LLM detected source: {result}")
            return result
            
        except Exception as e:
            logger.warning(f"LLM source detection failed: {e}")
            return {"source_type": "unknown", "file_path": None, "file_type": None, "csv_content": None}
    
    def _fallback_detect_source(self, text: str) -> Dict[str, Any]:
        """Fallback regex-based source detection."""
        import re
        
        # Look for file paths
        file_patterns = [
            (r'((?:\./|\.\./)?[a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-\.]+)*\.csv)', 'csv'),
            (r'((?:\./|\.\./)?[a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-\.]+)*\.xlsx?)', 'excel'),
            (r'((?:\./|\.\./)?[a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-\.]+)*\.sdf)', 'sdf'),
        ]
        
        for pattern, file_type in file_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {
                    "source_type": "file",
                    "file_path": match.group(1),
                    "file_type": file_type,
                    "csv_content": None,
                }
        
        return {"source_type": "unknown", "file_path": None, "file_type": None, "csv_content": None}

    def _resolve_file_path(self, file_path: str) -> Optional[str]:
        """Resolve file path using the tool."""
        if not self.resolve_path_tool:
            # Fallback manual resolution
            from pathlib import Path
            for candidate in [
                Path(file_path),
                Path.cwd() / file_path.lstrip('./'),
                Path.cwd() / file_path.lstrip('/'),
            ]:
                if candidate.exists():
                    return str(candidate)
            return None
        
        try:
            result = self.resolve_path_tool.invoke(file_path)
            if isinstance(result, str):
                result = json.loads(result)
            if result.get("success"):
                return result["resolved_path"]
            return None
        except Exception as e:
            logger.error(f"Path resolution error: {e}")
            return None

    def run(self, state: AgentState) -> AgentState:
        """Execute data extraction with LLM-assisted decisions.
        
        Workflow:
        1. LLM detects data source (file/inline)
        2. Resolve file path if needed
        3. Call appropriate extractor
        4. Clean extracted data
        5. Save results
        """
        console.print("\n[bold cyan]🧹 DataCleaner: AI-driven extraction...[/]")

        tid = state.get("current_task_id", "data_extraction")
        task = state.get("tasks", {}).get(tid, {})
        
        # Build input text
        description = task.get("description", "")
        inputs = task.get("inputs", {})
        user_query = state.get("user_query", "")
        combined_text = f"{user_query}\n{description}\n{inputs.get('data', '')}"
        
        console.print(f"   [dim]Analyzing input...[/]")
        
        # Step 1: LLM detects data source
        source_info = self._detect_data_source(combined_text)
        source_type = source_info.get("source_type", "unknown")
        
        extracted_data = None
        
        # Step 2 & 3: Extract based on source type
        if source_type == "file":
            file_path = source_info.get("file_path")
            file_type = source_info.get("file_type", "csv")
            
            console.print(f"   [dim]Detected file: {file_path} ({file_type})[/]")
            
            # Resolve path
            resolved = self._resolve_file_path(file_path)
            if not resolved:
                console.print(f"[red]✗ File not found: {file_path}[/]")
                state.setdefault("results", {})[tid] = {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "compounds": [],
                }
                return state
            
            console.print(f"   [dim]Resolved: {resolved}[/]")
            
            # Select and run extractor
            extractor = {
                "csv": self.extract_csv_tool,
                "excel": self.extract_excel_tool,
                "sdf": self.extract_sdf_tool,
            }.get(file_type)
            
            if extractor:
                try:
                    console.print(f"   [dim]Extracting with {extractor.name}...[/]")
                    extracted_data = extractor.invoke(resolved)
                except Exception as e:
                    console.print(f"[red]✗ Extraction error: {e}[/]")
                    extracted_data = None
        
        elif source_type == "inline_csv":
            csv_content = source_info.get("csv_content")
            if csv_content and self.parse_inline_csv_tool:
                console.print(f"   [dim]Parsing inline CSV...[/]")
                extracted_data = self.parse_inline_csv_tool.invoke(csv_content)
        
        # Step 4: Clean data if extracted
        if extracted_data and extracted_data.get("compounds") and self.clean_data_tool:
            try:
                console.print(f"   [dim]Cleaning {len(extracted_data['compounds'])} compounds...[/]")
                # Pass compounds as dict with list, not JSON string
                cleaned = self.clean_data_tool.invoke({"compounds": extracted_data["compounds"]})
                if isinstance(cleaned, str):
                    cleaned = json.loads(cleaned)
                if isinstance(cleaned, dict) and cleaned.get("compounds"):
                    extracted_data["compounds"] = cleaned["compounds"]
            except Exception as e:
                logger.warning(f"Data cleaning error: {e}")
        
        # Step 5: Save cleaned data to files
        output_files = {}
        compounds = extracted_data.get("compounds", []) if extracted_data else []
        
        if compounds and self.save_data_tool:
            try:
                console.print(f"   [dim]Saving {len(compounds)} compounds to files...[/]")
                save_result = self.save_data_tool.invoke({
                    "data": extracted_data,
                    "task_id": tid,
                })
                if isinstance(save_result, str):
                    save_result = json.loads(save_result)
                if isinstance(save_result, dict):
                    output_files = save_result
                    if "json" in output_files:
                        console.print(f"   [dim]Saved JSON: {output_files['json']}[/]")
                    if "csv" in output_files:
                        console.print(f"   [dim]Saved CSV: {output_files['csv']}[/]")
            except Exception as e:
                logger.warning(f"Data save error: {e}")
        
        # Store results with output files
        state.setdefault("results", {})[tid] = {
            "success": bool(compounds),
            "compounds": compounds,
            "activity_columns": extracted_data.get("activity_columns", []) if extracted_data else [],
            "source_file": extracted_data.get("source_file", "") if extracted_data else "",
            "extracted_data": extracted_data or {},
            "output_files": output_files,
        }
        
        if compounds:
            console.print(f"[green]✓ Extracted {len(compounds)} compounds[/]")
        else:
            console.print("[yellow]⚠ No compounds extracted[/]")
        
        return state
