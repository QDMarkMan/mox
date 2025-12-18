"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description DataCleanerAgent - AI-driven molecular data extraction.
*               Focuses on decision-making for standardized data extraction.
**************************************************************************
"""

import json
import logging
import os
import re
from typing import Optional

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
from molx_agent.tools.standardize import (
    CleanCompoundDataTool,
    SaveCleanedDataTool,
)

logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# System Prompt for AI-driven data extraction
# =============================================================================

DATA_CLEANER_PROMPT = """You are a molecular data extraction and standardization assistant.

Your task is to extract, clean and standardize molecular data from files.

## Available Tools:

### Extraction Tools:
1. `extract_from_csv` - Extract data from CSV files
2. `extract_from_excel` - Extract data from Excel files (.xlsx, .xls)
3. `extract_from_sdf` - Extract data from SDF/MOL files

### Standardization Tools:
4. `clean_compound_data` - Clean and validate compound data (canonicalize SMILES, filter invalid)
5. `save_cleaned_data` - Save cleaned data to JSON/CSV files

## Workflow:
1. Analyze the input to determine the file type
2. Call the appropriate EXTRACTION tool to get raw compound data
3. Call `clean_compound_data` to standardize the extracted data
4. Call `save_cleaned_data` to persist the results

## File Type Detection:
- `.csv` → use extract_from_csv
- `.xlsx`, `.xls` → use extract_from_excel  
- `.sdf`, `.mol`, `.sd` → use extract_from_sdf

Always extract first, then clean, then save.
"""


class DataCleanerAgent(BaseAgent):
    """AI-driven molecular data extraction and standardization agent.
    
    Focuses on decision-making: which extractor to use, how to clean data.
    Uses tools from extractor.py and standardize.py.
    """

    def __init__(self) -> None:
        super().__init__(
            name="data_cleaner",
            description="Extracts and standardizes molecular data using AI decision-making",
        )
        
        # Initialize LLM
        self.llm = get_llm()
        
        # Initialize ALL tools
        self.extractor_tools = [
            ExtractFromCSVTool(llm=self.llm),
            ExtractFromExcelTool(llm=self.llm),
            ExtractFromSDFTool(),
        ]
        
        self.standardize_tools = [
            CleanCompoundDataTool(),
            SaveCleanedDataTool(),
        ]
        
        self.tools = self.extractor_tools + self.standardize_tools
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

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

    def _get_extractor_for_file(self, file_path: str):
        """Get appropriate extractor tool based on file extension."""
        ext = file_path.lower().split('.')[-1]
        if ext == 'csv':
            return self.extractor_tools[0]  # CSV
        elif ext in ['xlsx', 'xls']:
            return self.extractor_tools[1]  # Excel
        elif ext in ['sdf', 'sd', 'mol']:
            return self.extractor_tools[2]  # SDF
        return None

    def run(self, state: AgentState) -> AgentState:
        """Execute data extraction and standardization.
        
        The agent decides:
        1. Which extractor tool to use based on file type
        2. How to clean and standardize the data
        3. Where to save the output
        
        Args:
            state: Agent state with task info.
            
        Returns:
            Updated state with extracted and cleaned data.
        """
        console.print("\n[bold cyan]🧹 DataCleaner: AI-driven extraction...[/]")

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
            # Step 1: Extract data using appropriate tool
            extractor = self._get_extractor_for_file(file_path)
            if not extractor:
                state["results"][tid] = {"error": f"Unsupported file type: {file_path}"}
                return state
            
            console.print(f"   [dim]Using extractor: {extractor.name}[/]")
            extracted_data = extractor.invoke(file_path)
            
            compounds = extracted_data.get("compounds", [])
            console.print(f"   [dim]Extracted: {len(compounds)} compounds[/]")
            
            # Step 2: Clean and standardize data
            if compounds:
                cleaner = self.standardize_tools[0]  # CleanCompoundDataTool
                clean_result = cleaner.invoke({"compounds": compounds})
                
                cleaned_compounds = clean_result.get("compounds", [])
                console.print(f"   [dim]Cleaned: {clean_result.get('cleaned_count')} compounds[/]")
                
                # Update extracted data with cleaned compounds
                extracted_data["compounds"] = cleaned_compounds
                extracted_data["cleaning_stats"] = {
                    "original_count": clean_result.get("original_count"),
                    "cleaned_count": clean_result.get("cleaned_count"),
                    "removed": clean_result.get("removed"),
                }
            
            # Step 3: Save output
            saver = self.standardize_tools[1]  # SaveCleanedDataTool
            output_files = saver.invoke({"data": extracted_data, "task_id": tid})
            extracted_data["output_files"] = output_files
            
            # Update state
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = extracted_data
            
            console.print(f"[green]✓ DataCleaner: Processed {len(extracted_data.get('compounds', []))} compounds[/]")
            for fmt, path in output_files.items():
                console.print(f"   [dim]{fmt.upper()}: {path}[/]")
                
        except Exception as e:
            console.print(f"[red]✗ DataCleaner error: {e}[/]")
            logger.error(f"DataCleaner error: {e}")
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {"error": str(e)}

        return state
