"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description DataCleanerAgent - AI-driven molecular data extraction.
*               Focuses on decision-making for standardized data extraction.
**************************************************************************
"""

import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

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

Your task is to extract, clean and standardize molecular data from files or inline data.

## Available Tools:

### Extraction Tools:
1. `extract_from_csv` - Extract data from CSV files
2. `extract_from_excel` - Extract data from Excel files (.xlsx, .xls)
3. `extract_from_sdf` - Extract data from SDF/MOL files

### Standardization Tools:
4. `clean_compound_data` - Clean and validate compound data (canonicalize SMILES, filter invalid)
5. `save_cleaned_data` - Save cleaned data to JSON/CSV files

## Data Sources:
1. **File Path**: Extract data from local files (.csv, .xlsx, .xls, .sdf, .mol)
2. **Inline CSV Data**: Parse CSV data directly from user input text

## Workflow:
1. Analyze the input to determine data source (file path or inline CSV)
2. If file path: Call the appropriate EXTRACTION tool
3. If inline CSV: Parse the CSV data directly from input
4. Call `clean_compound_data` to standardize the extracted data
5. Call `save_cleaned_data` to persist the results

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

    def _extract_inline_csv(self, text: str) -> Optional[str]:
        """Extract inline CSV data from user input text.
        
        Supports formats like:
        - CSV data in code blocks: ```csv ... ``` or ``` ... ```
        - Raw CSV data with headers containing common column names (smiles, compound, IC50, etc.)
        
        Args:
            text: User input text potentially containing CSV data.
            
        Returns:
            Extracted CSV string or None if not found.
        """
        # Pattern 1: CSV in code blocks (```csv ... ``` or ``` ... ```)
        code_block_pattern = r'```(?:csv)?\s*\n([\s\S]+?)\n```'
        match = re.search(code_block_pattern, text, re.IGNORECASE)
        if match:
            csv_content = match.group(1).strip()
            # Verify it looks like CSV (has commas and newlines)
            if ',' in csv_content and '\n' in csv_content:
                return csv_content
        
        # Pattern 2: Raw CSV data with molecular/activity headers
        # Look for lines that look like CSV headers
        csv_header_keywords = [
            'smiles', 'compound', 'molecule', 'name', 'id',
            'ic50', 'ec50', 'ki', 'kd', 'activity', 'potency',
            'quality', 'duplicate'
        ]
        
        lines = text.split('\n')
        csv_start_idx = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check if this line looks like a CSV header
            if ',' in line:
                matching_keywords = sum(1 for kw in csv_header_keywords if kw in line_lower)
                if matching_keywords >= 2:  # At least 2 keywords suggest CSV header
                    csv_start_idx = i
                    break
        
        if csv_start_idx is not None:
            # Find the end of CSV data (empty line or text without commas)
            csv_lines = []
            for line in lines[csv_start_idx:]:
                stripped = line.strip()
                if not stripped:  # Empty line - might be end of CSV
                    if csv_lines:  # Only break if we already have data
                        break
                    continue
                if ',' not in stripped and len(csv_lines) > 0:
                    # Non-CSV line after we started collecting
                    break
                csv_lines.append(stripped)
            
            if len(csv_lines) >= 2:  # Header + at least one data row
                return '\n'.join(csv_lines)
        
        return None

    def _parse_inline_csv_data(self, csv_content: str) -> Dict[str, Any]:
        """Parse inline CSV content into compound data.
        
        Uses LLM-assisted column detection similar to ExtractFromCSVTool.
        
        Args:
            csv_content: CSV string to parse.
            
        Returns:
            Extracted data dict with 'compounds' and 'columns' keys.
        """
        console.print("   [dim]Parsing inline CSV data...[/]")
        
        try:
            # Parse CSV with pandas
            df = pd.read_csv(io.StringIO(csv_content))
            console.print(f"   [dim]Parsed {len(df)} rows, columns: {list(df.columns)}[/]")
            
            # Use LLM to detect column mappings
            column_mapping = self._detect_columns_with_llm(df)
            
            smiles_col = column_mapping.get('smiles_col')
            name_col = column_mapping.get('name_col')
            activity_cols = column_mapping.get('activity_cols', [])
            
            if not smiles_col or smiles_col not in df.columns:
                # Try common SMILES column names
                for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'smi']:
                    if col in df.columns:
                        smiles_col = col
                        break
            
            if not smiles_col:
                console.print("[red]✗ No SMILES column found in CSV[/]")
                return {"compounds": [], "error": "No SMILES column found"}
            
            # Build compounds list
            compounds = []
            for idx, row in df.iterrows():
                smiles = str(row[smiles_col]).strip() if pd.notna(row[smiles_col]) else None
                if not smiles or smiles.lower() in ['nan', 'none', '']:
                    continue
                
                # Get compound ID from name column or generate one
                compound_id = None
                if name_col and pd.notna(row.get(name_col)):
                    compound_id = str(row[name_col])
                if not compound_id:
                    compound_id = f"Cpd-{idx+1}"
                
                # Build activities dict and get primary activity
                activities = {}
                primary_activity = None
                
                for act_col in activity_cols:
                    if act_col in df.columns and pd.notna(row.get(act_col)):
                        val = row[act_col]
                        try:
                            # Handle string values like ">200"
                            if isinstance(val, str):
                                val_clean = val.replace('>', '').replace('<', '').strip()
                                float_val = float(val_clean)
                            else:
                                float_val = float(val)
                            
                            activities[act_col] = float_val
                            if primary_activity is None:
                                primary_activity = float_val
                        except (ValueError, TypeError):
                            # Keep original value if cannot parse
                            activities[act_col] = val
                
                compound = {
                    "smiles": smiles,
                    "compound_id": compound_id,
                    "name": compound_id,
                    "activity": primary_activity,
                    "activities": activities,
                    "properties": {},
                }
                
                # Add other columns as properties (exclude smiles, name, and activity columns)
                for col in df.columns:
                    if col not in [smiles_col, name_col] + activity_cols:
                        if pd.notna(row.get(col)):
                            compound["properties"][col] = row[col]
                
                compounds.append(compound)
            
            console.print(f"   [dim]Extracted {len(compounds)} compounds from inline CSV[/]")
            
            return {
                "compounds": compounds,
                "columns": column_mapping,
                "source": "inline_csv",
                "activity_columns": activity_cols,  # Pass activity columns for downstream use
            }
            
        except Exception as e:
            console.print(f"[red]✗ Failed to parse inline CSV: {e}[/]")
            logger.error(f"Inline CSV parse error: {e}")
            return {"compounds": [], "error": str(e)}

    def _detect_columns_with_llm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM to detect SMILES, name, and activity columns.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Column mapping dict with smiles_col, name_col, activity_cols.
        """
        columns = list(df.columns)
        sample_rows = df.head(3).to_dict('records')
        
        prompt = f"""Analyze these CSV columns and sample data to identify:
1. SMILES column (contains molecular SMILES strings)
2. Name/ID column (compound identifier)
3. Activity columns (IC50, EC50, Ki, etc.)

Columns: {columns}
Sample data: {sample_rows}

Return JSON:
{{
  "smiles_col": "column_name or null",
  "name_col": "column_name or null", 
  "activity_cols": ["col1", "col2"]
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM column detection failed: {e}")
        
        # Fallback: heuristic detection
        mapping = {"smiles_col": None, "name_col": None, "activity_cols": []}
        
        for col in columns:
            col_lower = col.lower()
            if 'smiles' in col_lower or col_lower == 'smi':
                mapping["smiles_col"] = col
            elif any(k in col_lower for k in ['name', 'id', 'compound', 'number']):
                if not mapping["name_col"]:
                    mapping["name_col"] = col
            elif any(k in col_lower for k in ['ic50', 'ec50', 'ki', 'kd', 'activity', 'potency', 'inhibit']):
                mapping["activity_cols"].append(col)
        
        return mapping

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

    def _get_uploaded_file_path(self, state: AgentState) -> Optional[str]:
        """Return the first existing uploaded file path from agent state."""
        uploads = state.get("uploaded_files") or []
        for upload in uploads:
            if not isinstance(upload, dict):
                continue
            candidate = upload.get("file_path")
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def run(self, state: AgentState) -> AgentState:
        """Execute data extraction and standardization.
        
        The agent decides:
        1. Data source: file path or inline CSV data
        2. Which extractor tool to use based on file type (if file)
        3. How to clean and standardize the data
        4. Where to save the output
        
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
        
        # Combined text for analysis
        combined_text = f"{description}\n{user_query}\n{inputs.get('data', '')}"
        
        # Try to find file path first
        content = inputs.get("data") or inputs.get("file_path") or ""
        if not content:
            content = self._extract_file_path(description) or self._extract_file_path(user_query) or ""

        file_path = self._extract_file_path(content) if content else None
        if not file_path:
            uploaded_path = self._get_uploaded_file_path(state)
            if uploaded_path:
                file_path = uploaded_path
                console.print(f"   [dim]Using uploaded file: {uploaded_path}[/]")
        
        # Check if we have a valid file path
        extracted_data = None
        data_source = None
        
        if file_path and os.path.exists(file_path):
            # Source 1: File-based extraction
            console.print(f"   [dim]File: {file_path}[/]")
            data_source = "file"
            
            try:
                extractor = self._get_extractor_for_file(file_path)
                if not extractor:
                    if "results" not in state:
                        state["results"] = {}
                    state["results"][tid] = {"error": f"Unsupported file type: {file_path}"}
                    return state
                
                console.print(f"   [dim]Using extractor: {extractor.name}[/]")
                extracted_data = extractor.invoke(file_path)
            except Exception as e:
                console.print(f"[red]✗ File extraction error: {e}[/]")
                logger.error(f"File extraction error: {e}")
                extracted_data = None
        
        # Try inline CSV if no file found or file extraction failed
        if extracted_data is None or not extracted_data.get("compounds"):
            # Source 2: Inline CSV extraction
            csv_content = self._extract_inline_csv(combined_text)
            
            if csv_content:
                console.print("   [dim]Detected inline CSV data[/]")
                data_source = "inline_csv"
                extracted_data = self._parse_inline_csv_data(csv_content)
            elif file_path and not os.path.exists(file_path):
                console.print(f"[red]✗ File not found: {file_path}[/]")
                if "results" not in state:
                    state["results"] = {}
                state["results"][tid] = {"error": f"File not found: {file_path}"}
                return state
        
        # Check if we have any data
        if extracted_data is None or not extracted_data.get("compounds"):
            console.print("[red]✗ No data source found (no file path or inline CSV)[/]")
            if "results" not in state:
                state["results"] = {}
            state["results"][tid] = {"error": "No data source found. Please provide a file path or inline CSV data."}
            return state
        
        # Log data source info
        console.print(f"   [dim]Data source: {data_source}[/]")
        
        compounds = extracted_data.get("compounds", [])
        console.print(f"   [dim]Extracted: {len(compounds)} compounds[/]")
        
        try:
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
            
            # Preserve activity columns metadata for multi-activity report support
            extracted_data["activity_columns"] = extracted_data.get("columns", {}).get("activity_cols", [])
            
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
