"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-16].
*  @Description ReAct tools wrapping core agent functionalities.
**************************************************************************
"""

import logging
import os
from typing import Any, Optional

from langchain_core.tools import tool

from molx_agent.agents.data_cleaner import (
    detect_input_type,
    extract_file_path,
    parse_json_data,
    clean_compound_data,
    save_cleaned_data,
)
from molx_agent.tools.extractor import ExtractFromCSVTool, ExtractFromExcelTool, ExtractFromSDFTool
from molx_agent.agents.sar import SARAgent
from molx_agent.agents.reporter import ReporterAgent
from molx_agent.tools.html_builder import build_sar_html_report, save_html_report

logger = logging.getLogger(__name__)


@tool
def clean_data_tool(query: str = "", file_path: str = "") -> dict[str, Any]:
    """Clean and standardize molecular data from a file or query.
    
    Args:
        query: User query describing the data or file.
        file_path: Optional explicit file path.
        
    Returns:
        Dictionary with cleaned data and file paths.
    """
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold yellow]🛠️ Tool Call: clean_data_tool[/]")
        console.print(f"[dim]   Query: {query}[/]")
        console.print(f"[dim]   File Path: {file_path}[/]")
        
        # Get LLM for column identification
        from molx_agent.agents.modules.llm import get_llm
        llm = get_llm()
        
        content = file_path if file_path else query
        
        # 1. Detect input type
        input_type = detect_input_type(content)
        
        # 2. Extract data
        data = {}
        if input_type == 'json':
            data = parse_json_data(content)
        elif input_type == 'file':
            fpath = extract_file_path(content)
            if not fpath or not os.path.exists(fpath):
                return {"error": f"File not found: {fpath}"}
            
            ext = fpath.lower().split('.')[-1]
            if ext == 'csv':
                data = ExtractFromCSVTool(llm=llm).invoke(fpath)
            elif ext in ['xlsx', 'xls']:
                data = ExtractFromExcelTool(llm=llm).invoke(fpath)
            elif ext == 'sdf':
                data = ExtractFromSDFTool().invoke(fpath)
            else:
                return {"error": f"Unsupported file type: {ext}"}
        else:
            # Try to find file in text
            fpath = extract_file_path(content)
            if fpath and os.path.exists(fpath):
                ext = fpath.lower().split('.')[-1]
                if ext == 'csv':
                    data = ExtractFromCSVTool(llm=llm).invoke(fpath)
                elif ext in ['xlsx', 'xls']:
                    data = ExtractFromExcelTool(llm=llm).invoke(fpath)
                elif ext in ['sdf', 'sd']:
                    data = ExtractFromSDFTool().invoke(fpath)
                else:
                    return {"error": f"Unsupported file type: {ext}"}
            else:
                return {"error": "No valid data source found"}
        
        # 3. Clean data
        compounds = data.get("compounds", [])
        if compounds:
            cleaned_compounds = clean_compound_data(compounds)
            data["compounds"] = cleaned_compounds
            
            # Save intermediate files
            output_files = save_cleaned_data(data, "clean_data_tool")
            json_path = output_files.get("json")
            
            return {
                "summary": f"Cleaned {len(cleaned_compounds)} compounds.",
                "file_path": json_path,
                "compounds_sample": cleaned_compounds[:3] # Show a few for context
            }
        else:
            return {"error": "No compounds found in data"}
            
    except Exception as e:
        logger.error(f"clean_data_tool error: {e}")
        return {"error": str(e)}


@tool
def sar_analysis_tool(file_path: Optional[str] = None, compounds: Optional[list[dict]] = None, scaffold: Optional[str] = None) -> dict[str, Any]:
    """Perform SAR analysis including scaffold selection and R-group decomposition.
    
    Args:
        file_path: Path to JSON file containing compounds (preferred for large datasets).
        compounds: List of compound dictionaries (alternative to file_path).
        scaffold: Optional user-specified core scaffold (SMILES/SMARTS).
        
    Returns:
        Dictionary containing SAR analysis results.
    """
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold yellow]🛠️ Tool Call: sar_analysis_tool[/]")
        
        # Load compounds
        target_compounds = []
        if file_path and os.path.exists(file_path):
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                target_compounds = data.get("compounds", [])
            except Exception as e:
                return {"error": f"Failed to read file {file_path}: {e}"}
        elif compounds:
            target_compounds = compounds
        else:
            return {"error": "Must provide either file_path or compounds list"}
            
        if not target_compounds:
            return {"error": "No compounds found to analyze"}

        agent = SARAgent()
        
        # Construct a mock task for the agent method
        task = {"inputs": {}}
        if scaffold:
            task["inputs"]["scaffold"] = scaffold
            
        # 1. Select Scaffold
        scaffold_result = agent._select_scaffold(target_compounds, task)
        
        # 2. Decompose
        decomposed = agent._decompose_compounds(target_compounds, scaffold_result)
        
        # 3. OCAT Analysis
        # 3. OCAT Analysis
        from molx_agent.tools.sar import identify_ocat_series
        ocat_pairs = identify_ocat_series(decomposed)
        
        sar_results = {
            "scaffold": scaffold_result.get("scaffold"),
            "scaffold_strategy": scaffold_result.get("strategy"),
            "r_group_analysis": {
                "decomposed_compounds": decomposed,
                "ocat_pairs": ocat_pairs,
                "activity_range": {"min": 0, "max": 10}
            },
            "compounds": target_compounds # Pass through for reporter
        }
        
        # Save results to file to avoid context length issues
        import json
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.getcwd(), "output", "sar_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sar_results_{timestamp}.json")
        
        with open(output_path, "w") as f:
            json.dump(sar_results, f, indent=2)
            
        return {
            "summary": f"SAR analysis complete. Decomposed {len(decomposed)} compounds. Found {len(ocat_pairs)} OCAT pairs.",
            "file_path": output_path,
            "scaffold_strategy": scaffold_result.get("strategy")
        }
        
    except Exception as e:
        logger.error(f"sar_analysis_tool error: {e}")
        return {"error": str(e)}


@tool
def generate_report_tool(sar_results: Optional[dict] = None, sar_results_file: Optional[str] = None, title: str = "SAR Analysis Report") -> str:
    """Generate HTML SAR report from analysis results.
    
    Args:
        sar_results: Dictionary containing all SAR analysis results.
        sar_results_file: Path to JSON file containing SAR results (preferred).
        title: Title of the report.
        
    Returns:
        Path to the saved HTML report file.
    """
    try:
        from rich.console import Console
        console = Console()
        console.print(f"[bold yellow]🛠️ Tool Call: generate_report_tool[/]")
        
        data = {}
        if sar_results_file and os.path.exists(sar_results_file):
            import json
            with open(sar_results_file, 'r') as f:
                data = json.load(f)
        elif sar_results:
            data = sar_results
        else:
            return "Error: Must provide either sar_results or sar_results_file"
        
        html = build_sar_html_report(data, title)
        path = save_html_report(html)
        return path
        
    except Exception as e:
        logger.error(f"generate_report_tool error: {e}")
        return f"Error generating report: {str(e)}"
