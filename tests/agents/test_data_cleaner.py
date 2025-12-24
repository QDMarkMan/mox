"""
Tests for DataCleanerAgent LLM-driven extraction.
"""

import pytest
from pathlib import Path


class TestDataCleanerAgent:
    """Tests for the refactored LLM-driven DataCleanerAgent."""
    
    def test_agent_initialization(self):
        """Test that DataCleanerAgent initializes correctly with tools."""
        pytest.importorskip("langgraph")
        
        from molx_agent.agents.data_cleaner import DataCleanerAgent
        
        agent = DataCleanerAgent()
        
        assert agent.name == "data_cleaner"
        assert len(agent.tools) >= 5  # resolve, parse_csv, extract_csv, clean, save
        
        # Check for expected tools
        tool_names = [t.name for t in agent.tools]
        assert "resolve_file_path" in tool_names
        assert "extract_from_csv" in tool_names
        assert "clean_compound_data" in tool_names
    
    def test_resolve_file_path_tool(self):
        """Test the ResolveFilePathTool."""
        from molx_agent.tools.extractor import ResolveFilePathTool
        import json
        
        tool = ResolveFilePathTool()
        
        # Test with non-existent file
        result = json.loads(tool._run("nonexistent_file.csv"))
        assert result["success"] is False
        assert "error" in result
    
    def test_parse_inline_csv_tool(self):
        """Test the ParseInlineCSVTool."""
        from molx_agent.tools.extractor import ParseInlineCSVTool
        
        tool = ParseInlineCSVTool()
        
        # Test with simple CSV data
        csv_data = """smiles,compound_id,IC50
CCO,cpd1,10.5
CCC,cpd2,20.0
CCCC,cpd3,15.0"""
        
        result = tool._run(csv_data)
        
        assert result["success"] is True
        assert len(result["compounds"]) == 3
        assert result["compounds"][0]["smiles"] == "CCO"
    
    def test_parse_inline_csv_with_code_block(self):
        """Test parsing CSV from code blocks."""
        from molx_agent.tools.extractor import ParseInlineCSVTool
        
        tool = ParseInlineCSVTool()
        
        csv_data = """```csv
smiles,name,activity
CCO,ethanol,5.0
CCC,propane,10.0
```"""
        
        result = tool._run(csv_data)
        
        assert result["success"] is True
        assert len(result["compounds"]) == 2


class TestExtractorTools:
    """Tests for extractor tools."""
    
    def test_csv_extraction(self, tmp_path):
        """Test CSV extraction tool."""
        from molx_agent.tools.extractor import ExtractFromCSVTool
        
        # Create test CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""smiles,compound_id,IC50
CCO,cpd_001,10.5
CCC,cpd_002,20.0
CCCC,cpd_003,15.0""")
        
        tool = ExtractFromCSVTool()
        result = tool._run(str(csv_file))
        
        assert result["total_molecules"] == 3
        assert len(result["compounds"]) == 3
        assert result["compounds"][0]["smiles"] == "CCO"
    
    def test_file_path_resolution(self, tmp_path):
        """Test file path resolution with various path formats."""
        from molx_agent.tools.extractor import _resolve_file_path
        
        # Create test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("test")
        
        # Test with absolute path
        resolved = _resolve_file_path(str(test_file))
        assert resolved == str(test_file)
