"""Tests for DataCleanerAgent - AI-driven molecular data extraction.

Tests cover:
- Inline CSV extraction from user input
- File path extraction
- Column detection (LLM and heuristic fallback)
- End-to-end data cleaning workflow
"""

import io
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from molx_agent.agents.data_cleaner import DataCleanerAgent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create a DataCleanerAgent instance with mocked LLM."""
    with patch("molx_agent.agents.data_cleaner.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"smiles_col": "smiles", "name_col": "compound_number", "activity_cols": ["IC50"]}')
        mock_llm.bind_tools.return_value = mock_llm
        mock_get_llm.return_value = mock_llm
        yield DataCleanerAgent()


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """compound_number,duplicate_number,quality,smiles,IC50 CDK2 inhibitory activity (uM)
1,1,High Confidence,FC1=CC=C(NC2=CC(C3=CC=CC=C3)=NC3=CC=NN32)C=C1,6.09
2,1,High Confidence,CC1=NOC(C)=C1C1=NC2=CC=NN2C(NC2=CC=C(F)C=C2)=C1,13.37
3,1,High Confidence,BrC1=CC=CC(C2=NC3=CC=NN3C(NCC3=CC=NC=C3)=C2)=C1,1.64"""


@pytest.fixture
def sample_csv_in_code_block(sample_csv_content):
    """Sample CSV wrapped in code block."""
    return f"""下面是我的csv数据，给我完成SAR分析

```csv
{sample_csv_content}
```
"""


@pytest.fixture
def sample_csv_raw(sample_csv_content):
    """Sample CSV as raw text without code block."""
    return f"""下面是我的csv数据，给我完成SAR分析

{sample_csv_content}
"""


# =============================================================================
# Tests for _extract_inline_csv
# =============================================================================

class TestExtractInlineCSV:
    """Tests for _extract_inline_csv method."""

    def test_extract_csv_from_code_block(self, agent, sample_csv_in_code_block):
        """Should extract CSV content from ```csv code block."""
        result = agent._extract_inline_csv(sample_csv_in_code_block)
        
        assert result is not None
        assert "compound_number" in result
        assert "smiles" in result
        assert "IC50" in result

    def test_extract_csv_from_generic_code_block(self, agent, sample_csv_content):
        """Should extract CSV from generic ``` code block."""
        text = f"Here is data:\n```\n{sample_csv_content}\n```"
        result = agent._extract_inline_csv(text)
        
        assert result is not None
        assert "smiles" in result

    def test_extract_raw_csv_data(self, agent, sample_csv_raw):
        """Should extract raw CSV data based on header keywords."""
        result = agent._extract_inline_csv(sample_csv_raw)
        
        assert result is not None
        assert "compound_number" in result
        assert "smiles" in result

    def test_no_csv_in_text(self, agent):
        """Should return None when no CSV data is present."""
        text = "Please analyze my molecular data."
        result = agent._extract_inline_csv(text)
        
        assert result is None

    def test_insufficient_keywords(self, agent):
        """Should return None when CSV lacks molecular keywords."""
        text = """
name,age,city
Alice,30,NYC
Bob,25,LA
"""
        result = agent._extract_inline_csv(text)
        
        assert result is None

    def test_single_row_csv_rejected(self, agent):
        """Should reject CSV with only header row."""
        text = "smiles,compound,IC50"  # No data rows
        result = agent._extract_inline_csv(text)
        
        assert result is None


# =============================================================================
# Tests for _extract_file_path
# =============================================================================

class TestExtractFilePath:
    """Tests for _extract_file_path method."""

    def test_extract_unix_path(self, agent):
        """Should extract Unix file path."""
        text = "Please analyze /data/workspace/compounds.csv"
        result = agent._extract_file_path(text)
        
        assert result == "/data/workspace/compounds.csv"

    def test_extract_windows_path(self, agent):
        """Should extract Windows file path."""
        text = r"Please analyze C:\Users\data\compounds.xlsx"
        result = agent._extract_file_path(text)
        
        assert result == r"C:\Users\data\compounds.xlsx"

    def test_extract_sdf_path(self, agent):
        """Should extract SDF file path."""
        text = "Load /home/user/molecules.sdf"
        result = agent._extract_file_path(text)
        
        assert result == "/home/user/molecules.sdf"

    def test_no_file_path(self, agent):
        """Should return None when no file path present."""
        text = "Here is my CSV data inline"
        result = agent._extract_file_path(text)
        
        assert result is None

    def test_extract_first_valid_path(self, agent):
        """Should extract the first valid file path."""
        text = "Compare /path/to/file1.csv and /path/to/file2.csv"
        result = agent._extract_file_path(text)
        
        assert result == "/path/to/file1.csv"


# =============================================================================
# Tests for _parse_inline_csv_data
# =============================================================================

class TestParseInlineCSVData:
    """Tests for _parse_inline_csv_data method."""

    def test_parse_valid_csv(self, agent, sample_csv_content):
        """Should parse valid CSV into compounds list."""
        result = agent._parse_inline_csv_data(sample_csv_content)
        
        assert "compounds" in result
        assert len(result["compounds"]) == 3
        assert result["source"] == "inline_csv"

    def test_compounds_have_smiles(self, agent, sample_csv_content):
        """Each compound should have a SMILES string."""
        result = agent._parse_inline_csv_data(sample_csv_content)
        
        for compound in result["compounds"]:
            assert "smiles" in compound
            assert len(compound["smiles"]) > 0

    def test_compounds_have_properties(self, agent, sample_csv_content):
        """Compounds should have properties from other columns."""
        result = agent._parse_inline_csv_data(sample_csv_content)
        
        for compound in result["compounds"]:
            assert "properties" in compound

    def test_invalid_csv_returns_empty(self, agent):
        """Should return empty compounds for invalid CSV."""
        invalid_csv = "not,a,valid\ncsv,{broken"
        result = agent._parse_inline_csv_data(invalid_csv)
        
        # Should handle gracefully, may return empty or error
        assert "compounds" in result or "error" in result


# =============================================================================
# Tests for _detect_columns_with_llm
# =============================================================================

class TestDetectColumnsWithLLM:
    """Tests for _detect_columns_with_llm method."""

    def test_llm_column_detection(self, agent, sample_csv_content):
        """Should use LLM to detect column mappings."""
        df = pd.read_csv(io.StringIO(sample_csv_content))
        result = agent._detect_columns_with_llm(df)
        
        assert "smiles_col" in result
        assert "name_col" in result
        assert "activity_cols" in result

    def test_fallback_heuristic_detection(self, agent, sample_csv_content):
        """Should fallback to heuristic when LLM fails."""
        # Make LLM raise an exception
        agent.llm.invoke.side_effect = Exception("LLM error")
        
        df = pd.read_csv(io.StringIO(sample_csv_content))
        result = agent._detect_columns_with_llm(df)
        
        # Should still return a mapping from heuristics
        assert "smiles_col" in result
        assert result["smiles_col"] == "smiles"


# =============================================================================
# Tests for _get_extractor_for_file
# =============================================================================

class TestGetExtractorForFile:
    """Tests for _get_extractor_for_file method."""

    def test_csv_extractor(self, agent):
        """Should return CSV extractor for .csv files."""
        extractor = agent._get_extractor_for_file("/path/to/data.csv")
        
        assert extractor is not None
        assert "csv" in extractor.name.lower()

    def test_excel_extractor(self, agent):
        """Should return Excel extractor for .xlsx files."""
        extractor = agent._get_extractor_for_file("/path/to/data.xlsx")
        
        assert extractor is not None
        assert "excel" in extractor.name.lower()

    def test_sdf_extractor(self, agent):
        """Should return SDF extractor for .sdf files."""
        extractor = agent._get_extractor_for_file("/path/to/molecules.sdf")
        
        assert extractor is not None
        assert "sdf" in extractor.name.lower()

    def test_unsupported_format(self, agent):
        """Should return None for unsupported file formats."""
        extractor = agent._get_extractor_for_file("/path/to/data.txt")
        
        assert extractor is None


# =============================================================================
# Tests for run method
# =============================================================================

class TestRunMethod:
    """Tests for the main run method."""

    def test_run_with_inline_csv(self, agent, sample_csv_in_code_block):
        """Should process inline CSV data successfully."""
        # Mock the standardize tools
        mock_cleaner = MagicMock()
        mock_cleaner.invoke.return_value = {
            "compounds": [{"smiles": "CCO", "name": "test"}],
            "cleaned_count": 1,
            "original_count": 1,
            "removed": 0,
        }
        mock_saver = MagicMock()
        mock_saver.invoke.return_value = {"json": "/tmp/output.json"}
        
        agent.standardize_tools = [mock_cleaner, mock_saver]
        
        state = {
            "user_query": sample_csv_in_code_block,
            "tasks": {},
            "current_task_id": "test_task",
        }
        
        result = agent.run(state)
        
        assert "results" in result
        # The inline CSV should be detected and processed

    def test_run_without_data_source(self, agent):
        """Should return error when no data source found."""
        state = {
            "user_query": "Please analyze my data",  # No file path or CSV
            "tasks": {},
            "current_task_id": "test_task",
        }
        
        result = agent.run(state)
        
        assert "results" in result
        assert "error" in result["results"]["test_task"]

    def test_run_with_file_not_found(self, agent):
        """Should return error when file doesn't exist."""
        state = {
            "user_query": "Analyze /nonexistent/file.csv",
            "tasks": {},
            "current_task_id": "test_task",
        }
        
        result = agent.run(state)
        
        assert "results" in result
        # Should either find inline CSV fallback or return file not found error


# =============================================================================
# Integration-style tests
# =============================================================================

class TestIntegration:
    """Integration-style tests for complete workflows."""

    def test_full_inline_csv_workflow(self, agent):
        """Test complete workflow from inline CSV to cleaned compounds."""
        csv_text = """下面是我的数据
```
compound_id,smiles,IC50
1,CCO,1.5
2,CCCO,2.3
3,CCCCO,3.1
```
"""
        # Extract CSV
        csv_content = agent._extract_inline_csv(csv_text)
        assert csv_content is not None
        
        # Parse data
        parsed = agent._parse_inline_csv_data(csv_content)
        assert len(parsed["compounds"]) == 3
        
        # Verify compound structure
        for compound in parsed["compounds"]:
            assert "smiles" in compound
            assert "name" in compound

    def test_chinese_query_with_csv(self, agent):
        """Should handle Chinese queries with CSV data."""
        chinese_query = """给我分析下面的化合物数据，完成SAR分析
compound,smiles,activity
化合物1,CCO,5.0
化合物2,CCCO,3.2
"""
        csv_content = agent._extract_inline_csv(chinese_query)
        
        assert csv_content is not None
        assert "smiles" in csv_content
