"""Tests for tool output directory helpers."""

from pathlib import Path

import pytest

from molx_agent.tools.standardize import SaveCleanedDataTool
from molx_agent.utils import paths as path_utils
import molx_agent.config as agent_config


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Ensure settings caches are cleared between tests."""
    agent_config.get_settings.cache_clear()
    path_utils._artifacts_root.cache_clear()
    yield
    agent_config.get_settings.cache_clear()
    path_utils._artifacts_root.cache_clear()


def test_standardize_dir_respects_artifacts_root(tmp_path, monkeypatch):
    """Standardize output helper should honor ARTIFACTS_ROOT env."""
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))

    target = path_utils.get_standardize_output_dir()

    assert target.is_dir()
    assert target == Path(tmp_path) / agent_config.get_settings().standardize_subdir


def test_save_cleaned_data_uses_configured_output(tmp_path, monkeypatch):
    """SaveCleanedDataTool should emit files inside configured artifacts dir."""
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))

    tool = SaveCleanedDataTool()
    payload = {
        "compounds": [
            {"compound_id": "cmpd-1", "smiles": "CCO", "activity": 1.5},
        ]
    }

    outputs = tool._run(payload, task_id="unit_test")

    assert "json" in outputs
    json_path = Path(outputs["json"])
    assert json_path.exists()
    assert str(tmp_path) in outputs["json"]

    if "csv" in outputs:
        csv_path = Path(outputs["csv"])
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
