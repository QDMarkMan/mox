"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description Tools registration for LangGraph agents.
**************************************************************************
"""

import logging

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_all_tools() -> list[BaseTool]:
    """Get all available tools for the agent.

    Returns:
        List of LangChain tools.
    """
    tools: list[BaseTool] = []

    # Load RDKit basic tools
    try:
        from molx_agent.tools.rdkit import FuncGroups, MolSimilarity, SMILES2Weight

        tools.extend([
            MolSimilarity(),
            SMILES2Weight(),
            FuncGroups(),
        ])
        logger.info("Loaded RDKit tools (3)")
    except ImportError as e:
        logger.warning(f"Could not import RDKit tools: {e}")

    # Load converter tools
    try:
        from molx_agent.tools.standardize import StandardizeMolecule, BatchStandardize
        from molx_agent.tools.converters import Query2CAS, Query2SMILES, SMILES2Name

        tools.extend([
            Query2SMILES(),
            Query2CAS(),
            SMILES2Name(),
            StandardizeMolecule(),
            BatchStandardize(),
        ])
        logger.info("Loaded converter tools (3)")
    except ImportError as e:
        logger.warning(f"Could not import converter tools: {e}")

    # Load SAR core tools
    try:
        from molx_agent.tools.sar import get_sar_tools

        sar_tools = get_sar_tools()
        tools.extend(sar_tools)
        logger.info(f"Loaded SAR tools ({len(sar_tools)})")
    except ImportError as e:
        logger.warning(f"Could not import SAR tools: {e}")

    # Load SAR report tools
    try:
        from molx_agent.tools.report import get_report_tools

        report_tools = get_report_tools()
        tools.extend(report_tools)
        logger.info(f"Loaded report tools ({len(report_tools)})")
    except ImportError as e:
        logger.warning(f"Could not import report tools: {e}")

    logger.info(f"Total tools loaded: {len(tools)}")
    return tools


def get_tool_names() -> list[str]:
    """Get names of all available tools.

    Returns:
        List of tool names.
    """
    return [tool.name for tool in get_all_tools()]

