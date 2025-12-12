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

    # Load RDKit tools
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
        from molx_agent.tools.converters import Query2CAS, Query2SMILES, SMILES2Name

        tools.extend([
            Query2SMILES(),
            Query2CAS(),
            SMILES2Name(),
        ])
        logger.info("Loaded converter tools (3)")
    except ImportError as e:
        logger.warning(f"Could not import converter tools: {e}")

    logger.info(f"Total tools loaded: {len(tools)}")
    return tools


def get_tool_names() -> list[str]:
    """Get names of all available tools.

    Returns:
        List of tool names.
    """
    return [tool.name for tool in get_all_tools()]
