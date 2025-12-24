"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-24].
*  @Description Unified Tool Registry for LangGraph agents.
*               Provides centralized tool management with categories,
*               agent filtering, and lazy loading.
**************************************************************************
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Categories
# =============================================================================

class ToolCategory(Enum):
    """Tool categories for organization and filtering."""
    
    RDKIT = "rdkit"           # Basic RDKit utilities
    SAR = "sar"               # SAR analysis tools
    REPORT = "report"         # Report generation tools
    EXTRACTOR = "extractor"   # Data extraction tools
    STANDARDIZE = "standardize"  # Molecule standardization
    CONVERTER = "converter"   # Format converters
    MCP = "mcp"               # MCP external tools


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    
    name: str
    category: ToolCategory
    description: str = ""
    requires_llm: bool = False
    allowed_agents: list[str] = field(default_factory=list)  # Empty = all agents


# =============================================================================
# Tool Registry (Singleton)
# =============================================================================

class ToolRegistry:
    """Singleton registry for centralized tool management.
    
    Features:
    - Lazy loading of tools
    - Category-based organization
    - Agent-based access control
    - Tool caching to avoid re-instantiation
    
    Usage:
        registry = get_registry()
        tools = registry.get_tools(category=ToolCategory.SAR)
        tool = registry.get_tool_by_name("extract_from_csv")
    """
    
    _instance: Optional["ToolRegistry"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if ToolRegistry._initialized:
            return
        
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}
        self._loaders: list[Callable[["ToolRegistry"], None]] = []
        self._loaded: bool = False
        
        # Register default loaders
        self._register_default_loaders()
        
        ToolRegistry._initialized = True
        logger.debug("ToolRegistry initialized")
    
    def _register_default_loaders(self) -> None:
        """Register default tool loaders for lazy initialization."""
        self._loaders = [
            _load_rdkit_tools,
            _load_converter_tools,
            _load_sar_tools,
            _load_report_tools,
            _load_extractor_tools,
            _load_standardize_tools,
            _load_mcp_tools,
        ]
    
    def _ensure_loaded(self) -> None:
        """Ensure all tools are loaded (lazy loading)."""
        if self._loaded:
            return
        
        logger.info("Loading tool registry...")
        for loader in self._loaders:
            try:
                loader(self)
            except Exception as e:
                logger.warning(f"Failed to load tools with {loader.__name__}: {e}")
        
        self._loaded = True
        logger.info(f"Tool registry loaded: {len(self._tools)} tools")
    
    def register(
        self,
        tool: BaseTool,
        category: ToolCategory,
        requires_llm: bool = False,
        allowed_agents: list[str] | None = None,
    ) -> None:
        """Register a tool with the registry.
        
        Args:
            tool: The LangChain tool instance.
            category: Tool category for organization.
            requires_llm: Whether the tool needs LLM injection.
            allowed_agents: List of agent names that can use this tool.
                            Empty list means all agents can use it.
        """
        name = tool.name
        if name in self._tools:
            logger.debug(f"Tool '{name}' already registered, skipping")
            return
        
        self._tools[name] = tool
        self._metadata[name] = ToolMetadata(
            name=name,
            category=category,
            description=getattr(tool, "description", ""),
            requires_llm=requires_llm,
            allowed_agents=allowed_agents or [],
        )
        logger.debug(f"Registered tool: {name} ({category.value})")
    
    def get_tools(
        self,
        category: ToolCategory | None = None,
        agent: str | None = None,
    ) -> list[BaseTool]:
        """Get tools with optional filtering.
        
        Args:
            category: Filter by tool category.
            agent: Filter by agent name (only tools allowed for this agent).
        
        Returns:
            List of matching tools.
        """
        self._ensure_loaded()
        
        result = []
        for name, tool in self._tools.items():
            meta = self._metadata.get(name)
            if not meta:
                continue
            
            # Filter by category
            if category and meta.category != category:
                continue
            
            # Filter by agent permission
            if agent and meta.allowed_agents:
                if agent not in meta.allowed_agents:
                    continue
            
            result.append(tool)
        
        return result
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name.
        
        Args:
            name: The tool name.
        
        Returns:
            The tool instance or None if not found.
        """
        self._ensure_loaded()
        return self._tools.get(name)
    
    def list_tools(self) -> list[ToolMetadata]:
        """List all registered tool metadata."""
        self._ensure_loaded()
        return list(self._metadata.values())
    
    def get_tools_for_agent(self, agent_name: str) -> list[BaseTool]:
        """Get all tools available for a specific agent.
        
        This is a convenience method that filters tools by agent permission.
        """
        return self.get_tools(agent=agent_name)
    
    def inject_llm(self, llm: Any) -> None:
        """Inject LLM into tools that require it.
        
        Args:
            llm: The LLM instance to inject.
        """
        self._ensure_loaded()
        for name, tool in self._tools.items():
            meta = self._metadata.get(name)
            if meta and meta.requires_llm:
                if hasattr(tool, "llm"):
                    tool.llm = llm
                    logger.debug(f"Injected LLM into tool: {name}")


# =============================================================================
# Tool Loaders
# =============================================================================

def _load_rdkit_tools(registry: ToolRegistry) -> None:
    """Load RDKit basic tools."""
    try:
        from molx_agent.tools.rdkit import FuncGroups, MolSimilarity, SMILES2Weight

        registry.register(MolSimilarity(), ToolCategory.RDKIT)
        registry.register(SMILES2Weight(), ToolCategory.RDKIT)
        registry.register(FuncGroups(), ToolCategory.RDKIT)
        logger.info("Loaded RDKit tools (3)")
    except ImportError as e:
        logger.warning(f"Could not import RDKit tools: {e}")


def _load_converter_tools(registry: ToolRegistry) -> None:
    """Load converter tools."""
    try:
        from molx_agent.tools.standardize import StandardizeMolecule, BatchStandardize
        from molx_agent.tools.converters import Query2CAS, Query2SMILES, SMILES2Name

        registry.register(Query2SMILES(), ToolCategory.CONVERTER)
        registry.register(Query2CAS(), ToolCategory.CONVERTER)
        registry.register(SMILES2Name(), ToolCategory.CONVERTER)
        registry.register(StandardizeMolecule(), ToolCategory.STANDARDIZE)
        registry.register(BatchStandardize(), ToolCategory.STANDARDIZE)
        logger.info("Loaded converter tools (5)")
    except ImportError as e:
        logger.warning(f"Could not import converter tools: {e}")


def _load_sar_tools(registry: ToolRegistry) -> None:
    """Load SAR core tools."""
    try:
        from molx_agent.tools.sar import get_sar_tools

        sar_tools = get_sar_tools()
        for tool in sar_tools:
            registry.register(tool, ToolCategory.SAR)
        logger.info(f"Loaded SAR tools ({len(sar_tools)})")
    except ImportError as e:
        logger.warning(f"Could not import SAR tools: {e}")


def _load_report_tools(registry: ToolRegistry) -> None:
    """Load report generation tools."""
    try:
        from molx_agent.tools.report import get_report_tools

        report_tools = get_report_tools()
        for tool in report_tools:
            registry.register(
                tool, 
                ToolCategory.REPORT,
                allowed_agents=["reporter", "tool_agent"],
            )
        logger.info(f"Loaded report tools ({len(report_tools)})")
    except ImportError as e:
        logger.warning(f"Could not import report tools: {e}")


def _load_extractor_tools(registry: ToolRegistry) -> None:
    """Load data extractor tools."""
    try:
        from molx_agent.tools.extractor import (
            ExtractFromCSVTool,
            ExtractFromExcelTool,
            ExtractFromSDFTool,
        )

        registry.register(
            ExtractFromCSVTool(), 
            ToolCategory.EXTRACTOR,
            requires_llm=True,
            allowed_agents=["data_cleaner"],
        )
        registry.register(
            ExtractFromExcelTool(), 
            ToolCategory.EXTRACTOR,
            requires_llm=True,
            allowed_agents=["data_cleaner"],
        )
        registry.register(
            ExtractFromSDFTool(), 
            ToolCategory.EXTRACTOR,
            allowed_agents=["data_cleaner"],
        )
        logger.info("Loaded extractor tools (3)")
    except ImportError as e:
        logger.warning(f"Could not import extractor tools: {e}")


def _load_standardize_tools(registry: ToolRegistry) -> None:
    """Load standardization tools."""
    try:
        from molx_agent.tools.standardize import (
            CleanCompoundDataTool,
            SaveCleanedDataTool,
        )

        registry.register(
            CleanCompoundDataTool(), 
            ToolCategory.STANDARDIZE,
            allowed_agents=["data_cleaner"],
        )
        registry.register(
            SaveCleanedDataTool(), 
            ToolCategory.STANDARDIZE,
            allowed_agents=["data_cleaner"],
        )
        logger.info("Loaded standardize tools (2)")
    except ImportError as e:
        logger.warning(f"Could not import standardize tools: {e}")


def _load_mcp_tools(registry: ToolRegistry) -> None:
    """Load MCP tools if enabled."""
    try:
        from molx_agent.config import get_settings
        settings = get_settings()
        
        if settings.MCP_ENABLED:
            from molx_agent.agents.modules.mcp import get_mcp_tools
            mcp_tools = get_mcp_tools()
            if mcp_tools:
                for tool in mcp_tools:
                    registry.register(tool, ToolCategory.MCP)
                logger.info(f"Loaded MCP tools ({len(mcp_tools)})")
            else:
                logger.debug("No MCP tools loaded (no servers configured)")
    except ImportError as e:
        logger.debug(f"MCP tools not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to load MCP tools: {e}")


# =============================================================================
# Public API
# =============================================================================

def get_registry() -> ToolRegistry:
    """Get the singleton ToolRegistry instance.
    
    Returns:
        The global ToolRegistry instance.
    """
    return ToolRegistry()


def get_all_tools(
    category: ToolCategory | str | None = None,
    agent: str | None = None,
) -> list[BaseTool]:
    """Get all available tools with optional filtering.
    
    This is the main entry point for agents to get tools.
    Backward compatible with the old API when called without arguments.
    
    Args:
        category: Optional category filter (ToolCategory or string).
        agent: Optional agent name filter.
    
    Returns:
        List of LangChain tools.
    """
    registry = get_registry()
    
    # Convert string category to enum if needed
    cat_enum = None
    if category:
        if isinstance(category, str):
            try:
                cat_enum = ToolCategory(category)
            except ValueError:
                logger.warning(f"Unknown tool category: {category}")
        else:
            cat_enum = category
    
    return registry.get_tools(category=cat_enum, agent=agent)


def get_tool_names() -> list[str]:
    """Get names of all available tools.

    Returns:
        List of tool names.
    """
    return [tool.name for tool in get_all_tools()]
