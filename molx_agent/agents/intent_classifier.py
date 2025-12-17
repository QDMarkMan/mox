"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description IntentClassifierAgent - AI-based intent classification.
**************************************************************************
"""

import logging
from enum import Enum
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import invoke_llm

logger = logging.getLogger(__name__)


class Intent(Enum):
    """User intent categories."""

    SAR_ANALYSIS = "sar_analysis"  # SAR/drug design related queries
    DATA_PROCESSING = "data_processing"  # Data file processing requests
    MOLECULE_QUERY = "molecule_query"  # Simple molecule property queries
    GENERAL_CHAT = "general_chat"  # General conversation
    UNSUPPORTED = "unsupported"  # Unsupported request types


# Friendly responses for non-SAR intents
INTENT_RESPONSES = {
    Intent.GENERAL_CHAT: (
        "您好！👋 我是 SAR 分析助手，专门用于药物设计和分子分析。\n\n"
        "我可以帮您：\n"
        "• 分析分子的 SAR（结构-活性关系）\n"
        "• 查询分子属性（分子量、SMILES 等）\n"
        "• 处理分子数据文件（CSV、Excel）\n"
        "• 检查化学物质安全性\n\n"
        "请告诉我您想分析什么分子？"
    ),
    Intent.UNSUPPORTED: (
        "抱歉，这个请求超出了我的能力范围。😅\n\n"
        "我是一个专注于 **SAR 分析** 的助手，主要功能包括：\n"
        "• 分子结构-活性关系分析\n"
        "• 分子属性查询（分子量、官能团等）\n"
        "• 分子数据处理\n\n"
        "如果您有药物设计或分子分析相关的问题，请告诉我！"
    ),
}


INTENT_CLASSIFIER_PROMPT = """You are an AI intent classifier for a SAR (Structure-Activity Relationship) analysis system.

Your task is to analyze the user's query and classify their intent.

## Intent Categories:

1. **sar_analysis** - SAR analysis, drug design, structure-activity relationships
   - "Analyze SAR of aspirin"
   - "Compare activity of these compounds"
   - "Find R-group patterns"

2. **data_processing** - Processing molecular data files (CSV, Excel, SDF)
   - "Extract SMILES from this CSV"
   - "Read my data file"
   - "Process the Excel file"

3. **molecule_query** - Simple queries about molecules (MW, SMILES, properties)
   - "What is the molecular weight of aspirin?"
   - "Convert name to SMILES"

4. **general_chat** - General greetings, chitchat, off-topic conversation
   - "Hello", "How are you?", "What's the weather?"

5. **unsupported** - Requests clearly outside the system's capabilities
   - "Write me a poem", "Help me with my homework"

## Response Format:
Return ONLY a JSON object:
{
    "reasoning": "Brief explanation of your classification",
    "intent": "<category>",
    "confidence": <0.0-1.0>
}
"""


class IntentClassifierAgent(BaseAgent):
    """AI-based intent classifier agent.
    
    Uses LLM to classify user queries into predefined intent categories.
    """

    def __init__(self) -> None:
        super().__init__(
            name="intent_classifier",
            description="Classifies user queries using AI",
        )

    def run(self, state: AgentState) -> AgentState:
        """Classify user intent.
        
        Args:
            state: Agent state with user_query.
            
        Returns:
            Updated state with classified intent.
        """
        from rich.console import Console
        console = Console()
        
        user_query = state.get("user_query", "")
        console.print("\n[bold blue]🎯 IntentClassifier: Analyzing query...[/]")

        try:
            result = invoke_llm(
                INTENT_CLASSIFIER_PROMPT,
                f"User Query: {user_query}",
                parse_json=True,
            )

            intent_str = result.get("intent", "sar_analysis")
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")

            # Map to Intent enum
            intent_map = {
                "sar_analysis": Intent.SAR_ANALYSIS,
                "data_processing": Intent.DATA_PROCESSING,
                "molecule_query": Intent.MOLECULE_QUERY,
                "general_chat": Intent.GENERAL_CHAT,
                "unsupported": Intent.UNSUPPORTED,
            }

            intent = intent_map.get(intent_str, Intent.SAR_ANALYSIS)
            
            console.print(f"   [dim]Reasoning: {reasoning[:100]}...[/]" if len(reasoning) > 100 else f"   [dim]Reasoning: {reasoning}[/]")
            console.print(f"   [green]Intent: {intent.value} (confidence: {confidence:.2f})[/]")
            
            # Store in state
            state["intent"] = intent
            state["intent_confidence"] = confidence
            state["intent_reasoning"] = reasoning
            
            logger.info(f"Classified intent: {intent.value} ({confidence:.2f})")

        except Exception as e:
            console.print(f"[red]✗ IntentClassifier error: {e}[/]")
            logger.error(f"Intent classification error: {e}")
            # Default to SAR analysis on error
            state["intent"] = Intent.SAR_ANALYSIS
            state["intent_confidence"] = 0.5

        return state

    def is_supported(self, intent: Intent) -> bool:
        """Check if intent is supported for processing."""
        return intent in (
            Intent.SAR_ANALYSIS,
            Intent.DATA_PROCESSING,
            Intent.MOLECULE_QUERY,
        )

    def get_response(self, intent: Intent) -> Optional[str]:
        """Get friendly response for non-supported intents."""
        return INTENT_RESPONSES.get(intent)
