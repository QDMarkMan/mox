"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-15].
*  @Description Intent classifier for user query classification.
**************************************************************************
"""

import logging
from enum import Enum
from typing import Optional

from molx_agent.agents.modules.llm import invoke_llm

logger = logging.getLogger(__name__)


class Intent(Enum):
    """User intent categories."""

    SAR_ANALYSIS = "sar_analysis"  # SAR/drug design related queries
    DATA_PROCESSING = "data_processing"  # Data file processing requests
    MOLECULE_QUERY = "molecule_query"  # Simple molecule property queries
    GENERAL_CHAT = "general_chat"  # General conversation
    UNSUPPORTED = "unsupported"  # Unsupported request types


INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a SAR (Structure-Activity Relationship) analysis system.

Classify the user's query into ONE of these categories:

1. "sar_analysis" - SAR analysis, drug design, structure-activity relationships
   Examples: "Analyze SAR of aspirin", "Compare activity of these compounds"

2. "data_processing" - Processing molecular data files (CSV, Excel, SDF)
   Examples: "Extract SMILES from this CSV", "Read my data file"

3. "molecule_query" - Simple queries about molecules (MW, SMILES, properties)
   Examples: "What is the molecular weight of aspirin?", "Convert name to SMILES"

4. "general_chat" - General greetings, chitchat, off-topic conversation
   Examples: "Hello", "How are you?", "What's the weather?"

5. "unsupported" - Requests clearly outside the system's capabilities
   Examples: "Write me a poem", "Help me with my homework"

Return ONLY a JSON object:
{"intent": "<category>", "confidence": <0.0-1.0>}
"""

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


def classify_intent(query: str) -> tuple[Intent, float]:
    """Classify user query intent.

    Args:
        query: User query string.

    Returns:
        Tuple of (Intent, confidence_score).
    """
    try:
        result = invoke_llm(
            INTENT_CLASSIFIER_PROMPT,
            f"Query: {query}",
            parse_json=True,
        )

        intent_str = result.get("intent", "sar_analysis")
        confidence = result.get("confidence", 0.5)

        # Map to Intent enum
        intent_map = {
            "sar_analysis": Intent.SAR_ANALYSIS,
            "data_processing": Intent.DATA_PROCESSING,
            "molecule_query": Intent.MOLECULE_QUERY,
            "general_chat": Intent.GENERAL_CHAT,
            "unsupported": Intent.UNSUPPORTED,
        }

        intent = intent_map.get(intent_str, Intent.SAR_ANALYSIS)
        logger.info(f"Classified intent: {intent.value} ({confidence:.2f})")

        return intent, confidence

    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        # Default to SAR analysis on error
        return Intent.SAR_ANALYSIS, 0.5


def get_intent_response(intent: Intent) -> Optional[str]:
    """Get friendly response for non-supported intents.

    Args:
        intent: Classified intent.

    Returns:
        Response string if intent is not supported, None otherwise.
    """
    return INTENT_RESPONSES.get(intent)


def is_supported_intent(intent: Intent) -> bool:
    """Check if intent is supported for processing.

    Args:
        intent: Classified intent.

    Returns:
        True if intent should be processed, False otherwise.
    """
    return intent in (
        Intent.SAR_ANALYSIS,
        Intent.DATA_PROCESSING,
        Intent.MOLECULE_QUERY,
    )
