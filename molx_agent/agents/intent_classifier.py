"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-17].
*  @Description IntentClassifierAgent - AI-based intent classification.
**************************************************************************
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from molx_agent.agents.base import BaseAgent
from molx_agent.agents.modules.state import AgentState
from molx_agent.agents.modules.llm import invoke_llm
from molx_agent.config import get_settings

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
        "您好！👋 我是药物设计助手，专门用于药物设计和分子分析。\n\n"
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


@dataclass
class IntentClassificationResult:
    intent: Intent
    confidence: float
    reasoning: str
    reasoning_steps: Optional[list[str]] = None


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
    "reasoning_steps": [
        "Short bullet explaining how you interpreted the query",
        "Another short bullet leading to the final decision"
    ],
    "reasoning": "One-sentence justification referencing the steps above",
    "intent": "<category>",
    "confidence": <0.0-1.0>
}
"""


class IntentClassifierAgent(BaseAgent):
    """AI-based intent classifier agent.

    Uses LLM to classify user queries into predefined intent categories.
    Explicitly requires an available LLM and surfaces the reasoning it received.
    """

    def __init__(self, *, enable_llm: Optional[bool] = None) -> None:
        super().__init__(
            name="intent_classifier",
            description="Classifies user queries using AI",
        )
        self._enable_llm = enable_llm

    def run(self, state: AgentState) -> AgentState:
        """Classify user intent."""
        from rich.console import Console

        console = Console()
        user_query = state.get("user_query", "")
        console.print("\n[bold blue]🎯 IntentClassifier: Analyzing query...[/]")

        if not self._can_use_llm():
            message = (
                "LLM intent classification is required. Configure LOCAL_OPENAI_API_KEY "
                "or pass a custom settings object with a valid key."
            )
            console.print(f"[red]❌ {message}[/]")
            raise RuntimeError(message)

        result = self._classify_with_llm(user_query)
        return self._apply_result(state, console, result)

    def _can_use_llm(self) -> bool:
        if self._enable_llm is not None:
            return self._enable_llm
        settings = get_settings()
        api_key = getattr(settings, "LOCAL_OPENAI_API_KEY", "") or ""
        return bool(api_key.strip())

    def _classify_with_llm(self, user_query: str) -> IntentClassificationResult:
        try:
            payload = invoke_llm(
                INTENT_CLASSIFIER_PROMPT,
                f"User Query: {user_query}",
                parse_json=True,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Intent classification via LLM failed", exc_info=exc)
            raise RuntimeError("Intent classification failed") from exc

        intent_map = {
            "sar_analysis": Intent.SAR_ANALYSIS,
            "data_processing": Intent.DATA_PROCESSING,
            "molecule_query": Intent.MOLECULE_QUERY,
            "general_chat": Intent.GENERAL_CHAT,
            "unsupported": Intent.UNSUPPORTED,
        }

        intent = intent_map.get(payload.get("intent", "sar_analysis"), Intent.SAR_ANALYSIS)
        confidence = float(payload.get("confidence", 0.5) or 0.5)
        reasoning = payload.get("reasoning", "")
        raw_steps = payload.get("reasoning_steps") or payload.get("steps")
        reasoning_steps: Optional[list[str]] = None
        if isinstance(raw_steps, list):
            cleaned = [str(step).strip() for step in raw_steps if str(step).strip()]
            reasoning_steps = cleaned or None
        elif isinstance(raw_steps, str) and raw_steps.strip():
            reasoning_steps = [raw_steps.strip()]

        logger.info("Classified intent via LLM: %s (%.2f)", intent.value, confidence)
        return IntentClassificationResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
        )

    def _apply_result(
        self,
        state: AgentState,
        console,
        result: IntentClassificationResult,
    ) -> AgentState:
        if result.reasoning_steps:
            console.print("   [dim]Reasoning steps:[/]")
            for idx, step in enumerate(result.reasoning_steps, 1):
                trimmed = step[:160]
                suffix = "..." if len(step) > 160 else ""
                console.print(f"      {idx}. {trimmed}{suffix}")
        if result.reasoning:
            trimmed_reason = result.reasoning[:160]
            suffix = "..." if len(result.reasoning) > 160 else ""
            console.print(f"   [dim]Summary: {trimmed_reason}{suffix}[/]")
        console.print(
            f"   [green]Intent: {result.intent.value} (confidence: {result.confidence:.2f})[/]"
        )

        state["intent"] = result.intent
        state["intent_confidence"] = result.confidence
        state["intent_reasoning"] = result.reasoning
        if result.reasoning_steps:
            state["intent_reasoning_steps"] = result.reasoning_steps
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
