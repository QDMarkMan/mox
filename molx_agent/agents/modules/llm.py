"""
**************************************************************************
*  @Copyright [2025] Xtalpi Systems.
*  @Author tongfu.e@xtalpi.com
*  @Date [2025-12-12].
*  @Description LLM utilities for agents.
**************************************************************************
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from molx_agent.config import get_settings

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """Get configured LLM instance."""
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.LOCAL_OPENAI_MODEL,
        "api_key": settings.LOCAL_OPENAI_API_KEY,
    }
    if settings.LOCAL_OPENAI_BASE_URL:
        kwargs["base_url"] = settings.LOCAL_OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: The raw LLM response content.

    Returns:
        Parsed JSON as a dictionary.
    """
    content = content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return json.loads(content)


def invoke_llm(
    system_prompt: str,
    user_message: str,
    parse_json: bool = True,
) -> dict[str, Any] | str:
    """Invoke LLM with system prompt and user message.

    Args:
        system_prompt: The system prompt for the LLM.
        user_message: The user message/query.
        parse_json: Whether to parse the response as JSON.

    Returns:
        Parsed JSON dict or raw string response.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    resp = llm.invoke(messages)
    content = resp.content

    if parse_json:
        return parse_json_response(content)
    return content
