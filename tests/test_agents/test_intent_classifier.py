"""Tests for the AI-only intent classifier behavior."""

from types import SimpleNamespace

import pytest

from molx_agent.agents.intent_classifier import Intent, IntentClassifierAgent


def _fake_settings(api_key: str = ""):
    return SimpleNamespace(LOCAL_OPENAI_API_KEY=api_key)


def test_intent_classifier_requires_llm_key(monkeypatch):
    """Intent classification should abort when no API key is configured."""
    monkeypatch.setattr(
        "molx_agent.agents.intent_classifier.get_settings",
        lambda: _fake_settings(""),
    )

    agent = IntentClassifierAgent()
    with pytest.raises(RuntimeError):
        agent.run({"user_query": "完成 csv SAR 分析"})


def test_intent_classifier_uses_llm_when_key_is_available(monkeypatch):
    """LLM output should be forwarded into the agent state when enabled."""
    monkeypatch.setattr(
        "molx_agent.agents.intent_classifier.get_settings",
        lambda: _fake_settings("fake-key"),
    )

    captured = {}

    def _fake_llm(system_prompt: str, user_prompt: str, *, parse_json: bool):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["parse_json"] = parse_json
        return {
            "reasoning_steps": [
                "User asks for molecular weight, which maps to molecule properties",
                "Molecule property questions belong to molecule_query",
            ],
            "reasoning": "Looks like a molecule property question",
            "intent": "molecule_query",
            "confidence": 0.91,
        }

    monkeypatch.setattr(
        "molx_agent.agents.intent_classifier.invoke_llm",
        _fake_llm,
    )

    agent = IntentClassifierAgent()
    state = agent.run({"user_query": "What is the molecular weight of aspirin?"})

    assert state["intent"] == Intent.MOLECULE_QUERY
    assert state["intent_confidence"] == 0.91
    assert "molecule" in state["intent_reasoning"].lower()
    assert state["intent_reasoning_steps"] == [
        "User asks for molecular weight, which maps to molecule properties",
        "Molecule property questions belong to molecule_query",
    ]
    assert captured["parse_json"] is True
    assert "User Query" in captured["user_prompt"]
