"""Node implementations for SAR agent graph."""

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from molx_agent.config import get_settings

from .prompts import (
    BIO_WORKER_PROMPT,
    CHEMO_WORKER_PROMPT,
    LITERATURE_WORKER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)
from .state import AgentState, Task

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    """Get configured LLM instance."""
    settings = get_settings()
    kwargs = {
        "model": settings.LOCAL_OPENAI_MODEL,
        "api_key": settings.LOCAL_OPENAI_API_KEY,
    }
    if settings.LOCAL_OPENAI_BASE_URL:
        kwargs["base_url"] = settings.LOCAL_OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def pick_next_task(state: AgentState) -> Optional[str]:
    """Pick the next task to execute based on dependencies."""
    tasks = state.get("tasks", {})
    for tid, task in tasks.items():
        if task.get("status") != "pending":
            continue
        # Check if all dependencies are done
        depends_on = task.get("depends_on", [])
        if all(tasks.get(dep, {}).get("status") == "done" for dep in depends_on):
            return tid
    return None


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
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


def planner_node(state: AgentState) -> AgentState:
    """Plan and decompose the user query into tasks."""
    user_query = state.get("user_query", "")
    llm = get_llm()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User query:\n{user_query}\n\nReturn only the JSON DAG of tasks."
        ),
    ]

    try:
        resp = llm.invoke(messages)
        dag = parse_json_response(resp.content)

        tasks: dict[str, Task] = {}
        for t in dag.get("tasks", []):
            t["status"] = "pending"
            tasks[t["id"]] = t

        state["tasks"] = tasks
        state["results"] = {}
        state["current_task_id"] = pick_next_task(state)
        logger.info(f"Planner created {len(tasks)} tasks")

    except Exception as e:
        logger.error(f"Planner error: {e}")
        state["error"] = f"Planner error: {e}"
        state["tasks"] = {}
        state["current_task_id"] = None

    return state


def literature_worker_node(state: AgentState) -> AgentState:
    """Execute literature search and analysis task."""
    tid = state.get("current_task_id")
    if not tid:
        return state

    task = state.get("tasks", {}).get(tid)
    if not task:
        return state

    llm = get_llm()

    messages = [
        SystemMessage(content=LITERATURE_WORKER_PROMPT),
        HumanMessage(
            content=f"Task:\n{json.dumps(task, indent=2)}\n\n"
            "Return JSON with keys: summary, compounds, sar_insights, references."
        ),
    ]

    try:
        resp = llm.invoke(messages)
        result = parse_json_response(resp.content)
        state["results"][tid] = result
        state["tasks"][tid]["status"] = "done"
        logger.info(f"Literature worker completed task {tid}")
    except Exception as e:
        logger.error(f"Literature worker error: {e}")
        state["results"][tid] = {"error": str(e)}
        state["tasks"][tid]["status"] = "done"

    state["current_task_id"] = pick_next_task(state)
    return state


def chemo_worker_node(state: AgentState) -> AgentState:
    """Execute chemical analysis task."""
    tid = state.get("current_task_id")
    if not tid:
        return state

    task = state.get("tasks", {}).get(tid)
    if not task:
        return state

    llm = get_llm()

    messages = [
        SystemMessage(content=CHEMO_WORKER_PROMPT),
        HumanMessage(
            content=f"Task:\n{json.dumps(task, indent=2)}\n\n"
            "Return JSON with keys: sar_table, sar_rules, designed_analogs, summary."
        ),
    ]

    try:
        resp = llm.invoke(messages)
        result = parse_json_response(resp.content)
        state["results"][tid] = result
        state["tasks"][tid]["status"] = "done"
        logger.info(f"Chemo worker completed task {tid}")
    except Exception as e:
        logger.error(f"Chemo worker error: {e}")
        state["results"][tid] = {"error": str(e)}
        state["tasks"][tid]["status"] = "done"

    state["current_task_id"] = pick_next_task(state)
    return state


def bio_worker_node(state: AgentState) -> AgentState:
    """Execute biological analysis task."""
    tid = state.get("current_task_id")
    if not tid:
        return state

    task = state.get("tasks", {}).get(tid)
    if not task:
        return state

    llm = get_llm()

    messages = [
        SystemMessage(content=BIO_WORKER_PROMPT),
        HumanMessage(
            content=f"Task:\n{json.dumps(task, indent=2)}\n\n"
            "Return JSON with keys: target_info, binding_site, interactions, "
            "selectivity_analysis, summary."
        ),
    ]

    try:
        resp = llm.invoke(messages)
        result = parse_json_response(resp.content)
        state["results"][tid] = result
        state["tasks"][tid]["status"] = "done"
        logger.info(f"Bio worker completed task {tid}")
    except Exception as e:
        logger.error(f"Bio worker error: {e}")
        state["results"][tid] = {"error": str(e)}
        state["tasks"][tid]["status"] = "done"

    state["current_task_id"] = pick_next_task(state)
    return state


def reviewer_node(state: AgentState) -> AgentState:
    """Review all results and produce final report."""
    user_query = state.get("user_query", "")
    tasks = state.get("tasks", {})
    results = state.get("results", {})

    llm = get_llm()

    context = {"tasks": tasks, "results": results}

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User query:\n{user_query}\n\n"
            f"All tasks and results:\n{json.dumps(context, indent=2)}\n\n"
            "Return JSON with keys 'text_report' and 'structured'."
        ),
    ]

    try:
        resp = llm.invoke(messages)
        out = parse_json_response(resp.content)
        state["final_answer"] = out.get("text_report", "")
        state["results"]["final_structured"] = out.get("structured", {})
        logger.info("Reviewer completed final report")
    except Exception as e:
        logger.error(f"Reviewer error: {e}")
        state["final_answer"] = f"Error generating report: {e}"
        state["error"] = str(e)

    return state


def route_after_planner(state: AgentState) -> str:
    """Route to appropriate worker after planner."""
    current_task_id = state.get("current_task_id")

    if current_task_id is None:
        return "reviewer"

    task = state.get("tasks", {}).get(current_task_id)
    if not task:
        return "reviewer"

    task_type = task.get("type", "meta")

    if task_type == "literature":
        return "literature_worker"
    elif task_type == "chemo":
        return "chemo_worker"
    elif task_type == "bio":
        return "bio_worker"
    else:
        # Meta task: mark as done and pick next
        state["tasks"][current_task_id]["status"] = "done"
        state["current_task_id"] = pick_next_task(state)
        return route_after_planner(state)


def route_after_worker(state: AgentState) -> str:
    """Route after worker completion."""
    return route_after_planner(state)
