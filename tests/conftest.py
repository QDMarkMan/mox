"""Global pytest fixtures and configuration."""

import sys
from pathlib import Path
import types
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _ensure_langchain_stubs() -> None:
    """Provide lightweight langchain_core.messages stubs for unit tests."""
    if "langchain_core.messages" in sys.modules:
        return

    class _BaseMessage:
        def __init__(self, content: str = "") -> None:
            self.content = content
            self.type = "generic"

    class HumanMessage(_BaseMessage):
        type = "human"

    class AIMessage(_BaseMessage):
        type = "ai"

    class SystemMessage(_BaseMessage):
        type = "system"

    messages_module = types.ModuleType("langchain_core.messages")
    messages_module.BaseMessage = _BaseMessage
    messages_module.HumanMessage = HumanMessage
    messages_module.AIMessage = AIMessage
    messages_module.SystemMessage = SystemMessage

    langchain_module = types.ModuleType("langchain_core")
    langchain_module.messages = messages_module

    sys.modules["langchain_core"] = langchain_module
    sys.modules["langchain_core.messages"] = messages_module


def _ensure_langgraph_stubs() -> None:
    """Provide stub for langgraph.graph.message.add_messages."""
    if "langgraph.graph.message" in sys.modules:
        return

    def add_messages(messages, new_message):  # type: ignore[override]
        bucket = list(messages or [])
        if new_message is not None:
            bucket.append(new_message)
        return bucket

    message_module = types.ModuleType("langgraph.graph.message")
    message_module.add_messages = add_messages

    class _Compiled:
        def __init__(self, entry, nodes, edges, conditionals):
            self.entry = entry
            self.nodes = nodes
            self.edges = edges
            self.conditionals = conditionals

        def invoke(self, state):
            current = self.entry
            while current != "__end__":
                fn = self.nodes[current]
                state = fn(state)
                if current in self.conditionals:
                    fn_route, mapping = self.conditionals[current]
                    route = fn_route(state)
                    current = mapping[route]
                elif current in self.edges:
                    current = self.edges[current]
                else:
                    break
            return state

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.conditionals = {}
            self.entry = None

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_conditional_edges(self, name, fn, mapping):
            self.conditionals[name] = (fn, mapping)

        def add_edge(self, source, target):
            self.edges[source] = target

        def compile(self):
            return _Compiled(self.entry, self.nodes, self.edges, self.conditionals)

    graph_module = types.ModuleType("langgraph.graph")
    graph_module.message = message_module
    graph_module.StateGraph = StateGraph
    graph_module.END = "__end__"
    graph_module.START = "__start__"

    prebuilt_module = types.ModuleType("langgraph.prebuilt")

    class ToolNode:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, state):
            return state

    def tools_condition(*args, **kwargs):  # pragma: no cover - stub only
        return "default"

    prebuilt_module.ToolNode = ToolNode
    prebuilt_module.tools_condition = tools_condition

    langgraph_module = types.ModuleType("langgraph")
    langgraph_module.graph = graph_module
    langgraph_module.prebuilt = prebuilt_module

    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module
    sys.modules["langgraph.graph.message"] = message_module
    sys.modules["langgraph.prebuilt"] = prebuilt_module


_ensure_langchain_stubs()
_ensure_langgraph_stubs()


def _ensure_langchain_tools_stub() -> None:
    """Provide lightweight BaseTool stub used in unit tests."""
    if "langchain_core.tools" in sys.modules:
        return

    class _BaseTool:
        name: str = "tool"
        description: str = ""
        args_schema = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            pass

        def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - stub
            raise NotImplementedError

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return self._run(*args, **kwargs)

    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.BaseTool = _BaseTool
    sys.modules["langchain_core.tools"] = tools_module


_ensure_langchain_tools_stub()


def _ensure_langchain_openai_stub() -> None:
    """Provide stub ChatOpenAI for unit tests."""
    if "langchain_openai" in sys.modules:
        return

    class _ChatOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            pass

        def bind_tools(self, *args: Any, **kwargs: Any):  # pragma: no cover - simple stub
            return self

        def invoke(self, *args: Any, **kwargs: Any):  # pragma: no cover - simple stub
            class _Resp:
                content = "{}"

            return _Resp()

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = _ChatOpenAI
    sys.modules["langchain_openai"] = module


_ensure_langchain_openai_stub()
