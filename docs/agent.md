# molx-agent design

一套可落地的 **LangGraph 拓扑设计 + 关键代码骨架**，你可以在此基础上往里填具体工具调用（PubMed / RDKit / AlphaFold 等）。


我会按这种结构来写：

1. 整体 Graph 拓扑设计
2. Node 设计（Planner / Workers / Reviewer）
3. LangGraph 代码示例（含 state、边、入口函数）
4. 如何逐步把真实工具接进去

---

## 1. LangGraph 拓扑设计

### 1.1 角色 → Graph 节点

* `planner_node`

  * 输入：用户问题 + 当前 state
  * 输出：更新 state 中的 `tasks`（DAG，或者更简单的「待执行子任务列表」）和要执行的第一个/下一个子任务 ID。
* `literature_worker_node`
* `chemo_worker_node`
* `bio_worker_node`
* `reviewer_node`
* （可选）`router_node` 或直接用 `edges` 的条件逻辑从 planner/worker 跳转。

### 1.2 State 设计（用 `TypedDict`）

```python
from typing import List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict

class Task(TypedDict):
    id: str
    type: Literal["literature", "chemo", "bio", "meta"]
    description: str
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    depends_on: List[str]
    status: Literal["pending", "running", "done", "skipped"]

class AgentState(TypedDict, total=False):
    user_query: str
    tasks: Dict[str, Task]             # task_id -> Task
    current_task_id: Optional[str]
    results: Dict[str, Any]            # task_id -> worker output
    messages: List[Dict[str, Any]]     # 用于 LangGraph/LCEL 的 message 历史
    final_answer: Optional[str]        # reviewer 输出
```

> 简化点：一开始你也可以不用完整 DAG，仅用「Planner 一次性展开三个固定 worker，然后 reviewer 汇总」，后期再做依赖图。

### 1.3 Graph 流程（简单版）

1. `START` → `planner_node`
2. `planner_node` 根据用户问题产出 `tasks`，并设置 `current_task_id` 为第一个 worker 任务（如 literature）。
3. 根据 `current_task_id.type` 进入不同 worker：

   * `literature_worker_node`
   * `chemo_worker_node`
   * `bio_worker_node`
   * 或直接跳到 `reviewer_node`（如果没有 worker 任务）
4. 每个 worker 完成后：

   * 把结果写入 `state.results[task_id]`
   * 把该任务 `status = "done"`
   * Planner（或一个小的调度函数）选出下一个待执行子任务 → 更新 `current_task_id`
5. 若还有未完成 worker → 回到 worker 分发；
   都完成后 → `reviewer_node` → `END`

在 LangGraph 里用 **conditional edges** 来做 worker 路由即可。

---

## 2. 节点实现思路

### 2.1 Planner Node

使用一个 LLM（如 gpt-4.1）+ 前面你已经写好的 Planner system prompt。

伪代码：

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4.1")

PLANNER_SYSTEM_PROMPT = """
You are the Planner of a multi-agent SAR system.
[... 这里放你之前那段 Planner system prompt，要求输出 JSON DAG ...]
"""

def planner_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"User query:\n{user_query}\n\nReturn only a JSON object describing the DAG of tasks.")
    ]
    resp = llm.invoke(messages)
    text = resp.content
    dag = json.loads(text)    # 保证前面 prompt 要求“only JSON”
    
    tasks: Dict[str, Task] = {}
    for t in dag["tasks"]:
        t["status"] = "pending"
        tasks[t["id"]] = t

    # 挑选第一个可执行任务：没有 depends_on 或依赖已经满足（当前全空）
    def pick_next_task(tasks: Dict[str, Task]) -> Optional[str]:
        for tid, task in tasks.items():
            if task["status"] == "pending" and not task["depends_on"]:
                return tid
        return None

    next_id = pick_next_task(tasks)

    state["tasks"] = tasks
    state["current_task_id"] = next_id
    return state
```

### 2.2 Worker Nodes（以 Literature 为例）

先给一个「空壳，用 LLM + 伪工具」的版本，后面你可以把 PubMed API / 自己的 RAG 替换进去。

```python
LITERATURE_SYSTEM_PROMPT = """
You are the Literature Worker in a SAR system.
[... 这里放 Literature Worker 那段 system prompt ...]
"""

def literature_worker_node(state: AgentState) -> AgentState:
    task = state["tasks"][state["current_task_id"]]
    # 这里只把 task 信息作为上下文
    messages = [
        SystemMessage(content=LITERATURE_SYSTEM_PROMPT),
        HumanMessage(content=f"Task description:\n{task['description']}\n\nInputs:\n{json.dumps(task['inputs'], indent=2)}\n\nReturn a JSON object with keys: 'summary', 'compounds', 'raw_refs'.")
    ]
    resp = llm.invoke(messages)
    result = json.loads(resp.content)

    tid = task["id"]
    state["results"][tid] = result
    state["tasks"][tid]["status"] = "done"

    # 决定下一个任务
    state["current_task_id"] = pick_next_task(state)
    return state
```

`pick_next_task` 可以单独写一个函数，根据 `depends_on` 及 `status` 判断：

```python
def pick_next_task(state: AgentState) -> Optional[str]:
    tasks = state["tasks"]
    for tid, t in tasks.items():
        if t["status"] != "pending":
            continue
        # 如果所有依赖都已经 done，则可以执行
        if all(tasks[d]["status"] == "done" for d in t["depends_on"]):
            return tid
    # 如果没有 pending 任务，返回 None
    return None
```

Chemo / Bio worker 节点同理，只是 system prompt 和期望 JSON 输出不同（如 `sar_table`, `designed_analogs` 等）。

### 2.3 Reviewer Node

```python
REVIEWER_SYSTEM_PROMPT = """
You are the Reviewer of a multi-agent SAR system.
[... 放 Reviewer 那段 prompt ...]
"""

def reviewer_node(state: AgentState) -> AgentState:
    tasks = state.get("tasks", {})
    results = state.get("results", {})
    user_query = state["user_query"]

    # 把所有中间结果打包成一个概要文本喂给 Reviewer
    context = {
        "tasks": tasks,
        "results": results
    }
    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User query:\n{user_query}\n\n"
            "Here are all intermediate tasks and results (JSON):\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Please produce:\n"
            "1) A human-readable final SAR report.\n"
            "2) A machine-readable JSON with keys: 'key_SAR_rules', 'recommended_analogs', 'caveats', 'next_steps'.\n"
            "Return a JSON object with fields 'text_report' and 'structured'."
        ))
    ]
    resp = llm.invoke(messages)
    out = json.loads(resp.content)

    state["final_answer"] = out["text_report"]
    state["results"]["final_structured"] = out["structured"]
    return state
```

---

## 3. 用 LangGraph 串起来的完整骨架

下面是一份可以直接改成 `sar_graph.py` 的示例（你需要把 `...` 部分根据自己环境补全）。

```python
# sar_graph.py
from typing import Optional, Dict, Any
from typing_extensions import TypedDict
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

###################################
# 1. State & Task 定义
###################################

class Task(TypedDict, total=False):
    id: str
    type: str
    description: str
    inputs: Dict[str, Any]
    expected_outputs: list[str]
    depends_on: list[str]
    status: str

class AgentState(TypedDict, total=False):
    user_query: str
    tasks: Dict[str, Task]
    current_task_id: Optional[str]
    results: Dict[str, Any]

llm = ChatOpenAI(model="gpt-4.1")

###################################
# 2. Planner / Worker / Reviewer
###################################

PLANNER_SYSTEM_PROMPT = """..."""
LITERATURE_SYSTEM_PROMPT = """..."""
CHEMO_SYSTEM_PROMPT = """..."""
BIO_SYSTEM_PROMPT = """..."""
REVIEWER_SYSTEM_PROMPT = """..."""

def pick_next_task(state: AgentState) -> Optional[str]:
    tasks = state.get("tasks", {})
    for tid, t in tasks.items():
        if t["status"] != "pending":
            continue
        if all(tasks[d]["status"] == "done" for d in t["depends_on"]):
            return tid
    return None

def planner_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"User query:\n{user_query}\n\nReturn only the JSON DAG.")
    ]
    resp = llm.invoke(messages)
    dag = json.loads(resp.content)

    tasks: Dict[str, Task] = {}
    for t in dag["tasks"]:
        t["status"] = "pending"
        tasks[t["id"]] = t

    state["tasks"] = tasks
    state["results"] = {}
    state["current_task_id"] = pick_next_task(state)
    return state

def literature_worker_node(state: AgentState) -> AgentState:
    tid = state["current_task_id"]
    task = state["tasks"][tid]

    msgs = [
        SystemMessage(content=LITERATURE_SYSTEM_PROMPT),
        HumanMessage(content=f"Task:\n{json.dumps(task, indent=2)}\n\nReturn JSON with keys: summary, compounds, raw_refs.")
    ]
    resp = llm.invoke(msgs)
    result = json.loads(resp.content)

    state["results"][tid] = result
    state["tasks"][tid]["status"] = "done"
    state["current_task_id"] = pick_next_task(state)
    return state

def chemo_worker_node(state: AgentState) -> AgentState:
    tid = state["current_task_id"]
    task = state["tasks"][tid]

    msgs = [
        SystemMessage(content=CHEMO_SYSTEM_PROMPT),
        HumanMessage(content=f"Task:\n{json.dumps(task, indent=2)}\n\nReturn JSON with keys: sar_table, designed_analogs, summary.")
    ]
    resp = llm.invoke(msgs)
    result = json.loads(resp.content)

    state["results"][tid] = result
    state["tasks"][tid]["status"] = "done"
    state["current_task_id"] = pick_next_task(state)
    return state

def bio_worker_node(state: AgentState) -> AgentState:
    tid = state["current_task_id"]
    task = state["tasks"][tid]

    msgs = [
        SystemMessage(content=BIO_SYSTEM_PROMPT),
        HumanMessage(content=f"Task:\n{json.dumps(task, indent=2)}\n\nReturn JSON with keys: binding_site_description, residue_ligand_interaction_map, selectivity_analysis, summary.")
    ]
    resp = llm.invoke(msgs)
    result = json.loads(resp.content)

    state["results"][tid] = result
    state["tasks"][tid]["status"] = "done"
    state["current_task_id"] = pick_next_task(state)
    return state

def reviewer_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    context = {"tasks": state.get("tasks", {}), "results": state.get("results", {})}

    msgs = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User query:\n{user_query}\n\n"
            f"All tasks and results:\n{json.dumps(context, indent=2)}\n\n"
            "Return JSON with keys 'text_report' and 'structured'."
        ))
    ]
    resp = llm.invoke(msgs)
    out = json.loads(resp.content)

    state["results"]["final_structured"] = out["structured"]
    state["results"]["final_report"] = out["text_report"]
    return state

###################################
# 3. Router：根据当前任务类型选 Worker
###################################

def route_after_planner(state: AgentState) -> str:
    if state["current_task_id"] is None:
        return "reviewer"
    tid = state["current_task_id"]
    ttype = state["tasks"][tid]["type"]
    if ttype == "literature":
        return "literature_worker"
    elif ttype == "chemo":
        return "chemo_worker"
    elif ttype == "bio":
        return "bio_worker"
    else:
        # "meta" 任务：很多时候可以直接标记 done，然后 pick 下一任务
        state["tasks"][tid]["status"] = "done"
        state["current_task_id"] = pick_next_task(state)
        if state["current_task_id"] is None:
            return "reviewer"
        else:
            return route_after_planner(state)

def route_after_worker(state: AgentState) -> str:
    if state["current_task_id"] is None:
        return "reviewer"
    tid = state["current_task_id"]
    ttype = state["tasks"][tid]["type"]
    if ttype == "literature":
        return "literature_worker"
    elif ttype == "chemo":
        return "chemo_worker"
    elif ttype == "bio":
        return "bio_worker"
    else:
        state["tasks"][tid]["status"] = "done"
        state["current_task_id"] = pick_next_task(state)
        if state["current_task_id"] is None:
            return "reviewer"
        return route_after_worker(state)

###################################
# 4. 构建 LangGraph
###################################

def build_sar_graph():
    sg = StateGraph(AgentState)

    sg.add_node("planner", planner_node)
    sg.add_node("literature_worker", literature_worker_node)
    sg.add_node("chemo_worker", chemo_worker_node)
    sg.add_node("bio_worker", bio_worker_node)
    sg.add_node("reviewer", reviewer_node)

    sg.set_entry_point("planner")

    # planner 执行完后，根据任务类型路由
    sg.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "literature_worker": "literature_worker",
            "chemo_worker": "chemo_worker",
            "bio_worker": "bio_worker",
            "reviewer": "reviewer",
        }
    )

    # 任一 worker 执行完之后也用同一个路由函数
    for w in ["literature_worker", "chemo_worker", "bio_worker"]:
        sg.add_conditional_edges(
            w,
            route_after_worker,
            {
                "literature_worker": "literature_worker",
                "chemo_worker": "chemo_worker",
                "bio_worker": "bio_worker",
                "reviewer": "reviewer",
            }
        )

    sg.add_edge("reviewer", END)

    graph = sg.compile()
    return graph

###################################
# 5. 对外入口
###################################

sar_graph = build_sar_graph()

def run_sar_agent(user_query: str):
    initial_state: AgentState = {"user_query": user_query}
    final_state = sar_graph.invoke(initial_state)
    return final_state["results"]["final_report"], final_state["results"]["final_structured"]
```

> 这份骨架做的事：
>
> * Planner 用 LLM 生成 DAG → 存到 state
> * 通过路由函数在不同 worker 节点之间跳转，直到没有 pending task
> * Reviewer 读所有 `tasks + results`，输出最终报告

---

## 4. 下一步怎么把工具“挂”上去？

目前代码里 worker 都是「纯 LLM」，你可以逐步替换为 **工具 + LLM** 的组合：

1. 封装 PubMed / patent 搜索 → 在 Literature Worker 中：

   * 先根据 task 里的 `inputs` 调用搜索 API
   * 把检索结果摘要/段落送给 LLM，总结成 SAR JSON

2. 封装 RDKit / PyTorch Geometric → 在 Chemo Worker 中：

   * 先在 Python 里按 task.inputs 解析 SMILES、算描述符、构造改造 analog
   * 把结构 & 数值表 dump 成 JSON，连同「我想分析的问题」发给 LLM，让它解释 SAR 并补全文案

3. 封装 AlphaFold / BLAST → 在 Bio Worker 中：

   * 先跑序列工具，拿到 domain / homology / binding site 信息
   * 同样用 LLM 做结构化总结 + 文字解释

---

如果你愿意，我可以帮你把 **其中一个 worker（比如 Chemo Worker）改写成「RDKit 真调用 + LLM」的具体代码样例**，包括如何在 LangGraph state 里传 SMILES / 活性表。
