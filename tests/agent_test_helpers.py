"""Agent 离线测试共用的 mock 装配（单一分派点）。

3.4.3 重构后工具执行只有 `agent_tools.dispatch_tool` 一条路径；测试统一在
`agent_entry.dispatch_tool` 处打桩，返回与真实分派一致的结构化结果。
"""

import json
import os
from unittest.mock import Mock, patch

from lab_rag import agent_entry as ae

ASSOC_SCENARIOS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "agent_assoc_scenarios.json")


def load_assoc_scenarios(path=None):
    """装载可成长的关联性/链式任务场景库（tests/agent_assoc_scenarios.json）。

    新增场景只需往该 JSON 追加一条 {id, category, question, route, steps[]}，
    单测与评测脚本会自动纳入，无需改动 Python 代码。
    """
    with open(path or ASSOC_SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)


def steps_to_outputs(steps):
    """把结构化 step 转成 mock LLM 的输出字符串（Thought/Action/Answer）。"""
    outs = []
    for s in steps:
        if "answer" in s:
            outs.append("Answer: " + s["answer"])
        else:
            arg = json.dumps(s["arg"], ensure_ascii=False)
            outs.append(f"Thought: {s.get('thought', '继续检索')}\nAction: {s['tool']}[{arg}]")
    return outs


def drive_assoc(scenario):
    """驱动一条关联性场景，返回 (events, llm, rag_recorder, catalog_recorder)。"""
    cfg = {
        "AGENT_MAX_STEPS": scenario.get("max_steps", 4),
        "AGENT_REPEAT_ACTION_GUARD": True,
        "AGENT_TOTAL_TIMEOUT_S": 240,
        "AGENT_MAX_IDLE_STEPS": 2,
    }
    outputs = steps_to_outputs(scenario["steps"])
    return drive(outputs, scenario["question"], route=scenario.get("route", "hybrid"),
                 cfg=cfg, observation=scenario.get("observation", "目录：A设备/B设备"))


def _llm(outputs):
    from types import SimpleNamespace
    llm = Mock()
    llm.invoke.side_effect = [SimpleNamespace(content=o) for o in outputs]
    return llm


def make_dispatch(observation="目录：A设备/B设备", rag_answer="正文依据",
                  catalog_result=None):
    """返回 (dispatch_mock, rag_recorder, catalog_recorder)。

    rag_recorder / catalog_recorder 记录每次工具调用的参数，便于沿用既有断言。
    """
    rag = Mock(return_value={"answer": rag_answer, "file_result": catalog_result})
    catalog = Mock(return_value={"mode": "filesystem_directory"})

    def dispatch(name, arg, username, route_hint=None, tool_call_id=""):
        from lab_rag.agent_tools import tool_role
        role = tool_role(name)
        if role == "content":
            rag(arg, username=username, route_hint=route_hint)
            obs, cat = rag_answer, catalog_result
        elif role == "catalog":
            catalog(arg)
            obs, cat = observation, catalog_result
        else:
            obs, cat = f"[工具错误] 未找到名为 {name} 的工具。", None
        return {"ok": True, "observation": obs, "catalog_result": cat,
                "elapsed_s": 0.0, "tool_call_id": tool_call_id or "tc-test"}

    return Mock(side_effect=dispatch), rag, catalog


def drive(outputs, question, route="hybrid", cfg=None,
          observation="目录：A设备/B设备", rag_answer="正文依据"):
    """驱动流式单编排器，返回 (events, llm, rag_recorder, catalog_recorder)。"""
    llm = _llm(outputs)
    dispatch, rag, catalog = make_dispatch(observation=observation, rag_answer=rag_answer)
    cfg = cfg or {"AGENT_MAX_STEPS": 4, "AGENT_REPEAT_ACTION_GUARD": True,
                  "AGENT_TOTAL_TIMEOUT_S": 240, "AGENT_MAX_IDLE_STEPS": 2}
    with patch.object(ae, "rule_based_route", return_value={"route": route, "target": "VLC小组"}), \
            patch.object(ae, "_agent_config", return_value=cfg), \
            patch.object(ae, "build_qwen_llm", return_value=llm), \
            patch.object(ae, "dispatch_tool", dispatch), \
            patch.object(ae, "append_user_chat_history"):
        events = list(ae.ask_agent_stream(question, username="tester"))
    return events, llm, rag, catalog


def tool_steps(events):
    return [e for e in events if e["type"] == "step_tool"]
