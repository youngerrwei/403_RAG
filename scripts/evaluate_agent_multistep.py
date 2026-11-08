"""Agent 多轮推理编排层性能评测（离线、可复现，不依赖 GPU / Qdrant / vLLM）。

用 mock LLM 驱动真实的 `ask_agent_stream`，量化决定真实推理时延的编排指标：
- llm_calls：LLM 调用轮次（越少越快）
- context_chars_peak / context_chars_total：每轮喂给 LLM 的 messages 字符数
- idle_steps：未执行工具就消耗的 LLM 轮次（重复抑制 / 过早 Answer 拦截）
- converged：是否得到真正的 Answer（而非预算耗尽回显）

场景全部来自真实多轮问句类型（复合意图、跨小组、目录未命中换角度、
重复调用、过早 Answer、预算耗尽、长 Observation），不含测试集标注语义。
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_agent_module(path=None):
    """加载待评测的 agent_entry；path 指向历史版本源码时用于可比基线。"""
    if not path:
        from lab_rag import agent_entry
        return agent_entry
    spec = importlib.util.spec_from_file_location("lab_rag.agent_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lab_rag.agent_entry"] = module
    spec.loader.exec_module(module)
    return module


ae = None  # 由 main() 按 --agent-module 注入


class RecordingLLM:
    """记录每轮 messages 字符数，并按脚本输出 Thought/Action/Answer。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.context_chars = []
        self.index = 0

    def invoke(self, messages):
        self.context_chars.append(sum(len(m.get("content", "")) for m in messages))
        out = self.outputs[min(self.index, len(self.outputs) - 1)]
        self.index += 1
        return SimpleNamespace(content=out)


LONG_OBSERVATION = "目录：\n" + "\n".join(
    f"- 设备操作指南/VLC小组/设备{i:02d}/Manuals/manual_{i:02d}.md (chunks=12)"
    for i in range(120)
)


def _action(tool, arg, thought="继续检索"):
    return f"Thought: {thought}\nAction: {tool}[{json.dumps(arg, ensure_ascii=False)}]"


SCENARIOS = {
    "compound": {
        "question": "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤",
        "route": "hybrid",
        "outputs": [
            _action("list_group_files", "A2X小组", "先列设备"),
            _action("rag_qa", "USRP RIO 入门指南的安装步骤", "再读正文"),
            "Answer: 根据知识库内容可概括为：设备清单与安装步骤",
        ],
    },
    "cross_group": {
        "question": "VLC小组 与 A2X小组 的组会总结分别对设备平台提出了什么要求？",
        "route": "hybrid",
        "outputs": [
            _action("rag_qa", "VLC小组 组会总结 设备平台要求"),
            _action("rag_qa", "A2X小组 组会总结 设备平台要求"),
            "Answer: 根据知识库内容可概括为：两个小组的要求分别如下",
        ],
    },
    "nested_catalog": {
        "question": "VLC小组 的 Manuals 文件夹里有哪些文件？",
        "route": "file_list",
        "outputs": [
            _action("list_group_files", "VLC小组"),
            _action("list_group_files", "HOLOEYE LETO-3 Manuals", "目录未命中，换更窄关键词"),
            _action("rag_qa", "VLC小组 的 Manuals 文件夹里有哪些文件？"),
            "Answer: 根据知识库内容可概括为：Manuals 下的文件",
        ],
    },
    "repeat_action": {
        "question": "USRP-2943 和 USRP-2944 的规格参数有什么区别？",
        "route": "llm_fallback",
        "outputs": [
            _action("rag_qa", "USRP-2943 规格参数"),
            _action("rag_qa", "USRP-2943 规格参数", "再查一次"),
            _action("rag_qa", "USRP-2944 规格参数", "换角度"),
            "Answer: 根据知识库内容可概括为：两款设备参数差异",
        ],
    },
    "early_answer": {
        "question": "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤",
        "route": "hybrid",
        "outputs": [
            _action("list_group_files", "A2X小组"),
            "Answer: 设备有 A 和 B",
            _action("rag_qa", "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤"),
            "Answer: 根据知识库内容可概括为：设备清单与安装步骤",
        ],
    },
    "budget_exhausted": {
        "question": "请分别说明多个小组的全部研究内容与设备参数",
        "route": "llm_fallback",
        "outputs": [_action("rag_qa", f"第{i}个角度") for i in range(1, 7)],
    },
    "long_observation": {
        "question": "VLC小组 的 Manuals 文件夹里有哪些文件？",
        "route": "file_list",
        "outputs": [
            _action("list_group_files", "VLC小组"),
            _action("rag_qa", "VLC小组 Manuals 文件内容说明"),
            "Answer: 根据知识库内容可概括为：Manuals 下的文件",
        ],
        "observation": LONG_OBSERVATION,
    },
    # 小预算 + 一次空转：基线会因空转占掉步数而提前耗尽，优化后仍能完成两次有效检索
    "idle_then_recover": {
        "question": "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤",
        "route": "hybrid",
        "max_steps": 2,
        "outputs": [
            _action("list_group_files", "A2X小组"),
            _action("list_group_files", "A2X小组", "再查一次"),
            _action("rag_qa", "USRP RIO 入门指南的安装步骤"),
            "Answer: 根据知识库内容可概括为：设备清单与安装步骤",
        ],
    },
}


def run_scenario(name, spec, cfg):
    scenario_cfg = dict(cfg)
    if spec.get("max_steps"):
        scenario_cfg["AGENT_MAX_STEPS"] = spec["max_steps"]
    llm = RecordingLLM(spec["outputs"])
    rag_answer = "正文依据"
    catalog = {"mode": "filesystem_directory"}
    observation = spec.get("observation", "目录：A设备/B设备")
    rag = lambda *a, **kw: {"answer": rag_answer, "file_result": None}  # noqa: E731

    with patch.object(ae, "rule_based_route", return_value={"route": spec["route"], "target": "VLC小组"}), \
            patch.object(ae, "_agent_config", return_value=scenario_cfg), \
            patch.object(ae, "build_qwen_llm", return_value=llm), \
            patch.object(ae, "ask_rag", side_effect=rag), \
            patch.object(ae, "list_catalog_entries", return_value=catalog), \
            patch.object(ae, "build_file_context", return_value=observation), \
            patch.object(ae, "append_user_chat_history"):
        events = list(ae.ask_agent_stream(spec["question"], username="agent-perf"))

    final = [e for e in events if e["type"] == "final"]
    tool_steps = [e for e in events if e["type"] == "step_tool"]
    converged = bool(final) and "工具调用达到最大步数" not in final[-1]["content"]
    llm_calls = len(llm.context_chars)
    idle = max(0, llm_calls - len(tool_steps) - (1 if converged else 0))
    return {
        "llm_calls": llm_calls,
        "tool_calls": len(tool_steps),
        "idle_steps": idle,
        "context_chars_peak": max(llm.context_chars) if llm.context_chars else 0,
        "context_chars_total": sum(llm.context_chars),
        "converged": converged,
        "answer_chars": len(final[-1]["content"]) if final else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Agent 多轮推理编排层性能评测")
    parser.add_argument("--label", default="baseline", help="本次评测标签")
    parser.add_argument("--out", default="logs/agent_multistep_metrics.json")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--agent-module", default=None,
                        help="历史版本 agent_entry.py 源码路径，用于生成可比基线")
    args = parser.parse_args()

    global ae
    ae = load_agent_module(args.agent_module)

    cfg = {"AGENT_MAX_STEPS": args.max_steps,
           "AGENT_REPEAT_ACTION_GUARD": True,
           "AGENT_LEGACY_MAX_STEPS": 5}
    results = {name: run_scenario(name, spec, cfg) for name, spec in SCENARIOS.items()}
    n = len(results)
    summary = {
        "label": args.label,
        "scenarios": n,
        "avg_llm_calls": round(sum(v["llm_calls"] for v in results.values()) / n, 2),
        "avg_context_chars_peak": round(sum(v["context_chars_peak"] for v in results.values()) / n, 1),
        "avg_context_chars_total": round(sum(v["context_chars_total"] for v in results.values()) / n, 1),
        "max_context_chars_peak": max(v["context_chars_peak"] for v in results.values()),
        "total_idle_steps": sum(v["idle_steps"] for v in results.values()),
        "convergence_rate": round(sum(1 for v in results.values() if v["converged"]) / n, 3),
        # 只在已收敛场景上统计轮次，避免把「基线直接 fallback（轮次少但没答案）」
        # 与「优化后继续推理到收敛（轮次多但有答案）」混在一起比较
        "avg_llm_calls_converged": round(
            sum(v["llm_calls"] for v in results.values() if v["converged"]) /
            max(1, sum(1 for v in results.values() if v["converged"])), 2),
        "avg_context_chars_peak_converged": round(
            sum(v["context_chars_peak"] for v in results.values() if v["converged"]) /
            max(1, sum(1 for v in results.values() if v["converged"])), 1),
        "per_scenario": results,
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scenario"},
                     ensure_ascii=False, indent=2))
    print(f"\n报告已写出：{args.out}")


if __name__ == "__main__":
    main()
