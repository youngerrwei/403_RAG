"""ReAct Agent 多轮检索的离线行为测试（mock LLM / 工具，不依赖真实模型或 Qdrant）。

用例问句取自「可能产生多轮搜索」的真实场景：复合意图（清单+内容）、
跨小组分别检索、目录未命中、正文答非所问、重复调用空转、步数预算耗尽。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lab_rag import agent_entry as ae


MULTISTEP_QUERIES = {
    "compound": "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤",
    "cross_group": "VLC小组 与 A2X小组 的组会总结分别对设备平台提出了什么要求？",
    "nested_catalog": "VLC小组 的 Manuals 文件夹里有哪些文件？",
    "multi_doc": "USRP-2943 和 USRP-2944 的规格参数有什么区别？",
    "list_then_method": "列出 VLC小组 的设备，并说明 EDU36311A 电源的编程指南与快速入门分别讲了什么",
}


def _llm(outputs):
    llm = Mock()
    llm.invoke.side_effect = [SimpleNamespace(content=o) for o in outputs]
    return llm


def _events(outputs, question, route="hybrid", cfg=None, ask_rag_answer="正文依据"):
    """驱动流式 ReAct，返回 (events, llm, ask_rag_mock, list_catalog_mock)。"""
    llm = _llm(outputs)
    rag = Mock(return_value={"answer": ask_rag_answer, "file_result": None})
    catalog = Mock(return_value={"mode": "filesystem_directory"})
    cfg = cfg or {"AGENT_MAX_STEPS": 4, "AGENT_REPEAT_ACTION_GUARD": True}
    with patch.object(ae, "rule_based_route", return_value={"route": route, "target": "VLC小组"}), \
            patch.object(ae, "_agent_config", return_value=cfg), \
            patch.object(ae, "build_qwen_llm", return_value=llm), \
            patch.object(ae, "ask_rag", rag), \
            patch.object(ae, "list_catalog_entries", catalog), \
            patch.object(ae, "build_file_context", return_value="目录：A设备/B设备"), \
            patch.object(ae, "append_user_chat_history"):
        return list(ae.ask_agent_stream(question, username="tester")), llm, rag, catalog


def _tool_steps(events):
    return [e for e in events if e["type"] == "step_tool"]


class CompoundIntentMultiStepTests(unittest.TestCase):
    """复合意图：目录枚举只回答清单部分，必须再调 rag_qa 读正文。"""

    def test_list_then_rag_qa_before_answer(self):
        outputs = [
            'Thought: 先列设备\nAction: list_group_files["VLC小组"]',
            'Thought: 还要读安装步骤\nAction: rag_qa["USRP RIO 入门指南的安装步骤"]',
            'Answer: 根据知识库内容可概括为：设备清单与安装步骤',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["compound"])
        tools = [e["tool"] for e in _tool_steps(events)]
        self.assertEqual(tools, ["list_group_files", "rag_qa"])
        catalog.assert_called_once_with("VLC小组")
        rag.assert_called_once()
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(llm.invoke.call_count, 3)

    def test_early_answer_is_intercepted_for_hybrid(self):
        """只列目录就想 Answer 时，编排层强制继续 rag_qa。"""
        outputs = [
            'Thought: 列目录\nAction: list_group_files["VLC小组"]',
            'Answer: 设备有 A 和 B',
            'Thought: 补读正文\nAction: rag_qa["A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤"]',
            'Answer: 根据知识库内容可概括为：设备清单与安装步骤',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["compound"])
        tools = [e["tool"] for e in _tool_steps(events)]
        self.assertEqual(tools, ["list_group_files", "rag_qa"])
        rag.assert_called_once()
        self.assertEqual(len([e for e in events if e["type"] == "final"]), 1)
        # 确定性补检索使用原始问题，保证不丢失用户意图
        self.assertEqual(rag.call_args.args[0], MULTISTEP_QUERIES["compound"])


class CrossGroupMultiStepTests(unittest.TestCase):
    """跨小组/多实体：分别检索每个实体再综合。"""

    def test_two_groups_are_searched_with_distinct_args(self):
        outputs = [
            'Thought: 先查 VLC\nAction: rag_qa["VLC小组 组会总结 设备平台要求"]',
            'Thought: 再查 A2X\nAction: rag_qa["A2X小组 组会总结 设备平台要求"]',
            'Answer: 根据知识库内容可概括为：两个小组的要求分别如下',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["cross_group"])
        args = [c.args[0] for c in rag.call_args_list]
        self.assertEqual(len(args), 2)
        self.assertNotEqual(args[0], args[1])
        self.assertIn("VLC", args[0])
        self.assertIn("A2X", args[1])

    def test_three_distinct_angles_within_budget(self):
        outputs = [
            'Thought: 查规范一\nAction: rag_qa["团队规范 文档格式"]',
            'Thought: 查规范二\nAction: rag_qa["实验拍照要求 文档格式"]',
            'Thought: 查写作建议\nAction: rag_qa["组会总结 论文写作方法"]',
            'Answer: 根据知识库内容可概括为：三者对比',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["multi_doc"])
        self.assertEqual(rag.call_count, 3)
        self.assertEqual(llm.invoke.call_count, 4)


class AngleChangeMultiStepTests(unittest.TestCase):
    """目录未命中 / 正文答非所问时换检索角度，而非原地重试。"""

    def test_catalog_miss_then_narrower_keyword_then_rag_qa(self):
        outputs = [
            'Thought: 先列小组目录\nAction: list_group_files["VLC小组"]',
            'Thought: 目录未命中，换更窄关键词\nAction: list_group_files["HOLOEYE LETO-3 Manuals"]',
            'Thought: 用原始问题读正文\nAction: rag_qa["VLC小组 的 Manuals 文件夹里有哪些文件？"]',
            'Answer: 根据知识库内容可概括为：Manuals 下的文件',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["nested_catalog"])
        cat_args = [e["arg"] for e in _tool_steps(events) if e["tool"] == "list_group_files"]
        self.assertEqual(len(cat_args), 2)
        self.assertNotEqual(cat_args[0], cat_args[1])
        rag.assert_called_once()

    def test_repeated_identical_action_is_suppressed(self):
        """相同工具+完全相同参数不应被重复执行，避免多轮搜索空转。"""
        outputs = [
            'Thought: 第一次查\nAction: rag_qa["USRP-2943 规格参数"]',
            'Thought: 再查一次\nAction: rag_qa["USRP-2943 规格参数"]',
            'Thought: 换角度查 USRP-2944\nAction: rag_qa["USRP-2944 规格参数"]',
            'Answer: 根据知识库内容可概括为：两款设备参数差异',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["multi_doc"])
        args = [c.args[0] for c in rag.call_args_list]
        self.assertEqual(args, ["USRP-2943 规格参数", "USRP-2944 规格参数"])
        self.assertEqual(len(args), len(set(args)))

    def test_repeat_guard_can_be_disabled(self):
        cfg = {"AGENT_MAX_STEPS": 4, "AGENT_REPEAT_ACTION_GUARD": False}
        outputs = [
            'Thought: 第一次\nAction: rag_qa["同一问题"]',
            'Thought: 第二次\nAction: rag_qa["同一问题"]',
            'Answer: 完成',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["multi_doc"], cfg=cfg)
        self.assertEqual(rag.call_count, 2)


class StepBudgetTests(unittest.TestCase):
    """步数预算耗尽时收敛，不无限循环。"""

    def test_budget_is_configurable_and_bounded(self):
        cfg = {"AGENT_MAX_STEPS": 2, "AGENT_REPEAT_ACTION_GUARD": True}
        outputs = [
            'Thought: 第一跳\nAction: rag_qa["A"]',
            'Thought: 第二跳\nAction: rag_qa["B"]',
            'Thought: 收敛轮仍无依据\nAction: rag_qa["C"]',
        ]
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["multi_doc"], cfg=cfg)
        self.assertEqual(events[-1]["type"], "final")
        # 工具步数严格受预算约束（收敛轮不执行新工具）
        self.assertEqual(events[-1]["agent_metrics"]["tool_calls"], 2)
        # 预算耗尽时诚实收敛：不回显原始工具输出、不编造依据
        self.assertIn("暂未能形成完整回答", events[-1]["content"])
        self.assertIn("已尝试的检索角度", events[-1]["content"])
        self.assertFalse(events[-1]["agent_metrics"]["converged"])

    def test_no_action_falls_back_to_final_answer(self):
        outputs = ['Thought: 直接作答\n根据知识库内容可概括为：已知信息']
        events, llm, rag, catalog = _events(outputs, MULTISTEP_QUERIES["multi_doc"])
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(rag.call_count, 0)


class DeterministicRouteTests(unittest.TestCase):
    """明确内容型问题仍确定性直调 rag_qa，不进入 LLM 多轮循环。"""

    def test_rag_search_route_uses_single_direct_call(self):
        llm = Mock()
        rag = Mock(return_value={"answer": "周报格式正文", "file_result": None})
        with patch.object(ae, "rule_based_route", return_value={"route": "rag_search", "target": "周报"}), \
                patch.object(ae, "build_qwen_llm", return_value=llm), \
                patch.object(ae, "ask_rag", rag), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("周报怎么写", username="tester"))
        rag.assert_called_once_with("周报怎么写", username="tester")
        llm.invoke.assert_not_called()
        self.assertEqual([e["type"] for e in events][-1], "final")


class LegacyReActOnceTests(unittest.TestCase):
    """非流式入口同样受重复抑制与配置化步数约束。"""

    def test_repeat_action_suppressed_in_legacy_loop(self):
        llm = _llm([
            'Thought: 第一次\nAction: rag_qa["同一问题"]',
            'Thought: 重复\nAction: rag_qa["同一问题"]',
            'Thought: 换角度\nAction: rag_qa["换一个角度"]',
            'Answer: 完成',
        ])
        tool = Mock()
        tool.args = {"query": str}  # 非流式入口按 tool.args 取参数名
        tool.invoke.return_value = "工具结果"
        with patch.object(ae, "_agent_config", return_value={
                "AGENT_LEGACY_MAX_STEPS": 5, "AGENT_REPEAT_ACTION_GUARD": True}), \
                patch.object(ae, "TOOLS", {"rag_qa": tool}), \
                patch.object(ae, "agent_tool_context", return_value=_nullcontext()):
            answer = ae.run_react_once(llm, MULTISTEP_QUERIES["multi_doc"], debug=False)
        self.assertEqual(answer, "完成")
        self.assertEqual(tool.invoke.call_count, 2)


class PromptContractTests(unittest.TestCase):
    """提示词必须依据文档记录真实语料结构，并给出多轮检索决策规则。"""

    def test_prompt_documents_corpus_structure(self):
        p = ae.REACT_SYSTEM_PROMPT
        for token in ("小组/设备/Manuals", "组会资料", "参考书", "段落摘要",
                      "日期戳", "设备型号"):
            with self.subTest(token=token):
                self.assertIn(token, p)

    def test_prompt_defines_multi_step_decision_rules(self):
        p = ae.REACT_SYSTEM_PROMPT
        for token in ("多轮检索决策规则", "每一个子问", "必须再调用 rag_qa",
                      "分别检索", "换一个检索角度", "步数预算", "禁止用完全相同的参数"):
            with self.subTest(token=token):
                self.assertIn(token, p)

    def test_prompt_keeps_hard_constraints(self):
        p = ae.REACT_SYSTEM_PROMPT
        for token in ("Answer:", "Action:", "rag_qa", "list_group_files",
                      "不得凭空编造", "通用模板"):
            with self.subTest(token=token):
                self.assertIn(token, p)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
