"""ReAct Agent 多轮检索的离线行为测试（mock LLM / 工具，不依赖真实模型或 Qdrant）。

用例问句取自「可能产生多轮搜索」的真实场景：复合意图（清单+内容）、
跨小组分别检索、目录未命中、正文答非所问、重复调用空转、步数预算耗尽。
"""

import os
import unittest
from unittest.mock import Mock, patch

from lab_rag import agent_entry as ae
from lab_rag import rag_agent as ra
from agent_test_helpers import (_llm, drive as _drive, tool_steps as _tool_steps,
                                load_assoc_scenarios, drive_assoc, make_dispatch)


MULTISTEP_QUERIES = {
    "compound": "A2X小组有哪些设备，并说明 USRP RIO 入门指南的安装步骤",
    "cross_group": "VLC小组 与 A2X小组 的组会总结分别对设备平台提出了什么要求？",
    "nested_catalog": "VLC小组 的 Manuals 文件夹里有哪些文件？",
    "multi_doc": "USRP-2943 和 USRP-2944 的规格参数有什么区别？",
    "list_then_method": "列出 VLC小组 的设备，并说明 EDU36311A 电源的编程指南与快速入门分别讲了什么",
}


def _events(outputs, question, route="hybrid", cfg=None, ask_rag_answer="正文依据"):
    """兼容旧签名：驱动单一编排器，返回 (events, llm, rag_recorder, catalog_recorder)。"""
    return _drive(outputs, question, route=route, cfg=cfg, rag_answer=ask_rag_answer)


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


class RagSearchLoopTests(unittest.TestCase):
    """内容型问题（rag_search）不再确定性直调，而是进入 ReAct 循环由模型自主调用内容工具。"""

    def test_rag_search_enters_loop_and_calls_content_tool(self):
        outputs = [
            'Thought: 检索周报格式\nAction: rag_qa["周报格式规范"]',
            'Answer: 根据知识库内容可概括为：周报格式规范',
        ]
        events, llm, rag, catalog = _events(outputs, "周报怎么写", route="rag_search")
        # 首条仍是编排器路由决策事件，但随后真正进入 LLM 循环（不再短路直调）
        self.assertEqual(events[0]["type"], "route_direct")
        self.assertIn("step_llm", [e["type"] for e in events])
        llm.invoke.assert_called()
        # 由模型自主发起内容工具调用，并透传内容路由 hint 避免重复路由
        rag.assert_called_once_with("周报格式规范", username="tester",
                                    route_hint={"route": "rag_search"})
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertTrue(final["agent_metrics"]["converged"])
        self.assertTrue(final["agent_metrics"]["content_grounded"])
        self.assertNotIn("deterministic_route", final["agent_metrics"])

    def test_content_first_guard_rejects_answer_without_tool(self):
        """模型首轮直接 Answer（无依据）时，content-first 守卫挡回并要求先检索正文。"""
        outputs = [
            "Answer: 周报格式第一段写本周完成事项",
            'Thought: 需要先检索\nAction: rag_qa["周报格式规范"]',
            'Answer: 根据知识库内容可概括为：周报格式规范',
        ]
        events, llm, rag, catalog = _events(outputs, "周报怎么写", route="rag_search")
        rag.assert_called_once()
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertTrue(final["agent_metrics"]["converged"])
        self.assertTrue(final["agent_metrics"]["content_grounded"])
        self.assertGreaterEqual(final["agent_metrics"]["llm_calls"], 3)

    def test_content_route_without_evidence_converges_honestly(self):
        """内容型路由检索无命中时不得凭空作答，须诚实收敛。"""
        outputs = [
            'Thought: 检索\nAction: rag_qa["周报格式规范"]',
            'Answer: 根据知识库内容可概括为：周报格式规范',
        ]
        dispatch, rag, _catalog = make_dispatch(rag_coverage_level="none")
        events, llm, rag, catalog = _drive(outputs, "周报怎么写", route="rag_search",
                                           dispatch=dispatch)
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertFalse(final["agent_metrics"]["converged"])
        self.assertFalse(final["agent_metrics"]["content_grounded"])


class NonStreamAdapterTests(unittest.TestCase):
    """非流式入口只是消费同一事件流的薄适配器，不再另写一套循环。"""

    def test_adapter_uses_the_same_orchestrator(self):
        events, llm, rag, catalog = _events([
            'Thought: 检索\nAction: rag_qa["A"]',
            'Answer: 根据知识库内容可概括为：结论',
        ], MULTISTEP_QUERIES["multi_doc"])
        self.assertEqual(events[-1]["content"], "根据知识库内容可概括为：结论")
        with patch.object(ae, "ask_agent_stream", return_value=iter([
                {"type": "step_llm", "step": 1, "content": "Thought"},
                {"type": "final", "content": "适配器结果"}])) as stream:
            answer = ae.ask_agent("任意问题", username="tester")
        self.assertEqual(answer, "适配器结果")
        stream.assert_called_once()

    def test_repeat_suppression_applies_to_adapter_path(self):
        events, llm, rag, catalog = _events([
            'Thought: 第一次\nAction: rag_qa["同一问题"]',
            'Thought: 重复\nAction: rag_qa["同一问题"]',
            'Thought: 换角度\nAction: rag_qa["换一个角度"]',
            'Answer: 完成',
        ], MULTISTEP_QUERIES["multi_doc"])
        args = [c.args[0] for c in rag.call_args_list]
        self.assertEqual(args, ["同一问题", "换一个角度"])


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
        for token in ("多轮检索决策规则", "每一个子问", "必须再调用内容工具",
                      "分别检索", "换一个检索角度", "步数预算", "禁止用完全相同的参数"):
            with self.subTest(token=token):
                self.assertIn(token, p)

    def test_prompt_keeps_hard_constraints(self):
        p = ae.REACT_SYSTEM_PROMPT
        for token in ("Answer:", "Action:", "rag_qa", "list_group_files",
                      "不得凭空编造", "通用模板"):
            with self.subTest(token=token):
                self.assertIn(token, p)

    def test_prompt_guides_extract_then_summarize_chain(self):
        p = ae.REACT_SYSTEM_PROMPT
        for token in ("先定位", "分批处理", "提取", "摘要", "综合", "≤3"):
            with self.subTest(token=token):
                self.assertIn(token, p)


class AssociativeScenarioTests(unittest.TestCase):
    """关联性/链式任务：先定位一批条目，再对其中若干做提取/摘要/对比等处理。

    场景来自 tests/agent_assoc_scenarios.json，新增场景无需改动本文件。
    """

    def test_all_assoc_scenarios_drive_and_converge(self):
        for sc in load_assoc_scenarios():
            with self.subTest(id=sc["id"], category=sc.get("category")):
                events, llm, rag, catalog = drive_assoc(sc)
                final = [e for e in events if e["type"] == "final"][-1]
                # 实际工具序列与声明一致
                tools = [e["tool"] for e in _tool_steps(events)]
                expected = [s["tool"] for s in sc["steps"] if "tool" in s]
                self.assertEqual(tools, expected)
                # 正文检索参数彼此不同（无重复调用空转）
                rag_args = [c.args[0] for c in rag.call_args_list]
                self.assertEqual(len(rag_args), len(set(rag_args)))
                # 在预算内收敛，不触发确定性直调（首条不是 route_direct）
                self.assertTrue(final["agent_metrics"]["converged"])
                self.assertLessEqual(final["agent_metrics"]["tool_calls"],
                                     sc.get("max_steps", 4))
                self.assertNotEqual(events[0]["type"], "route_direct")

    def test_chain_preserves_original_intent_on_supplementary_search(self):
        """链式任务以 list 起手、过早 Answer 时，复合补检索必须保留原始问题。"""
        outputs = [
            'Thought: 先列组会\nAction: list_group_files["VLC小组 组会"]',
            'Answer: 有几篇组会',
            'Thought: 补提取写作方法\nAction: rag_qa["VLC小组 组会 论文写作方法 要点"]',
            'Answer: 根据知识库内容可概括为：写作方法要点',
        ]
        events, llm, rag, catalog = _events(outputs,
            "列出VLC小组近三年组会，提取其中的论文写作方法要点并总结")
        # 复合补检索（hybrid 下 list 后过早 Answer）使用原始问题，而非换固定话术
        self.assertEqual(rag.call_args_list[0].args[0],
                         "列出VLC小组近三年组会，提取其中的论文写作方法要点并总结")


class CompoundRouteTests(unittest.TestCase):
    """「定位一批条目 + 处理」型问题不应被折叠成单次确定性 rag_search。"""

    def test_compound_process_task_enters_loop(self):
        q = "A2X小组在水下方向有哪些论文，提取这些论文并对每篇做摘要总结"
        self.assertIsNone(ra.rule_based_route(q))

    def test_simple_content_stays_deterministic(self):
        self.assertEqual(ra.rule_based_route("周报怎么写")["route"], "rag_search")

    def test_pure_listing_stays_file_list(self):
        self.assertEqual(ra.rule_based_route("列出VLC小组设备")["route"], "file_list")

    def test_loop_passes_content_route_hint_to_rag_qa(self):
        """默认开启 AGENT_REUSE_ROUTE_HINT：循环内 rag_qa 子问题复用内容路由 hint。"""
        outputs = [
            'Thought: 检索\nAction: rag_qa["A2X小组 水下 论文"]',
            'Thought: 摘要\nAction: rag_qa["A2X小组 水下 论文一 摘要"]',
            'Answer: 根据知识库内容可概括为：摘要',
        ]
        events, llm, rag, catalog = _events(
            outputs, "A2X小组在水下方向有哪些论文，提取这些论文并对每篇做摘要总结")
        for c in rag.call_args_list:
            self.assertEqual(c.kwargs.get("route_hint"), {"route": "rag_search"})

    def test_loop_route_hint_can_be_disabled(self):
        """关闭后恢复历史逐轮子路由行为（route_hint=None）。"""
        cfg = {"AGENT_MAX_STEPS": 4, "AGENT_REPEAT_ACTION_GUARD": True,
               "AGENT_TOTAL_TIMEOUT_S": 240, "AGENT_MAX_IDLE_STEPS": 2,
               "AGENT_REUSE_ROUTE_HINT": False}
        outputs = [
            'Thought: 检索\nAction: rag_qa["A2X小组 水下 论文"]',
            'Answer: 根据知识库内容可概括为：摘要',
        ]
        events, llm, rag, catalog = _events(
            outputs, "A2X小组在水下方向有哪些论文，提取这些论文并对每篇做摘要总结", cfg=cfg)
        for c in rag.call_args_list:
            self.assertIsNone(c.kwargs.get("route_hint"))


class AgenticGroundingAndRecoveryTests(unittest.TestCase):
    """无依据 / 工具失败时的自愈与诚实收敛（agentic RAG 鲁棒性优化）。"""

    def _cfg(self, max_steps=4):
        return {"AGENT_MAX_STEPS": max_steps, "AGENT_REPEAT_ACTION_GUARD": True,
                "AGENT_TOTAL_TIMEOUT_S": 240, "AGENT_MAX_IDLE_STEPS": 2,
                "AGENT_REUSE_ROUTE_HINT": True}

    def test_tool_failure_recovers_to_fallback(self):
        """内容工具执行失败（ok=False）：注入恢复提示，最终无依据则诚实收敛。"""
        dispatch, _rag, _catalog = make_dispatch(rag_ok=False, rag_answer="")
        outputs = [
            'Thought: 检索\nAction: rag_qa["不存在的主题X"]',
            'Answer: 根据知识库内容可概括为：某结论',
        ]
        events, llm, _r, _c = _drive(outputs, "关于X的什么问题", route="hybrid",
                                    cfg=self._cfg(), dispatch=dispatch)
        final = events[-1]
        self.assertEqual(final["type"], "final")
        self.assertFalse(final["agent_metrics"]["converged"])
        self.assertIn("暂未能形成完整回答", final["content"])
        # 工具失败后注入恢复提示：第二次 LLM 调用发生在失败之后
        self.assertGreaterEqual(llm.invoke.call_count, 2)

    def test_no_evidence_content_triggers_fallback(self):
        """正文检索覆盖度为 none：不标记依据，最终无依据则诚实收敛。"""
        dispatch, _rag, _catalog = make_dispatch(rag_answer="未检索到",
                                                rag_coverage_level="none")
        outputs = [
            'Thought: 检索\nAction: rag_qa["不存在的主题X"]',
            'Answer: 根据知识库内容可概括为：某结论',
        ]
        events, llm, _r, _c = _drive(outputs, "关于X的问题", route="hybrid",
                                    cfg=self._cfg(), dispatch=dispatch)
        final = events[-1]
        self.assertFalse(final["agent_metrics"]["converged"])
        self.assertIn("暂未能形成完整回答", final["content"])

    def test_partial_evidence_medium_still_grounded(self):
        """覆盖度 medium（有依据）：不作无依据收敛，正常作答。"""
        dispatch, _rag, _catalog = make_dispatch(rag_answer="部分依据",
                                                rag_coverage_level="medium")
        outputs = [
            'Thought: 检索\nAction: rag_qa["A"]',
            'Answer: 根据知识库内容可概括为：结论',
        ]
        events, llm, _r, _c = _drive(outputs, "关于A的问题", route="hybrid",
                                    cfg=self._cfg(), dispatch=dispatch)
        final = events[-1]
        self.assertTrue(final["agent_metrics"]["converged"])
        self.assertNotIn("暂未能形成完整回答", final["content"])

    def test_tool_failure_then_success_grounds(self):
        """工具先失败后成功：最终获得依据并正常收敛。"""
        first = {"ok": False, "observation": "超时", "catalog_result": None,
                 "coverage_level": None, "elapsed_s": 0.0, "tool_call_id": "tc-1"}
        second = {"ok": True, "observation": "正文依据", "catalog_result": None,
                  "coverage_level": "high", "elapsed_s": 0.0, "tool_call_id": "tc-2"}
        state = {"n": 0}

        def dispatch(name, arg, username, route_hint=None, tool_call_id=""):
            from lab_rag.agent_tools import tool_role
            state["n"] += 1
            return second if state["n"] > 1 else first

        outputs = [
            'Thought: 第一次（失败）\nAction: rag_qa["A"]',
            'Thought: 换角度\nAction: rag_qa["A 更具体"]',
            'Answer: 根据知识库内容可概括为：结论',
        ]
        events, llm, _r, _c = _drive(outputs, "关于A的问题", route="hybrid",
                                    cfg=self._cfg(), dispatch=dispatch)
        final = events[-1]
        self.assertTrue(final["agent_metrics"]["converged"])
        self.assertNotIn("暂未能形成完整回答", final["content"])
        self.assertEqual(final["agent_metrics"]["tool_calls"], 2)


class DeeperChainTests(unittest.TestCase):
    """更深链式能力：默认步数上限提高到 6，可支撑 5 次工具调用 + 1 次作答。"""

    def test_five_tool_chain_converges(self):
        outputs = [
            'Thought: 1\nAction: rag_qa["A"]',
            'Thought: 2\nAction: rag_qa["B"]',
            'Thought: 3\nAction: rag_qa["C"]',
            'Thought: 4\nAction: rag_qa["D"]',
            'Thought: 5\nAction: rag_qa["E"]',
            'Answer: 根据知识库内容可概括为：五跳归纳结论',
        ]
        # 不依赖旧默认 4，显式给足步数上限以验证更深链路能力
        cfg = {"AGENT_MAX_STEPS": 6, "AGENT_REPEAT_ACTION_GUARD": True,
               "AGENT_TOTAL_TIMEOUT_S": 240, "AGENT_MAX_IDLE_STEPS": 2}
        events, llm, rag, catalog = _drive(outputs, "深度链式问题", route="hybrid", cfg=cfg)
        final = events[-1]
        self.assertTrue(final["agent_metrics"]["converged"])
        self.assertEqual(final["agent_metrics"]["tool_calls"], 5)


class HistoryCompressionTests(unittest.TestCase):
    """上下文压缩与历史管理：支持更长对话上下文（RAG 结果爆炸 / 更多轮次）。"""

    def test_initial_messages_keeps_short_history_intact(self):
        hist = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        msgs = ae._initial_messages("新", hist, {"AGENT_HISTORY_MAX_CHARS": 3000})
        contents = [m["content"] for m in msgs]
        self.assertIn("q", contents)
        self.assertIn("a", contents)
        self.assertFalse(any(c.startswith("[早期历史压缩]") for c in contents))

    def test_initial_messages_folds_old_turns_when_over_char_budget(self):
        hist = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
            {"role": "user", "content": "z" * 2000},
        ]
        msgs = ae._initial_messages(
            "新", hist, {"AGENT_HISTORY_MAX_CHARS": 1000, "AGENT_HISTORY_MAX_TURNS": 20})
        folded = [m for m in msgs if m["content"].startswith("[早期历史压缩]")]
        self.assertTrue(folded, "超字符预算的最旧历史轮次应被折叠")
        self.assertEqual(msgs[-1]["content"], "新")

    def test_truncate_history_folds_old_turns(self):
        # 用足量中文长文本，确保真实触发「超预算旧轮折叠」而非被递归全丢弃
        hist = [{"role": "user", "content": "问题%d" % i} for i in range(5)] + \
               [{"role": "assistant", "content": "内容" * 600} for _ in range(5)]
        out = ra.truncate_history(hist, max_tokens=1500, cfg={"HISTORY_COMPACT_ENABLED": True})
        folded = [t for t in out if "用户曾问" in t["content"] or "助手曾答" in t["content"]]
        self.assertTrue(folded, "超预算旧轮应折叠为摘要而非直接丢弃")

    def test_truncate_history_disabled_drops_old(self):
        hist = [{"role": "user", "content": "q" * 200} for _ in range(6)]
        out = ra.truncate_history(hist, max_tokens=20, cfg={"HISTORY_COMPACT_ENABLED": False})
        self.assertLess(len(out), len(hist))
        self.assertFalse(any("曾问" in t["content"] or "曾答" in t["content"] for t in out))

    def test_long_history_still_converges(self):
        hist = []
        for i in range(12):
            hist.append({"role": "user", "content": ("第%d轮问题：" % i) + "内容" * 300})
            hist.append({"role": "assistant", "content": ("第%d轮回答：" % i) + "内容" * 300})
        cfg = {"AGENT_MAX_STEPS": 4, "AGENT_REPEAT_ACTION_GUARD": True,
               "AGENT_TOTAL_TIMEOUT_S": 240, "AGENT_MAX_IDLE_STEPS": 2,
               "AGENT_REUSE_ROUTE_HINT": True, "AGENT_HISTORY_MAX_CHARS": 3000}
        outputs = ['Thought: 检索\nAction: rag_qa["某主题"]',
                   'Answer: 根据知识库内容可概括为：答案']
        events, llm, rag, catalog = _drive(
            outputs, "新问题", route="hybrid", cfg=cfg, chat_history=hist)
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["agent_metrics"]["converged"])

    def test_context_budget_guard_folds_old_messages(self):
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "A" * 5000},
            {"role": "assistant", "content": "B" * 5000},
            {"role": "user", "content": "C" * 5000},
        ]
        ae._trim_agent_context(msgs, budget=8000, protected_indices=set())
        folded = [m for m in msgs if m["content"].startswith("[已折叠]")]
        self.assertTrue(folded, "总上下文超预算时最旧消息应被折叠")
        # 系统消息不被折叠
        self.assertEqual(msgs[0]["content"], "S")


class ReadFileToolTests(unittest.TestCase):
    """read_file 工具：验证「先 file_list 定位，再 read_file 精读」链路可用。"""

    def test_read_file_after_list_used_and_converged(self):
        outputs = [
            'Thought: 列设备\nAction: list_group_files["VLC小组"]',
            'Thought: 读手册\nAction: read_file["设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/MSOX3054A-数据手册.md"]',
            'Answer: 根据知识库内容可概括为：使用方法',
        ]
        events, llm, rag, catalog = _drive(outputs, "VLC小组 MSO 怎么用", route="hybrid")
        tools = [e["tool"] for e in events if e["type"] == "step_tool"]
        self.assertIn("read_file", tools)
        self.assertTrue(events[-1]["agent_metrics"]["converged"])
        self.assertTrue(events[-1]["agent_metrics"]["content_grounded"])

    def test_safe_join_blocks_traversal(self):
        from lab_rag.agent_tools import _safe_join
        root = "/tmp/kb_root"
        self.assertIsNone(_safe_join(root, "../../etc/passwd"))
        self.assertIsNone(_safe_join(root, "/etc/passwd"))
        self.assertEqual(_safe_join(root, "sub/file.md"), os.path.abspath("/tmp/kb_root/sub/file.md"))

    def test_read_markdown_truncates(self):
        from lab_rag.agent_tools import _read_markdown_text
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("x" * 5000)
            path = f.name
        try:
            text = _read_markdown_text(path, 100)
            self.assertIn("已截断", text)
            self.assertLessEqual(len(text), 200)
        finally:
            os.unlink(path)


class ContextOverflowGuardTests(unittest.TestCase):
    """复现 2026-11-09 的 400「maximum context length」异常并验证修复。

    事故配置：VLLM_MAX_MODEL_LEN=6000、RESPONSE_MAX_TOKENS=2048，输入侧仅 3952 tokens；
    系统提示词约占 2546，list_group_files 返回大清单后 messages 达 4244 → 400 报错，
    旧逻辑还误报「模型服务暂时不可用」并丢弃整轮回答。
    """

    REAL_CFG = {
        "VLLM_MAX_MODEL_LEN": 6000,
        "RESPONSE_MAX_TOKENS": 2048,
        "AGENT_CONTEXT_SAFETY_TOKENS": 256,
        "AGENT_MAX_STEPS": 4,
        "AGENT_MAX_IDLE_STEPS": 2,
        "AGENT_TOTAL_TIMEOUT_S": 240,
        "AGENT_OBSERVATION_MAX_CHARS": 2000,
        "AGENT_KEEP_RECENT_OBSERVATIONS": 2,
    }

    def _overflow_exc(self):
        return RuntimeError(
            "Error code: 400 - {'message': \"This model's maximum context length is 6000 "
            "tokens. However, you requested 6292 tokens (4244 in the messages, 2048 in "
            "the completion). Please reduce the length of the messages or completion.\"}")

    def test_budget_derived_from_model_window(self):
        # 6000 - 2048 - 256 = 3696，明确小于 3952，为分词误差留余量
        self.assertEqual(ae._agent_input_token_budget(self.REAL_CFG), 3696)
        self.assertEqual(ae._agent_input_token_budget(
            dict(self.REAL_CFG, AGENT_CONTEXT_MAX_TOKENS=2000)), 2000)
        # 离线 mock 精简 cfg 不启用 token 护栏，避免污染字符压缩对照实验
        self.assertEqual(ae._agent_input_token_budget({"AGENT_MAX_STEPS": 4}), 0)

    def test_context_overflow_is_recognized(self):
        self.assertTrue(ae._is_context_overflow(self._overflow_exc()))
        self.assertFalse(ae._is_context_overflow(RuntimeError("boom")))

    def test_clip_to_tokens_handles_chinese(self):
        text = "中文设备清单" * 500
        out = ae._clip_to_tokens(text, 100)
        self.assertLess(len(out), len(text))
        self.assertLessEqual(ae._message_tokens({"content": out}), 160)

    def test_token_guard_bounds_messages(self):
        msgs = [{"role": "system", "content": ae.REACT_SYSTEM_PROMPT},
                {"role": "user", "content": "VLC小组有哪些设备"},
                {"role": "assistant", "content": 'Action: list_group_files["VLC小组"]'},
                {"role": "user", "content": "Observation: " + "设备手册路径" * 1200}]
        self.assertGreater(ae._messages_tokens(msgs), 3696)
        ae._fit_agent_context_tokens(msgs, 3696, {3})
        self.assertLessEqual(ae._messages_tokens(msgs), 3696)
        # 系统提示词与最后一条（当前问题/最新观察）必须保留
        self.assertIn("ReAct", msgs[0]["content"])
        self.assertIn("Observation", msgs[-1]["content"])

    def test_small_window_switches_to_compact_prompt(self):
        cfg = dict(self.REAL_CFG, VLLM_MAX_MODEL_LEN=4096)
        msgs = ae._initial_messages("周报怎么写", None, cfg)
        self.assertEqual(msgs[0]["content"], ae.REACT_SYSTEM_PROMPT_COMPACT)

    def test_overflow_error_retries_and_converges(self):
        """模拟 step=2 抛 400：紧急折叠后重试成功，不得误报服务不可用。"""
        from types import SimpleNamespace
        long_obs = "目录：\n" + "\n".join(
            f"- 设备操作指南/VLC小组/设备{i:03d}/Manuals/manual_{i:03d}.md"
            for i in range(200))
        call = {"n": 0}

        def invoke(messages):
            call["n"] += 1
            if call["n"] == 2:
                raise self._overflow_exc()
            out = ('Thought: 列目录\nAction: list_group_files["VLC小组"]'
                   if call["n"] == 1 else 'Answer: 根据知识库内容可概括为：设备清单')
            return SimpleNamespace(content=out)

        llm = Mock()
        llm.invoke.side_effect = invoke
        dispatch, _rag, _cat = make_dispatch(observation=long_obs)
        with patch.object(ae, "route_query", return_value={"route": "file_list", "target": "VLC小组"}), \
                patch.object(ae, "_agent_config", return_value=self.REAL_CFG), \
                patch.object(ae, "build_qwen_llm", return_value=llm), \
                patch.object(ae, "dispatch_tool", dispatch), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("VLC小组有哪些设备", username="tester"))
        self.assertFalse([e for e in events if e["type"] == "error"],
                         "上下文超长不得误报「模型服务暂时不可用」")
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["agent_metrics"]["converged"])
        self.assertGreaterEqual(call["n"], 3, "应发生折叠重试")

    def test_overflow_retry_exhausted_falls_back_honestly(self):
        """折叠重试仍失败：诚实收敛并写入历史，而非 error 丢弃整轮。"""
        llm = Mock()
        llm.invoke.side_effect = self._overflow_exc()
        dispatch, _rag, _cat = make_dispatch()
        with patch.object(ae, "route_query", return_value={"route": "file_list", "target": "VLC小组"}), \
                patch.object(ae, "_agent_config", return_value=self.REAL_CFG), \
                patch.object(ae, "build_qwen_llm", return_value=llm), \
                patch.object(ae, "dispatch_tool", dispatch), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("VLC小组有哪些设备", username="tester"))
        self.assertEqual(events[-1]["type"], "final")
        self.assertFalse(events[-1]["agent_metrics"]["converged"])
        self.assertNotIn("模型服务暂时不可用", events[-1]["content"])
        self.assertFalse([e for e in events if e["type"] == "error"])

    def test_non_context_llm_error_still_emits_error(self):
        """非上下文错误保持原契约：报 error，不吞异常。"""
        llm = Mock()
        llm.invoke.side_effect = RuntimeError("boom")
        with patch.object(ae, "route_query", return_value={"route": "file_list", "target": "VLC小组"}), \
                patch.object(ae, "_agent_config", return_value=self.REAL_CFG), \
                patch.object(ae, "build_qwen_llm", return_value=llm), \
                patch.object(ae, "dispatch_tool", make_dispatch()[0]), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("VLC小组有哪些设备", username="tester"))
        self.assertEqual([e["type"] for e in events][-1], "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
