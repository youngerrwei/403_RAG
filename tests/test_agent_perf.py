"""Agent 多轮推理性能保护的离线单测（不依赖模型 / Qdrant）。

覆盖：上下文有界压缩、早期 Observation 折叠、SSE 事件保留完整原文、
空转不占工具步数预算、LLM 硬上限防死循环、预算耗尽优雅收敛、结构化指标。
"""

import unittest
from unittest.mock import Mock, patch

from lab_rag import agent_entry as ae
from agent_test_helpers import _llm, make_dispatch


CFG = {
    "AGENT_MAX_STEPS": 4,
    "AGENT_MAX_IDLE_STEPS": 2,
    "AGENT_REPEAT_ACTION_GUARD": True,
    "AGENT_OBSERVATION_MAX_CHARS": 2000,
    "AGENT_KEEP_RECENT_OBSERVATIONS": 2,
    "AGENT_FINAL_CONVERGE_ROUND": True,
    "AGENT_TOTAL_TIMEOUT_S": 240,
}


def _drive(outputs, question="复合意图问题：有哪些设备并说明安装步骤",
           route="hybrid", cfg=None, observation="目录：A设备/B设备"):
    llm = _llm(outputs)
    dispatch, rag, _catalog = make_dispatch(observation=observation)
    with patch.object(ae, "route_query", return_value={"route": route, "target": "VLC小组"}), \
            patch.object(ae, "_agent_config", return_value=cfg or CFG), \
            patch.object(ae, "build_qwen_llm", return_value=llm), \
            patch.object(ae, "dispatch_tool", dispatch), \
            patch.object(ae, "append_user_chat_history"):
        events = list(ae.ask_agent_stream(question, username="tester"))
    return events, llm, rag


class ObservationCompactionTests(unittest.TestCase):

    def test_short_observation_is_untouched(self):
        text = "目录：A/B"
        self.assertEqual(ae._compact_observation(text, CFG), text)

    def test_long_observation_is_bounded(self):
        text = "行内容\n" * 3000
        out = ae._compact_observation(text, CFG)
        self.assertLess(len(out), len(text))
        self.assertIn("中间省略", out)
        self.assertIn("完整结果已在工具事件中展示", out)

    def test_zero_limit_disables_compaction(self):
        cfg = dict(CFG, AGENT_OBSERVATION_MAX_CHARS=0)
        text = "行内容\n" * 3000
        self.assertEqual(ae._compact_observation(text, cfg), text)

    def test_configurable_limit_is_respected(self):
        cfg = dict(CFG, AGENT_OBSERVATION_MAX_CHARS=200)
        out = ae._compact_observation("x" * 1000, cfg)
        self.assertLessEqual(len(out), 300)  # 含省略标记

    def test_history_keeps_recent_observations_and_folds_older(self):
        messages = [{"role": "system", "content": "SYS"},
                    {"role": "user", "content": "QUESTION"}]
        idx = []
        for i in range(4):
            messages.append({"role": "user", "content": f"Observation: 第{i}轮结果\n继续"})
            idx.append(len(messages) - 1)
        ae._compact_agent_history(messages, idx, CFG)
        self.assertTrue(messages[idx[0]]["content"].startswith("[已折叠]"))
        self.assertTrue(messages[idx[1]]["content"].startswith("[已折叠]"))
        self.assertIn("第2轮结果", messages[idx[2]]["content"])
        self.assertIn("第3轮结果", messages[idx[3]]["content"])
        self.assertEqual(messages[0]["content"], "SYS")
        self.assertEqual(messages[1]["content"], "QUESTION")

    def test_fold_is_idempotent(self):
        messages = [{"role": "user", "content": "Observation: 旧结果\n继续"}]
        ae._compact_agent_history(messages, [0], dict(CFG, AGENT_KEEP_RECENT_OBSERVATIONS=0))
        once = messages[0]["content"]
        ae._compact_agent_history(messages, [0], dict(CFG, AGENT_KEEP_RECENT_OBSERVATIONS=0))
        self.assertEqual(messages[0]["content"], once)


class SseObservationFidelityTests(unittest.TestCase):
    """压缩只作用于喂给 LLM 的 messages，SSE 事件必须保留完整原文。"""

    def test_step_tool_observation_keeps_full_text(self):
        long_obs = "目录：\n" + "\n".join(f"- 设备{i:03d}" for i in range(400))
        outputs = [
            'Thought: 列目录\nAction: list_group_files["VLC小组"]',
            "Answer: 根据知识库内容可概括为：设备清单",
        ]
        events, llm, rag = _drive(outputs, route="file_list", observation=long_obs)
        tool_events = [e for e in events if e["type"] == "step_tool"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["observation"], long_obs)

    def test_context_peak_stays_bounded_with_long_observation(self):
        long_obs = "目录：\n" + "\n".join(f"- 设备{i:03d}" for i in range(400))
        outputs = [
            'Thought: 第一跳\nAction: list_group_files["VLC小组"]',
            'Thought: 第二跳\nAction: list_group_files["HOLOEYE LETO-3 Manuals"]',
            "Answer: 根据知识库内容可概括为：结论",
        ]
        events, llm, rag = _drive(outputs, route="file_list", observation=long_obs)
        compacted_peak = events[-1]["agent_metrics"]["context_chars_peak"]
        # 对照：关闭压缩后，同样的两轮长 Observation 会原样累加
        no_compact_cfg = dict(CFG, AGENT_OBSERVATION_MAX_CHARS=0,
                              AGENT_KEEP_RECENT_OBSERVATIONS=99)
        events_raw, _, _ = _drive(outputs, route="file_list",
                                  cfg=no_compact_cfg, observation=long_obs)
        raw_peak = events_raw[-1]["agent_metrics"]["context_chars_peak"]
        self.assertGreater(raw_peak, compacted_peak)
        self.assertLess(compacted_peak, raw_peak * 0.8)


class IdleBudgetTests(unittest.TestCase):
    """空转不占工具步数预算，但受 LLM 硬上限保护。"""

    def test_repeat_suppression_does_not_consume_tool_budget(self):
        cfg = dict(CFG, AGENT_MAX_STEPS=2, AGENT_MAX_IDLE_STEPS=1)
        outputs = [
            'Thought: 第一次\nAction: rag_qa["A"]',
            'Thought: 重复\nAction: rag_qa["A"]',
            'Thought: 换角度\nAction: rag_qa["B"]',
            "Answer: 根据知识库内容可概括为：差异",
        ]
        events, llm, rag = _drive(outputs, cfg=cfg)
        metrics = events[-1]["agent_metrics"]
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["idle_steps"], 1)
        self.assertTrue(metrics["converged"])
        self.assertEqual([c.args[0] for c in rag.call_args_list], ["A", "B"])

    def test_idle_budget_bound_prevents_infinite_loop(self):
        cfg = dict(CFG, AGENT_MAX_STEPS=2, AGENT_MAX_IDLE_STEPS=1)
        outputs = ['Action: rag_qa["A"]'] * 10
        events, llm, rag = _drive(outputs, cfg=cfg)
        self.assertEqual(events[-1]["type"], "final")
        self.assertLessEqual(llm.invoke.call_count, 2 + 1 + 1)
        self.assertFalse(events[-1]["agent_metrics"]["converged"])

    def test_llm_calls_have_hard_upper_bound(self):
        outputs = [f'Action: rag_qa["角度{i}"]' for i in range(20)]
        events, llm, rag = _drive(outputs)
        # max_steps=4 + max_idle=2 + 1 = 7
        self.assertLessEqual(llm.invoke.call_count, 7)


class GracefulFallbackTests(unittest.TestCase):

    def test_fallback_lists_attempted_angles_without_raw_observation(self):
        out = ae._agent_fallback_answer("某问题", [("rag_qa", "角度A"), ("list_group_files", "小组")])
        self.assertIn("暂未能形成完整回答", out)
        self.assertIn("角度A", out)
        self.assertIn("某问题", out)

    def test_fallback_without_tool_log_is_honest(self):
        out = ae._agent_fallback_answer("某问题", [])
        self.assertIn("尚未成功执行工具", out)

    def test_final_converge_round_can_be_disabled(self):
        cfg = dict(CFG, AGENT_MAX_STEPS=1, AGENT_FINAL_CONVERGE_ROUND=False)
        outputs = [
            'Thought: 第一跳\nAction: rag_qa["A"]',
            "Answer: 根据知识库内容可概括为：结论",
        ]
        events, llm, rag = _drive(outputs, cfg=cfg)
        # 关闭后不再追加收敛轮，直接进入诚实收敛
        self.assertEqual(llm.invoke.call_count, 1)
        self.assertIn("暂未能形成完整回答", events[-1]["content"])

    def test_budget_exhaustion_answer_is_not_raw_observation_echo(self):
        cfg = dict(CFG, AGENT_MAX_STEPS=1)
        outputs = [
            'Thought: 第一跳\nAction: rag_qa["A"]',
            'Thought: 第二跳\nAction: rag_qa["B"]',
        ]
        events, llm, rag = _drive(outputs, cfg=cfg)
        content = events[-1]["content"]
        self.assertNotIn("最后一次观察结果为", content)
        self.assertIn("暂未能形成完整回答", content)


class AgentMetricsContractTests(unittest.TestCase):

    def test_final_event_exposes_metrics(self):
        outputs = [
            'Thought: 检索\nAction: rag_qa["A"]',
            "Answer: 根据知识库内容可概括为：结论",
        ]
        events, llm, rag = _drive(outputs)
        m = events[-1]["agent_metrics"]
        for key in ("llm_calls", "tool_calls", "idle_steps", "context_chars_peak",
                    "converged", "elapsed_s"):
            with self.subTest(key=key):
                self.assertIn(key, m)
        self.assertTrue(m["converged"])
        self.assertEqual(m["tool_calls"], 1)

    def test_deterministic_route_exposes_metrics(self):
        llm = Mock()
        dispatch, _rag, _catalog = make_dispatch(rag_answer="周报格式正文")
        with patch.object(ae, "route_query", return_value={"route": "rag_search", "target": "周报"}), \
                patch.object(ae, "build_qwen_llm", return_value=llm), \
                patch.object(ae, "dispatch_tool", dispatch), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("周报怎么写", username="tester"))
        m = events[-1]["agent_metrics"]
        self.assertTrue(m["converged"])
        self.assertTrue(m["deterministic_route"])
        self.assertEqual(events[0]["type"], "route_direct")


class UnifiedOrchestratorContractTests(unittest.TestCase):
    """单一编排器 + 统一工具注册表的契约。"""

    def test_no_duplicate_loop_or_legacy_tool_layer(self):
        import inspect
        src = inspect.getsource(ae)
        self.assertNotIn("def run_react_once", src)
        self.assertNotIn("agent_tool_context", src)
        self.assertEqual(src.count("def ask_agent_stream"), 1)
        self.assertEqual(src.count("def ask_agent("), 1)

    def test_prompt_tool_section_is_generated_from_registry(self):
        from lab_rag import agent_tools
        for spec in agent_tools.TOOL_SPECS:
            with self.subTest(tool=spec.name):
                self.assertIn(spec.name, ae.REACT_SYSTEM_PROMPT)
                self.assertIn(spec.description, ae.REACT_SYSTEM_PROMPT)

    def test_unknown_tool_is_degraded_by_registry(self):
        from lab_rag import agent_tools
        result = agent_tools.dispatch_tool("does_not_exist", "x", "tester")
        self.assertFalse(result["ok"])
        self.assertIn("可用工具", result["observation"])

    def test_dispatch_returns_traceable_result(self):
        dispatch, _rag, _catalog = make_dispatch()
        with patch.object(ae, "route_query", return_value={"route": "llm_fallback", "target": "x"}), \
                patch.object(ae, "_agent_config", return_value=CFG), \
                patch.object(ae, "build_qwen_llm", return_value=_llm([
                    'Action: rag_qa["A"]', "Answer: 完成"])), \
                patch.object(ae, "dispatch_tool", dispatch), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("问题", username="tester"))
        steps = [e for e in events if e["type"] == "step_tool"]
        self.assertTrue(all("tool_call_id" in e and "elapsed_s" in e and "ok" in e
                            for e in steps))

    def test_error_event_uses_unified_message_field(self):
        with patch.object(ae, "route_query", return_value={"route": "llm_fallback", "target": "x"}), \
                patch.object(ae, "_agent_config", return_value=CFG), \
                patch.object(ae, "build_qwen_llm", return_value=Mock(
                    invoke=Mock(side_effect=RuntimeError("boom")))), \
                patch.object(ae, "append_user_chat_history"):
            events = list(ae.ask_agent_stream("问题", username="tester"))
        err = [e for e in events if e["type"] == "error"]
        self.assertEqual(len(err), 1)
        self.assertIn("message", err[0])
        self.assertNotIn("content", err[0])

    def test_total_timeout_budget_stops_loop(self):
        cfg = dict(CFG, AGENT_TOTAL_TIMEOUT_S=0.001, AGENT_FINAL_CONVERGE_ROUND=False)
        import time as _time
        original = ae.time.perf_counter
        values = iter([0.0, 0.0, 999.0])
        try:
            ae.time.perf_counter = lambda: next(values, 999.0)
            events, llm, rag = _drive(["Action: rag_qa[\"A\"]"], cfg=cfg)
        finally:
            ae.time.perf_counter = original
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["agent_metrics"]["timed_out"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
