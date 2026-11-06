"""离线行为回归：执行源码函数，隔离模型/GPU/Linux 依赖，不代替服务集成测试。"""
import ast
import copy
import hashlib
import json
import os
import re
import time
import traceback
import unittest
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import __future__

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab_rag.document_structure import (
    evidence_window, preserve_html_tables, split_table_row, strip_ingest_prefix,
    has_structured_evidence, markdown_header_matches,
)


def load_functions(module, names, **overrides):
    path = ROOT / "src" / "lab_rag" / (module + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError("Missing source function")
    namespace = dict(globals())
    namespace.update(debug_log=lambda *args: None, Timer=lambda *args: nullcontext(), Document=Document)
    namespace.update(overrides)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions("rag_agent", {
            "retrieval_doc_key", "fuse_ranked_lists", "reciprocal_rank_fusion",
            "retrieve_multi_query", "dedup_docs", "rerank_docs", "build_context",
            "build_citations", "assess_coverage", "get_system_prompt_by_coverage",
            "ask_stream", "sanitize_generated_answer", "should_skip_rewrite",
        }, clean_retrieval_display_text=lambda text: text.strip())

    def doc(self, text, path="guide.md", **metadata):
        return Document(text, {"rel_path": path, "file_name": "guide.md", **metadata})

    def test_missing_channel_never_contributes_rrf_score(self):
        a, b = self.doc("A"), self.doc("B")
        dense = [a] + [self.doc(str(i)) for i in range(18)] + [b]
        result = self.ns["reciprocal_rank_fusion"](dense, [b])
        self.assertIs(result[0], b)

    def test_identity_keeps_same_filename_in_different_directories(self):
        a, b = self.doc("same", "one/guide.md"), self.doc("same", "two/guide.md")
        self.assertEqual(self.ns["dedup_docs"]([a, b, a]), [a, b])

    def test_identity_keeps_different_text_after_long_common_prefix(self):
        a, b = self.doc("x" * 1000 + "A"), self.doc("x" * 1000 + "B")
        self.assertEqual(self.ns["dedup_docs"]([a, b]), [a, b])

    def test_fusion_ties_are_stable_and_duplicates_vote_once(self):
        a, b = self.doc("A"), self.doc("B")
        for _ in range(10):
            self.assertEqual(self.ns["fuse_ranked_lists"]([[a, a], [b]], [1, 1], 60), [a, b])

    def test_multiquery_merges_ranks_not_completion_order(self):
        a, b, common = self.doc("A"), self.doc("B"), self.doc("shared")
        vectorstore = Mock()
        vectorstore.similarity_search.side_effect = lambda q, k: [a, common] if q == "first" else [b, common]
        self.assertEqual(self.ns["retrieve_multi_query"](vectorstore, ["first", "second", "first"], 2),
                         [common, a, b])
        self.assertEqual(vectorstore.similarity_search.call_count, 2)

    def test_low_rerank_scores_do_not_force_one_result(self):
        reranker = Mock()
        reranker.predict.return_value = [0.01, 0.02]
        self.assertEqual(self.ns["rerank_docs"](reranker, "query", [self.doc("A"), self.doc("B")], {}), [])

    def test_reranker_failure_does_not_return_unchecked_evidence(self):
        reranker = Mock()
        reranker.predict.side_effect = RuntimeError("offline")
        self.assertEqual(self.ns["rerank_docs"](reranker, "q", [self.doc("A")], {}), [])

    def test_reranker_keeps_supported_results(self):
        reranker = Mock()
        reranker.predict.return_value = [0.1, 0.8]
        a, b = self.doc("A"), self.doc("B")
        self.assertEqual(self.ns["rerank_docs"](reranker, "q", [a, b], {}), [b])

    def test_context_has_strict_budget_including_first_doc_and_separators(self):
        docs = [self.doc("a" * 200), self.doc("b" * 200)]
        for budget in (0, 10, 100, 280, 500):
            context, sources = self.ns["build_context"](docs, budget)
            self.assertLessEqual(len(context), budget)
            for index, source in sources.items():
                self.assertIn(f"[来源{index}]", context)
                self.assertIn(source["content"], context)

    def test_generated_summary_is_not_answer_evidence(self):
        doc = self.doc("原始记录", parent_summary="模型虚构内容")
        context, _ = self.ns["build_context"]([doc])
        self.assertNotIn("模型虚构内容", context)
        self.assertEqual(self.ns["build_citations"]([doc])[0]["summary"], "")

    def pipeline(self, docs, budget=100, route="rag_search"):
        config = {"INITIAL_RETRIEVAL_K": 4, "FINAL_TOP_K": 3, "RRF_K": 60,
                  "MAX_CONTEXT_CHARS": budget, "ENABLE_HYBRID_SEARCH": False}
        answer = Mock(return_value=iter([SimpleNamespace(content="证据回答")]))
        self.ns.update(
            _user_histories={}, load_history=lambda user: [], get_user_chat_history=lambda user: [],
            truncate_history=lambda history, limit: history, format_chat_history=lambda history: "",
            get_runtime=lambda: dict(config=config, llm=None, vectorstore=None, reranker=None, client=None),
            route_query=lambda llm, q: dict(route=route, target=q),
            _retrieval_cache=SimpleNamespace(get=lambda key: None, set=lambda key, value: None),
            retrieve_multi_query=lambda *args, **kwargs: docs,
            rerank_docs=lambda *args, **kwargs: docs, expand_to_parent_docs=lambda *args, **kwargs: docs,
            answer_stream=answer,
            list_catalog_entries=lambda target: {"files": ["item"]}, catalog_result_count=lambda result: 1,
            build_file_context=lambda result: "目录记录\n" * 300,
            build_catalog_answer_section=lambda result: "目录清单",
            resolve_catalog_target=lambda question, target: target,
            extract_catalog_content_intent=lambda question: "规范",
            build_catalog_retrieval_queries=lambda *args: ["路径 规范"],
        )
        events = list(self.ns["ask_stream"]("检索", username="", persist_history=False))
        self.assertFalse([event for event in events if event["type"] == "error"], events)
        return events, answer

    def test_no_evidence_pipeline_never_invokes_answer_model(self):
        events, answer = self.pipeline([])
        answer.assert_not_called()
        self.assertEqual(events[-1]["coverage"]["level"], "none")
        self.assertEqual(events[-1]["source_map"], {})

    def test_pipeline_citations_and_display_match_truncated_context(self):
        docs = [self.doc("a" * 200, "first.md", rerank_score=0.8),
                self.doc("b" * 200, "second.md", rerank_score=0.8)]
        events, answer = self.pipeline(docs)
        final = events[-1]
        retrieval = next(e for e in events if e.get("stage") == "retrieval")
        self.assertEqual(list(final["source_map"]), [1])
        self.assertEqual([c["source"] for c in final["citations"]], ["first.md"])
        self.assertEqual(retrieval["retrievals"][0]["content"], final["source_map"][1]["content"])
        self.assertLessEqual(len(answer.call_args.args[2]), 100)

    def test_hybrid_catalog_does_not_raise_body_coverage(self):
        events, answer = self.pipeline([], budget=150, route="hybrid")
        self.assertEqual(events[-1]["coverage"]["level"], "none")
        self.assertEqual(events[-1]["citations"], [])
        self.assertLessEqual(len(answer.call_args.args[2]), 150)

    def test_hybrid_total_context_budget(self):
        events, answer = self.pipeline([self.doc("证据" * 200, rerank_score=0.8)], budget=300, route="hybrid")
        self.assertTrue(events[-1]["source_map"])
        self.assertLessEqual(len(answer.call_args.args[2]), 300)

    def test_hyde_disabled_does_not_add_generated_hypotheses(self):
        chain = Mock()
        chain.invoke.return_value = SimpleNamespace(content=json.dumps({
            "standalone_question": "仪器要求", "keywords": ["仪器"],
            "expanded_queries": ["仪器操作要求"], "hypothetical_docs": "虚构答案" * 20,
        }))
        class Prompt:
            def __or__(self, llm):
                return chain
        ns = load_functions("rag_agent", {"rewrite_question"},
                            ChatPromptTemplate=SimpleNamespace(from_template=lambda text: Prompt()),
                            safe_parse_json=json.loads)
        result = ns["rewrite_question"](None, "仪器有什么使用要求", {"ENABLE_HYDE": False})
        self.assertFalse(any("虚构答案" in q for q in result["expanded_queries"]))
        self.assertIn("未启用", chain.invoke.call_args.args[0]["hyde_instruction"])


class TableTests(unittest.TestCase):
    def test_generic_tables_preserve_only_supplied_cells(self):
        ns = load_functions("ingest", {"clean_text", "mark_suspect_flattened_tables", "enrich_markdown_tables"})
        with patch.dict(os.environ, {"ENABLE_TABLE_SEMANTIC_ENRICHMENT": "true"}):
            text = "| 项目 | 限值 |\n| --- | --- |\n| 温度 | 25 |"
            output = ns["clean_text"](text)
        self.assertIn(text, output)
        self.assertIn("项目=温度；限值=25", output)
        flattened = "变量格式 变量类型 英文字母 罗马字母 标量 集合 向量、矩阵 正体+粗体"
        self.assertEqual(ns["clean_text"](flattened), flattened)

    def test_html_table_keeps_cells_and_empty_values(self):
        result = preserve_html_tables('<table><tr><th>参数</th><th>限值</th></tr>'
                                      '<tr><td>温度</td><td>25 &amp; 30</td></tr>'
                                      '<tr><td>压力</td><td></td></tr></table>')
        self.assertIn('| 参数 | 限值 |', result)
        self.assertIn('| 温度 | 25 & 30 |', result)
        self.assertIn('| 压力 |  |', result)

    def test_td_first_row_is_not_guessed_as_header(self):
        result = preserve_html_tables('<table><tr><td>值A</td><td>值B</td></tr></table>')
        self.assertIn('| 列1 | 列2 |', result)
        self.assertIn('| 值A | 值B |', result)

    def test_html_cell_retains_emphasis_scripts_and_image_uncertainty(self):
        result = preserve_html_tables('<table><tr><td><b>K</b></td><td><i>k</i></td>'
                                      '<td>m<sup>2</sup></td><td>x<sub>n</sub></td><td><img src="x.png"></td></tr></table>')
        for value in ('**K**', '*k*', 'm^{2}', 'x_{n}', '[图片]'):
            self.assertIn(value, result)

    def test_merged_table_records_literal_spans_without_filling_blanks(self):
        result = preserve_html_tables('<table><tr><td rowspan="2">型号</td><td>A</td></tr>'
                                      '<tr><td>B</td></tr></table>')
        self.assertIn('rowspan=2', result)
        self.assertIn('行2 单元格1', result)
        self.assertEqual(result.count('型号'), 1)

    def test_malformed_html_table_marks_uncertainty(self):
        result = preserve_html_tables('<table><tr><td>破损</table>')
        self.assertIn('[表格结构疑似丢失]', result)

    def test_html_table_and_equations_survive_cleaning(self):
        ns = load_functions('ingest', {'clean_text', 'mark_suspect_flattened_tables', 'enrich_markdown_tables'})
        text = '要求 x < y 且 y > z。\n<table><tr><th>字段</th><th>值</th></tr><tr><td>A|B</td><td>2</td></tr></table>'
        cleaned = ns['clean_text'](text)
        self.assertIn('x < y 且 y > z', cleaned)
        self.assertIn(r'字段=A\|B；值=2', cleaned)
        self.assertEqual(cleaned, ns['clean_text'](cleaned))


class LabDocumentTests(unittest.TestCase):
    def setUp(self):
        self.ingest = load_functions('ingest', {'analyze_bad_chunk_reason', 'split_by_markdown_headers'})
        self.rag = load_functions('rag_agent', {'expand_to_parent_docs', 'retrieval_doc_key', 'sanitize_summary',
                                               'rerank_docs', 'build_context'},
                                  clean_retrieval_display_text=lambda text: text.strip())

    def test_short_complete_clauses_and_parameter_tables_are_kept(self):
        cases = ['- 实验结束后应先关闭电源，再断开信号连接。',
                 '拍摄时应防止手抖，请尽量使用三脚架。',
                 '| 参数 | 值 |\n| --- | --- |\n| 温度 | 25 °C |']
        for text in cases:
            self.assertEqual(self.ingest['analyze_bad_chunk_reason'](text, {}), 'good', text)

    def test_long_prefix_does_not_make_empty_chunk_evidence(self):
        prefix = '【文档：' + '长路径/' * 100 + ' > 段落摘要：摘要】\n'
        self.assertEqual(self.ingest['analyze_bad_chunk_reason'](prefix, {}), 'empty')
        for text in ['# 空章节', '[图片]', '1. ................... 20']:
            self.assertNotEqual(self.ingest['analyze_bad_chunk_reason'](prefix + text, {}), 'good')

    def test_six_heading_levels_and_code_fences(self):
        text = '# 总览\n正文。\n```python\n# 代码注释\n```\n#### 配置\n具体说明。\n###### 注意事项\n操作说明。'
        docs = self.ingest['split_by_markdown_headers'](Document(text))
        self.assertEqual([d.metadata['header_level'] for d in docs], [1, 4, 6])
        self.assertEqual(docs[-1].metadata['header_path'], '总览 > 配置 > 注意事项')
        self.assertIn('# 代码注释', docs[0].page_content)

    def test_anchor_survives_parent_and_context_truncation(self):
        anchor = '实际命中的安全条款必须保留。'
        child = Document(anchor, {'parent_id': 'p', 'rel_path': '手册.md',
                                 'parent_content': '背景。' * 800 + anchor + '其他说明。' * 100,
                                 'rerank_score': 0.9})
        parents = self.rag['expand_to_parent_docs']([child], {'MAX_CONTEXT_CHARS': 200})
        self.assertIn(anchor, parents[0].page_content)
        context, _ = self.rag['build_context'](parents, 120)
        self.assertIn(anchor, context)
        self.assertLessEqual(len(context), 120)

    def test_legacy_child_without_parent_id_is_not_silently_dropped(self):
        child = Document('有效的原文片段', {'rel_path': '说明.md', 'rerank_score': 0.8})
        parents = self.rag['expand_to_parent_docs']([child], {'MAX_CONTEXT_CHARS': 200})
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].page_content, child.page_content)

    def test_reranker_sees_document_scope_but_not_generated_summary(self):
        reranker = Mock()
        reranker.predict.return_value = [0.8]
        doc = Document('开机前检查连接。', {'rel_path': '设备/示波器/手册.md', 'section_title': '安全',
                                        'parent_summary': '虚构内容'})
        self.rag['rerank_docs'](reranker, '使用要求', [doc], {})
        pair = reranker.predict.call_args.args[0][0]
        self.assertIn('设备/示波器/手册.md', pair[1])
        self.assertIn('安全', pair[1])
        self.assertNotIn('虚构内容', pair[1])


class AgentTests(unittest.TestCase):
    def run_agent(self, outputs, fail_first=False, streaming=True):
        llm, tool = Mock(), Mock()
        llm.invoke.side_effect = [SimpleNamespace(content=value) for value in outputs]
        tool.args = {"question": {}}
        tool.invoke.return_value = "原文依据"
        rag = Mock(return_value={"answer": "原文依据"})
        if fail_first:
            rag.side_effect = [RuntimeError("temporary"), {"answer": "原文依据"}]
            tool.invoke.side_effect = [RuntimeError("temporary"), "原文依据"]
        ns = load_functions("agent_entry", {"ask_agent_stream", "run_react_once", "expected_agent_tool",
                                               "parse_action", "parse_answer", "_initial_messages"},
                            REACT_SYSTEM_PROMPT="test", rule_based_route=lambda q: {"route": "rag_search"},
                            build_qwen_llm=lambda: llm, TOOLS={"rag_qa": tool}, ask_rag=rag,
                            sanitize_generated_answer=lambda value: value,
                            append_user_chat_history=lambda *args: None, _logger=Mock(),
                            agent_tool_context=lambda user: nullcontext())
        result = list(ns["ask_agent_stream"]("操作要求")) if streaming else ns["run_react_once"](llm, "操作要求")
        return result, rag if streaming else tool.invoke

    def test_agent_can_make_a_second_distinct_call(self):
        for streaming in (True, False):
            result, tool = self.run_agent(['Action: rag_qa["首次"]', 'Action: rag_qa["补充"]',
                                           'Answer: 完成'], streaming=streaming)
            self.assertEqual(tool.call_count, 2)
            self.assertEqual(result[-1]["content"] if streaming else result, "完成")

    def test_agent_rejects_duplicate_successful_calls(self):
        for streaming in (True, False):
            _, tool = self.run_agent(['Action: rag_qa["首次"]', 'Action: rag_qa["首次"]',
                                      'Answer: 完成'], streaming=streaming)
            self.assertEqual(tool.call_count, 1)

    def test_agent_can_retry_failed_call(self):
        for streaming in (True, False):
            _, tool = self.run_agent(['Action: rag_qa["首次"]', 'Action: rag_qa["首次"]',
                                      'Answer: 完成'], fail_first=True, streaming=streaming)
            self.assertEqual(tool.call_count, 2)


if __name__ == "__main__":
    unittest.main()
