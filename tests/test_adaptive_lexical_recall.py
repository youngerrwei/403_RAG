"""自适应词元兜底召回的离线行为测试（不依赖 Qdrant/GPU/模型）。

验证「按词元形态判定」这一通用机制：只有问题含长纯数字词元、
且该词元在首轮候选身份字段中零命中时才触发，不含任何用例语义。
"""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from lab_rag import rag_agent


def _doc(rel_path, title=""):
    return Document(page_content="正文", metadata={"rel_path": rel_path, "doc_title": title})


class AdaptiveLexicalTokenTests(unittest.TestCase):

    def test_long_digit_tokens_are_extracted_from_question_and_keywords(self):
        tokens = rag_agent._adaptive_lexical_tokens(
            "20230106 期末组会中 VLC小组 汇报了什么？", ["20230106", "期末组会"],
            {"ADAPTIVE_LEXICAL_MIN_DIGITS": 6})
        self.assertEqual(tokens, ["20230106"])

    def test_short_numbers_do_not_trigger(self):
        tokens = rag_agent._adaptive_lexical_tokens(
            "3054A 示波器的第 3 页写了什么", ["12", "3054A"],
            {"ADAPTIVE_LEXICAL_MIN_DIGITS": 6})
        self.assertEqual(tokens, [])

    def test_min_digits_is_configurable(self):
        tokens = rag_agent._adaptive_lexical_tokens(
            "编号 1234", [], {"ADAPTIVE_LEXICAL_MIN_DIGITS": 4})
        self.assertEqual(tokens, ["1234"])


class AdaptiveLexicalDecisionTests(unittest.TestCase):

    cfg = {"ENABLE_ADAPTIVE_LEXICAL_RECALL": True, "ADAPTIVE_LEXICAL_MIN_DIGITS": 6}

    def test_triggers_when_digit_token_present(self):
        docs = [_doc("组会资料/20220121期末组会/小组期末总结/WAI-总结会分享_v1.md")]
        self.assertTrue(rag_agent.needs_adaptive_lexical_recall(
            "20230106 期末组会中 VLC小组 汇报了什么？", ["20230106"], docs, self.cfg))

    def test_still_triggers_when_sibling_document_recalled(self):
        """同日期但非目标文档在候选池里，不代表目标文档已召回；仍须补召回。"""
        docs = [_doc("组会资料/20230106期末组会/组会分享-张楚元.md")]
        self.assertTrue(rag_agent.needs_adaptive_lexical_recall(
            "20230106 期末组会中 VLC小组 汇报了什么？", ["20230106"], docs, self.cfg))

    def test_does_not_trigger_without_digit_token(self):
        docs = [_doc("参考书/水下可见光通信关键技术.md")]
        self.assertFalse(rag_agent.needs_adaptive_lexical_recall(
            "论文写作方法有哪些建议？", ["论文写作方法"], docs, self.cfg))

    def test_disabled_by_config(self):
        cfg = dict(self.cfg, ENABLE_ADAPTIVE_LEXICAL_RECALL=False)
        self.assertFalse(rag_agent.needs_adaptive_lexical_recall(
            "20230106 期末组会", [], [], cfg))

    def test_degrades_when_lexical_module_unavailable(self):
        with patch.object(rag_agent, "_search_lexical_candidates", None):
            self.assertFalse(rag_agent.needs_adaptive_lexical_recall(
                "20230106 期末组会", [], [], self.cfg))

    def test_identity_coverage_includes_multiple_fields(self):
        d = Document(page_content="正文", metadata={
            "source": "a", "rel_path": "b", "file_name": "c", "doc_title": "d",
            "header_path": "e", "section_title": "f", "parent_source_label": "g"})
        self.assertEqual(rag_agent._identity_coverage_text([d]), "a b c d e f g")


class AdaptiveLexicalMergeTests(unittest.TestCase):

    cfg = {"ADAPTIVE_LEXICAL_MIN_DIGITS": 6}

    def test_exact_match_candidates_are_prepended(self):
        recalled = [_doc("参考书/水下可见光通信关键技术.md")]
        lexical = [
            _doc("组会资料/20230106期末组会/组会分享-张楚元.md"),
            _doc("组会资料/20230106期末组会/20230106期末组会_许健鹏.md"),
        ]
        merged = rag_agent.merge_adaptive_lexical_docs(
            recalled, lexical, "20230106 期末组会汇报了什么？", [], self.cfg)
        self.assertEqual(len(merged), 3)
        self.assertTrue(all("20230106" in (d.metadata or {}).get("rel_path", "")
                            for d in merged[:2]))

    def test_non_matching_candidates_are_appended(self):
        recalled = [_doc("参考书/水下可见光通信关键技术.md")]
        lexical = [_doc("组会资料/20220121期末组会/小组期末总结/WAI-总结会分享_v1.md")]
        merged = rag_agent.merge_adaptive_lexical_docs(
            recalled, lexical, "20230106 期末组会汇报了什么？", [], self.cfg)
        self.assertEqual(merged[0].metadata["rel_path"], "参考书/水下可见光通信关键技术.md")
        self.assertEqual(merged[-1].metadata["rel_path"], lexical[0].metadata["rel_path"])

    def test_no_digit_token_falls_back_to_append(self):
        recalled = [_doc("参考书/A.md")]
        lexical = [_doc("参考书/B.md")]
        merged = rag_agent.merge_adaptive_lexical_docs(
            recalled, lexical, "论文写作方法有哪些建议？", [], self.cfg)
        self.assertEqual([d.metadata["rel_path"] for d in merged],
                         ["参考书/A.md", "参考书/B.md"])

    def test_empty_lexical_keeps_recalled_intact(self):
        recalled = [_doc("参考书/A.md")]
        merged = rag_agent.merge_adaptive_lexical_docs(
            recalled, [], "20230106 期末组会", [], self.cfg)
        self.assertEqual(len(merged), 1)


class LexicalChannelForceTests(unittest.TestCase):

    def test_global_switch_off_and_force_off_returns_empty(self):
        # 不接触 Qdrant：开关关闭且未 force 时函数直接返回
        self.assertEqual(rag_agent.retrieve_lexical_candidates(
            object(), "col", "q", [], 10, cfg={"ENABLE_LEXICAL_RETRIEVAL": False}), [])

    def test_force_respects_missing_module(self):
        with patch.object(rag_agent, "_search_lexical_candidates", None):
            self.assertEqual(rag_agent.retrieve_lexical_candidates(
                object(), "col", "q", [], 10,
                cfg={"ENABLE_LEXICAL_RETRIEVAL": False}, force=True), [])


class NewConfigKeysRegisteredTests(unittest.TestCase):

    def test_adaptive_keys_are_registered(self):
        cfg = rag_agent.load_config()
        for key in ("ENABLE_ADAPTIVE_LEXICAL_RECALL", "ADAPTIVE_LEXICAL_MIN_DIGITS",
                    "ADAPTIVE_LEXICAL_TOP_K"):
            with self.subTest(key=key):
                self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
