"""全文候选的离线行为测试，不需要 Qdrant、GPU 或模型依赖。"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lab_rag.lexical_retrieval import (
    lexical_terms, lexical_match, lexical_rank, search_lexical_candidates,
)


def point(pid, body, **metadata):
    return {"id": pid, "payload": {"page_content": body, "metadata": metadata}}


class LexicalTests(unittest.TestCase):
    def search(self, scroll, **kwargs):
        options = dict(candidate_limit=64, max_terms=8, timeout=1, log=lambda msg: None)
        options.update(kwargs)
        return search_lexical_candidates(scroll, "ZX9 reset", ["ZX9", "reset"], 16, **options)

    def test_terms_deterministic_and_query_preserved(self):
        self.assertEqual(lexical_terms(" Q ", ["q", None, "A", "b", "c"], 3), ["Q", "A", "b"])

    def test_model_boundaries_and_command_punctuation(self):
        self.assertFalse(lexical_match("ZX9", "ZX99"))
        self.assertTrue(lexical_match("MODE:SET", "mode:set"))
        self.assertTrue(lexical_match("安装 操作", "安装 / 操作"))

    def test_summary_alone_cannot_contribute(self):
        bad = point("1", "【文档：manual > 段落摘要：ZX9 reset】\nUnrelated text.")
        self.assertEqual(lexical_rank(bad, ["ZX9", "reset"]), (0, 0, 0))

    def test_cross_field_coverage_beats_scope_only(self):
        relevant = point("9", "Press reset.", rel_path="ZX9/manual.md", section_title="Reset")
        noise = point("1", "Warranty.", rel_path="ZX9/manual.md")
        self.assertGreater(lexical_rank(relevant, ["ZX9", "reset"]), lexical_rank(noise, ["ZX9", "reset"]))

    def test_independent_lanes_rescue_later_id(self):
        target = point("z", "reset", rel_path="ZX9/manual.md")
        noise = point("a", "warranty", rel_path="ZX9/manual.md")
        def scroll(f, limit, timeout):
            return [target] if "must" in f else [noise]
        self.assertEqual(self.search(scroll), [target, noise])

    def test_total_candidate_budget_and_term_field_coverage(self):
        scroll = Mock(return_value=[])
        self.search(scroll, candidate_limit=19)
        self.assertEqual(sum(c.args[1] for c in scroll.call_args_list), 19)
        filters = [c.args[0] for c in scroll.call_args_list]
        self.assertEqual(len(filters), 4)
        self.assertEqual(len(filters[0]["must"]), 2)
        self.assertTrue(all(len(f["should"]) == 5 for f in filters[1:]))

    def test_partial_failure_keeps_other_lanes(self):
        target = point("a", "reset", source="ZX9")
        def scroll(f, limit, timeout):
            if "must" in f:
                raise TimeoutError()
            return [target]
        self.assertEqual(self.search(scroll), [target])

    def test_timeout_is_bounded(self):
        def slow(f, limit, timeout):
            time.sleep(0.1)
            return []
        started = time.monotonic()
        self.assertEqual(self.search(slow, timeout=0.01), [])
        self.assertLess(time.monotonic() - started, 0.08)

    def test_duplicates_and_ties_stable(self):
        a = point("a", "reset", source="ZX9")
        z = point("z", "reset", source="ZX9", rel_path="other.md")
        self.assertEqual(self.search(lambda *args: [z, a, a]), [a, z])

    def test_invalid_budgets_do_not_query(self):
        scroll = Mock()
        for option in ({"candidate_limit": 0}, {"max_terms": 0}, {"timeout": 0}):
            self.assertEqual(self.search(scroll, **option), [])
        scroll.assert_not_called()

    def test_empty_terms_do_not_query(self):
        scroll = Mock()
        self.assertEqual(search_lexical_candidates(scroll, "", [], 10,
                         candidate_limit=64, max_terms=8, timeout=1, log=lambda msg: None), [])
        scroll.assert_not_called()

    def test_tokenizer_only_matches_remain_for_semantic_reranking(self):
        candidate = point("1", "安装前，应按要求进行操作。")
        result = search_lexical_candidates(lambda *args: [candidate], "安装操作", [], 5,
                                          candidate_limit=10, max_terms=8, timeout=1, log=lambda msg: None)
        self.assertEqual(result, [candidate])

    def test_prefix_only_points_cannot_become_evidence(self):
        candidate = point("1", "【文档：ZX9 > 段落摘要：reset】")
        self.assertEqual(self.search(lambda *args: [candidate]), [])

    @unittest.skip("依赖 origin 版 rag_agent.retrieve_sparse wrapper；3.4.0 基线下待适配")
    def test_wrapper_preserves_all_metadata_and_disables_vectors(self):
        from types import SimpleNamespace
        from test_evidence_pipeline import load_functions
        target = SimpleNamespace(id="1", payload={"page_content": "reset",
                                 "metadata": {"source": "ZX9", "header_path": "Safety > Reset"}})
        client = Mock()
        client.scroll.return_value = ([target], None)
        ns = load_functions("rag_agent", {"retrieve_sparse"},
                            search_lexical_candidates=search_lexical_candidates,
                            rest=SimpleNamespace(Filter=lambda **kw: kw))
        docs = ns["retrieve_sparse"](client, "test", "ZX9 reset", ["ZX9", "reset"], 5, cfg={})
        self.assertEqual(docs[0].metadata, target.payload["metadata"])
        self.assertTrue(all(not c.kwargs["with_vectors"] for c in client.scroll.call_args_list))


if __name__ == "__main__":
    unittest.main()
