"""只读 Qdrant 难样本回放。固定查询词，隔离全文召回，不冒充端到端问答评测。"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab_rag.document_structure import strip_ingest_prefix
from lab_rag.lexical_retrieval import lexical_filter, search_lexical_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "tests/fixtures/qdrant_hard_cases.json")
    parser.add_argument("--top-k", type=int, default=64)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k 必须为正整数")
    cfg = dotenv_values(ROOT / ".env")
    base = f"http://{cfg.get('QDRANT_HOST', '172.18.216.71')}:{cfg.get('QDRANT_PORT', '6333')}"
    endpoint = base + "/collections/" + cfg.get("QDRANT_COLLECTION_NAME", "lab_knowledge_base") + "/points"

    def post(path, body, timeout):
        response = requests.post(endpoint + path, json=body, timeout=timeout)
        response.raise_for_status()
        return response.json()["result"]

    def scroll(search_filter, limit, timeout):
        return post("/scroll", {"filter": search_filter, "limit": limit,
                               "with_payload": True, "with_vector": False}, timeout)["points"]

    report = {"scope": "fixed-query lexical retrieval only", "top_k": args.top_k, "cases": []}
    for case in json.loads(args.cases.read_text(encoding="utf-8")):
        target = post("", {"ids": [case["point_id"]], "with_payload": True, "with_vector": False}, 10)
        if not target or case["expected_body"] not in strip_ingest_prefix(target[0]["payload"].get("page_content", "")):
            raise RuntimeError("评测 point 缺失或正文已变化，请重新标注样本，不能视为召回失败")
        # 基线保留旧 should+首屏逻辑，仅固定 set 顺序；本样本不触及旧 30 条件裁剪。
        terms = list(dict.fromkeys(case["keywords"] + [case["query"]]))
        old_filter = {"should": [f for term in terms for f in lexical_filter(term)["should"]][:30]}
        started = time.monotonic()
        old = scroll(old_filter, args.top_k, 10)
        old_seconds = time.monotonic() - started
        budget = int(cfg.get("SPARSE_CANDIDATE_LIMIT", 256))
        # 同预算的旧 OR 通道用于区分“只扩大首屏”与“独立召回通道”的作用。
        wide_old = scroll(old_filter, budget, 10)
        started = time.monotonic()
        new = search_lexical_candidates(
            scroll, case["query"], case["keywords"], args.top_k,
            candidate_limit=budget,
            max_terms=int(cfg.get("SPARSE_MAX_TERMS", 8)),
            timeout=float(cfg.get("SPARSE_TIMEOUT", 10)), log=lambda msg: None,
        )
        def rank(points):
            return next((i for i, point in enumerate(points, 1) if str(point["id"]) == case["point_id"]), None)
        report["cases"].append({**case, "rel_path": target[0]["payload"].get("metadata", {}).get("rel_path"),
                                "before_rank": rank(old), "after_rank": rank(new),
                                "wide_baseline_budget": budget, "wide_baseline_rank": rank(wide_old),
                                "before_seconds": round(old_seconds, 3),
                                "after_seconds": round(time.monotonic() - started, 3)})
    report["before_hits"] = sum(c["before_rank"] is not None for c in report["cases"])
    report["after_hits"] = sum(c["after_rank"] is not None for c in report["cases"])
    # stdout 为机器可读报告，不输出配置/Token；不落盘、不初始化模型、不写索引。
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
