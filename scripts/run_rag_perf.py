#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 链路性能评估 runner（非系统压测）。

功能：
- 读取 tests/rag_perf_queries.json（全部文档类型 + 多重路由 + 边界案例）。
- 复用运行中的 runtime（embedding / reranker / vLLM / Qdrant），逐条调用
  ask_stream 迭代 SSE 事件，用事件 wall-clock 时间戳拆分阶段时延。
- 质量指标：路由准确率、grounding 命中率、Top-1 命中率、召回文档数分布、
  无答案问题的拒答正确率、覆盖度分布、引用数；并采集每条引用的来源明细。
- 时延指标：路由 / 目录枚举 / 检索 / 生成 / 总耗时。
- 按 expected_route、doc_type、case_tag 聚合；可与 --baseline 报告对比前后变化。

仅做单查询链路性能评估，不做并发 / 预热 / 吞吐等系统级压测。

用法：
    python scripts/run_rag_perf.py                                   # 跑全部
    python scripts/run_rag_perf.py --route rag_search                # 只跑某路由
    python scripts/run_rag_perf.py --case-tag 跨设备同类手册           # 只跑某类边界案例
    python scripts/run_rag_perf.py --limit 10                        # 只跑前 N 条
    python scripts/run_rag_perf.py --out logs/rag_perf_baseline.json # 保存基线
    python scripts/run_rag_perf.py --baseline logs/rag_perf_baseline.json  # 复测并对比
"""
import sys
import os
import json
import time
import argparse
import unicodedata
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lab_rag import rag_agent

REAL_ROUTES = {"rag_search", "file_list", "hybrid"}
DOCS_ROOT = "/mnt/cpu_share"
REFUSE_COVERAGE = {"none", "very_low"}


def fmt_ms(seconds):
    return f"{seconds * 1000:.0f}ms" if seconds is not None else "  -  "


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "max": None, "min": None}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": round(sum(s) / n, 3),
        "p50": round(s[int(n * 0.5)], 3),
        "max": round(s[-1], 3),
        "min": round(s[0], 3),
    }


def norm_path(value):
    """把绝对路径/兼容字符统一为相对 DOCS_ROOT 的 NFKC 归一路径。"""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().replace("\\", "/")
    for prefix in (DOCS_ROOT + "/", DOCS_ROOT.lstrip("/") + "/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.lstrip("/").rstrip("/")


def source_matches(sources, part, gtype):
    """判断证据来源集合是否命中 grounding 的单个落点。"""
    target = norm_path(part)
    if not target:
        return False
    base = target.rsplit("/", 1)[-1]
    for raw in sources:
        src = norm_path(raw)
        if not src:
            continue
        if gtype == "dir":
            if src.startswith(target) or target in src:
                return True
        else:
            if src.endswith(target) or target in src:
                return True
            sbase = src.rsplit("/", 1)[-1]
            if sbase and (sbase == base or (len(base) > 8 and base in sbase)
                          or (len(sbase) > 8 and sbase in base)):
                return True
    return False


def evaluate_grounding(citations, sources, grounding, gtype, expected_behavior,
                       coverage_level, n_retrievals):
    """给出 hit / top1_hit / n_sources / refused 等判定。

    citations: final 事件的引用列表（含 source/section/score）
    sources:   引用 source + catalog 枚举 rel_path 的并集
    """
    # 支持 `|` 分隔的多个可接受来源（所有类型），因为很多问题的正确来源本就不唯一；
    # 既消除过严标注造成的假 MISS，也契合「同主题/同设备等价文档」的真实语义。
    parts = [p.strip() for p in grounding.split("|") if p.strip()] or [grounding]
    hit_parts = [p for p in parts if source_matches(sources, p, gtype if gtype != "multi" else "file")]
    hit = bool(hit_parts) if gtype != "none" else None

    top1_source = norm_path(citations[0].get("source")) if citations else ""
    top1_hit = bool(top1_source) and any(
        source_matches([top1_source], p, gtype if gtype != "multi" else "file") for p in parts
    ) if gtype != "none" else None

    refused = (coverage_level in REFUSE_COVERAGE and not citations)
    if expected_behavior == "refuse":
        refused_correct = refused
    elif expected_behavior == "hit_source":
        refused_correct = None
    else:
        refused_correct = None

    # 次指标（诊断用，不影响主指标 hit）：同设备/同主题等价文档命中。
    # 例：grounding 指向 Newport 目录下的 Understanding-...md，而引用命中了同目录的
    # Power-Meter-Datasheet.md，语义上等价、答案仍可信，但严格口径记为 MISS。
    equivalent_hit = None
    if gtype != "none" and hit is False:
        for part in parts:
            target = norm_path(part)
            parent = target.rsplit("/", 1)[0] if "/" in target else ""
            if not parent:
                continue
            for raw in sources:
                src = norm_path(raw)
                if src and src.rsplit("/", 1)[0] == parent and src != target:
                    equivalent_hit = True
                    break
            if equivalent_hit:
                break
        if equivalent_hit is None:
            equivalent_hit = False

    return {
        "hit": hit,
        "equivalent_hit": equivalent_hit,
        "hit_parts": len(hit_parts),
        "n_grounding_parts": len(parts),
        "top1_hit": top1_hit,
        "n_sources": len({norm_path(s) for s in sources if norm_path(s)}),
        "refused": refused,
        "refused_correct": refused_correct,
        "n_retrievals": n_retrievals,
    }


def collect_catalog_sources(file_result):
    """从 list_catalog_entries 结果提取路径证据（file_list 无 citations，靠目录枚举判定）。"""
    if not isinstance(file_result, dict):
        return []
    out = []
    for key in ("matched_dir",):
        if file_result.get(key):
            out.append(str(file_result[key]))
    for key in ("directories", "files", "entries"):
        for item in file_result.get(key) or []:
            if isinstance(item, dict):
                val = item.get("rel_path") or item.get("path") or item.get("name")
                if val:
                    out.append(str(val))
    return out


def run_one(query):
    """运行单条 query，返回指标 dict（含引用明细）。"""
    t0 = time.perf_counter()
    t_route = t_tool = t_retrieval = t_first_chunk = t_final = None
    actual_route = None
    n_retrievals = None
    n_queries = None
    n_keywords = None
    coverage_level = None
    coverage_hint = None
    citations = []
    catalog_sources = []
    queries_seen = []
    keywords_seen = []
    error = False
    error_msg = ""

    try:
        for item in rag_agent.ask_stream(query, username="rag-perf", persist_history=False):
            now = time.perf_counter()
            itype = item.get("type")
            if itype == "metadata":
                stage = item.get("stage")
                if stage == "route":
                    t_route = now
                    actual_route = item.get("route")
                elif stage == "retrieval":
                    t_retrieval = now
                    n_retrievals = len(item.get("retrievals") or [])
                    n_queries = len(item.get("queries") or [])
                    n_keywords = len(item.get("keywords") or [])
                    queries_seen = [str(q) for q in (item.get("queries") or [])]
                    keywords_seen = [str(k) for k in (item.get("keywords") or [])]
                elif stage == "coverage":
                    cov = item.get("coverage") or {}
                    coverage_level = cov.get("level")
                    coverage_hint = cov.get("hint")
                    citations = item.get("citations") or []
            elif itype == "tool" and item.get("tool_name") == "list_catalog_entries":
                t_tool = now
                catalog_sources = collect_catalog_sources(item.get("content"))
            elif itype == "chunk":
                if t_first_chunk is None:
                    t_first_chunk = now
            elif itype == "final":
                t_final = now
                actual_route = item.get("route") or actual_route
                cov = item.get("coverage") or {}
                coverage_level = cov.get("level")
                coverage_hint = cov.get("hint")
                citations = item.get("citations") or []
                queries_seen = [str(q) for q in (item.get("queries") or [])] or queries_seen
                keywords_seen = [str(k) for k in (item.get("keywords") or [])] or keywords_seen
                if n_queries is None:
                    n_queries = len(queries_seen)
                if n_keywords is None:
                    n_keywords = len(keywords_seen)
            elif itype == "error":
                error = True
                error_msg = item.get("content", "")
    except Exception as e:  # 异常统一记为错误，不中断整体聚合
        error = True
        error_msg = repr(e)

    if t_final is None:
        t_final = time.perf_counter()
        error = True
        error_msg = error_msg or "no final event"

    route_time = (t_route - t0) if t_route is not None else None
    catalog_time = (t_tool - t_route) if (t_tool is not None and t_route is not None) else None
    retrieval_time = (
        (t_retrieval - (t_tool if t_tool is not None else t_route))
        if t_retrieval is not None and (t_tool is not None or t_route is not None)
        else None
    )
    generate_time = (
        (t_final - t_first_chunk)
        if t_first_chunk is not None
        else ((t_final - t_retrieval) if t_retrieval is not None else None)
    )

    citation_detail = [
        {
            "index": c.get("index"),
            "source": c.get("source"),
            "section": c.get("section"),
            "score": c.get("score"),
        }
        for c in citations
    ]
    sources = [c.get("source") for c in citations if c.get("source")] + catalog_sources

    return {
        "query": query,
        "actual_route": actual_route,
        "route_match": None,
        "error": error,
        "error_msg": error_msg,
        "t_total": t_final - t0,
        "t_route": route_time,
        "t_catalog": catalog_time,
        "t_retrieval": retrieval_time,
        "t_generate": generate_time,
        "n_retrievals": n_retrievals,
        "n_queries": n_queries,
        "n_keywords": n_keywords,
        "coverage_level": coverage_level,
        "coverage_hint": coverage_hint,
        "n_citations": len(citations),
        "citations": citation_detail,
        "catalog_sources": catalog_sources[:20],
        "sources": sorted({norm_path(s) for s in sources if norm_path(s)})[:30],
        "queries_seen": queries_seen[:20],
        "keywords_seen": keywords_seen[:20],
    }


def rate(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else "n/a"


def aggregate(results, key_fn, label):
    groups = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cov_dist = Counter(r["coverage_level"] or "none" for r in items)
        errs = sum(1 for r in items if r["error"])
        rows.append({
            label: key,
            "count": len(items),
            "route_acc": rate([r["route_match"] for r in items]),
            "grounding_hit_rate": rate([r["grounding_hit"] for r in items]),
            "equivalent_hit_rate": rate([r.get("equivalent_hit") for r in items]),
            "top1_hit_rate": rate([r["top1_hit"] for r in items]),
            "refuse_acc": rate([r["refused_correct"] for r in items]),
            "avg_n_sources": round(sum(r["n_sources"] for r in items) / len(items), 2),
            "avg_n_retrievals": rate([r["n_retrievals"] for r in items]),
            "retrieval_count_dist": dict(Counter(r["n_retrievals"] for r in items)),
            "total": stats([r["t_total"] for r in items]),
            "route": stats([r["t_route"] for r in items if r["t_route"] is not None]),
            "catalog": stats([r["t_catalog"] for r in items if r["t_catalog"] is not None]),
            "retrieval": stats([r["t_retrieval"] for r in items if r["t_retrieval"] is not None]),
            "generate": stats([r["t_generate"] for r in items if r["t_generate"] is not None]),
            "coverage": dict(cov_dist),
            "avg_citations": round(sum(r["n_citations"] for r in items) / len(items), 2),
            "error_rate": round(errs / len(items), 3),
        })
    return rows


def print_table(title, rows, label):
    print(f"\n=== {title} ===")
    header = (f"{label:<16} {'n':>3} {'route':>6} {'hit%':>6} {'top1%':>6} "
              f"{'refuse':>6} {'src':>4} {'ret_n':>5} {'total':>8} {'ret':>7} "
              f"{'gen':>7} {'cit':>4} {'err':>4}")
    print(header)
    print("-" * len(header))
    for r in rows:
        def pct(v):
            return f"{v*100:>5.0f}%" if isinstance(v, (int, float)) else f"{'n/a':>6}"
        t, ret, gen = r["total"], r["retrieval"], r["generate"]
        print(
            f"{str(r[label])[:16]:<16} {r['count']:>3} {pct(r['route_acc']):>6} "
            f"{pct(r['grounding_hit_rate']):>6} {pct(r['top1_hit_rate']):>6} "
            f"{pct(r['refuse_acc']):>6} {r['avg_n_sources']:>4} {str(r['avg_n_retrievals']):>5} "
            f"{fmt_ms(t['mean']):>8} {fmt_ms(ret['mean']):>7} {fmt_ms(gen['mean']):>7} "
            f"{r['avg_citations']:>4} {r['error_rate']*100:>3.0f}%"
        )


def compare_with_baseline(current, baseline):
    """按 id 对比前后报告，返回质量与时延变化。"""
    base = {q["id"]: q for q in (baseline or {}).get("per_query", [])}
    rows = []
    for q in current.get("per_query", []):
        b = base.get(q["id"])
        if not b:
            continue
        rows.append({
            "id": q["id"],
            "expected_route": q.get("expected_route"),
            "case_tag": q.get("case_tag", ""),
            "hit_before": b.get("grounding_hit"),
            "hit_after": q.get("grounding_hit"),
            "top1_before": b.get("top1_hit"),
            "top1_after": q.get("top1_hit"),
            "cit_before": b.get("n_citations"),
            "cit_after": q.get("n_citations"),
            "ret_n_before": b.get("n_retrievals"),
            "ret_n_after": q.get("n_retrievals"),
            "total_before_s": round(b.get("t_total", 0), 3),
            "total_after_s": round(q.get("t_total", 0), 3),
            "total_delta_s": round(q.get("t_total", 0) - b.get("t_total", 0), 3),
            "retrieval_before_s": round(b.get("t_retrieval") or 0, 3),
            "retrieval_after_s": round(q.get("t_retrieval") or 0, 3),
            "coverage_before": b.get("coverage_level"),
            "coverage_after": q.get("coverage_level"),
        })
    common = len(rows)
    def avg(field, pred=lambda r: True):
        vals = [r[field] for r in rows if pred(r) and isinstance(r[field], (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {
        "n_common": common,
        "hit_rate_before": rate([r["hit_before"] for r in rows]),
        "hit_rate_after": rate([r["hit_after"] for r in rows]),
        "top1_rate_before": rate([r["top1_before"] for r in rows]),
        "top1_rate_after": rate([r["top1_after"] for r in rows]),
        "avg_citations_before": avg("cit_before"),
        "avg_citations_after": avg("cit_after"),
        "avg_total_before_s": avg("total_before_s"),
        "avg_total_after_s": avg("total_after_s"),
        "avg_total_delta_s": avg("total_delta_s"),
        "avg_retrieval_before_s": avg("retrieval_before_s"),
        "avg_retrieval_after_s": avg("retrieval_after_s"),
        "per_query": rows,
    }


def print_comparison(cmp_rows):
    print("\n=== 基线对比（优化前 → 优化后）===")
    c = cmp_rows
    print(f"  共同 query 数: {c['n_common']}")
    print(f"  grounding 命中率: {c['hit_rate_before']} → {c['hit_rate_after']}")
    print(f"  Top-1 命中率:    {c['top1_rate_before']} → {c['top1_rate_after']}")
    print(f"  平均引用数:      {c['avg_citations_before']} → {c['avg_citations_after']}")
    print(f"  平均总时延(s):   {c['avg_total_before_s']} → {c['avg_total_after_s']} "
          f"(Δ {c['avg_total_delta_s']})")
    print(f"  平均检索时延(s): {c['avg_retrieval_before_s']} → {c['avg_retrieval_after_s']}")
    improved = [r for r in c["per_query"] if r["hit_before"] is False and r["hit_after"] is True]
    regressed = [r for r in c["per_query"] if r["hit_before"] is True and r["hit_after"] is False]
    print(f"  命中改善: {len(improved)} 条 {[r['id'] for r in improved]}")
    print(f"  命中退化: {len(regressed)} 条 {[r['id'] for r in regressed]}")


def rate_of(results, field):
    """返回 float 或 None；与 rate() 不同，不用字符串 'n/a' 表示未测。"""
    vals = [v for v in (r.get(field) for r in results) if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def distribution_balance(results, dataset_meta):
    """测试集分布均衡度（构建期 meta + 运行期实际覆盖）。"""
    dist = dataset_meta.get("doc_type_distribution", {})
    tags = dataset_meta.get("case_tag_distribution", {})
    routes = dataset_meta.get("route_distribution", {})
    total = dataset_meta.get("total", len(results)) or len(results)
    return {
        "total": total,
        "device_manual_ratio": round(dist.get("设备手册", 0) / total, 4) if total else None,
        "min_doc_type_count": min(dist.values()) if dist else None,
        "case_tag_count": len(tags),
        "min_case_tag_count": min(tags.values()) if tags else None,
        "non_rag_route_ratio": round(1 - routes.get("rag_search", 0) / total, 4) if total else None,
        "doc_type_distribution": dist,
        "case_tag_distribution": tags,
        "route_distribution": routes,
    }


def evaluate_criteria(report, criteria_path, external_metrics=None):
    """按机器可读门槛逐项判定；未达成项列入 passed=False，不放宽口径。

    external_metrics: scripts/check_engineering_criteria.py 产出的工程侧指标，
    用于把回归数/lint/依赖/定制化点四项从 N/A 变为可测。
    """
    if not criteria_path or not os.path.isfile(criteria_path):
        return None
    with open(criteria_path, encoding="utf-8") as f:
        spec = json.load(f)
    ext = external_metrics or {}

    main = report["per_query"]
    by_route = {r["expected_route"]: r for r in report["by_expected_route"]}
    hold = report.get("per_holdout_query", [])
    dist = report.get("distribution_balance", {})
    cmp_rows = report.get("baseline_comparison") or {}
    regressions = [r["id"] for r in cmp_rows.get("per_query", [])
                   if r["hit_before"] is True and r["hit_after"] is False]
    total_s = stats([r["t_total"] for r in main])["mean"]
    ret_s = stats([r["t_retrieval"] for r in main if r["t_retrieval"] is not None])["mean"]
    fl_s = by_route.get("file_list", {}).get("total", {}).get("mean")
    hy_s = by_route.get("hybrid", {}).get("total", {}).get("mean")
    main_hit = rate_of(main, "grounding_hit")
    hold_hit = rate_of(hold, "grounding_hit") if hold else None
    gap = round(abs(main_hit - hold_hit), 4) if (main_hit is not None and hold_hit is not None) else None

    values = {
        "n_queries": len(main),
        "device_manual_ratio": dist.get("device_manual_ratio"),
        "min_doc_type_count": dist.get("min_doc_type_count"),
        "case_tag_count": dist.get("case_tag_count"),
        "min_case_tag_count": dist.get("min_case_tag_count"),
        "non_rag_route_ratio": dist.get("non_rag_route_ratio"),
        "n_holdout": len(hold),
        "holdout_query_overlap": len({r["query"] for r in main} & {r["query"] for r in hold}),
        "grounding_hit_rate": main_hit,
        "top1_hit_rate": rate_of(main, "top1_hit"),
        "per_id_regression_count": len(regressions),
        "refuse_acc": rate_of(main, "refused_correct"),
        "error_count": sum(1 for r in main if r["error"]),
        "holdout_grounding_hit_rate": hold_hit,
        "generalization_gap": gap,
        "avg_total_s": total_s,
        "file_list_total_s": fl_s,
        "hybrid_total_s": hy_s,
        "avg_retrieval_s": ret_s,
        "reliability_test_count": ext.get("reliability_test_count"),
        "lint_error_count": ext.get("lint_error_count"),
        "new_third_party_import_count": ext.get("new_third_party_import_count"),
        "inlined_customization_count": ext.get("inlined_customization_count"),
    }
    results = []
    for c in spec["criteria"]:
        got = values.get(c["metric"])
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            got = None
        if got is None:
            status, reason = "not_measured", "需外部检查（回归/lint/静态扫描）"
        else:
            op, th = c["op"], c["threshold"]
            ok = ({"==": got == th, ">=": got >= th, "<=": got <= th,
                   ">": got > th, "<": got < th}).get(op, False)
            status = "pass" if ok else "fail"
            reason = f"{got} {op} {th}"
        results.append({"id": c["id"], "dimension": c["dimension"], "metric": c["metric"],
                        "op": c["op"], "threshold": c["threshold"], "target": c.get("target"),
                        "actual": got, "status": status, "reason": reason})
    measured = [r for r in results if r["status"] in ("pass", "fail")]
    return {
        "criteria_file": criteria_path,
        "measured": len(measured),
        "passed": sum(1 for r in measured if r["status"] == "pass"),
        "failed": sum(1 for r in measured if r["status"] == "fail"),
        "not_measured": [r["id"] for r in results if r["status"] == "not_measured"],
        "regressions": regressions,
        "generalization_gap": gap,
        "holdout_hit_rate": hold_hit,
        "main_hit_rate": main_hit,
        "criteria": results,
    }


def print_criteria(evaluated):
    if not evaluated:
        return
    print("\n=== 量化门槛达成情况 ===")
    print(f"  已测 {evaluated['measured']} 项：通过 {evaluated['passed']}，未达成 {evaluated['failed']}，"
          f"需外部检查 {len(evaluated['not_measured'])} 项")
    for c in evaluated["criteria"]:
        mark = {"pass": "PASS", "fail": "FAIL", "not_measured": "N/A "}[c["status"]]
        print(f"  [{mark}] {c['id']:<24} {str(c['actual']):<10} {c['op']} {c['threshold']}"
              f"{'  (target ' + str(c['target']) + ')' if c.get('target') else ''}")
    if evaluated["failed"]:
        print("  未达成项（残留短板）：", [c["id"] for c in evaluated["criteria"] if c["status"] == "fail"])
    if evaluated["regressions"]:
        print("  逐 id 退化：", evaluated["regressions"])


def run_queries(selected, label):
    """运行一批 query 并补齐评估字段，返回逐条结果。"""
    results = []
    for i, q in enumerate(selected, 1):
        print(f"[{label} {i}/{len(selected)}] ({q['id']} | {q['expected_route']:<11} | "
              f"{q.get('case_tag') or '基线'}) {q['query']}")
        r = run_one(q["query"])
        exp = q["expected_route"]
        r["route_match"] = (r["actual_route"] == exp) if exp in REAL_ROUTES else None
        g = evaluate_grounding(
            r["citations"], r["sources"], q["grounding"], q.get("grounding_type", "file"),
            q.get("expected_behavior", "hit_source"), r["coverage_level"], r["n_retrievals"],
        )
        r.update(g)
        r["grounding_hit"] = g["hit"]
        r.update({
            "id": q["id"], "expected_route": exp, "doc_type": q["doc_type"],
            "grounding": q["grounding"], "grounding_type": q.get("grounding_type", "file"),
            "expected_behavior": q.get("expected_behavior", "hit_source"),
            "case_tag": q.get("case_tag", ""), "expected_topic": q["expected_topic"],
        })
        flag = "ERR" if r["error"] else ("HIT" if r["hit"] else ("MISS" if r["hit"] is False else "-"))
        if r["expected_behavior"] == "refuse":
            flag = "REFUSE_OK" if r["refused_correct"] else "REFUSE_FAIL"
        print(f"    -> {flag:<10} route={r['actual_route']} total={fmt_ms(r['t_total'])} "
              f"ret={fmt_ms(r['t_retrieval'])} gen={fmt_ms(r['t_generate'])} "
              f"cov={r['coverage_level']} cit={r['n_citations']} src={r['n_sources']} "
              f"ret_n={r['n_retrievals']} equiv={r.get('equivalent_hit')} err={r['error_msg'] or '-'}")
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "tests", "rag_perf_queries.json"))
    ap.add_argument("--route", default=None, help="按 expected_route 过滤")
    ap.add_argument("--doc-type", dest="doc_type", default=None, help="按 doc_type 过滤")
    ap.add_argument("--case-tag", dest="case_tag", default=None, help="按 case_tag 过滤边界案例")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（过滤后）")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "rag_perf_report.json"))
    ap.add_argument("--baseline", default=None, help="基线报告路径，提供则输出前后对比")
    ap.add_argument("--holdout", default=None, help="独立留出集路径，提供则额外评估泛化")
    ap.add_argument("--criteria", default=os.path.join(ROOT, "config", "iteration_criteria.json"),
                    help="机器可读量化门槛文件")
    ap.add_argument("--no-criteria", action="store_true", help="跳过门槛判定")
    ap.add_argument("--external-metrics", default=os.path.join(ROOT, "logs", "engineering_metrics.json"),
                    help="工程侧指标 JSON（scripts/check_engineering_criteria.py 产出）")
    args = ap.parse_args()

    external = {}
    if args.external_metrics and os.path.isfile(args.external_metrics):
        with open(args.external_metrics, encoding="utf-8") as f:
            external = json.load(f)

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    dataset_meta = data.get("meta", {})

    selected = queries
    if args.route:
        selected = [q for q in selected if q["expected_route"] == args.route]
    if args.doc_type:
        selected = [q for q in selected if q["doc_type"] == args.doc_type]
    if args.case_tag:
        selected = [q for q in selected if q.get("case_tag") == args.case_tag]
    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        print("没有匹配的 query，请检查过滤参数。")
        sys.exit(1)

    baseline = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)

    holdout_data = None
    if args.holdout:
        with open(args.holdout, encoding="utf-8") as f:
            holdout_data = json.load(f)
    holdout_selected = (holdout_data or {}).get("queries", [])

    print(f"加载测试集：{len(queries)} 条，筛选后运行 {len(selected)} 条"
          f"{'；留出集 ' + str(len(holdout_selected)) + ' 条' if holdout_selected else ''}")
    print("初始化 runtime（embedding / reranker / vLLM / Qdrant）...")
    rag_agent.build_runtime()
    print("runtime 就绪，开始逐条评估...\n")

    results = run_queries(selected, "MAIN")
    holdout_results = run_queries(holdout_selected, "HOLDOUT") if holdout_selected else []

    by_route = aggregate(results, lambda r: r["expected_route"], "expected_route")
    by_doc = aggregate(results, lambda r: r["doc_type"], "doc_type")
    by_tag = aggregate(results, lambda r: r["case_tag"] or "原有基线", "case_tag")

    print_table("按 expected_route 聚合", by_route, "expected_route")
    print_table("按 doc_type 聚合", by_doc, "doc_type")
    print_table("按 case_tag 聚合", by_tag, "case_tag")

    llm_actual = Counter(r["actual_route"] for r in results if r["expected_route"] == "llm_fallback")
    if llm_actual:
        print(f"\nllm_fallback 实际路由分布（规则返回 None，LLM 决定）：{dict(llm_actual)}")

    misses = [r for r in results if r["expected_behavior"] == "hit_source" and r["hit"] is False]
    if misses:
        print(f"\n=== grounding 未命中案例（{len(misses)} 条）===")
        for r in misses:
            print(f"  [{r['id']}] {r['query'][:48]}")
            print(f"      grounding: {r['grounding'][:110]}")
            print(f"      sources:   {r['sources'][:4]}")

    balance = distribution_balance(results, dataset_meta)
    holdout_agg = aggregate(holdout_results, lambda r: r["doc_type"], "doc_type") if holdout_results else []
    main_hit = rate_of(results, "grounding_hit")
    hold_hit = rate_of(holdout_results, "grounding_hit") if holdout_results else None
    gap = round(abs(main_hit - hold_hit), 4) if (main_hit is not None and hold_hit is not None) else None

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "holdout_dataset": args.holdout,
        "filter": {"route": args.route, "doc_type": args.doc_type,
                   "case_tag": args.case_tag, "limit": args.limit},
        "n_run": len(results),
        "n_holdout": len(holdout_results),
        "by_expected_route": by_route,
        "by_doc_type": by_doc,
        "by_case_tag": by_tag,
        "distribution_balance": balance,
        "holdout_hit_rate": hold_hit,
        "main_hit_rate": main_hit,
        "generalization_gap": gap,
        "by_holdout_doc_type": holdout_agg,
        "equivalent_hit_rate": rate_of(results, "equivalent_hit"),
        "llm_fallback_actual_routes": dict(llm_actual),
        "per_query": results,
        "per_holdout_query": holdout_results,
    }

    if baseline is not None:
        cmp_rows = compare_with_baseline(report, baseline)
        report["baseline_comparison"] = cmp_rows
        report["regressions"] = [r["id"] for r in cmp_rows["per_query"]
                                 if r["hit_before"] is True and r["hit_after"] is False]
        print_comparison(cmp_rows)

    if holdout_results:
        print(f"\n=== 留出集泛化 ===")
        print(f"  主集命中率 {main_hit} / 留出集命中率 {hold_hit} / 差距 {gap}")
        print_table("留出集按 doc_type", holdout_agg, "doc_type")

    if not args.no_criteria:
        evaluated = evaluate_criteria(report, args.criteria, external)
        if evaluated:
            evaluated["external_metrics"] = external
            report["criteria_evaluation"] = evaluated
            print_criteria(evaluated)

    print(f"\n=== 分布均衡 ===")
    print(f"  总数={balance['total']} 设备手册占比={balance['device_manual_ratio']} "
          f"最少 doc_type={balance['min_doc_type_count']} case_tag 类数={balance['case_tag_count']} "
          f"最少 case_tag={balance['min_case_tag_count']} 非 rag 路由占比={balance['non_rag_route_ratio']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写出：{args.out}")


if __name__ == "__main__":
    main()
