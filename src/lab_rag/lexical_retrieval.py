"""有界全文候选召回；不加载模型，不写索引，也不将 scroll 顺序当相关性。"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait

from .document_structure import strip_ingest_prefix


TEXT_FIELDS = ("page_content", "metadata.doc_title", "metadata.section_title",
               "metadata.file_name", "metadata.rel_path")


def lexical_terms(query, keywords, max_terms):
    """保留输入顺序与完整型号；大小写去重，不拆分中文或猜测缩写。"""
    terms, seen = [], set()
    for value in [query, *keywords]:
        if not isinstance(value, str):
            continue
        term = " ".join(value.split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def lexical_filter(term):
    return {"should": [{"key": field, "match": {"text": term}} for field in TEXT_FIELDS]}


def lexical_match(term, text):
    """英文型号按词边界匹配，中文按原短语匹配；不修改公式或源正文。"""
    units = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", term.casefold())
    text = text.casefold()
    return bool(units) and all(
        unit in text if re.search(r"[\u3400-\u9fff]", unit)
        else re.search(r"(?<![a-z0-9])" + re.escape(unit) + r"(?![a-z0-9])", text)
        for unit in units
    )


def lexical_rank(point, terms):
    """只用原文、章节和来源覆盖率排序，生成摘要不能贡献匹配分。"""
    payload = point.get("payload") or {}
    metadata = payload.get("metadata") or {}
    body = strip_ingest_prefix(payload.get("page_content") or metadata.get("page_content") or "")
    if not body.strip():
        return (0, 0, 0)
    heading = "\n".join(str(metadata.get(key) or "") for key in ("section_title", "header_path"))
    scope = "\n".join(str(metadata.get(key) or "") for key in ("doc_title", "file_name", "rel_path"))
    # 同一术语在多个字段出现只计一次；优先覆盖对象和意图，再比较章节及正文。
    body_hits = [bool(lexical_match(term, body)) for term in terms]
    heading_hits = [bool(lexical_match(term, heading)) for term in terms]
    scope_hits = [bool(lexical_match(term, scope)) for term in terms]
    return (sum(a or b or c for a, b, c in zip(body_hits, heading_hits, scope_hits)),
            sum(heading_hits), sum(body_hits))


def search_lexical_candidates(scroll, query, keywords, top_k, *, candidate_limit,
                              max_terms, timeout, log):
    """scroll(filter, limit, timeout) 返回 point 字典列表；限制总候选和总等待时间。"""
    if top_k <= 0 or candidate_limit <= 0 or max_terms <= 0 or timeout <= 0:
        return []
    terms = lexical_terms(query, keywords, max_terms)
    if not terms:
        return []
    # 完整问句通常比关键词更严格，不把它加入关键词交集，否则跨字段命中会被排除。
    query_key = " ".join(query.split()).casefold()
    facets = [term for term in terms if term.casefold() != query_key]
    filters = []
    if len(facets) > 1:
        filters.append({"must": [lexical_filter(term) for term in facets]})
    filters.extend(lexical_filter(term) for term in terms)
    filters = filters[:candidate_limit]
    per_lane, remainder = divmod(candidate_limit, len(filters))
    deadline = time.monotonic() + timeout

    def fetch(search_filter, limit):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return []
        try:
            return scroll(search_filter, limit, remaining)[:limit]
        except Exception as exc:
            log(f"[SPARSE] 候选通道失败: {type(exc).__name__}")
            return []

    executor = ThreadPoolExecutor(max_workers=min(4, len(filters)))
    futures = [executor.submit(fetch, search_filter, per_lane + (i < remainder))
               for i, search_filter in enumerate(filters)]
    done, pending = wait(futures, timeout=max(0, deadline - time.monotonic()))
    for future in pending:
        future.cancel()
    executor.shutdown(wait=False, cancel_futures=True)
    candidates = {}
    # 按计划顺序合并，完成时序不影响排序；不同词各有额度，不会被宽泛型号吞没。
    for future in futures:
        if future not in done:
            continue
        for point in future.result():
            payload = point.get("payload") or {}
            md = payload.get("metadata") or {}
            identity = (md.get("rel_path") or md.get("source") or md.get("file_name", ""),
                        payload.get("page_content") or md.get("page_content", ""))
            candidates.setdefault(identity, point)
    ranked = []
    for point in candidates.values():
        payload = point.get("payload") or {}
        metadata = payload.get("metadata") or {}
        if strip_ingest_prefix(payload.get("page_content") or metadata.get("page_content") or "").strip():
            ranked.append((lexical_rank(point, terms), point))
    # 本地匹配不是 Qdrant multilingual 分词器；零分候选保留在末尾交给语义重排，
    # 不因中文词间存在标点/间隔而误删。摘要只能触发召回，不能贡献本地排序分。
    ranked.sort(key=lambda item: (tuple(-value for value in item[0]), str(item[1].get("id", ""))))
    log(f"[SPARSE] lanes={len(filters)} completed={len(done)} timed_out={len(pending)} "
        f"candidates={len(candidates)} nonempty={len(ranked)} returned={min(top_k, len(ranked))}")
    return [point for _, point in ranked[:top_k]]
