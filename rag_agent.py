import os
import re
import json
import time
import traceback
from typing import List, Dict, Optional, Generator

from dotenv import load_dotenv
from difflib import SequenceMatcher

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore

from sentence_transformers import CrossEncoder

load_dotenv()

# =========================
# 全局状态
# =========================
chat_history: List[Dict[str, str]] = []
_runtime: Optional[dict] = None


# =========================
# Debug
# =========================
def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y", "on"}


DEBUG_MODE = _env_bool("DEBUG_MODE", "true")


def debug_log(*args):
    if DEBUG_MODE:
        print("[DEBUG]", *args, flush=True)


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        debug_log(f"{self.name} START")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        debug_log(f"{self.name} END elapsed={elapsed:.3f}s")


# =========================
# 配置
# =========================
def load_config() -> dict:
    return {
        "DEBUG_MODE": _env_bool("DEBUG_MODE", "true"),

        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "EMPTY"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", ""),

        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cpu"),

        "QDRANT_HOST": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base"),

        "INITIAL_RETRIEVAL_K": int(os.getenv("INITIAL_RETRIEVAL_K", "20")),
        "FINAL_TOP_K": int(os.getenv("FINAL_TOP_K", "5")),

        "MAX_HISTORY_TOKENS": int(os.getenv("MAX_HISTORY_TOKENS", "1500")),

        "RERANKER_MODEL_NAME": os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "RERANKER_DEVICE": os.getenv("RERANKER_DEVICE", "cpu"),

        "KNOWLEDGE_BASE_ROOT": os.getenv("KNOWLEDGE_BASE_ROOT", "").strip(),
        "ENABLE_FILESYSTEM_TOOL": os.getenv("ENABLE_FILESYSTEM_TOOL", "true").lower() == "true",
        "FILE_SEARCH_LIMIT": int(os.getenv("FILE_SEARCH_LIMIT", "1000")),
        "DIRECTORY_CHILD_LIMIT": int(os.getenv("DIRECTORY_CHILD_LIMIT", "200")),
    }


# =========================
# 初始化
# =========================
def build_embeddings(model_name: str, device: str):
    debug_log(f"build_embeddings model={model_name} device={device}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_llm(base_url: str, api_key: str, model_name: str):
    debug_log(f"build_llm model={model_name} base_url={base_url}")
    return ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024,
        streaming=True,
        timeout=120,
    )


def build_vectorstore(host: str, port: int, collection_name: str, embeddings):
    debug_log(f"build_vectorstore host={host} port={port} collection={collection_name}")
    client = QdrantClient(host=host, port=port, timeout=30)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def build_reranker(model_name: str, device: str):
    debug_log(f"build_reranker model={model_name} device={device}")
    return CrossEncoder(model_name, device=device)


def build_runtime():
    with Timer("build_runtime"):
        config = load_config()
        debug_log("config=", json.dumps(config, ensure_ascii=False))

        embeddings = build_embeddings(config["EMBEDDING_MODEL_NAME"], config["EMBEDDING_DEVICE"])
        vectorstore = build_vectorstore(
            config["QDRANT_HOST"],
            config["QDRANT_PORT"],
            config["QDRANT_COLLECTION_NAME"],
            embeddings,
        )
        llm = build_llm(
            config["VLLM_BASE_URL"],
            config["VLLM_API_KEY"],
            config["VLLM_MODEL_NAME"],
        )
        reranker = build_reranker(
            config["RERANKER_MODEL_NAME"],
            config["RERANKER_DEVICE"],
        )

        return {
            "config": config,
            "vectorstore": vectorstore,
            "client": vectorstore.client,
            "llm": llm,
            "reranker": reranker,
        }


def get_runtime():
    global _runtime
    if _runtime is None:
        debug_log("runtime is None, building runtime...")
        _runtime = build_runtime()
    return _runtime


# =========================
# 历史管理
# =========================
def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) * 0.6))


def estimate_history_tokens(history: List[Dict[str, str]]) -> int:
    return sum(4 + estimate_tokens(x.get("content", "")) for x in history)


def truncate_history(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    if estimate_history_tokens(history) <= max_tokens:
        return history

    kept = []
    total = 0
    for msg in reversed(history):
        cost = 4 + estimate_tokens(msg.get("content", ""))
        if total + cost > max_tokens:
            break
        kept.insert(0, msg)
        total += cost
    return kept


def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "无历史对话"
    return "\n".join([f'{x["role"]}: {x["content"]}' for x in history])


# =========================
# 工具函数
# =========================
def safe_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.I).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def normalize_retrieval_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compress_repeated_text(text: str) -> str:
    text = normalize_retrieval_text(text)
    words = text.split()
    if len(words) < 20:
        return text
    half = len(words) // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:half * 2])
    if first and second and SequenceMatcher(None, first, second).ratio() > 0.96:
        return first
    return text


def strip_structured_prefix(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*\[文档标题\][^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"^\s*\[路径\][^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"^\s*\[页码\][^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"^\s*[:：\-\s]+", "", text).strip()
    return text


def path_to_unix(p: str) -> str:
    return str(p or "").replace("\\", "/")


def clean_retrieval_display_text(text: str) -> str:
    """
    清洗前端展示用的检索正文：
    1. 压缩重复
    2. 去掉结构前缀（文档标题/章节标题/页码）
    3. 规范空白
    """
    if not text:
        return ""

    raw = text
    text = compress_repeated_text(text)
    text = strip_structured_prefix(text)
    text = normalize_retrieval_text(text)

    return text if text else raw.strip()
# =========================
# 查询路由
# =========================
def route_query(llm, question: str) -> dict:
    prompt = ChatPromptTemplate.from_template("""
    你是实验室知识库问答系统的查询路由助手。
    你只能输出一个 JSON 对象。
    禁止输出解释、前缀、后缀、Markdown、代码块、注释。

    路由类型说明：
    - rag_search：查询原理、步骤、参数、说明、使用方法、故障分析等文档内容
    - file_list：查询目录、文件、设备清单、路径、某目录下有什么
    - hybrid：既要目录/文件清单，又要文档内容说明

    输出格式必须严格如下：
    {{"route":"rag_search 或 file_list 或 hybrid","target":"检索目标（如VLC小组）","reason":"使用原因（如查询原理）"}}

    当前问题：
    {question}
    """.strip())

    try:
        with Timer("route_query_llm"):
            result = (prompt | llm).invoke({"question": question})
        debug_log("ROUTE_RAW_OUTPUT:", repr(result.content))

        data = safe_parse_json(result.content)
        if not data:
            raise ValueError("invalid route json")

        route = data.get("route", "rag_search")
        if route not in {"rag_search", "file_list", "hybrid"}:
            route = "rag_search"

        target = data.get("target", question)
        if not isinstance(target, str) or not target.strip():
            target = question

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        debug_log(f"route_query parsed route={route} target={target} reason={reason}")
        return {
            "route": route,
            "target": target.strip(),
            "reason": reason.strip(),
        }
    except Exception as e:
        debug_log("route_query error:", repr(e))
        return {
            "route": "rag_search",
            "target": question,
            "reason": "默认使用 RAG 检索路由，因为路由 JSON 解析失败。",
        }


# =========================
# 查询改写
# =========================
def rewrite_question(llm, question: str) -> dict:
    prompt = ChatPromptTemplate.from_template("""
    你是实验室知识库检索改写助手。
    请把当前问题改写成更适合检索的结构化 JSON。
    不要回答问题，只输出 JSON。

    格式：
    {{
      "standalone_question": "完整独立问题",
      "keywords": ["关键词1", "关键词2"],
      "expanded_queries": ["改写1", "改写2"]
    }}

    当前问题：
    {question}
    """.strip())

    try:
        with Timer("rewrite_question_llm"):
            result = (prompt | llm).invoke({"question": question})
        debug_log("REWRITE_RAW_OUTPUT:", repr(result.content))

        data = safe_parse_json(result.content)
        if not data:
            raise ValueError("invalid rewrite json")

        standalone_question = data.get("standalone_question", question)
        keywords = data.get("keywords", [])
        expanded_queries = data.get("expanded_queries", [])

        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(expanded_queries, list):
            expanded_queries = []

        queries = [question, standalone_question] + expanded_queries
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        queries = list(dict.fromkeys(queries))

        parsed = {
            "standalone_question": standalone_question if isinstance(standalone_question,
                                                                     str) and standalone_question.strip() else question,
            "keywords": [str(k).strip() for k in keywords if str(k).strip()][:6],
            "expanded_queries": queries[:3],
        }
        debug_log("rewrite_question parsed:", parsed)
        return parsed
    except Exception as e:
        debug_log("rewrite_question error:", repr(e))
        return {
            "standalone_question": question,
            "keywords": [],
            "expanded_queries": [question],
        }


# =========================
# 检索与重排
# =========================
def retrieve_multi_query(vectorstore, queries: List[str], top_k_each: int):
    all_docs = []
    for q in queries:
        t0 = time.perf_counter()
        try:
            docs = vectorstore.similarity_search(q, k=top_k_each)
            elapsed = time.perf_counter() - t0
            debug_log(f"similarity_search query={repr(q)} k={top_k_each} docs={len(docs)} elapsed={elapsed:.3f}s")
            all_docs.extend(docs)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            debug_log(f"similarity_search error query={repr(q)} elapsed={elapsed:.3f}s err={repr(e)}")
    return all_docs


def dedup_docs(docs):
    """
    严格按 topk 返回时，只做完全重复去重，不做相似度阈值去重。
    避免多个高相关片段因内容相似被压缩成 1 个。
    """
    with Timer(f"dedup_docs_exact_only input={len(docs)}"):
        unique_docs = []
        seen = set()

        for doc in docs:
            file_name = doc.metadata.get("file_name", "")
            page = doc.metadata.get("page", "")
            content = clean_retrieval_display_text(doc.page_content)

            exact_key = (file_name, page, content[:1000])
            if exact_key in seen:
                continue

            seen.add(exact_key)
            unique_docs.append(doc)

        debug_log(f"dedup_docs_exact_only output={len(unique_docs)}")
        return unique_docs


def rerank_docs(reranker, query: str, docs, top_k: int):
    if not docs:
        return []
    try:
        with Timer(f"rerank_docs input={len(docs)} top_k={top_k}"):
            pairs = [(query, clean_retrieval_display_text(d.page_content)) for d in docs]
            scores = reranker.predict(pairs)
            scored = list(zip(docs, scores))
            scored.sort(key=lambda x: float(x[1]), reverse=True)

            final_docs = []
            for doc, score in scored[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                final_docs.append(doc)

            debug_log(f"rerank_docs output={len(final_docs)}")
            return final_docs
    except Exception as e:
        debug_log("rerank_docs error:", repr(e))
        return docs[:top_k]


def build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"[片段{i}]\n"
            f"[文档标题] {doc.metadata.get('doc_title', '未知标题')}\n"
            f"[路径] {doc.metadata.get('rel_path', '未知路径')}\n"
            f"[页码] {doc.metadata.get('page', '未知页码')}\n"
            f"{clean_retrieval_display_text(doc.page_content)}"
        )
    return "\n\n".join(parts)


# =========================
# 文件系统工具
# =========================
def safe_join(root: str, rel_path: str) -> str:
    root = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root, rel_path))
    if os.path.commonpath([root, target]) != root:
        raise ValueError("非法路径")
    return target


def normalize_query_tokens(text: str) -> List[str]:
    text = path_to_unix(text).strip().lower()
    parts = re.split(r"[\/\s_\-\(\)\[\]，。,；;：:]+", text)
    return [p for p in parts if p]


def path_match_score(query: str, candidate_path: str) -> float:
    q = path_to_unix(query).lower().strip()
    c = path_to_unix(candidate_path).lower().strip()
    if not q or not c:
        return 0.0

    if q == c:
        return 1.0
    if q in c:
        return 0.95
    if os.path.basename(c) == q:
        return 0.98

    q_tokens = normalize_query_tokens(q)
    c_tokens = normalize_query_tokens(c)
    if not q_tokens or not c_tokens:
        return SequenceMatcher(None, q, c).ratio()

    hit = sum(1 for t in q_tokens if t in c_tokens or t in c)
    token_score = hit / max(1, len(q_tokens))
    seq_score = SequenceMatcher(None, q, os.path.basename(c)).ratio()
    return max(token_score * 0.9, seq_score * 0.8)


def find_best_matching_dirs(keyword: str, root_dir: str, limit: int = 20) -> List[Dict]:
    if not root_dir or not os.path.exists(root_dir):
        return []

    with Timer(f"find_best_matching_dirs keyword={keyword}"):
        matches = []
        count_dirs = 0
        for current_root, dirs, _ in os.walk(root_dir):
            rel_current = os.path.relpath(current_root, root_dir)
            if rel_current == ".":
                rel_current = ""

            for d in dirs:
                count_dirs += 1
                rel_path = os.path.join(rel_current, d) if rel_current else d
                rel_path = path_to_unix(rel_path)
                score = path_match_score(keyword, rel_path)
                if score >= 0.45:
                    matches.append({"rel_path": rel_path, "score": round(score, 4)})

        matches.sort(key=lambda x: (-x["score"], len(x["rel_path"])))
        debug_log(f"find_best_matching_dirs scanned_dirs={count_dirs} matched={len(matches)}")
        return matches[:limit]


def list_immediate_children(root_dir: str, rel_dir: str, limit: int = 200) -> Dict:
    with Timer(f"list_immediate_children rel_dir={rel_dir}"):
        abs_dir = safe_join(root_dir, rel_dir)
        if not os.path.isdir(abs_dir):
            return {"directories": [], "files": []}

        directories, files = [], []
        names = sorted(os.listdir(abs_dir))

        for name in names:
            full_path = os.path.join(abs_dir, name)
            rel_path = path_to_unix(os.path.join(rel_dir, name))
            if os.path.isdir(full_path):
                directories.append({"name": name, "rel_path": rel_path})
            else:
                files.append({"name": name, "rel_path": rel_path})

        debug_log(f"list_immediate_children dirs={len(directories)} files={len(files)}")
        return {
            "directories": directories[:limit],
            "files": files[:limit],
        }


def list_fs_entries_by_keyword(keyword: str, root_dir: str, limit: int = 200) -> List[Dict]:
    if not root_dir or not os.path.exists(root_dir):
        return []

    with Timer(f"list_fs_entries_by_keyword keyword={keyword}"):
        results = []
        count_dirs = 0
        count_files = 0

        for current_root, dirs, files in os.walk(root_dir):
            rel_current = os.path.relpath(current_root, root_dir)
            if rel_current == ".":
                rel_current = ""

            for d in dirs:
                count_dirs += 1
                rel_path = path_to_unix(os.path.join(rel_current, d) if rel_current else d)
                score = path_match_score(keyword, rel_path)
                if score >= 0.45:
                    results.append({
                        "type": "directory",
                        "name": d,
                        "rel_path": rel_path,
                        "score": round(score, 4),
                    })

            for f in files:
                count_files += 1
                rel_path = path_to_unix(os.path.join(rel_current, f) if rel_current else f)
                score = path_match_score(keyword, rel_path)
                if score >= 0.45:
                    results.append({
                        "type": "file",
                        "name": f,
                        "rel_path": rel_path,
                        "score": round(score, 4),
                    })

        debug_log(
            f"list_fs_entries_by_keyword scanned_dirs={count_dirs} scanned_files={count_files} matched={len(results)}")
        return sorted(results, key=lambda x: (-x["score"], len(x["rel_path"])))[:limit]


def is_directory_intent(text: str) -> bool:
    q = normalize_retrieval_text(text).lower()
    hints = ["目录", "目录下", "有什么", "子目录", "文件夹", "有哪些", "清单", "列表"]
    return any(h in q for h in hints)


# =========================
# Qdrant 路径聚合工具
# =========================
def list_paths_by_keyword(keyword: str, limit: int = 1000) -> List[Dict]:
    runtime = get_runtime()
    client = runtime["client"]
    collection_name = runtime["config"]["QDRANT_COLLECTION_NAME"]

    search_filter = rest.Filter(
        should=[
            rest.FieldCondition(key="metadata.rel_path", match=rest.MatchText(text=keyword)),
            rest.FieldCondition(key="metadata.file_name", match=rest.MatchText(text=keyword)),
            rest.FieldCondition(key="metadata.doc_title", match=rest.MatchText(text=keyword)),
        ]
    )

    offset = None
    files = {}

    with Timer(f"list_paths_by_keyword keyword={keyword}"):
        page_count = 0
        while True:
            t0 = time.perf_counter()
            results, next_page_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=min(limit, 256),
                offset=offset,
                with_payload=True,
            )
            elapsed = time.perf_counter() - t0
            page_count += 1
            debug_log(f"qdrant scroll page={page_count} results={len(results)} elapsed={elapsed:.3f}s")

            for hit in results:
                payload = hit.payload or {}
                md = payload.get("metadata", payload)

                file_name = md.get("file_name", "")
                doc_title = md.get("doc_title", "")
                rel_path = md.get("rel_path", "")

                key = rel_path or f"{file_name}|{doc_title}"
                if key not in files:
                    files[key] = {
                        "type": "file",
                        "file_name": file_name,
                        "doc_title": doc_title,
                        "name": doc_title or file_name,
                        "rel_path": rel_path,
                        "count": 0,
                        "score": round(path_match_score(keyword, rel_path or file_name or doc_title), 4),
                    }
                files[key]["count"] += 1

            if not next_page_offset or len(files) >= limit:
                break
            offset = next_page_offset

        debug_log(f"list_paths_by_keyword unique_files={len(files)}")
        return sorted(list(files.values()), key=lambda x: (-x["score"], -x["count"], len(x["rel_path"])))


def list_catalog_entries(target: str) -> Dict:
    runtime = get_runtime()
    config = runtime["config"]

    root_dir = config.get("KNOWLEDGE_BASE_ROOT", "")
    enable_fs = config.get("ENABLE_FILESYSTEM_TOOL", False)

    with Timer(f"list_catalog_entries target={target}"):
        if enable_fs and root_dir and os.path.exists(root_dir):
            matched_dirs = find_best_matching_dirs(target, root_dir, limit=10)
            entries = list_fs_entries_by_keyword(
                target,
                root_dir=root_dir,
                limit=config["FILE_SEARCH_LIMIT"]
            )

            best_dir = matched_dirs[0] if matched_dirs else None
            best_entry = entries[0] if entries else None

            directory_intent = is_directory_intent(target)

            if best_dir and (
                    directory_intent or
                    not best_entry or
                    best_dir["score"] >= best_entry.get("score", 0.0)
            ):
                children = list_immediate_children(
                    root_dir,
                    best_dir["rel_path"],
                    limit=config.get("DIRECTORY_CHILD_LIMIT", 200),
                )
                return {
                    "mode": "filesystem_directory",
                    "matched_dir": best_dir["rel_path"],
                    "matched_score": best_dir["score"],
                    "directories": children["directories"],
                    "files": children["files"],
                }

            return {
                "mode": "filesystem_search",
                "matched_dir": None,
                "entries": entries,
            }

        files = list_paths_by_keyword(target, limit=config["FILE_SEARCH_LIMIT"])
        return {
            "mode": "qdrant_search",
            "matched_dir": None,
            "entries": files,
        }


def build_file_context(file_result: Dict) -> str:
    mode = file_result.get("mode", "")

    if mode == "filesystem_directory":
        matched_dir = file_result.get("matched_dir", "")
        dirs = file_result.get("directories", [])
        files = file_result.get("files", [])

        parts = [f"已定位目录: {matched_dir}"]

        if dirs:
            parts.append("下一级子目录：")
            for i, d in enumerate(dirs, 1):
                parts.append(f"{i}. {d['name']} (路径: {d['rel_path']})")

        if files:
            parts.append("该目录下文件：")
            for i, f in enumerate(files, 1):
                parts.append(f"{i}. {f['name']} (路径: {f['rel_path']})")

        if not dirs and not files:
            parts.append("该目录下未发现子目录或文件。")

        return "\n".join(parts)

    entries = file_result.get("entries", [])
    if not entries:
        return "未找到匹配的目录或文件。"

    parts = [f"找到 {len(entries)} 个相关条目："]
    for i, item in enumerate(entries[:100], 1):
        if item.get("type") == "directory":
            parts.append(f"{i}. [目录] {item['name']} (路径: {item['rel_path']})")
        elif item.get("type") == "file" and "count" not in item:
            parts.append(f"{i}. [文件] {item['name']} (路径: {item['rel_path']})")
        else:
            parts.append(
                f"{i}. {item.get('doc_title') or item.get('file_name') or item.get('name', '未知文件')} "
                f"(路径: {item.get('rel_path', '')}, chunks: {item.get('count', 0)})"
            )

    return "\n".join(parts)


# =========================
# 回答生成
# =========================
def build_file_list_answer(question: str, file_result: Dict, route_reason: str) -> str:
    mode = file_result.get("mode", "")
    parts = [""]

    if mode == "filesystem_directory":
        matched_dir = file_result.get("matched_dir", "")
        dirs = file_result.get("directories", [])
        files = file_result.get("files", [])

        if dirs:
            parts.append("子目录如下：")
            for i, d in enumerate(dirs, 1):
                parts.append(f"{i}. {d['name']}")

        if files:
            parts.append("文件如下：")
            for i, f in enumerate(files, 1):
                parts.append(f"{i}. {f['name']}")

        if not dirs and not files:
            parts.append("该目录下未发现子目录或文件。")

        parts.append(f"\n主要来源：{matched_dir}")
        return "\n".join(parts)

    entries = file_result.get("entries", [])
    if entries:
        parts.append("找到以下相关条目：")
        for i, item in enumerate(entries[:50], 1):
            parts.append(f"{i}. {item.get('name', '未知条目')}（路径：{item.get('rel_path', '')}）")
        return "\n".join(parts)

    parts.append("知识库中未找到足够相关内容。")
    return "\n".join(parts)


def answer_stream(llm, question: str, context: str, route: str, route_reason: str):
    prompt = ChatPromptTemplate.from_template("""
    你是实验室内部知识库问答助手。

    请严格遵守以下规则：
    1. 必须优先依据“上下文结果”回答问题。
    2. 如果用户问“有哪些设备/有哪些文件/目录下有什么/某组有什么设备”，优先根据目录/文件枚举结果作答，完整列出。
    3. 如果用户问的是说明、原理、参数、步骤、操作方法、用途等内容，优先根据文档检索片段回答。
    4. 如果上下文没有足够信息，请明确说明：“知识库中未找到足够相关内容”。
    5. 可以基于你的通用知识补充，但必须明确标注：“以下补充非知识库内容”。
    6. 回答要准确、专业、清晰，注意滤除你认为是文档解析错误带来的结果。
    7. 如果能够识别来源路径或文档标题，回答结尾列出“主要来源”。
    8. 如果是目录枚举类问题，优先用条目列表回答。
    9. 涉及数学符号或公式时，必须使用标准 LaTeX 格式：$公式$。

    上下文结果：
    {context}

    用户问题：
    {question}
    """.strip())

    chain = prompt | llm
    return chain.stream({
        "route": route,
        "route_reason": route_reason if route_reason.strip() else "无法解析原因",
        "context": context if context.strip() else "无可用上下文",
        "question": question,
    })


# =========================
# 主入口：流式
# =========================
def ask_stream(question: str) -> Generator[dict, None, None]:
    global chat_history

    total_start = time.perf_counter()

    try:
        debug_log("=" * 80)
        debug_log(f"ask_stream question={repr(question)}")

        runtime = get_runtime()
        config = runtime["config"]
        llm = runtime["llm"]
        vectorstore = runtime["vectorstore"]
        reranker = runtime["reranker"]

        with Timer("stage_route"):
            route_result = route_query(llm, question)
        route = route_result["route"]
        target = route_result["target"]
        route_reason = route_result.get("reason", "")

        rag_retrievals = []
        file_result = None
        context_parts = []
        rewritten_question = question
        keywords = []
        queries = []

        yield {
            "type": "metadata",
            "stage": "route",
            "route": route,
            "route_target": target,
            "route_reason": route_reason,
        }

        if route in {"file_list", "hybrid"}:
            with Timer("stage_file_list"):
                file_result = list_catalog_entries(target)
                file_context = build_file_context(file_result)
                context_parts.append("[目录/文件结果]\n" + file_context)

            yield {
                "type": "tool",
                "tool_name": "list_catalog_entries",
                "content": file_result,
            }

        if route in {"rag_search", "hybrid"}:
            with Timer("stage_rewrite"):
                rewrite_result = rewrite_question(llm, question)
            rewritten_question = rewrite_result["standalone_question"]
            keywords = rewrite_result["keywords"]
            expanded_queries = rewrite_result["expanded_queries"]

            queries = [question, rewritten_question] + expanded_queries
            if keywords:
                queries.append(" ".join([str(k).strip() for k in keywords if str(k).strip()]))
            queries = list(dict.fromkeys([q.strip() for q in queries if q and q.strip()]))

            debug_log(f"queries_count={len(queries)} queries={queries}")

            with Timer("stage_retrieval"):
                recalled_docs = retrieve_multi_query(
                    vectorstore,
                    queries,
                    config["INITIAL_RETRIEVAL_K"]
                )

                debug_log(f"recalled_docs_before_dedup={len(recalled_docs)}")
                recalled_docs = dedup_docs(recalled_docs)
                debug_log(f"recalled_docs_after_dedup={len(recalled_docs)}")

            with Timer("stage_rerank"):
                final_docs = rerank_docs(reranker, rewritten_question, recalled_docs, config["FINAL_TOP_K"])

            debug_log(f"final_docs={len(final_docs)}")

            for i, doc in enumerate(final_docs, start=1):
                raw_content = (doc.page_content or "").strip()
                display_content = clean_retrieval_display_text(raw_content)
                preview = display_content[:220]

                rag_retrievals.append({
                    "index": i,
                    "doc_title": doc.metadata.get("doc_title", "未知标题"),
                    "file_name": doc.metadata.get("file_name", "未知文件"),
                    "rel_path": doc.metadata.get("rel_path", "未知路径"),
                    "page": doc.metadata.get("page", "未知页码"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "preview": preview,
                    "content": display_content,
                })

                debug_log("raw_content=", repr(raw_content[:200]))
                debug_log("display_content=", repr(display_content[:200]))

            yield {
                "type": "metadata",
                "stage": "retrieval",
                "rewritten_question": rewritten_question,
                "keywords": keywords,
                "queries": queries,
                "retrievals": rag_retrievals,
            }

            with Timer("stage_build_context"):
                rag_context = build_context(final_docs)
                context_parts.append("[知识库检索结果]\n" + rag_context)

        final_context = "\n\n".join([x for x in context_parts if x.strip()])
        debug_log(f"final_context_len={len(final_context)}")

        full_answer = ""

        with Timer("stage_answer_stream"):
            first_chunk_time = None
            for chunk in answer_stream(llm, question, final_context, route, route_reason):
                content = getattr(chunk, "content", "")
                if content:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter() - total_start
                        debug_log(f"first_answer_chunk_elapsed={first_chunk_time:.3f}s")
                    full_answer += content
                    yield {
                        "type": "chunk",
                        "content": content,
                    }

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": full_answer})

        total_elapsed = time.perf_counter() - total_start
        debug_log(f"ask_stream done total_elapsed={total_elapsed:.3f}s answer_len={len(full_answer)}")

        yield {
            "type": "final",
            "content": full_answer,
            "retrievals": rag_retrievals,
            "file_result": file_result,
            "route": route,
            "route_reason": route_reason,
            "rewritten_question": rewritten_question,
            "keywords": keywords,
            "queries": queries,
        }

    except Exception as e:
        debug_log("ask_stream error:", repr(e))
        debug_log(traceback.format_exc())
        yield {
            "type": "error",
            "content": str(e),
        }


# =========================
# 非流式包装
# =========================
def ask(question: str) -> Dict:
    answer = ""
    retrievals = []
    rewritten_question = question
    keywords = []
    queries = []
    route = "rag_search"
    route_reason = ""
    file_result = None

    for item in ask_stream(question):
        if item["type"] == "metadata":
            stage = item.get("stage")
            if stage == "route":
                route = item.get("route", route)
                route_reason = item.get("route_reason", route_reason)
            elif stage == "retrieval":
                retrievals = item.get("retrievals", retrievals)
                rewritten_question = item.get("rewritten_question", rewritten_question)
                keywords = item.get("keywords", keywords)
                queries = item.get("queries", queries)
        elif item["type"] == "tool":
            file_result = item.get("content")
        elif item["type"] == "chunk":
            answer += item["content"]
        elif item["type"] == "final":
            if item.get("content"):
                answer = item["content"]
            file_result = item.get("file_result", file_result)
            route = item.get("route", route)
            route_reason = item.get("route_reason", route_reason)
            retrievals = item.get("retrievals", retrievals)
            rewritten_question = item.get("rewritten_question", rewritten_question)
            keywords = item.get("keywords", keywords)
            queries = item.get("queries", queries)

    return {
        "answer": answer,
        "history": chat_history,
        "retrievals": retrievals,
        "rewritten_question": rewritten_question,
        "keywords": keywords,
        "queries": queries,
        "route": route,
        "route_reason": route_reason,
        "file_result": file_result,
    }


def clear_history():
    global chat_history
    chat_history = []