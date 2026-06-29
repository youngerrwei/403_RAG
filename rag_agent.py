import os
import re
import json
import time
import fcntl
import shutil
import hashlib
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Generator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

from logger import get_logger

load_dotenv()

# =========================
# 全局状态
# =========================
_runtime: Optional[dict] = None

# =========================
# 用户级历史隔离（线程安全）
# =========================
_user_histories_lock = threading.Lock()
_user_histories: Dict[str, List[Dict[str, str]]] = {}


def get_user_chat_history(username: str) -> List[Dict[str, str]]:
    """获取用户的聊天历史（线程安全）"""
    with _user_histories_lock:
        return _user_histories.get(username, []).copy()


def append_user_chat_history(username: str, role: str, content: str):
    """追加用户历史记录（线程安全），同时持久化到文件"""
    with _user_histories_lock:
        if username not in _user_histories:
            _user_histories[username] = []
        _user_histories[username].append({"role": role, "content": content})
        # 限制内存中的历史长度
        max_history = 50  # 内存中保留最近50轮
        if len(_user_histories[username]) > max_history * 2:
            _user_histories[username] = _user_histories[username][-max_history * 2:]
    # 同时持久化到当日文件
    append_to_user_history(username, role, content)


def clear_user_chat_history(username: str):
    """清空用户历史（线程安全）"""
    with _user_histories_lock:
        _user_histories.pop(username, None)


# =========================
# 历史记录本地文件管理（按日期分离存储）
# =========================
HISTORY_DIR = os.path.join(os.getcwd(), "data", "chat_histories")
os.makedirs(HISTORY_DIR, exist_ok=True)


def get_history_dir(username: str) -> str:
    """获取用户历史记录目录"""
    if not username:
        username = "anonymous"
    safe_name = "".join(c for c in str(username) if c.isalnum() or c in "._-")
    user_dir = os.path.join(HISTORY_DIR, safe_name)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_history_path(username: str, date_str: str = None) -> str:
    """获取指定用户指定日期的历史记录文件路径"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    user_dir = get_history_dir(username)
    return os.path.join(user_dir, f"{date_str}.json")


def load_history(username: str) -> list:
    """加载用户近 N 天的历史记录（合并多个日期文件）"""
    retention_days = int(os.getenv("HISTORY_RETENTION_DAYS", "7"))
    user_dir = get_history_dir(username)
    if not os.path.isdir(user_dir):
        return []
    today = datetime.now().date()
    all_history = []
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".json"):
            continue
        date_str = fname.replace(".json", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if (today - file_date).days <= retention_days:
            path = os.path.join(user_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                if isinstance(data, list):
                    all_history.extend(data)
            except (json.JSONDecodeError, FileNotFoundError, IOError):
                continue
    return all_history


def save_history(username: str, history: list):
    """保存历史到当日文件"""
    today = datetime.now().strftime("%Y-%m-%d")
    path = get_history_path(username, today)
    daily_max = int(os.getenv("HISTORY_DAILY_MAX", "200"))
    if len(history) > daily_max:
        history = history[-daily_max:]
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(history, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_to_user_history(username: str, role: str, content: str):
    """向指定用户当日历史追加一条带时间戳的对话记录"""
    with _user_histories_lock:
        today = datetime.now().strftime("%Y-%m-%d")
        path = get_history_path(username, today)
        # 读取当日已有记录
        history = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    try:
                        history = json.load(f)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except (json.JSONDecodeError, IOError):
                history = []

        # 追加新记录（带时间戳）
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # 单日上限
        daily_max = int(os.getenv("HISTORY_DAILY_MAX", "200"))
        if len(history) > daily_max:
            history = history[-daily_max:]

        # 写入
        with open(path, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(history, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def clear_user_history(username: str):
    """清空用户的所有历史记录（删除用户历史目录）"""
    with _user_histories_lock:
        user_dir = get_history_dir(username)
        if os.path.isdir(user_dir):
            shutil.rmtree(user_dir)
        # 同时清理内存中的历史
        if username in _user_histories:
            del _user_histories[username]


def cleanup_expired_history(username: str = None, retention_days: int = None):
    """清理超过保留天数的历史文件"""
    if retention_days is None:
        retention_days = int(os.getenv("HISTORY_RETENTION_DAYS", "7"))
    today = datetime.now().date()

    if username:
        targets = [get_history_dir(username)]
    else:
        if not os.path.isdir(HISTORY_DIR):
            return
        targets = [os.path.join(HISTORY_DIR, d) for d in os.listdir(HISTORY_DIR)
                   if os.path.isdir(os.path.join(HISTORY_DIR, d))]

    removed_count = 0
    for user_dir in targets:
        if not os.path.isdir(user_dir):
            continue
        for fname in os.listdir(user_dir):
            if not fname.endswith(".json"):
                continue
            date_str = fname.replace(".json", "")
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (today - file_date).days > retention_days:
                os.remove(os.path.join(user_dir, fname))
                removed_count += 1

    if removed_count > 0:
        print(f"[INFO] 已清理 {removed_count} 个过期历史文件")


# =========================
# Debug
# =========================
def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y", "on"}


def batch_get_parents(client, collection_name: str, parent_ids: list) -> dict:
    """批量从 parent collection 获取父块内容及摘要"""
    result = {}
    if not parent_ids or not client:
        return result
    try:
        search_filter = rest.Filter(
            should=[
                rest.FieldCondition(
                    key="parent_id",
                    match=rest.MatchValue(value=pid)
                ) for pid in parent_ids
            ]
        )
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=len(parent_ids) + 10,
            with_payload=True,
            timeout=10,  # 父块查询超时保护，防止网络异常时长时间阻塞
        )
        for point in points:
            payload = point.payload or {}
            pid = payload.get("parent_id")
            content = payload.get("parent_content", "")
            if pid and content:
                # 返回包含 content 和 summary 的字典
                result[pid] = {
                    "content": content,
                    "summary": payload.get("parent_summary", ""),
                }
    except Exception as e:
        debug_log(f"[WARN] 批量获取父块失败: {e}")
    return result


def expand_to_parent_docs(child_docs: List[Document], cfg: dict, top_k: int = None, qdrant_client=None) -> List[Document]:
    """子块展开为父块，带长度控制"""
    if top_k is None:
        top_k = int(cfg.get("FINAL_TOP_K", "16"))
    max_context_chars = int(cfg.get("MAX_CONTEXT_CHARS", 12000))

    # 收集需要的 parent_ids
    needed_parent_ids = set()
    best_children = {}
    for child in child_docs:
        pid = child.metadata.get("parent_id")
        score = child.metadata.get("rerank_score", 0)
        if pid and (pid not in best_children or score > best_children[pid][1]):
            best_children[pid] = (child, score)
            needed_parent_ids.add(pid)

    # 批量从 parent collection 获取父块内容及摘要（带降级策略）
    parent_contents = {}
    parent_summaries = {}
    if qdrant_client and needed_parent_ids:
        parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
        try:
            parent_data = batch_get_parents(qdrant_client, parent_collection, list(needed_parent_ids))
            # 解析新的返回结构，分别获取 content 和 summary
            for pid, data in parent_data.items():
                if isinstance(data, dict):
                    parent_contents[pid] = data.get("content", "")
                    parent_summaries[pid] = data.get("summary", "")
                else:
                    # 向后兼容旧格式（直接是字符串）
                    parent_contents[pid] = data
            debug_log(f"从 parent collection 获取到 {len(parent_contents)}/{len(needed_parent_ids)} 个父块")
        except Exception as e:
            debug_log(f"[WARNING] parent collection 查询失败: {e}，降级使用子块")
            parent_contents = {}
            parent_summaries = {}
            # 降级：使用 child metadata 中的 parent_content 和 parent_summary
            for child in child_docs:
                pid = child.metadata.get("parent_id")
                if pid:
                    if "parent_content" in child.metadata:
                        parent_contents[pid] = child.metadata.get("parent_content", "")
                    parent_summaries[pid] = child.metadata.get("parent_summary", "")

    # 按 rerank_score 排序
    sorted_children = sorted(best_children.values(), key=lambda x: x[1], reverse=True)

    parent_docs = []
    seen_parent_ids = set()
    accumulated_length = 0

    for child, score in sorted_children:
        parent_id = child.metadata.get("parent_id")

        # 优先从 parent collection 获取，降级从子块 metadata 读取（向后兼容）
        parent_content = parent_contents.get(parent_id, "") or child.metadata.get("parent_content", "")
        # 获取父块摘要
        parent_summary = parent_summaries.get(parent_id, "") or child.metadata.get("parent_summary", "")

        # 最终降级：如果父块内容仍为空（旧数据无 parent_content），使用子块内容兜底
        if not parent_content:
            parent_content = child.page_content
            debug_log(f"[WARN] parent_id={parent_id} 无父块内容，降级使用子块内容兜底")

        # [FIX] 单个 parent_content 字段级截断，防止异常大文档占用整个 context 预算
        max_single_content = max_context_chars // 2  # 单个文档最多占一半预算
        if len(parent_content) > max_single_content:
            parent_content = parent_content[:max_single_content]
            debug_log(f"[WARN] parent_content 过大，已截断至 {max_single_content} 字符")

        if not parent_id or parent_id in seen_parent_ids:
            continue

        # 长度控制：预估加入该父块后是否超限
        content_length = len(parent_content)
        if accumulated_length + content_length > max_context_chars and parent_docs:
            debug_log(f"上下文长度控制: 已达 {accumulated_length} 字符, 停止展开")
            break

        # 构建父文档，确保 parent_summary 被正确设置到 metadata
        parent_meta = {k: v for k, v in child.metadata.items() if k != "parent_content"}
        parent_meta["is_parent"] = True
        if parent_summary:
            parent_meta["parent_summary"] = parent_summary

        parent_doc = Document(
            page_content=parent_content,
            metadata=parent_meta
        )

        parent_docs.append(parent_doc)
        seen_parent_ids.add(parent_id)
        accumulated_length += content_length

        if len(parent_docs) >= top_k:
            break

    debug_log(f"父块展开: {len(child_docs)} 子块 → {len(parent_docs)} 父块 (总长={accumulated_length})")
    return parent_docs


DEBUG_MODE = _env_bool("DEBUG_MODE", "true")

_logger = get_logger("agent")


def debug_log(*args):
    """调试日志 - 同时输出到控制台和文件"""
    if DEBUG_MODE:
        msg = " ".join(str(a) for a in args)
        _logger.debug(msg)


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

class TTLCache:
    """线程安全的 TTL 缓存，带主动清理"""
    MAX_CACHE_SIZE = 200  # 最大缓存条目数

    def __init__(self, maxsize=128, ttl=300):
        self._cache = OrderedDict()
        self._ttl = ttl
        self._maxsize = min(maxsize, self.MAX_CACHE_SIZE)
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._cache.move_to_end(key)
                    return value
                else:
                    del self._cache[key]
        return None

    def set(self, key, value):
        with self._lock:
            self._cleanup()
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time())

    def _cleanup(self):
        """主动清理过期缓存（需在锁内调用）"""
        now = time.time()
        expired_keys = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired_keys:
            del self._cache[k]
        # 如果仍然超出容量，删除最旧的
        if len(self._cache) > self._maxsize:
            excess = len(self._cache) - self._maxsize
            for _ in range(excess):
                self._cache.popitem(last=False)


# 模块级缓存实例
_retrieval_cache = TTLCache(maxsize=128, ttl=300)


def load_config() -> dict:
    cfg = {
        "DEBUG_MODE": _env_bool("DEBUG_MODE", "true"),

        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "lab-secret-key"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct"),

        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "./models/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cuda:2"),

        "QDRANT_HOST": os.getenv("QDRANT_HOST", "172.18.216.71"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base"),
        "QDRANT_PARENT_COLLECTION": os.getenv("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents"),

        "INITIAL_RETRIEVAL_K": int(os.getenv("INITIAL_RETRIEVAL_K", "64")),
        "FINAL_TOP_K": int(os.getenv("FINAL_TOP_K", "16")),

        "MAX_HISTORY_TOKENS": int(os.getenv("MAX_HISTORY_TOKENS", "1500")),

        "RERANKER_MODEL_NAME": os.getenv("RERANKER_MODEL_NAME", "./models/bge-reranker-v2-m3"),
        "RERANKER_DEVICE": os.getenv("RERANKER_DEVICE", "cuda:2"),

        "KNOWLEDGE_BASE_ROOT": os.getenv("KNOWLEDGE_BASE_ROOT", "").strip(),
        "ENABLE_FILESYSTEM_TOOL": os.getenv("ENABLE_FILESYSTEM_TOOL", "true").lower() == "true",
        "FILE_SEARCH_LIMIT": int(os.getenv("FILE_SEARCH_LIMIT", "200")),
        "DIRECTORY_CHILD_LIMIT": int(os.getenv("DIRECTORY_CHILD_LIMIT", "200")),

        "ENABLE_HYBRID_SEARCH": os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
        "BM25_WEIGHT": float(os.getenv("BM25_WEIGHT", "0.3")),
        "DENSE_WEIGHT": float(os.getenv("DENSE_WEIGHT", "0.7")),

        "MAX_CONTEXT_CHARS": int(os.getenv("MAX_CONTEXT_CHARS", "3000")),

        "RESPONSE_MAX_TOKENS": int(os.getenv("RESPONSE_MAX_TOKENS", "2048")),
        "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "RERANKER_SCORE_THRESHOLD": float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.3")),

        "RRF_K": int(os.getenv("RRF_K", "60")),
        "KEYWORD_BOOST": float(os.getenv("KEYWORD_BOOST", "1.05")),
        "TITLE_BOOST": float(os.getenv("TITLE_BOOST", "1.08")),

        "ENABLE_HYDE": os.getenv("ENABLE_HYDE", "true").lower() == "true",
    }

    # 配置验证
    log = _logger.warning
    critical_configs = {
        "VLLM_BASE_URL": cfg.get("VLLM_BASE_URL"),
        "QDRANT_HOST": cfg.get("QDRANT_HOST"),
        "EMBEDDING_MODEL_NAME": cfg.get("EMBEDDING_MODEL_NAME"),
    }
    for key, val in critical_configs.items():
        if not val:
            _logger.critical(f"关键配置 {key} 未设置，系统可能无法正常运行！")

    # 设备警告
    if cfg.get("EMBEDDING_DEVICE", "cpu") == "cpu":
        debug_log("EMBEDDING_DEVICE=cpu，推理速度将非常慢，建议设置为 cuda:2")
    if cfg.get("RERANKER_DEVICE", "cpu") == "cpu":
        debug_log("RERANKER_DEVICE=cpu，推理速度将非常慢，建议设置为 cuda:2")

    return cfg


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


def build_llm(base_url: str, api_key: str, model_name: str, cfg: dict = None):
    temperature = (cfg or {}).get("LLM_TEMPERATURE", 0.1)
    max_tokens = (cfg or {}).get("RESPONSE_MAX_TOKENS", 2048)
    debug_log(f"build_llm model={model_name} base_url={base_url} temperature={temperature} max_tokens={max_tokens}")
    return ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        streaming=True,
        timeout=120,
    )


def _connect_qdrant_with_retry(host: str, port: int, max_retries: int = 3) -> QdrantClient:
    """带重试机制的 Qdrant 客户端连接，防止短暂网络抖动导致启动失败"""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=host, port=port, timeout=30)
            client.get_collections()  # 验证连接可用
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                _logger.warning(f"[WARNING] Qdrant 连接失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                _logger.error(f"[ERROR] Qdrant 连接失败，已重试 {max_retries} 次: {e}")
                raise


def build_vectorstore(host: str, port: int, collection_name: str, embeddings):
    debug_log(f"build_vectorstore host={host} port={port} collection={collection_name}")
    client = _connect_qdrant_with_retry(host, port)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="page_content",
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
            cfg=config,
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


_runtime_lock = threading.Lock()


def get_runtime():
    """获取运行时实例，使用 Double-check Locking 防止竞态条件"""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:  # Double-check
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
    text = re.sub(r"^\s*\[章节\][^\n]*\n?", "", text, flags=re.M)
    text = re.sub(r"^\s*\[页码\][^\n]*\n?", "", text, flags=re.M)
    # 清洗 build_contextual_prefix() 注入的上下文前缀，格式为整行 【...】
    text = re.sub(r"^\s*【[^】]*】\s*$\n?", "", text, flags=re.M)
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
def rule_based_route(question: str) -> Optional[dict]:
    """
    基于规则的快速路由。
    如果能确定意图则返回路由结果 dict，否则返回 None 表示需要 LLM 判断。
    """
    # 去重后的关键词列表（移除被短关键词覆盖的长关键词）
    # 包含计数/列举类关键词，使"有多少篇论文"等查询能正确路由到 file_list 或 hybrid
    file_list_keywords = [
        "目录", "有哪些文件", "文件夹", "子目录", "文件列表", "清单", "有什么设备", "列出",
        "有多少", "多少篇", "几篇", "统计", "数量", "一共有", "总共有", "共有", "全部", "所有",
    ]
    rag_search_keywords = [
        "原理", "步骤", "参数", "如何", "什么是", "解释", "说明", "方法", "教程",
        "操作", "区别", "对比", "优缺点",
        "论文", "实验", "公式", "算法", "仿真", "测试", "性能",
    ]

    # 歧义词需要额外条件才触发
    ambiguous_words = {"为什么", "怎么", "什么", "哪些", "几个", "几台"}
    # 只有当歧义词与专业词共现时才触发 RAG
    academic_context = {"论文", "实验", "系统", "方法", "技术", "原理", "参数", "算法", "结果", "数据", "研究", "设计", "方案", "通信", "信号", "光", "水下"}

    hit_file = any(kw in question for kw in file_list_keywords)
    hit_rag = any(kw in question for kw in rag_search_keywords)

    # 歧义词上下文判断
    hit_ambiguous = any(w in question for w in ambiguous_words)
    if hit_ambiguous:
        if hit_rag and hit_file:
            return {"route": "hybrid", "target": question, "reason": "规则匹配"}
        elif hit_rag:
            return {"route": "rag_search", "target": question, "reason": "规则匹配"}
        elif hit_file:
            return {"route": "file_list", "target": question, "reason": "规则匹配"}
        else:
            # 歧义词单独出现，检查学术上下文（要求至少2个学术词共现）
            academic_context_count = sum(1 for w in academic_context if w in question)
            if academic_context_count >= 2:
                return {"route": "rag_search", "target": question, "reason": "规则匹配(学术上下文)"}
            return None  # 交给LLM兜底

    if hit_file and hit_rag:
        return {"route": "hybrid", "target": question, "reason": "规则匹配"}
    elif hit_file:
        return {"route": "file_list", "target": question, "reason": "规则匹配"}
    elif hit_rag:
        return {"route": "rag_search", "target": question, "reason": "规则匹配"}

    return None


def route_query(llm, question: str) -> dict:
    # 规则优先：尝试基于关键词快速路由
    rule_result = rule_based_route(question)
    if rule_result is not None:
        debug_log(f"rule_based_route hit: route={rule_result['route']} target={rule_result['target']}")
        return rule_result

    # LLM 兜底：规则未命中时调用 LLM 路由
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
        _logger.warning(f"LLM 路由判断失败，降级为默认检索: {e}")
        debug_log("route_query error:", repr(e))
        return {
            "route": "rag_search",
            "target": question,
            "reason": "LLM 路由判断失败，降级为默认 RAG 检索路由。",
        }


# =========================
# 查询改写
# =========================
def should_skip_rewrite(question: str) -> bool:
    """判断是否可以跳过LLM改写，直接使用原始问题检索"""
    # 条件1：极短问题（<=8字符）且无代词/模糊表达，视为独立关键词查询
    if len(question) <= 8 and not any(w in question for w in ["它", "这个", "那个", "上面", "之前", "它们", "那些", "其中", "后者", "前者", "刚才"]):
        return True
    return False


def rewrite_question(llm, question: str, cfg: dict = None) -> dict:
    # [P0优化] 长问题（>80字符）已包含充分上下文，无代词指代时跳过改写
    pronoun_indicators = ["这个", "那个", "它", "他们", "上面", "之前", "刚才", "前面"]
    if len(question) > 80 and not any(p in question for p in pronoun_indicators):
        debug_log(f"[ROUTE] 长问题跳过改写: {question[:30]}...")
        return {
            "standalone_question": question,
            "keywords": [],
            "expanded_queries": [question],
        }

    prompt = ChatPromptTemplate.from_template("""你是实验室知识库的检索改写专家。将用户问题改写为最优检索形式。

【输出格式】严格JSON：
{{
  "standalone_question": "补全代词和省略后的完整独立问题",
  "expanded_queries": [
    "原问题的精准改写（保留专业术语）",
    "更宽泛的上位概念查询",
    "关键术语的同义词/缩写展开查询"
  ],
  "hypothetical_docs": "生成2-3个可能包含答案的学术文献摘要片段，每个50-80字，模拟真实文献内容，用'|||'分隔"
}}

【规则】
1. 保留所有专业术语、设备型号、参数名、英文缩写
2. 补全代词（这个/那个/它）为具体指代对象
3. expanded_queries 必须多样化，覆盖不同检索角度
4. 不要解释问题，仅做改写
5. 若问题已清晰完整，standalone_question保持原意
6. hypothetical_docs: 假设你已经知道答案，写2-3个简短的模拟文献摘要片段（每个50-80字），用"|||"分隔

用户问题：{question}""")

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
        # [FIX] 限制单个 query 最大长度，防止 LLM 返回超长改写
        queries = [q[:200].strip() for q in queries if isinstance(q, str) and q.strip() and len(q.strip()) > 0]
        queries = list(dict.fromkeys(queries))

        # [HyDE] 提取假设性文档片段，作为额外检索 query 增强召回
        hyde_text = ""
        try:
            if cfg and cfg.get("ENABLE_HYDE", True):
                # 优先使用新格式 hypothetical_docs（多个假设文档，|||分隔）
                if data.get("hypothetical_docs"):
                    hyde_text = data["hypothetical_docs"]
                    hyde_docs = [d.strip() for d in hyde_text.split("|||") if d.strip()]
                    for doc in hyde_docs[:3]:  # 最多取3个假设文档片段
                        if len(doc) > 20:  # 过滤过短的无效片段
                            queries.append(doc[:200])
                    debug_log(f"[HyDE] 生成 {len(hyde_docs)} 个假设文档片段")
                elif data.get("hypothetical_answer"):
                    # 向后兼容：旧格式单个假设回答
                    hypo = data["hypothetical_answer"].strip()[:200]
                    if hypo:
                        queries.append(hypo)
                        debug_log(f"[HyDE] 已提取假设回答用于检索增强: {hypo[:80]}...")
        except Exception as hyde_err:
            debug_log(f"[HyDE] 提取假设文档失败（静默跳过）: {hyde_err}")

        parsed = {
            "standalone_question": standalone_question if isinstance(standalone_question,
                                                                     str) and standalone_question.strip() else question,
            "keywords": [str(k).strip() for k in keywords if str(k).strip()][:6],
            "expanded_queries": queries[:5],
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
def retrieve_multi_query(vectorstore, queries: List[str], top_k_each: int, max_workers: int = 4):
    """
    🔥 升级版（并行化）：
    - 不再均分K
    - 每个query都拿大K（大召回）
    - 使用 ThreadPoolExecutor 并行执行多路检索
    - 后面再统一裁剪
    """
    all_docs = []

    # [FIX] 修正 Top-K 计算逻辑：每个 query 使用固定大 K 召回，后续由去重和重排裁剪
    # 旧逻辑 num_queries 越多 K 越小（反了），现在每个 query 统一使用 INITIAL_RETRIEVAL_K
    per_query_k = top_k_each  # 每个query使用完整K，后续统一去重裁剪

    def search_one(q):
        t0 = time.perf_counter()
        try:
            docs = vectorstore.similarity_search(q, k=per_query_k)
            elapsed = time.perf_counter() - t0
            debug_log(f"[RECALL] q={repr(q)} k={per_query_k} docs={len(docs)} time={elapsed:.3f}s")
            return docs
        except Exception as e:
            debug_log(f"[RECALL ERROR] q={repr(q)} err={repr(e)}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(search_one, q): q for q in queries}
        for future in as_completed(futures, timeout=30):
            try:
                result = future.result(timeout=10)
                all_docs.extend(result)
            except Exception as e:
                debug_log(f"[WARNING] 检索超时或失败: {e}")

    debug_log(f"[RECALL TOTAL] before merge = {len(all_docs)}")
    return all_docs


def retrieve_sparse(client, collection_name: str, query: str, keywords: List[str], top_k: int) -> List[Document]:
    """
    基于 Qdrant 全文搜索的稀疏检索。
    利用 FieldCondition + MatchText 对 page_content 进行关键词匹配。
    """
    try:
        with Timer(f"retrieve_sparse query={repr(query[:50])} keywords={keywords[:10]}"):
            # 构建搜索词列表：query 分词 + keywords（限制最多10个关键词）
            keywords = keywords[:10]
            search_terms = list(set([t.strip() for t in keywords if t.strip()] + [query.strip()]))
            if not search_terms:
                return []

            # 构建 should 条件：对多个字段做 MatchText
            should_conditions = []
            for term in search_terms:
                if not term:
                    continue
                should_conditions.append(
                    rest.FieldCondition(key="page_content", match=rest.MatchText(text=term))
                )
                should_conditions.append(
                    rest.FieldCondition(key="metadata.doc_title", match=rest.MatchText(text=term))
                )
                should_conditions.append(
                    rest.FieldCondition(key="metadata.section_title", match=rest.MatchText(text=term))
                )
                should_conditions.append(
                    rest.FieldCondition(key="metadata.file_name", match=rest.MatchText(text=term))
                )
                should_conditions.append(
                    rest.FieldCondition(key="metadata.rel_path", match=rest.MatchText(text=term))
                )

            # 限制 MatchText 条件数量，防止条件过多导致超时
            should_conditions = should_conditions[:30]

            if not should_conditions:
                return []

            search_filter = rest.Filter(should=should_conditions)

            # 设置超时保护，防止大量条件导致 Qdrant 查询过慢
            try:
                results, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=top_k,
                    with_payload=True,
                    timeout=10,  # 稀疏检索超时保护
                )
            except Exception as timeout_err:
                debug_log(f"[WARN] retrieve_sparse scroll 超时或失败: {timeout_err}")
                return []

            docs = []
            for hit in results:
                payload = hit.payload or {}
                md = payload.get("metadata", {})

                # 提取 page_content
                content = payload.get("page_content", "") or md.get("page_content", "")
                if not content:
                    continue

                # 构建 Document 对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "file_name": md.get("file_name", ""),
                        "doc_title": md.get("doc_title", ""),
                        "rel_path": md.get("rel_path", ""),
                        "section_title": md.get("section_title", ""),
                        "parent_id": md.get("parent_id", ""),
                        "parent_content": md.get("parent_content", ""),
                    }
                )
                docs.append(doc)

            debug_log(f"retrieve_sparse results={len(docs)}")
            return docs

    except Exception as e:
        debug_log(f"retrieve_sparse error: {repr(e)}")
        return []


def reciprocal_rank_fusion(dense_docs: List[Document], sparse_docs: List[Document],
                           dense_weight: float = 0.7, sparse_weight: float = 0.3,
                           k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) 融合两路检索结果。
    score = dense_weight * 1/(k+rank_dense) + sparse_weight * 1/(k+rank_sparse)
    """
    try:
        def doc_key(doc: Document) -> str:
            """以 page_content 前1000字符 + file_name 作为去重 key"""
            content_prefix = (doc.page_content or "")[:1000]
            file_name = doc.metadata.get("file_name", "") if doc.metadata else ""
            return f"{file_name}||{content_prefix}"

        # 构建 dense 排名映射
        dense_rank = {}
        for rank, doc in enumerate(dense_docs, start=1):
            key = doc_key(doc)
            if key not in dense_rank:
                dense_rank[key] = rank

        # 构建 sparse 排名映射
        sparse_rank = {}
        for rank, doc in enumerate(sparse_docs, start=1):
            key = doc_key(doc)
            if key not in sparse_rank:
                sparse_rank[key] = rank

        # 收集所有唯一文档
        all_keys = set(dense_rank.keys()) | set(sparse_rank.keys())

        # 建立 key -> doc 的映射
        key_to_doc = {}
        for doc in dense_docs:
            key = doc_key(doc)
            if key not in key_to_doc:
                key_to_doc[key] = doc
        for doc in sparse_docs:
            key = doc_key(doc)
            if key not in key_to_doc:
                key_to_doc[key] = doc

        # 计算 RRF 融合分数
        # 对于只在单路出现的文档，使用固定默认排名，避免因两路返回数量不同导致不对称降权
        default_rank = int(os.getenv("INITIAL_RETRIEVAL_K", "64")) + 1
        scored = []
        for key in all_keys:
            d_rank = dense_rank.get(key, default_rank)
            s_rank = sparse_rank.get(key, default_rank)
            rrf_score = dense_weight * (1.0 / (k + d_rank)) + sparse_weight * (1.0 / (k + s_rank))
            scored.append((key, rrf_score))

        # 按融合得分降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的文档列表
        fused_docs = []
        for key, score in scored:
            doc = key_to_doc.get(key)
            if doc:
                fused_docs.append(doc)

        debug_log(f"reciprocal_rank_fusion: dense={len(dense_docs)} sparse={len(sparse_docs)} fused={len(fused_docs)}")
        return fused_docs

    except Exception as e:
        debug_log(f"reciprocal_rank_fusion error: {repr(e)}")
        # 出错时回退为只返回 dense_docs
        return dense_docs


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
            content = clean_retrieval_display_text(doc.page_content)

            exact_key = (file_name, content[:1000])
            if exact_key in seen:
                continue

            seen.add(exact_key)
            unique_docs.append(doc)

        debug_log(f"dedup_docs_exact_only output={len(unique_docs)}")
        return unique_docs


def rerank_docs(reranker, query: str, docs, cfg: dict, top_k: int = None):
    """改进版重排：乘法 boost + 分数阈值过滤"""
    if not docs:
        return []

    if top_k is None:
        top_k = int(cfg.get("FINAL_TOP_K", "16"))

    score_threshold = cfg.get("RERANKER_SCORE_THRESHOLD", 0.3)

    try:
        pairs = [(query, clean_retrieval_display_text(d.page_content)) for d in docs]
        # [FIX] Reranker 分批推理，防止高并发显存溢出
        batch_size = 64
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            all_scores.extend(batch_scores if hasattr(batch_scores, '__iter__') else [batch_scores])
        scores = all_scores

        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        boosted = []
        for doc, score in zip(docs, scores):
            score = float(score)

            # 乘法 Boost（而非加法）
            text_lower = doc.page_content.lower()
            doc_title_lower = (doc.metadata.get("doc_title", "") or "").lower()
            section_title_lower = (doc.metadata.get("section_title", "") or "").lower()

            # Keyword boost: 至少 2 个有意义的词重合即触发
            text_tokens = set(text_lower.split())
            meaningful_overlap = query_tokens & text_tokens
            if len(meaningful_overlap) >= 2:
                score *= cfg.get("KEYWORD_BOOST", 1.05)

            # Title boost: 同时检查 doc_title 和 section_title
            combined_title = f"{doc_title_lower} {section_title_lower}"
            if combined_title.strip() and any(t in combined_title for t in query_tokens if len(t) > 2):
                score *= cfg.get("TITLE_BOOST", 1.08)

            doc.metadata["rerank_score"] = score
            boosted.append((doc, score))

        # 按分数排序
        boosted.sort(key=lambda x: x[1], reverse=True)

        # 分数阈值过滤
        filtered = [(d, s) for d, s in boosted if s >= score_threshold]

        if not filtered:
            # 如果全部低于阈值，至少保留最高分的一个
            filtered = boosted[:1]

        final_docs = [d for d, _ in filtered[:top_k]]

        debug_log(f"Rerank: {len(docs)} → {len(final_docs)} (threshold={score_threshold})")
        return final_docs

    except Exception as e:
        debug_log(f"Rerank error: {repr(e)}")
        return docs[:top_k]


def build_context(docs, max_chars: int = 12000) -> tuple:
    """
    构建带引用编号的上下文，返回 (context_str, source_map)。
    source_map: {1: {"doc_title": "...", "rel_path": "...", "section_title": "..."}, ...}
    """
    parts = []
    source_map = {}
    current_length = 0

    for idx, doc in enumerate(docs, start=1):
        doc_title = doc.metadata.get('doc_title', '未知标题')
        rel_path = doc.metadata.get('rel_path', '未知路径')
        section_title = doc.metadata.get('section_title', '')

        section_line = f"[章节] {section_title}\n" if section_title else ""
        # 如果有父块摘要，展示给 LLM 帮助理解宏观上下文
        parent_summary = doc.metadata.get("parent_summary", "")
        summary_line = f"[段落摘要] {parent_summary}\n" if parent_summary else ""
        content = (
            f"[来源{idx}] [文档标题] {doc_title}\n"
            f"[路径] {rel_path}\n"
            f"{section_line}"
            f"{summary_line}"
            f"{clean_retrieval_display_text(doc.page_content)}"
        )

        doc_len = len(content)

        # 长度校验：如果加入当前块会严重超标，则丢弃排名靠后的父块
        if current_length + doc_len > max_chars and current_length > 0:
            debug_log(f"上下文已达 {current_length} 字符，截断后续 {len(docs) - len(parts)} 个排名较低的文档以防爆炸。")
            break

        parts.append(content)
        source_map[idx] = {
            "doc_title": doc_title,
            "rel_path": rel_path,
            "section_title": section_title,
        }
        current_length += doc_len

    debug_log(f"build_context: source_map size={len(source_map)}")
    context_str = "\n\n====================\n\n".join(parts)
    return context_str, source_map


# =========================
# 覆盖度评估与引用溯源
# =========================
def assess_coverage(reranked_docs: list, config: dict) -> dict:
    """评估知识库对当前查询的覆盖度（综合 max_score + avg_score + doc_count）"""
    if not reranked_docs:
        return {
            "level": "none",
            "hint": "⚠️ 知识库中未找到相关内容，以下回答基于模型通用知识，仅供参考。"
        }

    # 综合评估指标
    scores = [doc.metadata.get("rerank_score", 0) for doc in reranked_docs]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    doc_count = len(reranked_docs)

    debug_log(f"[覆盖度评估] max_score={max_score:.3f} avg_score={avg_score:.3f} doc_count={doc_count}")

    # 综合判断：max_score + avg_score + doc_count
    if max_score >= 0.7 and avg_score >= 0.4:
        return {"level": "high", "hint": None}
    elif max_score >= 0.5 or (avg_score >= 0.35 and doc_count >= 4):
        return {"level": "medium", "hint": "ℹ️ 知识库中的相关内容有限，回答可能不够全面。"}
    elif max_score >= 0.3:
        return {"level": "low", "hint": "⚠️ 知识库覆盖不足，回答可能不够准确，建议查阅原始文档。"}
    else:
        return {
            "level": "very_low",
            "hint": "⚠️ 该问题可能不在知识库范围内，以下回答仅供参考。"
        }


def build_citations(used_docs: list) -> list:
    """从使用的文档中提取引用信息"""
    citations = []
    seen = set()

    for doc in used_docs:
        source = doc.metadata.get("source", "") or doc.metadata.get("rel_path", "") or doc.metadata.get("file_name", "未知来源")
        section = doc.metadata.get("section_title", "") or doc.metadata.get("header_path", "")

        # 去重：同一文档同一章节只引用一次
        key = f"{source}|{section}"
        if key in seen:
            continue
        seen.add(key)

        citation = {
            "source": source,
            "section": section,
            "score": round(doc.metadata.get("rerank_score", 0), 3),
            "summary": doc.metadata.get("parent_summary", ""),  # 传递父块摘要
        }
        citations.append(citation)

    # 按分数排序，最多保留5条引用
    citations.sort(key=lambda x: x["score"], reverse=True)
    return citations[:5]


def get_system_prompt_by_coverage(coverage_level: str) -> str:
    """根据覆盖度级别返回不同的system prompt策略"""
    base_prompt = "你是实验室知识库助手，根据提供的参考资料回答用户问题。"

    if coverage_level == "high":
        return base_prompt + "请基于参考资料准确回答，引用具体内容。"
    elif coverage_level == "medium":
        return base_prompt + "参考资料可能不够全面，请基于已有内容尽可能回答，对不确定的部分请明确指出。"
    elif coverage_level == "low":
        return base_prompt + "参考资料相关性较低，请谨慎回答。如果参考资料无法支撑回答，请明确告知用户并建议查阅原始文档。"
    else:  # very_low or none
        return base_prompt + "当前没有找到高度相关的参考资料。请基于你的通用知识简要回答，并明确告知用户该回答未经知识库验证。"


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

    MAX_DIR_TRAVERSAL = 10000

    with Timer(f"find_best_matching_dirs keyword={keyword}"):
        matches = []
        count_dirs = 0
        traversal_count = 0
        for current_root, dirs, _ in os.walk(root_dir, followlinks=False):
            traversal_count += 1
            if traversal_count > MAX_DIR_TRAVERSAL:
                debug_log(f"[WARN] 目录遍历超过 {MAX_DIR_TRAVERSAL} 限制，停止遍历")
                break

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

    MAX_DIR_TRAVERSAL = 10000

    with Timer(f"list_fs_entries_by_keyword keyword={keyword}"):
        results = []
        count_dirs = 0
        count_files = 0
        traversal_count = 0

        for current_root, dirs, files in os.walk(root_dir, followlinks=False):
            traversal_count += 1
            if traversal_count > MAX_DIR_TRAVERSAL:
                debug_log(f"[WARN] 目录遍历超过 {MAX_DIR_TRAVERSAL} 限制，停止遍历")
                break

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
            try:
                results, next_page_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=min(limit, 256),
                    offset=offset,
                    with_payload=True,
                    timeout=10,  # 文件列表查询超时保护
                )
            except Exception as e:
                debug_log(f"[WARN] list_paths_by_keyword scroll 超时或失败: {e}")
                break
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


# [FIX] answer_stream 增加 chat_history 参数，确保截断后的历史实际参与 prompt 生成
# [OPT] 增加 coverage_level 参数，支持分级 Fallback 策略
def answer_stream(llm, question: str, context: str, route: str, route_reason: str, chat_history: str = "无历史对话", coverage_level: str = "high"):
    # 根据覆盖度级别动态选择 system prompt 策略
    system_instruction = get_system_prompt_by_coverage(coverage_level)

    prompt = ChatPromptTemplate.from_template("""
    {system_instruction}

    请严格遵守以下规则：
    1. 必须且只能依据"上下文结果"回答问题。
    2. 如果用户问"有哪些设备/有哪些文件/目录下有什么/某组有什么设备"，优先根据目录/文件枚举结果作答，完整列出。
    3. 如果用户问的是说明、原理、参数、步骤、操作方法、用途等内容，优先根据文档检索片段回答。
    4. 如果上下文没有足够信息，请明确说明："知识库中未找到足够相关内容"。
    5. 回答要准确、专业、清晰，注意滤除你认为是文档解析错误带来的结果。
    6. 如果依据知识库内容进行回答，结尾必须列出"主要来源"。
    7. 涉及数学符号或公式时，必须使用标准 LaTeX 格式：$公式$。
    8. 在引用知识库内容时，请在相关内容后标注来源编号，格式为 [来源N]。
    9. 如果回答综合了多个来源，请分别标注。
    10. 可以参考"历史对话"理解用户意图的上下文，但回答内容必须基于知识库。
    11. 如果用户问的是"有哪些/有多少/列出"等列举型问题，且参考资料中包含多个相关条目：
        - 首先完整列出所有找到的条目（标题或名称），使用序号列表
        - 然后对排名前3-5个条目提供简要说明或摘要
        - 如果条目过多（超过10个），列出全部标题后建议用户"可输入具体名称查询更详细的内容"

    历史对话：
    {chat_history}

    上下文结果：
    {context}

    用户问题：
    {question}
    """.strip())

    chain = prompt | llm
    return chain.stream({
        "system_instruction": system_instruction,
        "route": route,
        "route_reason": route_reason if route_reason.strip() else "无法解析原因",
        "context": context if context.strip() else "无可用上下文",
        "chat_history": chat_history,
        "question": question,
    })


# =========================
# 主入口：流式
# =========================
def ask_stream(question: str, username: str = "anonymous") -> Generator[dict, None, None]:
    total_start = time.perf_counter()

    try:
        debug_log("=" * 80)
        debug_log(f"ask_stream question={repr(question)}")

        # [FIX] 从磁盘加载用户历史到内存，确保 app 重启后多轮对话不丢失
        if username and username not in _user_histories:
            loaded = load_history(username)
            if loaded:
                with _user_histories_lock:
                    _user_histories[username] = loaded
                debug_log(f"[HISTORY] 从磁盘加载用户 {username} 的历史记录: {len(loaded)} 条")

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

        # [FIX] 获取用户历史并截断，防止历史token无限累积超出预算
        user_history = get_user_chat_history(username)
        user_history = truncate_history(user_history, config.get("MAX_HISTORY_TOKENS", 1500))

        rag_retrievals = []
        file_result = None
        context_parts = []
        rewritten_question = question
        keywords = []
        queries = []
        source_map = {}
        coverage_info = {"level": "high", "hint": None}
        citations = []

        yield {
            "type": "metadata",
            "stage": "route",
            "route": route,
            "route_target": target,
            "route_reason": route_reason,
        }

        if route in {"file_list", "hybrid"}:
            # 通知前端：正在检索知识库（文件搜索）
            if route == "file_list":
                yield {"type": "status", "stage": "searching"}

            with Timer("stage_file_list"):
                file_result = list_catalog_entries(target)
                file_context = build_file_context(file_result)

                # [OPT] hybrid 路由下对文件列表部分进行预算截断，防止占用过多上下文空间
                if route == "hybrid":
                    file_context_budget = int(config.get("HYBRID_FILE_CONTEXT_BUDGET", 1500))
                    if len(file_context) > file_context_budget:
                        file_context = file_context[:file_context_budget] + "\n...（更多结果请使用文件浏览功能查看）"
                        debug_log(f"[HYBRID] file_context 超出预算 {file_context_budget}，已截断至 {len(file_context)} 字符")

                context_parts.append("[目录/文件结果]\n" + file_context)

            yield {
                "type": "tool",
                "tool_name": "list_catalog_entries",
                "content": file_result,
            }

        if route in {"rag_search", "hybrid"}:
            # 通知前端：正在检索知识库
            yield {"type": "status", "stage": "searching"}

            # [智能路由] 判断是否跳过 LLM 改写，直接使用原始问题
            if should_skip_rewrite(question):
                debug_log(f"[智能路由] 跳过改写，直接使用原始问题: {question}")
                rewritten_question = question
                keywords = []
                expanded_queries = []
                queries = [question]
            else:
                with Timer("stage_rewrite"):
                    rewrite_result = rewrite_question(llm, question, cfg=config)
                rewritten_question = rewrite_result["standalone_question"]
                keywords = rewrite_result["keywords"]
                expanded_queries = rewrite_result["expanded_queries"]

                queries = [question, rewritten_question] + expanded_queries
                if keywords:
                    queries.append(" ".join([str(k).strip() for k in keywords if str(k).strip()]))
                queries = list(dict.fromkeys([q.strip() for q in queries if q and q.strip()]))

            debug_log(f"queries_count={len(queries)} queries={queries}")

            with Timer("stage_retrieval"):
                # 检索缓存：基于 queries 的 hash 判断是否命中
                cache_key = hashlib.md5(str(sorted(queries)).encode()).hexdigest()
                cached = _retrieval_cache.get(cache_key)
                if cached is not None:
                    debug_log(f"[CACHE HIT] key={cache_key}")
                    recalled_docs = cached
                else:
                    # 原有稠密检索
                    recalled_docs = retrieve_multi_query(
                        vectorstore,
                        queries,
                        config["INITIAL_RETRIEVAL_K"]
                    )

                    # 如果开启混合检索，增加稀疏检索并融合
                    if config.get("ENABLE_HYBRID_SEARCH", False):
                        # 当 keywords 为空时，使用 query 本身作为唯一搜索词
                        if not keywords:
                            sparse_search_terms = [rewritten_question]
                        else:
                            sparse_search_terms = [rewritten_question] + keywords

                        sparse_docs = retrieve_sparse(
                            runtime["client"],
                            config["QDRANT_COLLECTION_NAME"],
                            rewritten_question,
                            sparse_search_terms,
                            config["INITIAL_RETRIEVAL_K"]
                        )
                        debug_log(f"sparse_docs_count={len(sparse_docs)}")
                        recalled_docs = reciprocal_rank_fusion(
                            recalled_docs, sparse_docs,
                            dense_weight=config["DENSE_WEIGHT"],
                            sparse_weight=config["BM25_WEIGHT"],
                            k=config["RRF_K"]
                        )
                        debug_log(f"after_rrf_fusion={len(recalled_docs)}")

                    # 缓存检索结果
                    _retrieval_cache.set(cache_key, recalled_docs)

                debug_log(f"recalled_docs_before_dedup={len(recalled_docs)}")
                recalled_docs = dedup_docs(recalled_docs)
                debug_log(f"recalled_docs_after_dedup={len(recalled_docs)}")

            with Timer("stage_rerank"):
                child_top_k = config["FINAL_TOP_K"] * 3
                reranked_child_docs = rerank_docs(reranker, rewritten_question, recalled_docs, cfg=config, top_k=child_top_k)

                # 【新增】将重排后的子块展开为大召回的父块
                final_docs = expand_to_parent_docs(reranked_child_docs, cfg=config, top_k=config["FINAL_TOP_K"], qdrant_client=runtime["client"])

            debug_log(f"final_docs_after_parent_expansion={len(final_docs)}")

            # [OPT] 覆盖度评估：基于重排分数判断知识库覆盖情况
            coverage_info = assess_coverage(reranked_child_docs, config)
            debug_log(f"coverage_assessment: level={coverage_info['level']} hint={coverage_info.get('hint')}")

            # [OPT] 引用溯源：收集引用信息
            citations = build_citations(final_docs)
            debug_log(f"citations_count={len(citations)}")

            for i, doc in enumerate(final_docs, start=1):
                raw_content = (doc.page_content or "").strip()
                display_content = clean_retrieval_display_text(raw_content)
                # 因为变成了父块，内容变长，截取预览长度可以稍微放宽
                preview = display_content[:300]

                rag_retrievals.append({
                    "index": i,
                    "doc_title": doc.metadata.get("doc_title", "未知标题"),
                    "file_name": doc.metadata.get("file_name", "未知文件"),
                    "rel_path": doc.metadata.get("rel_path", "未知路径"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "summary": doc.metadata.get("parent_summary", ""),  # 新增：父块摘要
                    "preview": preview,
                    "content": display_content,  # 这里前端展示和传入LLM的已经是完整的父块
                })

                debug_log(f"[DOC {i}] title={doc.metadata.get('doc_title')}")

            with Timer("stage_build_context"):
                max_ctx = config.get("MAX_CONTEXT_CHARS", 12000)
                # [OPT] hybrid 路由下动态分配 RAG 上下文预算：总预算减去已用的文件列表部分
                if route == "hybrid" and context_parts:
                    file_context_used = sum(len(p) for p in context_parts)
                    rag_budget = max(max_ctx - file_context_used, 2000)  # 至少保留 2000 字符给 RAG
                    debug_log(f"[HYBRID] 上下文预算分配: 总预算={max_ctx}, 文件部分已用={file_context_used}, RAG预算={rag_budget}")
                else:
                    rag_budget = max_ctx
                rag_context, source_map = build_context(final_docs, max_chars=rag_budget)
                context_parts.append("[知识库检索结果]\n" + rag_context)

            yield {
                "type": "metadata",
                "stage": "retrieval",
                "rewritten_question": rewritten_question,
                "keywords": keywords,
                "queries": queries,
                "retrievals": rag_retrievals,
                "source_map": source_map,
            }

        final_context = "\n\n".join([x for x in context_parts if x.strip()])
        debug_log(f"final_context_len={len(final_context)}")

        full_answer = ""

        # 通知前端：正在生成回答
        yield {"type": "status", "stage": "generating"}

        # [FIX] 将截断后的 user_history 格式化并传入 answer_stream，确保历史实际生效
        history_text = format_chat_history(user_history)

        with Timer("stage_answer_stream"):
            first_chunk_time = None
            for chunk in answer_stream(llm, question, final_context, route, route_reason, chat_history=history_text, coverage_level=coverage_info["level"]):
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

        # [OPT] 流式回答结束后，发送覆盖度和引用元数据事件
        yield {
            "type": "metadata",
            "stage": "coverage",
            "coverage": coverage_info,
            "citations": citations,
        }

        append_user_chat_history(username, "user", question)
        append_user_chat_history(username, "assistant", full_answer)

        total_elapsed = time.perf_counter() - total_start
        debug_log(f"ask_stream done total_elapsed={total_elapsed:.3f}s answer_len={len(full_answer)}")

        yield {
            "type": "final",
            "content": full_answer,
            "file_result": file_result,
            "route": route,
            "route_reason": route_reason,
            "rewritten_question": rewritten_question,
            "keywords": keywords,
            "queries": queries,
            "source_map": source_map,
            "coverage": coverage_info,
            "citations": citations,
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
    coverage_info = {"level": "high", "hint": None}
    citations = []

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
            elif stage == "coverage":
                coverage_info = item.get("coverage", coverage_info)
                citations = item.get("citations", citations)
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
            rewritten_question = item.get("rewritten_question", rewritten_question)
            keywords = item.get("keywords", keywords)
            queries = item.get("queries", queries)
            coverage_info = item.get("coverage", coverage_info)
            citations = item.get("citations", citations)

    return {
        "answer": answer,
        "history": [],
        "retrievals": retrievals,
        "rewritten_question": rewritten_question,
        "keywords": keywords,
        "queries": queries,
        "route": route,
        "route_reason": route_reason,
        "file_result": file_result,
        "coverage": coverage_info,
        "citations": citations,
    }
