# 核心 RAG 运行时是进程内唯一模型与检索资源所有者。
import os
import re
import json
import time
import fcntl
import shutil
import hashlib
import threading
import traceback
import tempfile
import urllib.error
import urllib.request
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
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

from .logger import get_logger
from .paths import PROJECT_ROOT
from . import lexicon

load_dotenv(PROJECT_ROOT / ".env")

# =========================
# 全局状态
# =========================
_runtime: Optional[dict] = None
_runtime_building = False
_runtime_prewarm_scheduled = False
_runtime_last_error: Optional[str] = None
_runtime_last_attempt = 0.0

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
HISTORY_DIR = str(PROJECT_ROOT / "data" / "chat_histories")
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
                with open(path + ".lock", "a+", encoding="utf-8") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_SH)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
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
    with open(path + ".lock", "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            _atomic_write_json(path, history)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _atomic_write_json(path: str, data) -> None:
    """在同目录写临时文件后原子替换，避免并发或崩溃留下半截 JSON。"""
    fd, temp_path = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def append_to_user_history(username: str, role: str, content: str):
    """向指定用户当日历史追加一条带时间戳的对话记录"""
    with _user_histories_lock:
        today = datetime.now().strftime("%Y-%m-%d")
        path = get_history_path(username, today)
        # 读取当日已有记录
        history = []
        with open(path + ".lock", "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        history = []

                history.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
                daily_max = int(os.getenv("HISTORY_DAILY_MAX", "200"))
                if len(history) > daily_max:
                    history = history[-daily_max:]
                _atomic_write_json(path, history)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


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
        _logger.info(f"已清理 {removed_count} 个过期历史文件")


# =========================
# Debug
# =========================
def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y", "on"}


def sanitize_summary(value: object) -> str:
    """移除模型思考块及已知的整段回答污染，避免摘要泄漏到最终上下文。"""
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    polluted_markers = ("知识库中未找到足够相关内容", "用户现在问的是", "上下文结果：")
    if any(marker in text for marker in polluted_markers):
        return ""
    return text.strip()


def sanitize_generated_answer(value: object) -> str:
    """清理回答中泄漏的思考块，以及模型在答案末尾复述的原始上下文。"""
    text = str(value or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(
        r"\n\s*(?:```[^\n]*\n\s*)?上下文结果：.*$",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    return text.rstrip("` \n")


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
                # 返回内容、摘要与路径层级，供上下文组装标注来源
                result[pid] = {
                    "content": content,
                    "summary": sanitize_summary(payload.get("parent_summary", "")),
                    "header_path": payload.get("header_path", ""),
                    "source_label": payload.get("parent_source_label", ""),
                    "rel_path": payload.get("rel_path", ""),
                }
    except Exception as e:
        debug_log(f"[DEGRADED][parent-collection] 批量获取父块失败: {e}")
    return result


def _truncate_at_boundary(text: str, limit: int) -> str:
    """按段落/换行边界截断，避免把表格行或型号参数从中间切断。"""
    if len(text) <= limit:
        return text
    window = text[:limit]
    boundary = max(window.rfind("\n\n"), window.rfind("\n"))
    if boundary >= limit // 2:
        return window[:boundary].rstrip()
    return window.rstrip()


def expand_to_parent_docs(child_docs: List[Document], cfg: dict, top_k: int = None, qdrant_client=None) -> List[Document]:
    """子块展开为父块，按行/表格边界截断并保留命中子块的分数证据。"""
    cfg = cfg or {}
    if top_k is None:
        top_k = int(cfg.get("FINAL_TOP_K", 16))
    # 父块展开预算与最终上下文预算分离：前者需容纳多个来源的候选，
    # 后者由 build_context 控制入 prompt 的字符数，两者不必相同。
    max_context_chars = int(cfg.get("PARENT_EXPAND_MAX_CHARS", 9000))
    # 同一来源文件最多保留的父块数，避免单一长手册垄断候选榜
    max_per_source = max(1, int(cfg.get("PARENT_MAX_PER_SOURCE", 3)))

    # 收集 parent_id、最佳子块，以及同一父块下所有命中子块的分数
    needed_parent_ids = set()
    best_children = {}
    child_scores_by_parent = {}
    for child in child_docs:
        pid = child.metadata.get("parent_id")
        if not pid:
            continue
        score = float(child.metadata.get("rerank_score", 0) or 0)
        child_scores_by_parent.setdefault(pid, []).append(score)
        if pid not in best_children or score > best_children[pid][1]:
            best_children[pid] = (child, score)
            needed_parent_ids.add(pid)

    # 批量从 parent collection 获取父块内容、摘要与路径层级（带降级策略）
    parent_contents = {}
    parent_summaries = {}
    parent_extras = {}
    if qdrant_client and needed_parent_ids:
        parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
        try:
            parent_data = batch_get_parents(qdrant_client, parent_collection, list(needed_parent_ids))
            for pid, data in parent_data.items():
                if isinstance(data, dict):
                    parent_contents[pid] = data.get("content", "")
                    parent_summaries[pid] = sanitize_summary(data.get("summary", ""))
                    parent_extras[pid] = {
                        "header_path": data.get("header_path", ""),
                        "parent_source_label": data.get("source_label", ""),
                        "rel_path": data.get("rel_path", ""),
                    }
                else:
                    # 向后兼容旧格式（直接是字符串）
                    parent_contents[pid] = data
            debug_log(f"从 parent collection 获取到 {len(parent_contents)}/{len(needed_parent_ids)} 个父块")
        except Exception as e:
            debug_log(f"[DEGRADED][parent-collection] 查询失败，降级使用子块 metadata: {e}")
            parent_contents = {}
            parent_summaries = {}
            for child in child_docs:
                pid = child.metadata.get("parent_id")
                if pid:
                    if "parent_content" in child.metadata:
                        parent_contents[pid] = child.metadata.get("parent_content", "")
                    parent_summaries[pid] = sanitize_summary(child.metadata.get("parent_summary", ""))

    sorted_children = sorted(best_children.values(), key=lambda x: x[1], reverse=True)

    parent_docs = []
    seen_parent_ids = set()
    source_counts = {}
    accumulated_length = 0

    for child, score in sorted_children:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        # 来源配额：同一文件的父块已被优先高分块占满时，把名额让给其他文档
        source_key = (child.metadata.get("source") or child.metadata.get("rel_path")
                      or child.metadata.get("file_name") or "").lower()
        if source_key and source_counts.get(source_key, 0) >= max_per_source:
            continue

        parent_content = parent_contents.get(parent_id, "") or child.metadata.get("parent_content", "")
        parent_summary = parent_summaries.get(parent_id, "") or sanitize_summary(
            child.metadata.get("parent_summary", "")
        )

        # 最终降级：旧数据无 parent_content 时使用子块内容兜底
        if not parent_content:
            parent_content = child.page_content
            debug_log(f"[DEGRADED][parent-collection] parent_id={parent_id} 无父块内容，使用子块兜底")

        # 单个父块独立限长（不再随总预算折半放大）；按行边界截断，避免腰斩表格行与参数表
        max_single_content = int(cfg.get("PARENT_MAX_SINGLE_CHARS", 1500))
        if len(parent_content) > max_single_content:
            parent_content = _truncate_at_boundary(parent_content, max_single_content)
            debug_log(f"[WARN] parent_content 过大，已按行边界截断至 {len(parent_content)} 字符")

        content_length = len(parent_content)
        if accumulated_length + content_length > max_context_chars and parent_docs:
            debug_log(f"上下文长度控制: 已达 {accumulated_length} 字符, 停止展开")
            break

        parent_meta = {k: v for k, v in child.metadata.items() if k != "parent_content"}
        parent_meta["is_parent"] = True
        if parent_summary:
            parent_meta["parent_summary"] = parent_summary
        # 父集合的章节路径/出处优先于子块 metadata，保证引用准确
        for key, value in parent_extras.get(parent_id, {}).items():
            if value:
                parent_meta[key] = value
        parent_meta["child_scores"] = sorted(
            child_scores_by_parent.get(parent_id, []), reverse=True
        )[:5]

        parent_docs.append(Document(page_content=parent_content, metadata=parent_meta))
        seen_parent_ids.add(parent_id)
        if source_key:
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        accumulated_length += content_length

        if len(parent_docs) >= top_k:
            break

    debug_log(f"父块展开: {len(child_docs)} 子块 → {len(parent_docs)} 父块 "
              f"(来源数={len(source_counts)} 总长={accumulated_length} 配额={max_per_source})")
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
# 目录树快照缓存：value 为 (timestamp, snapshot)，实际 TTL 由 CATALOG_SNAPSHOT_TTL 在读取时判定
_catalog_snapshot_cache = TTLCache(maxsize=8, ttl=86400)

# 目录树快照的遍历上限（目录数）；配置化后默认大幅提高以覆盖完整语料目录树，
# 否则大共享盘下靠后的目录（如 设备操作指南/VLP小组）不会被纳入快照，
# 导致 file_list / hybrid 的目录枚举错配到名称子串相近的噪声路径。
_CATALOG_MAX_TRAVERSAL = 200000
_CATALOG_EXCLUDED_DIRS = {".svn", ".git", ".hg", ".bzr", "__pycache__", "node_modules"}


def _is_excluded_catalog_dir(name: str) -> bool:
    return name in _CATALOG_EXCLUDED_DIRS or name.startswith(".")


def _build_catalog_snapshot(root_dir: str, exclude_hidden: bool = True, cfg: dict = None) -> list:
    """单次 os.walk 产出目录树快照。

    元素为 (kind, name, rel_path)，顺序与原先两次独立遍历一致：每个目录下先子目录后文件。
    排除 .svn/.git 等版本控制目录后不再递归，既消除噪声也大幅减少网络盘 IO。
    traversal 上限由 `CATALOG_MAX_TRAVERSAL` 控制（默认大幅提高以覆盖完整语料目录树）。
    """
    max_traversal = int((cfg or {}).get("CATALOG_MAX_TRAVERSAL", _CATALOG_MAX_TRAVERSAL))
    entries = []
    traversal_count = 0
    for current_root, dirs, files in os.walk(root_dir, followlinks=False):
        traversal_count += 1
        if traversal_count > max_traversal:
            debug_log(f"[WARN] 目录遍历超过 {max_traversal} 限制，停止遍历（请调大 CATALOG_MAX_TRAVERSAL）")
            break
        if exclude_hidden:
            dirs[:] = [d for d in dirs if not _is_excluded_catalog_dir(d)]
        rel_current = os.path.relpath(current_root, root_dir)
        if rel_current == ".":
            rel_current = ""
        for d in dirs:
            rel_path = path_to_unix(os.path.join(rel_current, d) if rel_current else d)
            entries.append(("directory", d, rel_path))
        for f in files:
            if exclude_hidden and f.startswith("."):
                continue
            rel_path = path_to_unix(os.path.join(rel_current, f) if rel_current else f)
            entries.append(("file", f, rel_path))
    debug_log(f"[CATALOG] snapshot built entries={len(entries)} traversal={traversal_count}")
    return entries


def _get_catalog_snapshot(root_dir: str, cfg: dict = None) -> list:
    """取目录树快照；缓存键纳入根目录、索引版本与排除开关，TTL 由配置控制。"""
    cfg = cfg or {}
    enabled = bool(cfg.get("ENABLE_CATALOG_SNAPSHOT_CACHE", True))
    ttl = max(0, int(cfg.get("CATALOG_SNAPSHOT_TTL", 300)))
    exclude = bool(cfg.get("CATALOG_EXCLUDE_HIDDEN_DIRS", True))
    version = str(cfg.get("RETRIEVAL_INDEX_VERSION", "1"))
    key = hashlib.md5(f"{root_dir}|{version}|{exclude}".encode()).hexdigest()

    if enabled:
        cached = _catalog_snapshot_cache.get(key)
        if cached is not None:
            ts, snapshot = cached
            if time.time() - ts < ttl:
                debug_log(f"[CATALOG][CACHE HIT] root={root_dir} entries={len(snapshot)}")
                return snapshot

    with Timer("build_catalog_snapshot"):
        snapshot = _build_catalog_snapshot(root_dir, exclude, cfg=cfg)
    if enabled:
        _catalog_snapshot_cache.set(key, (time.time(), snapshot))
    return snapshot


def load_config() -> dict:
    cfg = {
        "DEBUG_MODE": _env_bool("DEBUG_MODE", "true"),

        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "lab-secret-key"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct"),
        "VLLM_INFERENCE_PROBE_TIMEOUT": float(os.getenv("VLLM_INFERENCE_PROBE_TIMEOUT", "15")),
        "VLLM_ENABLE_THINKING": os.getenv("VLLM_ENABLE_THINKING", "false").lower() == "true",

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

        "KNOWLEDGE_BASE_ROOT": os.getenv("KNOWLEDGE_BASE_ROOT", "/mnt/cpu_share").strip(),
        "DOCS_PATH": os.getenv("DOCS_PATH", "/mnt/cpu_share").strip(),
        "ENABLE_FILESYSTEM_TOOL": os.getenv("ENABLE_FILESYSTEM_TOOL", "true").lower() == "true",
        "FILE_SEARCH_LIMIT": int(os.getenv("FILE_SEARCH_LIMIT", "200")),
        "CATALOG_SCAN_LIMIT": int(os.getenv("CATALOG_SCAN_LIMIT", "20000")),
        "DIRECTORY_CHILD_LIMIT": int(os.getenv("DIRECTORY_CHILD_LIMIT", "200")),
        # 目录树单次遍历快照：消除 catalog 每次请求两次全盘 os.walk 的 IO
        "ENABLE_CATALOG_SNAPSHOT_CACHE": os.getenv("ENABLE_CATALOG_SNAPSHOT_CACHE", "true").lower() == "true",
        "CATALOG_SNAPSHOT_TTL": int(os.getenv("CATALOG_SNAPSHOT_TTL", "300")),
        # 排除 .svn/.git 等版本控制与隐藏目录，避免路径匹配把噪声目录当成设备
        "CATALOG_EXCLUDE_HIDDEN_DIRS": os.getenv("CATALOG_EXCLUDE_HIDDEN_DIRS", "true").lower() == "true",
        # 目录树快照遍历上限（目录数）；默认大幅提高以覆盖完整语料目录树，避免靠后
        # 目录（如 设备操作指南/VLP小组）缺失导致 file_list / hybrid 目录枚举错配
        "CATALOG_MAX_TRAVERSAL": int(os.getenv("CATALOG_MAX_TRAVERSAL", "200000")),
        # hybrid 目录驱动查询去重后的最大路数（完整设备清单仍由目录枚举结果提供）
        "MAX_HYBRID_QUERIES": int(os.getenv("MAX_HYBRID_QUERIES", "8")),

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
        "HYBRID_FILE_CONTEXT_BUDGET": int(os.getenv("HYBRID_FILE_CONTEXT_BUDGET", "1500")),
        "HEALTHCHECK_TIMEOUT": float(os.getenv("HEALTHCHECK_TIMEOUT", "3")),
        "RUNTIME_RETRY_INTERVAL": float(os.getenv("RUNTIME_RETRY_INTERVAL", "30")),

        # 多路检索预算：Dense 与 HyDE 分开计数，避免 queries[:5] 截掉假设文档
        "MAX_DENSE_QUERIES": int(os.getenv("MAX_DENSE_QUERIES", "4")),
        "MAX_HYDE_QUERIES": int(os.getenv("MAX_HYDE_QUERIES", "2")),
        "MAX_SPARSE_CONDITIONS": int(os.getenv("MAX_SPARSE_CONDITIONS", "30")),
        "ENABLE_HISTORY_AWARE_REWRITE": os.getenv("ENABLE_HISTORY_AWARE_REWRITE", "true").lower() == "true",
        # 索引版本参与缓存键；入库逻辑变更后递增即可失效旧缓存
        "RETRIEVAL_INDEX_VERSION": os.getenv("RETRIEVAL_INDEX_VERSION", "2"),
        # 覆盖度阈值与重排分数同口径，集中到配置避免散落硬编码
        "COVERAGE_HIGH_THRESHOLD": float(os.getenv("COVERAGE_HIGH_THRESHOLD", "0.7")),
        "COVERAGE_MEDIUM_THRESHOLD": float(os.getenv("COVERAGE_MEDIUM_THRESHOLD", "0.5")),
        "COVERAGE_LOW_THRESHOLD": float(os.getenv("COVERAGE_LOW_THRESHOLD", "0.3")),
        "RERANK_QUERY_MODE": os.getenv("RERANK_QUERY_MODE", "both"),
        "RERANKER_ALLOW_LOW_SCORE_FALLBACK": os.getenv("RERANKER_ALLOW_LOW_SCORE_FALLBACK", "false").lower() == "true",
        # 重排过滤策略：绝对阈值 + 相对 Top-1 比例 + 保底条数（后两项默认关闭，保持既有行为）
        "RERANKER_RELATIVE_RATIO": float(os.getenv("RERANKER_RELATIVE_RATIO", "0")),
        "RERANKER_MIN_KEEP": int(os.getenv("RERANKER_MIN_KEEP", "0")),
        # 拒答闸门：最终父块中关键词零命中且 Top-1 分数偏低时判为知识库无答案
        "ENABLE_ANSWERABILITY_GATE": os.getenv("ENABLE_ANSWERABILITY_GATE", "true").lower() == "true",
        "ANSWERABILITY_TOP_SCORE_FLOOR": float(os.getenv("ANSWERABILITY_TOP_SCORE_FLOOR", "0.90")),
        "ANSWERABILITY_MIN_KEYWORD_HITS": int(os.getenv("ANSWERABILITY_MIN_KEYWORD_HITS", "1")),
        "ANSWERABILITY_MIN_OVERLAP_HITS": int(os.getenv("ANSWERABILITY_MIN_OVERLAP_HITS", "2")),
        # 上下文前检索词元预算（型号词元化修复后需略放大）
        "MAX_SPARSE_TERMS": int(os.getenv("MAX_SPARSE_TERMS", "12")),
        # 父块展开：预算与最终上下文解耦，并限制单一来源垄断候选
        "PARENT_EXPAND_MAX_CHARS": int(os.getenv("PARENT_EXPAND_MAX_CHARS", "9000")),
        "PARENT_MAX_SINGLE_CHARS": int(os.getenv("PARENT_MAX_SINGLE_CHARS", "1500")),
        "PARENT_MAX_PER_SOURCE": int(os.getenv("PARENT_MAX_PER_SOURCE", "3")),
        "NOTATION_MATCH_BONUS": float(os.getenv("NOTATION_MATCH_BONUS", "0.15")),
        "ENABLE_CONTEXT_NOISE_FILTER": os.getenv("ENABLE_CONTEXT_NOISE_FILTER", "true").lower() == "true",
        # v3.3.0 配置化检索规则与语料驱动词典
        "LEXICON_PATH": os.getenv("LEXICON_PATH", "config/retrieval_lexicon.json").strip(),
        "CORPUS_LEXICON_PATH": os.getenv("CORPUS_LEXICON_PATH", "data/corpus_lexicon.json").strip(),
        "ENABLE_CORPUS_LEXICON": os.getenv("ENABLE_CORPUS_LEXICON", "true").lower() == "true",
        "ENABLE_NOTATION_SPECIALIZATION": os.getenv("ENABLE_NOTATION_SPECIALIZATION", "true").lower() == "true",
        "ENABLE_EVIDENCE_RERANK": os.getenv("ENABLE_EVIDENCE_RERANK", "true").lower() == "true",
        "EVIDENCE_RERANK_WEIGHT": float(os.getenv("EVIDENCE_RERANK_WEIGHT", "0.12")),
        # 强证据先验：日期/文档标题词元/型号精确命中身份字段时的加性加分（封顶）
        "EVIDENCE_PRIOR_WEIGHT": float(os.getenv("EVIDENCE_PRIOR_WEIGHT", "0.12")),
        "EVIDENCE_PRIOR_CAP": float(os.getenv("EVIDENCE_PRIOR_CAP", "0.24")),

        # 本机 MCP 旁路配置；Token 为空表示内部接口禁用。
        "MCP_CONDA_ENV": os.getenv("MCP_CONDA_ENV", "rag-mcp"),
        "MCP_INTERNAL_TOKEN": os.getenv("MCP_INTERNAL_TOKEN", "").strip(),
        "MCP_INTERNAL_BASE_URL": os.getenv("MCP_INTERNAL_BASE_URL", "http://127.0.0.1:5000").rstrip("/"),
        "MCP_HTTP_TIMEOUT": float(os.getenv("MCP_HTTP_TIMEOUT", "120")),
        "MCP_QUERY_MAX_CHARS": int(os.getenv("MCP_QUERY_MAX_CHARS", "4000")),
        "MCP_RESULT_LIMIT": int(os.getenv("MCP_RESULT_LIMIT", "6")),
        "MCP_RESULT_MAX_CHARS": int(os.getenv("MCP_RESULT_MAX_CHARS", "1500")),

    }
    # 配置验证
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
    enable_thinking = (cfg or {}).get("VLLM_ENABLE_THINKING", False)
    debug_log(f"build_llm model={model_name} base_url={base_url} temperature={temperature} max_tokens={max_tokens}")
    return ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        streaming=True,
        timeout=120,
    )


def _connect_qdrant_with_retry(host: str, port: int, max_retries: int = 1, timeout: float = 3) -> QdrantClient:
    """带重试机制的 Qdrant 客户端连接，防止短暂网络抖动导致启动失败"""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=host, port=port, timeout=timeout)
            client.get_collections()  # 验证连接可用
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                _logger.warning(f"[WARNING] Qdrant 连接失败 (尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                _logger.error(f"[ERROR] Qdrant 连接失败，已重试 {max_retries} 次: {e}")
                raise


def build_vectorstore(host: str, port: int, collection_name: str, embeddings, client=None):
    debug_log(f"build_vectorstore host={host} port={port} collection={collection_name}")
    client = client or _connect_qdrant_with_retry(host, port)
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
        secret_keys = {"VLLM_API_KEY", "MCP_INTERNAL_TOKEN"}
        safe_config = {key: value for key, value in config.items() if key not in secret_keys}
        debug_log("config=", json.dumps(safe_config, ensure_ascii=False))

        # 先验证远程依赖，避免 Qdrant 不可用时仍占用 GPU 加载模型。
        client = _connect_qdrant_with_retry(
            config["QDRANT_HOST"],
            config["QDRANT_PORT"],
            max_retries=1,
            timeout=config["HEALTHCHECK_TIMEOUT"],
        )
        required_collections = (
            config["QDRANT_COLLECTION_NAME"],
            config["QDRANT_PARENT_COLLECTION"],
        )
        missing_collections = []
        for collection_name in required_collections:
            try:
                client.get_collection(collection_name)
            except Exception:
                missing_collections.append(collection_name)
        if missing_collections:
            raise RuntimeError(f"Qdrant 缺少必要集合: {', '.join(missing_collections)}")
        embeddings = build_embeddings(config["EMBEDDING_MODEL_NAME"], config["EMBEDDING_DEVICE"])
        vectorstore = build_vectorstore(
            config["QDRANT_HOST"],
            config["QDRANT_PORT"],
            config["QDRANT_COLLECTION_NAME"],
            embeddings,
            client=client,
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


class RuntimeUnavailableError(RuntimeError):
    """运行时依赖尚未就绪。对外只应返回通用服务提示。"""


def get_runtime():
    """获取单例运行时；失败后进入冷却，避免请求反复触发模型加载。"""
    global _runtime, _runtime_building, _runtime_last_error, _runtime_last_attempt
    if _runtime is not None:
        return _runtime
    cfg = load_config()
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        now = time.monotonic()
        if _runtime_building:
            raise RuntimeUnavailableError("运行时正在初始化")
        if _runtime_last_error and now - _runtime_last_attempt < cfg["RUNTIME_RETRY_INTERVAL"]:
            raise RuntimeUnavailableError("运行时依赖暂不可用")
        _runtime_building = True
        _runtime_last_attempt = now
    try:
        runtime = build_runtime()
    except Exception as exc:
        with _runtime_lock:
            _runtime_last_error = repr(exc)
        raise RuntimeUnavailableError("运行时依赖暂不可用") from exc
    else:
        with _runtime_lock:
            _runtime = runtime
            _runtime_last_error = None
        return runtime
    finally:
        with _runtime_lock:
            _runtime_building = False


def start_runtime_prewarm() -> None:
    """后台触发一次运行时初始化；已有实例、初始化中或冷却期内均直接返回。"""
    def _worker():
        global _runtime_prewarm_scheduled
        try:
            get_runtime()
            _logger.info("RAG 运行时后台预热完成")
        except RuntimeUnavailableError as exc:
            _logger.warning(f"RAG 运行时后台预热暂不可用: {exc}")
        finally:
            with _runtime_lock:
                _runtime_prewarm_scheduled = False

    global _runtime_prewarm_scheduled
    with _runtime_lock:
        if _runtime is not None or _runtime_building or _runtime_prewarm_scheduled:
            return
        _runtime_prewarm_scheduled = True
    threading.Thread(target=_worker, daemon=True, name="rag-runtime-prewarm").start()


def _http_json_reachable(url: str, timeout: float, api_key: str = "") -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8"))
            return isinstance(data, dict)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _qdrant_collections_ready(host: str, port: int, collection_names, timeout: float) -> bool:
    """Qdrant 仅在连通且所有必要集合存在时才视为可用于检索。"""
    request = urllib.request.Request(f"http://{host}:{port}/collections")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8"))
            collections = data.get("result", {}).get("collections", [])
            existing = {
                item.get("name")
                for item in collections
                if isinstance(item, dict) and item.get("name")
            }
            return set(collection_names).issubset(existing)
    except (AttributeError, OSError, TypeError, ValueError, urllib.error.URLError):
        return False


def get_runtime_status() -> dict:
    """返回快速健康快照，不触发同步模型加载。"""
    cfg = load_config()
    timeout = cfg["HEALTHCHECK_TIMEOUT"]
    runtime = _runtime
    with ThreadPoolExecutor(max_workers=2) as executor:
        qdrant_future = executor.submit(
            _qdrant_collections_ready,
            cfg["QDRANT_HOST"],
            cfg["QDRANT_PORT"],
            (cfg["QDRANT_COLLECTION_NAME"], cfg["QDRANT_PARENT_COLLECTION"]),
            timeout,
        )
        llm_future = executor.submit(
            _http_json_reachable,
            cfg["VLLM_BASE_URL"].rstrip("/") + "/models",
            timeout,
            cfg["VLLM_API_KEY"],
        )
        components = {
            "embedding": bool(runtime and runtime.get("vectorstore")),
            "reranker": bool(runtime and runtime.get("reranker")),
            "qdrant": qdrant_future.result(),
            "llm": llm_future.result(),
        }
    if components["qdrant"] and runtime is None:
        start_runtime_prewarm()
    return components


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
    has_markdown_structure = bool(re.search(r"(?m)^\s*(?:#{1,6}\s+|\|.*\|\s*$|[-*+]\s+|```)", text))
    if has_markdown_structure:
        # 表格、标题和列表必须保留换行，否则单元格关系会再次退化为扁平文本。
        text = strip_structured_prefix(text)
        text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines())
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    else:
        text = compress_repeated_text(text)
        text = strip_structured_prefix(text)
        text = normalize_retrieval_text(text)

    return text if text else raw.strip()


# =========================
# 查询路由
# =========================
# 小组缩写归一改由语料派生词典（data/corpus_lexicon.json）驱动，
# 词典缺失时回退 config/retrieval_lexicon.json 的静态别名；不再在代码里写死三元组。


def extract_catalog_target(question: str, cfg: dict = None) -> str:
    """从列举型问题中提取适合目录匹配的实体，避免把完整意图句当作路径关键词。

    cfg 为可选参数，仅用于读取 ENABLE_CORPUS_LEXICON 开关；不传时沿用运行时配置。
    """
    text = normalize_retrieval_text(question).strip()
    patterns = (
        r"([A-Za-z][A-Za-z0-9._-]{0,31}\s*小组)",
        r"([\u4e00-\u9fff]{1,12}(?:小组|课题组|实验室))",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            target = re.sub(r"^(?:请问|请|关于|查询|查找|列出|介绍|我想了解)+", "", match.group(1))
            if target:
                return target.strip()

    # 「A2X 有哪些设备」这类不带"小组"的问法归一到真实目录名。
    # 别名来自语料派生词典，按长度降序避免短别名抢先匹配长别名。
    if "小组" not in text:
        upper_text = text.upper()
        aliases = lexicon.get_group_aliases(cfg)
        for group in sorted(aliases, key=len, reverse=True):
            if group.upper() in upper_text:
                resolved = aliases[group]
                return resolved if resolved.endswith("小组") else f"{resolved}小组"

    prefix = re.split(r"(?:的)?(?:设备|文件|目录|资料|清单|使用规范|操作规范)", text, maxsplit=1)[0]
    prefix = re.sub(r"^(?:请问|请|关于|查询|查找|列出|介绍|我想了解)+", "", prefix).strip(" ：:，,")
    # 去掉列举型疑问词，避免把「有哪些」等残留当成路径关键词
    prefix = re.sub(r"(?:有哪些|有什么|列出|共有|总共有|一共有|所有的|全部的)$", "", prefix).strip()
    return prefix if len(prefix) >= 2 else text


def rule_based_route(question: str) -> Optional[dict]:
    """
    基于规则的快速路由。
    如果能确定意图则返回路由结果 dict，否则返回 None 表示需要 LLM 判断。
    """
    # 四张词表统一来自检索词典，代码只保留通用判定逻辑
    keywords = lexicon.load_lexicon()["route_keywords"]
    file_list_keywords = keywords["file_list"]
    rag_search_keywords = keywords["rag_search"]
    ambiguous_words = keywords["ambiguous"]
    academic_context = keywords["academic_context"]

    hit_file = any(kw in question for kw in file_list_keywords)
    hit_rag = any(kw in question for kw in rag_search_keywords)
    route_target = extract_catalog_target(question) if hit_file else question

    # 歧义词上下文判断
    hit_ambiguous = any(w in question for w in ambiguous_words)
    if hit_ambiguous:
        if hit_rag and hit_file:
            return {"route": "hybrid", "target": route_target, "reason": "规则匹配"}
        elif hit_rag:
            return {"route": "rag_search", "target": question, "reason": "规则匹配"}
        elif hit_file:
            return {"route": "file_list", "target": route_target, "reason": "规则匹配"}
        else:
            # 歧义词单独出现，检查学术上下文（要求至少2个学术词共现）
            academic_context_count = sum(1 for w in academic_context if w in question)
            if academic_context_count >= 2:
                return {"route": "rag_search", "target": question, "reason": "规则匹配(学术上下文)"}
            return None  # 交给LLM兜底

    if hit_file and hit_rag:
        return {"route": "hybrid", "target": route_target, "reason": "规则匹配"}
    elif hit_file:
        return {"route": "file_list", "target": route_target, "reason": "规则匹配"}
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

    边界规则：
    - “周报格式是什么”“报告怎么写”“提交要求是什么”是在询问文档内容，必须选择 rag_search。
    - 不能因为答案可能存在某个文件或模板中，就选择 file_list；只有用户明确要求列出/定位文件、目录、路径或设备清单时才选择 file_list。

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

        # LLM 偶尔会把“格式/要求是什么”误解成查找模板文件；对无明确目录意图的内容问题做保守纠偏。
        content_terms = ("格式", "规范", "要求", "步骤", "方法", "如何", "怎么", "说明", "原理", "参数")
        explicit_catalog_terms = (
            "目录", "文件列表", "文件清单", "有哪些文件", "文件有哪些", "列出文件", "列出目录",
            "路径", "在哪里", "位置", "有哪些设备", "有什么设备", "设备清单", "多少篇", "几篇",
        )
        if (
                route in {"file_list", "hybrid"}
                and any(term in question for term in content_terms)
                and not any(term in question for term in explicit_catalog_terms)
        ):
            route = "rag_search"
            target = question
            reason = "查询格式、规范或方法等文档内容，且未明确要求枚举目录或文件。"

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
def is_notation_format_question(question: str, cfg: dict = None) -> bool:
    """识别排版规范问题；词表来自检索词典，能力由 ENABLE_NOTATION_SPECIALIZATION 控制。"""
    cfg = cfg if cfg is not None else {}
    if not bool(cfg.get("ENABLE_NOTATION_SPECIALIZATION", True)):
        return False
    notation = lexicon.load_lexicon()["notation"]
    if not bool(notation.get("enabled", True)):
        return False
    has_object = any(term in question for term in notation["object_terms"])
    has_format = any(term in question for term in notation["format_terms"])
    return has_object and has_format and any(term in question for term in notation["context_terms"])


def build_notation_retrieval_plan() -> Dict[str, object]:
    """返回词典中的确定性检索计划，避免 HyDE 引入通用惯例污染。"""
    return dict(lexicon.load_lexicon()["notation"]["retrieval_plan"])


def should_skip_rewrite(question: str, chat_history: str = "") -> bool:
    """判断是否可以跳过 LLM 改写；有对话历史时需要改写来消解代词指代。

    合并原先分散在 should_skip_rewrite / rewrite_question 里的两套跳过规则，
    保持单参数调用的既有行为不变。
    """
    history = (chat_history or "").strip()
    if history and history != "无历史对话":
        return False
    pronouns = ("它", "这个", "那个", "上面", "之前", "它们", "那些", "其中", "后者", "前者", "刚才", "他们", "前面")
    question_markers = ("是什么", "怎么", "如何", "为什么", "哪些", "多少", "是否", "吗", "？", "?")
    # 纯关键词短语跳过改写；“是什么/怎么/如何”等短问句仍需扩展检索表达
    if (
            len(question) <= 8
            and not any(word in question for word in pronouns)
            and not any(marker in question for marker in question_markers)
    ):
        return True
    # 短主题短语（≤14 字、无疑问词/代词/标点）本就是明确的检索实体，
    # 直接 Dense + 词项检索即可，省去一次 LLM 改写往返（通用规则，非用例专用）
    max_topic_chars = 14
    if (
            len(question) <= max_topic_chars
            and not any(word in question for word in pronouns)
            and not any(marker in question for marker in question_markers)
            and not any(ch in question for ch in ("？", "?", "。", "，", ","))
    ):
        return True
    # 长问题已包含充分上下文，无代词指代时跳过改写
    if len(question) > 80 and not any(word in question for word in pronouns):
        return True
    return False


def rewrite_question(llm, question: str, cfg: dict = None, chat_history: str = "") -> dict:
    """把问题改写为独立问题、精确关键词、Dense 查询与独立的 HyDE 查询。"""
    cfg = cfg or {}
    max_dense = max(1, int(cfg.get("MAX_DENSE_QUERIES", 4)))
    max_hyde = max(0, int(cfg.get("MAX_HYDE_QUERIES", 2)))

    if should_skip_rewrite(question, chat_history):
        debug_log(f"[ROUTE] 跳过改写，直接使用原始问题: {question[:30]}...")
        return {
            "standalone_question": question,
            "keywords": [],
            "expanded_queries": [question],
            "hyde_queries": [],
        }

    prompt = ChatPromptTemplate.from_template("""你是实验室知识库的检索改写专家。将用户问题改写为最优检索形式。

【最近对话历史】
{chat_history}

【输出格式】严格JSON：
{{
  "standalone_question": "补全代词和省略后的完整独立问题",
  "keywords": ["精确检索词：设备型号、参数名、英文缩写、专业术语，最多6个"],
  "expanded_queries": [
    "原问题的精准改写（保留专业术语）",
    "更宽泛的上位概念查询",
    "关键术语的同义词/缩写展开查询"
  ],
  "hypothetical_docs": "生成2-3个可能包含答案的学术文献摘要片段，每个50-80字，模拟真实文献内容，用'|||'分隔"
}}

【规则】
1. 保留所有专业术语、设备型号、参数名、英文缩写
2. 结合对话历史补全代词（这个/那个/它）为具体指代对象；历史为空时不要臆造指代对象
3. expanded_queries 必须多样化，覆盖不同检索角度
4. 不要解释问题，仅做改写
5. 若问题已清晰完整，standalone_question保持原意
6. hypothetical_docs: 假设你已经知道答案，写2-3个简短的模拟文献摘要片段（每个50-80字），用"|||"分隔
7. keywords 必须是从问题中提取的精确短词（型号/参数名/术语），不要整句

用户问题：{question}""")

    try:
        with Timer("rewrite_question_llm"):
            result = (prompt | llm).invoke({
                "question": question,
                "chat_history": chat_history or "无历史对话",
            })
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

        # Dense 查询：原始问题 + 独立问题 + 扩展查询，去重保序并按配置限制路数
        dense_queries = [question, standalone_question] + expanded_queries
        dense_queries = [q[:200].strip() for q in dense_queries if isinstance(q, str) and q.strip()]
        dense_queries = list(dict.fromkeys(dense_queries))[:max_dense]

        # HyDE 单独返回，不再与 Dense 争抢同一路数预算（旧实现被 queries[:5] 全部截掉）
        hyde_queries = []
        try:
            if bool(cfg.get("ENABLE_HYDE", True)):
                hyde_text = str(data.get("hypothetical_docs") or data.get("hypothetical_answer") or "")
                hyde_queries = [
                    part.strip()[:200]
                    for part in hyde_text.split("|||")
                    if len(part.strip()) > 20
                ][:max_hyde]
                debug_log(f"[HyDE] 独立假设文档片段={len(hyde_queries)}")
        except Exception as hyde_err:
            debug_log(f"[HyDE] 提取假设文档失败（静默跳过）: {hyde_err}")

        parsed = {
            "standalone_question": standalone_question if isinstance(standalone_question,
                                                                     str) and standalone_question.strip() else question,
            "keywords": [str(k).strip()[:40] for k in keywords if str(k).strip()][:6],
            "expanded_queries": dense_queries,
            "hyde_queries": hyde_queries,
        }
        debug_log("rewrite_question parsed:", parsed)
        return parsed
    except Exception as e:
        debug_log("rewrite_question error:", repr(e))
        return {
            "standalone_question": question,
            "keywords": [],
            "expanded_queries": [question],
            "hyde_queries": [],
        }


# =========================
# 检索与重排
# =========================
def canonical_doc_key(doc) -> str:
    """统一去重键：优先用入库时的 parent_id + chunk_hash，缺失时回退文件名 + 规范化正文。"""
    metadata = doc.metadata or {}
    parent_id = metadata.get("parent_id", "")
    chunk_hash = metadata.get("chunk_hash", "")
    if parent_id and chunk_hash:
        return f"{parent_id}:{chunk_hash}"
    content = clean_retrieval_display_text(doc.page_content or "")
    return f"{metadata.get('file_name', '')}||{content[:1000]}"


def retrieve_multi_query(vectorstore, queries: List[str], top_k_each: int, max_workers: int = 4):
    """
    并行多路 Dense 检索，并按相似度分数归并排序。

    旧实现把并发完成顺序（set done 的遍历顺序）当作相关性顺序，
    导致后续 RRF 的排名基础不可靠；这里改用 similarity_search_with_score
    取每个文档在各路查询中的最高分，再按分数降序输出。
    """
    per_query_k = top_k_each

    def search_one(q):
        t0 = time.perf_counter()
        try:
            if hasattr(vectorstore, "similarity_search_with_score"):
                pairs = vectorstore.similarity_search_with_score(q, k=per_query_k)
                docs = []
                for doc, score in pairs:
                    doc.metadata = dict(doc.metadata or {})
                    doc.metadata["dense_score"] = float(score)
                    docs.append(doc)
            else:
                docs = vectorstore.similarity_search(q, k=per_query_k)
            elapsed = time.perf_counter() - t0
            debug_log(f"[RECALL] q={repr(q)} k={per_query_k} docs={len(docs)} time={elapsed:.3f}s")
            return docs
        except Exception as e:
            debug_log(f"[RECALL ERROR] q={repr(q)} err={repr(e)}")
            return []

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(search_one, q): q for q in queries}
    done, pending = wait(futures, timeout=30)

    merged = {}
    for future in done:
        try:
            for doc in future.result():
                key = canonical_doc_key(doc)
                score = float((doc.metadata or {}).get("dense_score", 0.0))
                previous = merged.get(key)
                if previous is None or score > float((previous.metadata or {}).get("dense_score", 0.0)):
                    merged[key] = doc
        except Exception as e:
            debug_log(f"[WARNING] 检索归并失败: {e}")
    for future in pending:
        future.cancel()
    if pending:
        debug_log(f"[WARNING] {len(pending)} 个检索任务超过 30 秒，已取消等待")
    executor.shutdown(wait=False, cancel_futures=True)

    all_docs = sorted(
        merged.values(),
        key=lambda doc: float((doc.metadata or {}).get("dense_score", 0.0)),
        reverse=True,
    )
    debug_log(f"[RECALL TOTAL] after score merge = {len(all_docs)}")
    return all_docs


def _build_sparse_terms(query: str, keywords: List[str], cfg: dict = None) -> List[str]:
    """产出 MatchText 词元：设备型号/英文术语优先，中文按标点切成短片段。

    型号形态覆盖：连字符/下划线复合（MSO-X 3054A、USRP-2944）、字母数字混合
    （N9322C、EDU36311A、SVA1000X）与纯数字型号带字母后缀（3054A）。
    旧正则会把 N9322C 整个丢弃、把 EDU36311A 截成 EDU36311，导致纯英文手册漏召回。
    此外补充两类通用词元以提升身份字段命中率：
    - 纯数字日期戳（≥6 位，如 20230106 / 20220121），让「某日期组会」精确命中文件名；
    - 中文片段的 2/3-gram 滑窗，让「论文写作方法」产出「论文写作」这类子串命中文件名。
    """
    cfg = cfg or {}
    max_terms = max(1, int(cfg.get("MAX_SPARSE_TERMS", 12)))
    # 优先级桶：型号 > 日期 > 英文 > 中文 n-gram；前者优先占用预算
    models, dates, english, cjk_ngrams = [], [], [], []
    sources = [str(k).strip() for k in (keywords or []) if str(k).strip()] + [str(query or "").strip()]
    for source in sources:
        # 复合型号（含 -/_）优先整体保留，避免拆出 MSO 这类无区分度前缀
        compound = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+", source)
        for token in compound:
            token = token.strip()
            if len(token) >= 2 and token not in models:
                models.append(token)
        # 从剩余文本提取字母数字混合型号、纯英文词与纯数字日期戳
        residual = source
        for token in compound:
            residual = residual.replace(token, " ")
        for token in re.findall(r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z][A-Za-z0-9]*|[A-Za-z]{2,}|\d{6,}", residual):
            token = token.strip()
            if not token or len(token) < 2:
                continue
            if token.isdigit():
                if token not in dates:
                    dates.append(token)  # 纯数字日期戳（20230106 等）
            elif token not in english:
                english.append(token)
        # 中文片段：整段（≤10 字）作为精确词，并补充 2/3-gram 滑窗以命中文件名/标题
        for part in re.split(r"[\s,，。；;、？?！!：:（）()\[\]【】/|]+", source):
            part = part.strip()
            if not part or not re.search(r"[\u4e00-\u9fff]", part):
                continue
            clean = re.sub(r"(的|了|如何|怎么|是什么|有哪些|请|？|\?)", "", part)[:12]
            if clean and clean not in cjk_ngrams:
                cjk_ngrams.append(clean)
            for n in (3, 2):
                for i in range(len(clean) - n + 1):
                    g = clean[i:i + n]
                    if g and g not in cjk_ngrams:
                        cjk_ngrams.append(g)
    # 按优先级组装，型号/日期/英文优先，中文 n-gram 填充剩余预算
    ordered = []
    for bucket in (models, dates, english, cjk_ngrams):
        for token in sorted(bucket, key=len, reverse=True):
            if token not in ordered:
                ordered.append(token)
    return ordered[:max_terms]


# MatchText 字段优先级：路径/文件名对「某小组有哪些设备」最关键，正文放最后
_SPARSE_FIELD_PRIORITY = (
    "metadata.rel_path",
    "metadata.file_name",
    "metadata.doc_title",
    "metadata.section_title",
    "metadata.header_path",
    "metadata.parent_summary",
    "page_content",
)


def retrieve_sparse(client, collection_name: str, query: str, keywords: List[str],
                    top_k: int, cfg: dict = None) -> List[Document]:
    """
    基于 Qdrant MatchText 的全文词项检索（非 BM25 打分，返回 point 顺序）。

    条件预算按「词 × 字段优先级」分配，保证路径/文件名字段不会被无序截断；
    词元化让设备型号与参数名能作为独立词命中，而非整句中文匹配。
    """
    cfg = cfg or {}
    max_conditions = max(1, int(cfg.get("MAX_SPARSE_CONDITIONS", 30)))
    try:
        with Timer(f"retrieve_sparse query={repr(query[:50])} keywords={len(keywords or [])}"):
            search_terms = _build_sparse_terms(query, keywords, cfg=cfg)
            if not search_terms:
                return []

            should_conditions = []
            for term in search_terms:
                for field_name in _SPARSE_FIELD_PRIORITY:
                    if len(should_conditions) >= max_conditions:
                        break
                    should_conditions.append(
                        rest.FieldCondition(key=field_name, match=rest.MatchText(text=term))
                    )
                if len(should_conditions) >= max_conditions:
                    break

            if not should_conditions:
                return []

            search_filter = rest.Filter(should=should_conditions)

            # 超时保护，防止大量条件导致 Qdrant 查询过慢
            try:
                results, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=search_filter,
                    limit=top_k,
                    with_payload=True,
                    timeout=10,
                )
            except Exception as timeout_err:
                debug_log(f"[DEGRADED][qdrant-sparse] scroll 超时或失败: {timeout_err}")
                return []

            docs = []
            for hit in results:
                payload = hit.payload or {}
                md = payload.get("metadata", {})

                content = payload.get("page_content", "") or md.get("page_content", "")
                if not content:
                    continue

                # 补齐与 Dense 一致的 metadata，保证 canonical_doc_key 两路可对齐
                doc = Document(
                    page_content=content,
                    metadata={
                        "file_name": md.get("file_name", ""),
                        "doc_title": md.get("doc_title", ""),
                        "rel_path": md.get("rel_path", ""),
                        "section_title": md.get("section_title", ""),
                        "header_path": md.get("header_path", ""),
                        "parent_id": md.get("parent_id", ""),
                        "parent_content": md.get("parent_content", ""),
                        "parent_summary": md.get("parent_summary", ""),
                        "chunk_hash": md.get("chunk_hash", ""),
                        "chunk_index": md.get("chunk_index", 0),
                        "summary_status": md.get("summary_status", ""),
                    }
                )
                docs.append(doc)

            debug_log(f"retrieve_sparse results={len(docs)} conditions={len(should_conditions)}")
            return docs

    except Exception as e:
        debug_log(f"[DEGRADED][qdrant-sparse] retrieve_sparse error: {repr(e)}")
        return []


def reciprocal_rank_fusion(dense_docs: List[Document], sparse_docs: List[Document],
                           dense_weight: float = 0.7, sparse_weight: float = 0.3,
                           k: int = 60, cfg: dict = None) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) 融合两路检索结果。
    score = dense_weight * 1/(k+rank_dense) + sparse_weight * 1/(k+rank_sparse)

    融合分数写入 metadata.rrf_score，作为重排前的先验与可观测信息；
    去重键与 dedup_docs 统一为 canonical_doc_key，避免两处结论不一致。
    """
    try:
        cfg = cfg or {}
        # 构建 dense 排名映射
        dense_rank = {}
        for rank, doc in enumerate(dense_docs, start=1):
            key = canonical_doc_key(doc)
            if key not in dense_rank:
                dense_rank[key] = rank

        # 构建 sparse 排名映射
        sparse_rank = {}
        for rank, doc in enumerate(sparse_docs, start=1):
            key = canonical_doc_key(doc)
            if key not in sparse_rank:
                sparse_rank[key] = rank

        all_keys = set(dense_rank.keys()) | set(sparse_rank.keys())

        # 建立 key -> doc 的映射（Dense 优先，保留 dense_score）
        key_to_doc = {}
        for doc in dense_docs + sparse_docs:
            key = canonical_doc_key(doc)
            if key not in key_to_doc:
                key_to_doc[key] = doc

        # 只在单路出现的文档使用固定默认排名，避免两路数量不同造成不对称降权
        default_rank = int(cfg.get("INITIAL_RETRIEVAL_K", os.getenv("INITIAL_RETRIEVAL_K", "64"))) + 1
        scored = []
        for key in all_keys:
            d_rank = dense_rank.get(key, default_rank)
            s_rank = sparse_rank.get(key, default_rank)
            rrf_score = dense_weight * (1.0 / (k + d_rank)) + sparse_weight * (1.0 / (k + s_rank))
            scored.append((key, rrf_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        fused_docs = []
        for key, score in scored:
            doc = key_to_doc.get(key)
            if doc:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["rrf_score"] = round(float(score), 6)
                fused_docs.append(doc)

        debug_log(f"reciprocal_rank_fusion: dense={len(dense_docs)} sparse={len(sparse_docs)} fused={len(fused_docs)}")
        return fused_docs

    except Exception as e:
        debug_log(f"[DEGRADED][rrf] reciprocal_rank_fusion error: {repr(e)}")
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
            # 与 RRF / retrieve_multi_query 共用同一去重键，避免两处结论相反
            exact_key = canonical_doc_key(doc)
            if exact_key in seen:
                continue

            seen.add(exact_key)
            unique_docs.append(doc)

        debug_log(f"dedup_docs_exact_only output={len(unique_docs)}")
        return unique_docs


def _query_match_tokens(query: str) -> set:
    """中文按 2/3 字 n-gram、英文按词切分，避免中文整句只产生 1 个 token。"""
    lowered = (query or "").lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{1,}", lowered))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    for size in (2, 3):
        for index in range(len(chinese) - size + 1):
            tokens.add(chinese[index:index + size])
    return tokens


def select_reranked(boosted: list, cfg: dict, top_k: int) -> list:
    """重排过滤策略：绝对阈值 + 相对 Top-1 比例 + 保底条数。

    boosted: [(Document, score)]，已按分数降序。返回过滤并截取后的 Document 列表。
    - 绝对阈值 RERANKER_SCORE_THRESHOLD 始终是主判据；
    - RERANKER_RELATIVE_RATIO>0 时额外放行 score >= top1*ratio 的近优候选；
    - RERANKER_MIN_KEEP>0 时保证至少保留 N 条（仅在存在过阈值候选时生效）；
    - 全部低分时的行为仍由 RERANKER_ALLOW_LOW_SCORE_FALLBACK 决定，语义不变。
    """
    if not boosted:
        return []
    cfg = cfg or {}
    abs_threshold = float(cfg.get("RERANKER_SCORE_THRESHOLD", 0.3))
    relative_ratio = max(0.0, float(cfg.get("RERANKER_RELATIVE_RATIO", 0)))
    min_keep = max(0, int(cfg.get("RERANKER_MIN_KEEP", 0)))
    allow_fallback = bool(cfg.get("RERANKER_ALLOW_LOW_SCORE_FALLBACK", False))

    top_score = float(boosted[0][1])
    relative_floor = top_score * relative_ratio if relative_ratio > 0 else None
    effective = abs_threshold if relative_floor is None else min(abs_threshold, relative_floor)

    filtered = [(d, s) for d, s in boosted if s >= effective]
    absolute_hits = [(d, s) for d, s in filtered if s >= abs_threshold]

    if not absolute_hits:
        if allow_fallback:
            debug_log("[DEGRADED][reranker] 全部候选低于绝对阈值，按配置保底返回 1 条")
            return [boosted[0][0]]
        debug_log("[DEGRADED][reranker] 全部候选低于绝对阈值，返回空结果走 none 覆盖度")
        return []

    if min_keep > 0 and len(filtered) < min_keep:
        filtered = boosted[:min_keep]
        debug_log(f"[RERANK] 保底条数生效: 保留前 {len(filtered)} 条")

    return [d for d, _ in filtered[:top_k]]


def _corpus_entity_hits(question: str) -> list:
    """原始问题中的实体词是否出现在语料派生词典里（语料侧证据，非用例专用规则）。"""
    if not question:
        return []
    corpus = None
    try:
        corpus = lexicon.load_corpus_lexicon()
    except Exception as e:  # 词典缺失/损坏时降级为无语料侧证据
        debug_log(f"[ANSWERABILITY] 语料词典不可用，跳过语料侧证据: {e!r}")
        return []
    if not corpus:
        return []
    doc_terms = [str(t).lower() for t in (corpus.get("document_terms") or []) if len(str(t)) >= 3]
    models = [str(t).lower() for t in (corpus.get("device_models") or []) if len(str(t)) >= 3]
    aliases = [str(k).lower() for k in (corpus.get("group_aliases") or {}) if len(str(k)) >= 2]
    q = str(question).lower()
    return [t for t in dict.fromkeys(doc_terms + models + aliases) if t in q]


def evaluate_answerability(keywords: list, docs: list, cfg: dict = None,
                           question: str = "") -> dict:
    """判断知识库是否有能力回答：检索侧证据 + 语料侧证据 + Top-1 分数三线判定。

    动机：bge-reranker 对域外问题也会给出高分；而改写产出的关键词常是抽象短语
    （「照片画质」「数值规定」），在文档身份字段里零命中，单靠关键词会把正常的
    主题级问法误判为无答案。因此补两条通用证据：
    - 检索侧：改写关键词是否命中最终父块的身份类字段（排除生成式摘要）；
    - 语料侧：原始问题是否包含语料派生词典中的文档实体/型号/小组别名。
    只有两侧都零证据且 Top-1 分数低于下限时，才判为知识库无答案。
    """
    cfg = cfg or {}
    details = {"enabled": bool(cfg.get("ENABLE_ANSWERABILITY_GATE", True)),
               "answerable": True, "keyword_hits": 0, "corpus_hits": 0,
               "top_score": None, "reason": ""}
    if not details["enabled"] or not docs:
        return details

    scores = [float((d.metadata or {}).get("rerank_score", 0) or 0) for d in docs]
    top_score = max(scores) if scores else 0.0
    details["top_score"] = round(top_score, 4)

    terms = []
    for kw in (keywords or []):
        for token in re.split(r"[\s/_\-]+", str(kw).strip().lower()):
            if len(token) >= 2 and token not in terms:
                terms.append(token)
    # 只用文档身份类字段：parent_summary 是 LLM 生成文本，含「方法/信息」等通用词，
    # 会让域外问题的关键词出现假命中，削弱拒答闸门
    evidence = " ".join(
        " ".join(str((d.metadata or {}).get(f, "") or "") for f in
                 ("source", "rel_path", "file_name", "doc_title", "header_path",
                  "section_title", "parent_source_label"))
        for d in docs
    ).lower()
    hits = [t for t in terms if t in evidence]
    corpus_hits = _corpus_entity_hits(question) if bool(cfg.get("ENABLE_CORPUS_LEXICON", True)) else []
    # 第三条通用证据：原始问题与召回文档身份字段的中文 2-gram 重合数。
    # 改写关键词常是抽象短语（「命名」「提交方式」），但原问题本身含真实实体
    # （「团队文档要求」），bigram 重合能捕捉这种字面近似而无需列举任何词表。
    q_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", unicodedata.normalize("NFKC", str(question or "")).lower()))
    q_grams = {q_cjk[i:i + 2] for i in range(len(q_cjk) - 1)}
    ev_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", unicodedata.normalize("NFKC", evidence)))
    ev_grams = {ev_cjk[i:i + 2] for i in range(len(ev_cjk) - 1)}
    overlap_hits = sorted(q_grams & ev_grams)

    details["keyword_hits"] = len(hits)
    details["hit_terms"] = hits[:8]
    details["corpus_hits"] = len(corpus_hits)
    details["corpus_hit_terms"] = corpus_hits[:8]
    details["overlap_hits"] = len(overlap_hits)
    details["overlap_terms"] = overlap_hits[:8]

    floor = float(cfg.get("ANSWERABILITY_TOP_SCORE_FLOOR", 0.90))
    min_hits = max(0, int(cfg.get("ANSWERABILITY_MIN_KEYWORD_HITS", 1)))
    min_overlap = max(1, int(cfg.get("ANSWERABILITY_MIN_OVERLAP_HITS", 2)))
    # 关键词为空（跳过改写/notation/hybrid）时不启用闸门，避免误判
    if not terms:
        details["reason"] = "no_keywords_skip_gate"
        return details
    has_evidence = (len(hits) >= min_hits or bool(corpus_hits)
                    or len(overlap_hits) >= min_overlap)
    if not has_evidence and top_score < floor:
        details["answerable"] = False
        details["reason"] = (f"zero_evidence retrieval+corpus+overlap "
                             f"top1={top_score:.3f}<{floor}")
    return details


def rerank_docs(reranker, query: str, docs, cfg: dict, top_k: int = None,
                original_question: str = ""):
    """重排：原始+改写双 query、中文 n-gram boost、阈值过滤与 RRF 先验并列打破。"""
    if not docs:
        return []
    cfg = cfg or {}
    if top_k is None:
        top_k = int(cfg.get("FINAL_TOP_K", 16))
    score_threshold = float(cfg.get("RERANKER_SCORE_THRESHOLD", 0.3))

    # 候选上限：避免 dedup 后的全量 predict 推高显存与延迟
    max_candidates = max(top_k, top_k * 3)
    docs = docs[:max_candidates]

    # both 模式拼接原始问题与改写问题，既保留用户原意又利用改写消歧
    if cfg.get("RERANK_QUERY_MODE", "both") == "both" and original_question.strip():
        rerank_query = f"{original_question.strip()} {query.strip()}"
    else:
        rerank_query = query

    try:
        pairs = [(rerank_query, clean_retrieval_display_text(d.page_content)) for d in docs]
        # Reranker 分批推理，防止高并发显存溢出
        batch_size = 64
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_scores = reranker.predict(pairs[i:i + batch_size])
            all_scores.extend(batch_scores if hasattr(batch_scores, '__iter__') else [batch_scores])
        scores = all_scores

        query_tokens = _query_match_tokens(rerank_query)
        boosted = []
        for doc, score in zip(docs, scores):
            score = float(score)
            text_lower = doc.page_content.lower()
            doc_title_lower = (doc.metadata.get("doc_title", "") or "").lower()
            section_title_lower = (doc.metadata.get("section_title", "") or "").lower()

            # Keyword boost：中文 n-gram / 英文词命中至少 2 个才触发
            overlap = sum(1 for token in query_tokens if len(token) >= 2 and token in text_lower)
            if overlap >= 2:
                score *= float(cfg.get("KEYWORD_BOOST", 1.05))

            # Title boost：文档标题与章节标题合并判断
            combined_title = f"{doc_title_lower} {section_title_lower}"
            if combined_title.strip() and any(
                    token in combined_title for token in query_tokens if len(token) >= 2
            ):
                score *= float(cfg.get("TITLE_BOOST", 1.08))

            doc.metadata["rerank_score"] = score
            boosted.append((doc, score))

        # RRF 分数作为并列时的次级排序依据，避免融合排序信息被完全丢弃
        boosted.sort(
            key=lambda item: (item[1], float((item[0].metadata or {}).get("rrf_score", 0) or 0)),
            reverse=True,
        )

        # 强证据先验：日期/文档标题词元/型号精确命中身份字段时附加加分，让语义重排
        #在长尾意图（具体日期、专名）失真时仍能回归正确文档。仅在用户原问题/关键词
        #实际包含该实体时生效，不会因通用词误提升；加性且封顶，不改写入 metadata 的
        #原始 rerank_score（覆盖度判定仍用原始分）。
        prior_weight = float(cfg.get("EVIDENCE_PRIOR_WEIGHT", 0.12))
        prior_cap = float(cfg.get("EVIDENCE_PRIOR_CAP", 0.24))
        if prior_weight > 0 and original_question.strip():
            ents = _strong_evidence_entities(original_question, [], cfg)
            if ents:
                identity_fields = ("rel_path", "file_name", "doc_title", "header_path",
                                   "section_title", "parent_source_label", "source")
                adjusted = []
                for doc, score in boosted:
                    id_text = " ".join(
                        str((doc.metadata or {}).get(f, "") or "") for f in identity_fields
                    ).lower()
                    hits = sum(1 for e in ents if e and e in id_text)
                    if hits:
                        score = score + min(hits * prior_weight, prior_cap)
                    adjusted.append((doc, score))
                adjusted.sort(key=lambda item: -item[1])
                boosted = adjusted
                debug_log(f"[EVIDENCE_PRIOR] entities={sorted(ents)} applied to {len(adjusted)} candidates")

        final_docs = select_reranked(boosted, cfg, top_k)
        debug_log(f"Rerank: {len(docs)} → {len(final_docs)} (threshold={score_threshold})")
        return final_docs

    except Exception as e:
        debug_log(f"[DEGRADED][reranker] Rerank error: {repr(e)}")
        return docs[:top_k]


def _filter_context_noise(text: str) -> str:
    """移除超长链接与多余空行，避免 OCR 噪声白占上下文预算。"""
    text = re.sub(r"https?://\S{80,}", "[长链接已省略，请查阅原文]", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def build_context(docs, max_chars: int = 3000, cfg: dict = None) -> tuple:
    """
    构建带引用编号的上下文，返回 (context_str, source_map)。
    source_map: {1: {"doc_title": "...", "rel_path": "...", "section_title": "...", "header_path": "..."}, ...}
    """
    cfg = cfg or {}
    filter_noise = bool(cfg.get("ENABLE_CONTEXT_NOISE_FILTER", True))
    parts = []
    source_map = {}
    current_length = 0

    for idx, doc in enumerate(docs, start=1):
        doc_title = doc.metadata.get('doc_title', '未知标题')
        rel_path = doc.metadata.get('rel_path', '未知路径')
        section_title = doc.metadata.get('section_title', '')
        # 完整章节路径（组 > 设备 > Manuals > 小节）优先于单级 section_title
        header_path = doc.metadata.get('header_path', '')

        section_line = f"[章节] {header_path or section_title}\n" if (header_path or section_title) else ""
        # 如果有父块摘要，展示给 LLM 帮助理解宏观上下文
        parent_summary = sanitize_summary(doc.metadata.get("parent_summary", ""))
        summary_line = f"[段落摘要] {parent_summary}\n" if parent_summary else ""
        body = clean_retrieval_display_text(doc.page_content)
        if filter_noise:
            body = _filter_context_noise(body)
        content = (
            f"[来源{idx}] [文档标题] {doc_title}\n"
            f"[路径] {rel_path}\n"
            f"{section_line}"
            f"{summary_line}"
            f"{body}"
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
            "header_path": header_path,
        }
        current_length += doc_len

    debug_log(f"build_context: source_map size={len(source_map)}")
    context_str = "\n\n====================\n\n".join(parts)
    return context_str, source_map


# =========================
# 覆盖度评估与引用溯源
# =========================
def assess_coverage(reranked_docs: list, config: dict) -> dict:
    """评估知识库覆盖度（max_score + avg_score + doc_count），阈值统一走配置。"""
    config = config or {}
    if not reranked_docs:
        return {
            "level": "none",
            "hint": "⚠️ 知识库中未找到相关内容，本次回答基于模型通用知识，仅供参考。"
        }

    scores = [float(doc.metadata.get("rerank_score", 0) or 0) for doc in reranked_docs]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    doc_count = len(reranked_docs)
    high = float(config.get("COVERAGE_HIGH_THRESHOLD", 0.7))
    medium = float(config.get("COVERAGE_MEDIUM_THRESHOLD", 0.5))
    low = float(config.get("COVERAGE_LOW_THRESHOLD", 0.3))

    debug_log(f"[覆盖度评估] max_score={max_score:.3f} avg_score={avg_score:.3f} doc_count={doc_count}")

    if max_score >= high and avg_score >= high * 0.55:
        return {"level": "high", "hint": None}
    elif max_score >= medium or (avg_score >= low and doc_count >= 4):
        return {"level": "medium", "hint": "ℹ️ 知识库中的相关内容有限，回答可能不够全面。"}
    elif max_score >= low:
        return {"level": "low", "hint": "⚠️ 知识库覆盖不足，回答可能不够准确，建议查阅原始文档。"}
    else:
        return {
            "level": "very_low",
            "hint": "⚠️ 该问题可能不在知识库范围内，本次回答仅供参考。"
        }


def _strong_evidence_entities(question: str, keywords: list, cfg: dict) -> set:
    """高特异性证据实体：日期、语料词典文档标题词元、型号。

    仅当用户原问题/关键词**实际包含**该实体时才保留（调用方再做身份字段命中判定），
    因此不会因通用词（小组/系统/设备…）误提升。用于把「精确命中文档身份字段」的
    候选提到 Top，弥补语义重排在具体日期、专名等长尾意图上的失真。
    """
    ents = set()
    sources = [str(k).strip() for k in (keywords or []) if str(k).strip()] + [str(question or "").strip()]
    text = " ".join(sources)
    # 1) 日期：6-8 位数字串（20230106）或「YYYY年」
    for m in re.findall(r"\d{6,8}|\d{4}年", text):
        ents.add(m)
    # 2) 语料词典文档标题词元（排除过于通用的目录/类型词）
    if bool(cfg.get("ENABLE_CORPUS_LEXICON", True)):
        corpus = lexicon.load_corpus_lexicon()
        if corpus:
            low = text.lower()
            generic = {"小组", "系统", "资料", "目录", "手册", "设备", "设备操作指南",
                       "参考书", "组会资料", "论文", "报告"}
            for t in corpus.get("document_terms", []):
                t = str(t).lower()
                if t in generic or len(t) < 3:
                    continue
                if t in low:
                    ents.add(t)
    # 3) 型号/英文术语（含数字或连字符，区别于普通中文片段）
    for t in _evidence_terms(question, keywords):
        if re.search(r"[\d\-_]", t):
            ents.add(t)
    return ents


def _evidence_terms(question: str, keywords: list) -> list:
    """抽取用于「精确命中」判定的词元：型号与英文术语优先，中文长片段次之。"""
    terms = []
    sources = [str(k).strip() for k in (keywords or []) if str(k).strip()] + [str(question or "").strip()]
    # 型号形态：连字符/下划线复合、字母数字混合、数字带字母后缀（843-R）、纯英文术语
    model_patterns = (
        r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+",
        r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+[A-Za-z][A-Za-z0-9]*|\d+[-_][A-Za-z0-9]+",
        r"[A-Za-z]{3,}",
    )
    stop = {"the", "and", "for", "with", "what", "how", "are", "does", "about",
            "manual", "guide", "spec", "datasheet", "sheet", "user", "series"}
    for source in sources:
        for pattern in model_patterns:
            for token in re.findall(pattern, source):
                token = token.strip("-_ ").lower()
                if len(token) >= 3 and token not in terms and token not in stop and not token.isdigit():
                    terms.append(token)
    for source in sources:
        for part in re.split(r"[\s,，。；;、？?！!：:（）()\[\]【】/|]+", source):
            part = part.strip()
            if len(part) >= 3 and re.search(r"[\u4e00-\u9fff]", part) and part not in terms:
                terms.append(part.lower())
    return terms[:24]


def reorder_final_docs_by_evidence(docs: list, question: str, keywords: list,
                                   cfg: dict = None) -> list:
    """仅在 rerank 分数并列时，用「型号/路径精确命中」稳定重排 final_docs。

    动机：同设备多份等价手册的 rerank 分数差常在 0.005 内，谁排第一取决于
    RRF 先验等次要因素；问题里的型号若能精确命中某文档的路径/标题，应优先。

    严格约束（保证上下文内容、覆盖度、引用集合均不变）：
    - 不改变文档集合与 page_content，只调整顺序；
    - 只在分数差 <= EVIDENCE_RERANK_WEIGHT 的相邻组内换位；
    - 组内按精确命中数降序，命中数相同保持原顺序（稳定排序）；
    - 开关关闭或词典缺失时原样返回。
    """
    cfg = cfg or {}
    if not bool(cfg.get("ENABLE_EVIDENCE_RERANK", True)) or len(docs) < 2:
        return docs
    epsilon = max(0.0, float(cfg.get("EVIDENCE_RERANK_WEIGHT", 0.12)))
    terms = _evidence_terms(question, keywords)
    if not terms:
        return docs

    # 语料词典里的型号也纳入证据词，增强纯英文/别名型号的泛化
    if bool(cfg.get("ENABLE_CORPUS_LEXICON", True)):
        corpus = lexicon.load_corpus_lexicon()
        if corpus:
            for model in corpus.get("device_models", []):
                token = str(model).lower()
                if len(token) >= 3 and token not in terms:
                    terms.append(token)

    identity_fields = ("rel_path", "file_name", "doc_title", "header_path",
                       "section_title", "parent_source_label", "source")

    def hit_count(doc) -> int:
        md = doc.metadata or {}
        text = " ".join(str(md.get(f, "") or "") for f in identity_fields).lower()
        return sum(1 for t in terms if t in text)

    scored = [(i, d, float((d.metadata or {}).get("rerank_score", 0) or 0), hit_count(d))
              for i, d in enumerate(docs)]
    # 已按 rerank 分数降序；在分数差 <= epsilon 的连续组内按命中数降序稳定重排
    result = []
    start = 0
    while start < len(scored):
        end = start + 1
        while end < len(scored) and (scored[start][2] - scored[end][2]) <= epsilon:
            end += 1
        group = scored[start:end]
        if len(group) > 1:
            group.sort(key=lambda x: (-x[3], x[0]))
        result.extend(group)
        start = end
    reordered = [d for _, d, _, _ in result]
    if any(a is not b for a, b in zip(docs, reordered)):
        debug_log(f"[EVIDENCE_RERANK] 并列组重排生效: n={len(docs)} "
                  f"max_hits={max(x[3] for x in scored)}")
    return reordered


def build_citations(used_docs: list) -> list:
    """从实际入 prompt 的文档提取引用；index 与上下文 [来源N] 严格对齐。"""
    citations = []
    seen = set()

    for idx, doc in enumerate(used_docs, start=1):
        source = doc.metadata.get("source", "") or doc.metadata.get("rel_path", "") or doc.metadata.get("file_name", "未知来源")
        # 完整章节路径优先，sparse 构造的文档也带上了 header_path
        section = doc.metadata.get("header_path", "") or doc.metadata.get("section_title", "")

        # 去重：同一文档同一章节只引用一次
        key = f"{source}|{section}"
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "index": idx,
            "source": source,
            "section": section,
            "score": round(float(doc.metadata.get("rerank_score", 0) or 0), 3),
            "summary": sanitize_summary(doc.metadata.get("parent_summary", "")),
        })

    # 保持与 source_map 相同的排名顺序，最多 5 条，避免引用到被预算丢弃的文档
    return citations[:5]


def get_system_prompt_by_coverage(coverage_level: str, hint: str = "") -> str:
    """根据覆盖度级别返回 system prompt 策略，并把覆盖度提示注入给 LLM。"""
    base_prompt = "你是实验室知识库助手，根据提供的参考资料回答用户问题。"

    if coverage_level == "high":
        prompt = base_prompt + "请基于参考资料准确回答，引用具体内容。"
    elif coverage_level == "medium":
        prompt = base_prompt + "参考资料可能不够全面，请基于已有内容尽可能回答，对不确定的部分请明确指出。"
    elif coverage_level == "low":
        prompt = base_prompt + "参考资料相关性较低，请谨慎回答。如果参考资料无法支撑回答，请明确告知用户并建议查阅原始文档。"
    else:  # very_low or none
        prompt = base_prompt + "当前没有找到高度相关的参考资料。请基于你的通用知识简要回答，并明确告知用户该回答未经知识库验证。"

    if hint:
        prompt += f" 当前覆盖度提示：{hint}"
    return prompt


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
    parts = re.split(r"[\/\s_\-\(\)\[\]，。,；;：:]+|(?:的|和|与|及)", text)
    stopwords = {"请", "请问", "帮我", "查询", "查找", "列出", "介绍", "了解", "相关"}
    return [p for p in parts if p and p not in stopwords]


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


def find_best_matching_dirs(keyword: str, root_dir: str, limit: int = 20,
                            snapshot: list = None, cfg: dict = None) -> List[Dict]:
    """从目录树快照筛出最佳匹配目录；snapshot 省略时自行构建（保持旧调用方式可用）。"""
    if not root_dir or not os.path.exists(root_dir):
        return []
    entries = snapshot if snapshot is not None else _get_catalog_snapshot(root_dir, cfg)

    with Timer(f"find_best_matching_dirs keyword={keyword}"):
        matches = []
        for kind, _name, rel_path in entries:
            if kind != "directory":
                continue
            score = path_match_score(keyword, rel_path)
            if score >= 0.45:
                matches.append({"rel_path": rel_path, "score": round(score, 4)})

        matches.sort(key=lambda x: (-x["score"], len(x["rel_path"])))
        debug_log(f"find_best_matching_dirs matched={len(matches)} from_snapshot={len(entries)}")
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


def list_fs_entries_by_keyword(keyword: str, root_dir: str, limit: int = 200,
                              snapshot: list = None, cfg: dict = None) -> List[Dict]:
    """从目录树快照筛出匹配的文件/目录；snapshot 省略时自行构建（保持旧调用方式可用）。"""
    if not root_dir or not os.path.exists(root_dir):
        return []
    entries = snapshot if snapshot is not None else _get_catalog_snapshot(root_dir, cfg)

    with Timer(f"list_fs_entries_by_keyword keyword={keyword}"):
        results = []
        for kind, name, rel_path in entries:
            score = path_match_score(keyword, rel_path)
            if score >= 0.45:
                results.append({
                    "type": kind,
                    "name": name,
                    "rel_path": rel_path,
                    "score": round(score, 4),
                })

        debug_log(f"list_fs_entries_by_keyword matched={len(results)} from_snapshot={len(entries)}")
        return sorted(results, key=lambda x: (-x["score"], len(x["rel_path"])))[:limit]


def is_directory_intent(text: str) -> bool:
    q = normalize_retrieval_text(text).lower()
    hints = ["目录", "目录下", "有什么", "子目录", "文件夹", "有哪些", "清单", "列表", "设备", "使用规范", "操作规范"]
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

        # 中文路径在不同 Qdrant tokenizer/旧索引版本下可能无法命中 MatchText。
        # 仅在过滤结果为空时做有界 payload 扫描，并在应用侧进行路径匹配。
        if not files:
            scan_limit = int(runtime["config"].get("CATALOG_SCAN_LIMIT", 20000))
            scanned = 0
            offset = None
            debug_log(f"[CATALOG] MatchText 无结果，开始本地路径回退扫描，limit={scan_limit}")
            while scanned < scan_limit and len(files) < limit:
                try:
                    results, next_page_offset = client.scroll(
                        collection_name=collection_name,
                        limit=min(256, scan_limit - scanned),
                        offset=offset,
                        with_payload=True,
                        timeout=10,
                    )
                except Exception as e:
                    debug_log(f"[WARN] catalog payload 回退扫描失败: {e}")
                    break
                scanned += len(results)
                for hit in results:
                    payload = hit.payload or {}
                    md = payload.get("metadata", payload)
                    file_name = md.get("file_name", "")
                    doc_title = md.get("doc_title", "")
                    rel_path = md.get("rel_path", "")
                    candidate = rel_path or file_name or doc_title
                    score = path_match_score(keyword, candidate)
                    if score < 0.45:
                        continue
                    key = rel_path or f"{file_name}|{doc_title}"
                    if key not in files:
                        files[key] = {
                            "type": "file", "file_name": file_name, "doc_title": doc_title,
                            "name": doc_title or file_name, "rel_path": rel_path,
                            "count": 0, "score": round(score, 4),
                        }
                    files[key]["count"] += 1
                if not next_page_offset:
                    break
                offset = next_page_offset
            debug_log(f"[CATALOG] payload scan scanned={scanned} unique_files={len(files)}")

        debug_log(f"list_paths_by_keyword unique_files={len(files)}")
        return sorted(list(files.values()), key=lambda x: (-x["score"], -x["count"], len(x["rel_path"])))


def collapse_qdrant_entries_to_directory(entries: List[Dict], target: str) -> Optional[Dict]:
    """将 Qdrant 的深层文档路径折叠为目标目录的直接子级。"""
    target_norm = normalize_retrieval_text(target).lower().strip()
    directories = {}
    direct_files = {}
    matched_prefixes = {}
    for item in entries:
        rel_path = path_to_unix(item.get("rel_path", "")).strip("/")
        parts = [part for part in rel_path.split("/") if part]
        match_index = next(
            (idx for idx, part in enumerate(parts) if target_norm and target_norm in normalize_retrieval_text(part).lower()),
            None,
        )
        if match_index is None:
            continue
        prefix = "/".join(parts[:match_index + 1])
        matched_prefixes[prefix] = matched_prefixes.get(prefix, 0) + int(item.get("count", 1) or 1)
        if match_index + 1 >= len(parts):
            continue
        child_name = parts[match_index + 1]
        child_path = "/".join(parts[:match_index + 2])
        if match_index + 2 < len(parts):
            entry = directories.setdefault(child_path, {
                "type": "directory", "name": child_name, "rel_path": child_path, "count": 0,
            })
        else:
            entry = direct_files.setdefault(child_path, {
                "type": "file", "name": child_name, "rel_path": child_path, "count": 0,
            })
        entry["count"] += int(item.get("count", 1) or 1)

    if not directories and not direct_files:
        return None
    matched_dir = max(matched_prefixes, key=matched_prefixes.get) if matched_prefixes else target
    return {
        "mode": "qdrant_directory",
        "matched_dir": matched_dir,
        "directories": sorted(directories.values(), key=lambda item: item["name"].lower()),
        "files": sorted(direct_files.values(), key=lambda item: item["name"].lower()),
        "raw_entry_count": len(entries),
    }


def list_catalog_entries(target: str) -> Dict:
    # 文件系统目录查询不依赖 Qdrant，降级状态下仍应可用。
    config = _runtime["config"] if _runtime is not None else load_config()

    enable_fs = config.get("ENABLE_FILESYSTEM_TOOL", False)
    root_candidates = []
    for value in (config.get("KNOWLEDGE_BASE_ROOT", ""), config.get("DOCS_PATH", "")):
        value = str(value or "").strip()
        if value and value not in root_candidates:
            root_candidates.append(value)

    with Timer(f"list_catalog_entries target={target}"):
        checked_roots = []
        for root_dir in root_candidates if enable_fs else []:
            root_state = {
                "path": root_dir,
                "exists": os.path.exists(root_dir),
                "is_dir": os.path.isdir(root_dir),
                "readable": os.access(root_dir, os.R_OK),
            }
            checked_roots.append(root_state)
            if not (root_state["exists"] and root_state["is_dir"] and root_state["readable"]):
                continue
            # 单次遍历 + 快照缓存：原先两个函数各自全盘 os.walk，改为共用一份快照
            snapshot = _get_catalog_snapshot(root_dir, config)
            matched_dirs = find_best_matching_dirs(
                target, root_dir, limit=10, snapshot=snapshot, cfg=config)
            entries = list_fs_entries_by_keyword(
                target,
                root_dir=root_dir,
                limit=config["FILE_SEARCH_LIMIT"],
                snapshot=snapshot,
                cfg=config,
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
                    limit=config.get("DIRECTORY_CHILD_LIMIT", 10),
                )
                return {
                    "mode": "filesystem_directory",
                    "matched_dir": best_dir["rel_path"],
                    "matched_score": best_dir["score"],
                    "directories": children["directories"],
                    "files": children["files"],
                    "filesystem_root": root_dir,
                    "checked_roots": checked_roots,
                }

            if entries:
                return {
                    "mode": "filesystem_search", "matched_dir": None, "entries": entries,
                    "filesystem_root": root_dir, "checked_roots": checked_roots,
                }

        # 所有候选根目录均无匹配时，再使用 Qdrant；不能因第一个旧挂载点存在而提前返回空结果。
        debug_log(f"[CATALOG] filesystem 无匹配，回退 Qdrant: roots={checked_roots!r} target={target!r}")
        files = list_paths_by_keyword(target, limit=config["FILE_SEARCH_LIMIT"])
        collapsed = collapse_qdrant_entries_to_directory(files, target)
        if collapsed:
            collapsed["checked_roots"] = checked_roots
            collapsed["query"] = target
            return collapsed
        return {
            "mode": "qdrant_search_fallback" if enable_fs else "qdrant_search",
            "matched_dir": None,
            "entries": files,
            "checked_roots": checked_roots,
            "query": target,
        }


def build_file_context(file_result: Dict) -> str:
    mode = file_result.get("mode", "")

    if mode in {"filesystem_directory", "qdrant_directory"}:
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


def catalog_result_count(file_result: Optional[Dict]) -> int:
    """返回目录枚举命中数，供混合路由综合评估覆盖度。"""
    if not file_result:
        return 0
    if file_result.get("mode") in {"filesystem_directory", "qdrant_directory"}:
        return len(file_result.get("directories", [])) + len(file_result.get("files", []))
    return len(file_result.get("entries", []))


def extract_catalog_content_intent(question: str) -> str:
    """从原问题保留正文检索意图；触发词与输出词表来自检索词典，不引入模型扩写。"""
    ci = lexicon.load_lexicon()["catalog_intent"]
    intent_terms = []
    for rule in ci["rules"]:
        if any(term in question for term in rule["match_terms"]):
            intent_terms.extend(rule["intent_terms"])
    return " ".join(dict.fromkeys(intent_terms)) or ci["default_intent"]


def build_catalog_retrieval_queries(
    file_result: Optional[Dict], fallback_target: str, question: str = "",
    cfg: dict = None,
) -> List[str]:
    """使用目录级实体和用户正文意图构造检索词，避免深入枚举 Manual 文件。

    目录枚举的完整设备清单由 build_catalog_answer_section 直接注入答案，Dense 查询
    只需覆盖「小组级 + 与问题最相关的设备」；因此对同一 intent 的重复路径做归一化
    去重并限流，避免 16 路几乎同义的 Dense 往返（保持可追溯性、不改变答案区块）。
    """
    cfg = cfg or {}
    intent = extract_catalog_content_intent(question)
    max_queries = max(1, int(cfg.get("MAX_HYBRID_QUERIES", 8)))
    q_norm = normalize_retrieval_text(question).strip()

    candidates = []
    if file_result:
        candidates.append(file_result.get("matched_dir", ""))
        keys = ("directories", "files") if file_result.get("mode") in {"filesystem_directory", "qdrant_directory"} else ("entries",)
        for key in keys:
            for item in file_result.get(key, []) or []:
                candidates.append(item.get("rel_path", "") or item.get("name", ""))
    candidates.append(fallback_target)

    norm_values = []
    seen = set()
    for value in candidates:
        value = path_to_unix(str(value or "").strip())
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        norm_values.append(value)

    # 小组级/短路径优先，其次按与问题实体的重合度排序；稳定排序保证同分保持目录顺序
    def relevance(rel_path: str):
        tokens = [t for t in re.split(r"[\s/_\-]+", q_norm.lower()) if len(t) >= 2]
        base = rel_path.lower()
        return (sum(1 for t in tokens if t in base), -len(rel_path))

    ranked = sorted(norm_values, key=relevance, reverse=True)
    selected = ranked[:max_queries]
    # 兜底：路数允许时补入小组级检索词；max=1 时保留最相关设备（清单已由答案区块提供）
    group_value = path_to_unix(str(fallback_target or "").strip())
    if max_queries > 1 and group_value and not any(s.lower() == group_value.lower() for s in selected):
        selected = selected[: max_queries - 1] + [group_value]

    result = [f"{value} {intent}".strip() for value in selected]
    debug_log(f"[HYBRID] catalog queries {len(norm_values)} → {len(result)} (max={max_queries})")
    return result


def build_catalog_answer_section(file_result: Optional[Dict]) -> str:
    """生成不可被模型省略的目录结果区块，保证混合回答具备可追溯文件清单。"""
    if not file_result:
        return "## 目录 / 文件检索结果\n\n目录检索未返回有效结果。"
    return "## 目录 / 文件检索结果\n\n" + build_file_context(file_result or {})


# =========================
# 回答生成
# =========================
def build_grounded_notation_answer(context: str) -> str:
    """从格式表的结构化转写直接生成回答；模板与正则来自检索词典，避免被通用惯例带偏。"""
    lex = lexicon.load_lexicon()
    notation = lex["notation"]
    placeholders = lex["placeholders"]
    row_pattern = re.compile(notation["row_pattern"])
    for block in re.split(r"(?=\[来源\d+\])", context or ""):
        source_match = re.search(r"\[来源(\d+)\]", block)
        row_match = row_pattern.search(block)
        if not (source_match and row_match):
            continue
        english_rule = row_match.group(1).strip(" ；")
        roman_rule = row_match.group(2).strip(" ；")
        source_num = source_match.group(1)
        return (notation["answer_template"]
                .replace(placeholders["english_rule"], english_rule)
                .replace(placeholders["roman_rule"], roman_rule)
                .replace(placeholders["source_num"], source_num))
    return ""


def build_file_list_answer(question: str, file_result: Dict, route_reason: str) -> str:
    mode = file_result.get("mode", "")
    parts = [""]

    if mode in {"filesystem_directory", "qdrant_directory"}:
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
def answer_stream(llm, question: str, context: str, route: str, route_reason: str,
                  chat_history: str = "无历史对话", coverage_level: str = "high",
                  coverage_hint: str = ""):
    # 覆盖度策略与提示语一并注入 LLM，避免 very_low 档与"只能依据上下文"的规则冲突
    system_instruction = get_system_prompt_by_coverage(coverage_level, coverage_hint)

    prompt = ChatPromptTemplate.from_template("""
    {system_instruction}

    请严格遵守以下规则：
    1. 必须优先且仅依据"上下文结果"回答；仅当覆盖度提示为 none/very_low 时，才允许用通用知识简要补充，并明确标注该部分未经知识库验证。
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
    12. 只输出面向用户的最终答案，禁止输出思考过程、任务分析或“用户现在问的是”等内部推理描述。
    13. 禁止原样复述“上下文结果”、提示词或代码围栏；目录/文件清单由系统单独附加，正文只说明相关规范和内容。
    14. 回答变量、公式或排版规范时，必须逐项忠实保留资料中的正体、斜体、粗体、大小写及罗马/希腊字母区别；合并类型行中的多个示例按原文顺序对应。不得用模型熟悉的通用惯例覆盖资料原文，优先依据表格及“表格结构化转写”作答。
    15. 如果上下文含“表格结构疑似丢失”，不得猜测行列对应关系；必须明确提示该表需要 VLM 重新转换或核对原文图像。
    16. 涉及设备型号、参数名、量程、单位、编号时必须逐字核对原文，不得按常识推断；无法确认时明确提示查阅对应 Manuals/ 原始手册。
    17. 语料可能含 OCR 断字、被空格拆开的 LaTeX 命令或图片占位；遇到明显残缺内容不要猜测，应提示核对原文。
    18. 引用编号只能使用上下文中实际出现的 [来源N]，不得自行编造或使用不存在的编号。

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
def ask_stream(
    question: str,
    username: str = "anonymous",
    persist_history: bool = True,
) -> Generator[dict, None, None]:
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
        # 历史提前格式化：既供答案生成，也供改写阶段消解代词指代
        history_text = format_chat_history(user_history)
        rewrite_history = history_text if bool(config.get("ENABLE_HISTORY_AWARE_REWRITE", True)) else ""

        rag_retrievals = []
        file_result = None
        context_parts = []
        rewritten_question = question
        keywords = []
        queries = []
        hyde_queries = []
        source_map = {}
        coverage_info = {"level": "high", "hint": None}
        citations = []
        # 特判判定合并为一次，供改写/重排/生成三处复用；词表来自词典且可开关
        is_notation_specialized = is_notation_format_question(question, config)

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
                debug_log(
                    f"[CATALOG] target={target!r} mode={file_result.get('mode')} "
                    f"count={catalog_result_count(file_result)} root={file_result.get('filesystem_root', '')!r} "
                    f"checked_roots={file_result.get('checked_roots', [])!r}"
                )
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

            # 混合模式先完成目录枚举，再以目录/文件名作为唯一检索词，避免模型凭空扩写。
            if route == "hybrid":
                catalog_target = target or extract_catalog_target(question)
                content_intent = extract_catalog_content_intent(question)
                catalog_queries = build_catalog_retrieval_queries(
                    file_result, catalog_target, question, cfg=config)
                rewritten_question = f"{catalog_target}各设备的{content_intent}".strip()
                keywords = content_intent.split()
                expanded_queries = []
                queries = catalog_queries or [rewritten_question]
                debug_log(f"[HYBRID] catalog-driven queries={queries}")
            elif is_notation_specialized:
                notation_plan = build_notation_retrieval_plan()
                rewritten_question = str(notation_plan["rewritten_question"])
                keywords = list(notation_plan["keywords"])
                expanded_queries = []
                queries = list(notation_plan["queries"])
                debug_log(f"[NOTATION] deterministic retrieval plan={queries}")
            # [智能路由] 判断是否跳过 LLM 改写，直接使用原始问题
            elif should_skip_rewrite(question, rewrite_history):
                debug_log(f"[智能路由] 跳过改写，直接使用原始问题: {question}")
                rewritten_question = question
                keywords = []
                expanded_queries = []
                queries = [question]
            else:
                with Timer("stage_rewrite"):
                    rewrite_result = rewrite_question(llm, question, cfg=config, chat_history=rewrite_history)
                rewritten_question = rewrite_result["standalone_question"]
                keywords = rewrite_result["keywords"]
                expanded_queries = rewrite_result["expanded_queries"]
                hyde_queries = list(rewrite_result.get("hyde_queries") or [])

                queries = [question, rewritten_question] + expanded_queries
                if keywords:
                    queries.append(" ".join([str(k).strip() for k in keywords if str(k).strip()]))
                queries = list(dict.fromkeys([q.strip() for q in queries if q and q.strip()]))

            # Dense 与 HyDE 分开计数：假设文档不再被 queries[:5] 截掉，也避免 7~8 路 Dense
            max_dense = int(config.get("MAX_DENSE_QUERIES", 4))
            max_hyde = int(config.get("MAX_HYDE_QUERIES", 2))
            if route != "hybrid":
                # hybrid 是目录驱动的确定性多查询，保留完整设备清单，不做路数裁剪
                queries = queries[:max_dense]
            queries = list(dict.fromkeys(
                [q.strip() for q in queries if q and q.strip()]
                + [q.strip() for q in hyde_queries[:max_hyde] if q and q.strip()]
            ))
            debug_log(f"queries_count={len(queries)} queries={queries}")

            with Timer("stage_retrieval"):
                # 缓存键加入用户、路由与索引版本；入库逻辑变更后递增版本即可失效旧缓存
                cache_key = hashlib.md5(
                    f"{username}|{route}|{config.get('RETRIEVAL_INDEX_VERSION', '1')}|{sorted(queries)}".encode()
                ).hexdigest()
                cached = _retrieval_cache.get(cache_key)
                if cached is not None:
                    debug_log(f"[CACHE HIT] key={cache_key}")
                    recalled_docs = cached
                else:
                    # 稠密检索：内部按相似度分数归并，不再依赖并发完成顺序
                    recalled_docs = retrieve_multi_query(
                        vectorstore,
                        queries,
                        config["INITIAL_RETRIEVAL_K"]
                    )

                    # 混合检索：MatchText 全文词项路与 Dense 路做 RRF 融合
                    if config.get("ENABLE_HYBRID_SEARCH", False):
                        if route == "hybrid":
                            sparse_search_terms = list(dict.fromkeys(keywords + queries))[:10]
                        elif not keywords:
                            sparse_search_terms = [rewritten_question]
                        else:
                            sparse_search_terms = [rewritten_question] + keywords

                        sparse_docs = retrieve_sparse(
                            runtime["client"],
                            config["QDRANT_COLLECTION_NAME"],
                            rewritten_question,
                            sparse_search_terms,
                            config["INITIAL_RETRIEVAL_K"],
                            cfg=config,
                        )
                        debug_log(f"sparse_docs_count={len(sparse_docs)}")
                        recalled_docs = reciprocal_rank_fusion(
                            recalled_docs, sparse_docs,
                            dense_weight=config["DENSE_WEIGHT"],
                            sparse_weight=config["BM25_WEIGHT"],
                            k=config["RRF_K"],
                            cfg=config,
                        )
                        debug_log(f"after_rrf_fusion={len(recalled_docs)}")

                    # 缓存检索结果
                    _retrieval_cache.set(cache_key, recalled_docs)

                debug_log(f"recalled_docs_before_dedup={len(recalled_docs)}")
                recalled_docs = dedup_docs(recalled_docs)
                debug_log(f"recalled_docs_after_dedup={len(recalled_docs)}")

            with Timer("stage_rerank"):
                child_top_k = config["FINAL_TOP_K"] * 3
                reranked_child_docs = rerank_docs(
                    reranker, rewritten_question, recalled_docs, cfg=config,
                    top_k=child_top_k, original_question=question,
                )

                if is_notation_specialized:
                    exact_notation_docs = []
                    match_terms = lexicon.load_lexicon()["notation"]["match_terms"]
                    for doc in recalled_docs:
                        content = doc.page_content or ""
                        if all(term in content for term in match_terms):
                            # 用可配置加分取代写死 1.0，避免制造假满分引用
                            current_score = float(doc.metadata.get("rerank_score", 0) or 0)
                            bonus = float(config.get("NOTATION_MATCH_BONUS", 0.15))
                            doc.metadata["rerank_score"] = min(1.0, current_score + bonus)
                            exact_notation_docs.append(doc)
                    if exact_notation_docs:
                        reranked_child_docs = dedup_docs(exact_notation_docs + reranked_child_docs)[:child_top_k]
                        debug_log(f"[NOTATION] exact format chunks prioritized={len(exact_notation_docs)}")

                # 【新增】将重排后的子块展开为大召回的父块
                final_docs = expand_to_parent_docs(reranked_child_docs, cfg=config, top_k=config["FINAL_TOP_K"], qdrant_client=runtime["client"])
                # 仅在 rerank 分数并列时用型号/路径精确命中打破并列；不改文档集合与内容
                final_docs = reorder_final_docs_by_evidence(final_docs, question, keywords, config)

            debug_log(f"final_docs_after_parent_expansion={len(final_docs)}")

            # 拒答闸门：域外问题在父块结构化证据中零命中且 Top-1 分数偏低时，
            # 不把无关文档包装成引用；清空 final_docs 后 context/引用/覆盖度统一走 none 档。
            answerability = evaluate_answerability(keywords, final_docs, config, question)
            if route == "rag_search" and not answerability["answerable"]:
                debug_log(f"[ANSWERABILITY] 判为知识库无答案: {answerability['reason']} "
                          f"hits={answerability['keyword_hits']} top1={answerability['top_score']}")
                final_docs = []
            else:
                debug_log(f"[ANSWERABILITY] answerable: hits={answerability['keyword_hits']} "
                          f"terms={answerability.get('hit_terms')} "
                          f"top1={answerability['top_score']} route={route}")

            # [OPT] 覆盖度评估：基于实际入 prompt 的父块，避免评估与回答依据脱节
            coverage_info = assess_coverage(final_docs, config)
            if route == "hybrid" and catalog_result_count(file_result) > 0 and coverage_info["level"] in {"none", "very_low"}:
                coverage_info = {
                    "level": "medium",
                    "hint": "ℹ️ 已找到相关目录或文件，但正文检索内容有限；设备/文件名称以目录枚举结果为准。",
                }
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
                rag_context, source_map = build_context(final_docs, max_chars=rag_budget, cfg=config)
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

        # history_text 已在检索前计算（改写阶段同样复用），此处直接用于答案生成
        grounded_notation_answer = build_grounded_notation_answer(final_context) if is_notation_specialized else ""

        if grounded_notation_answer:
            full_answer = grounded_notation_answer
            yield {"type": "chunk", "content": grounded_notation_answer}
            debug_log("[NOTATION] 使用格式表结构化转写直接回答")
        else:
            with Timer("stage_answer_stream"):
                first_chunk_time = None
                for chunk in answer_stream(
                        llm, question, final_context, route, route_reason,
                        chat_history=history_text,
                        coverage_level=coverage_info["level"],
                        coverage_hint=coverage_info.get("hint") or ""):
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

        full_answer = sanitize_generated_answer(full_answer)
        if route == "hybrid":
            catalog_section = build_catalog_answer_section(file_result)
            if catalog_section:
                full_answer = f"{catalog_section}\n\n## 文档内容说明\n\n{full_answer}".strip()

        # [OPT] 流式回答结束后，发送覆盖度和引用元数据事件
        yield {
            "type": "metadata",
            "stage": "coverage",
            "coverage": coverage_info,
            "citations": citations,
        }

        if persist_history:
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
            "content": "服务暂时不可用，请稍后重试。",
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
