import os
import re
import sys
import hashlib
import time
import uuid as uuid_module
from pathlib import Path
from typing import List, Optional

import requests  # 用于调用 vLLM API 生成摘要
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import threading

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, TextIndexParams, TextIndexType, TokenizerType
from qdrant_client.http.models import PointStruct

from langchain_qdrant import QdrantVectorStore

from logger import get_logger


# ================= LOG =================

_logger = get_logger("ingest")


def log(msg: str):
    """入库日志 - 同时输出到控制台和文件"""
    _logger.info(msg)


def debug_log(msg: str):
    """调试级日志 - 仅在 DEBUG 级别时输出"""
    _logger.debug(msg)


# ================= CONFIG =================

def load_config():
    """
    加载环境变量配置。

    说明：
    - DOCS_PATH：知识库原始文件目录（仅限 Markdown）
    - CHUNK_SIZE / CHUNK_OVERLAP：文本切块参数
    - EMBEDDING_MODEL_NAME：向量模型名称
    - EMBEDDING_DEVICE：embedding 所使用的设备，如 cuda / cpu
    - QDRANT_HOST / QDRANT_PORT：远程 Qdrant 地址
    - QDRANT_COLLECTION_NAME：目标集合名称
    - QDRANT_RECREATE_COLLECTION：是否重建集合（true 会先删后建）
    """
    load_dotenv(override=False)

    cfg = {
        "DOCS_PATH": os.getenv("DOCS_PATH", "./data"),
        "PARENT_CHUNK_SIZE": int(os.getenv("PARENT_CHUNK_SIZE", "1500")),
        "PARENT_CHUNK_OVERLAP": int(os.getenv("PARENT_CHUNK_OVERLAP", "200")),
        "CHILD_CHUNK_SIZE": int(os.getenv("CHILD_CHUNK_SIZE", "300")),
        "CHILD_CHUNK_OVERLAP": int(os.getenv("CHILD_CHUNK_OVERLAP", "50")),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "./models/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cuda:2"),

        "QDRANT_HOST": os.getenv("QDRANT_HOST", "172.18.216.71"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base"),
        "QDRANT_PARENT_COLLECTION": os.getenv("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents"),
        "QDRANT_RECREATE_COLLECTION": os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true",

        "MIN_CHUNK_LENGTH": int(os.getenv("MIN_CHUNK_LENGTH", "120")),
        "INGEST_BATCH_SIZE": int(os.getenv("INGEST_BATCH_SIZE", "64")),

        # vLLM 配置（供摘要调用使用）
        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "lab-secret-key"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct"),

        # 摘要增强配置
        "ENABLE_SUMMARY_AUGMENTATION": os.getenv("ENABLE_SUMMARY_AUGMENTATION", "true").lower() == "true",
        "SUMMARY_VLLM_TIMEOUT": int(os.getenv("SUMMARY_VLLM_TIMEOUT", "30")),
        "SUMMARY_VLLM_RETRIES": int(os.getenv("SUMMARY_VLLM_RETRIES", "3")),
        "SUMMARY_MAX_WORKERS": int(os.getenv("SUMMARY_MAX_WORKERS", "3")),
        "SUMMARY_MAX_TOKENS": int(os.getenv("SUMMARY_MAX_TOKENS", "300")),
        "SUMMARY_INJECTION_MODE": os.getenv("SUMMARY_INJECTION_MODE", "both"),
        "ENABLE_SUMMARY_CACHE": os.getenv("ENABLE_SUMMARY_CACHE", "true").lower() == "true",
        "SUMMARY_CACHE_DIR": os.getenv("SUMMARY_CACHE_DIR", "./data/summary_cache"),
    }

    log(f"配置加载完成: {cfg}")
    return cfg


# ================= TEXT CLEAN =================

def clean_text(text: str) -> str:
    """增强文本清洗，适用于实验室论文/规范/教程类 Markdown"""
    if not text:
        return ""
    # 1. 移除 HTML 注释 (如 <!-- image -->)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 2. 移除 Markdown 图片语法，保留 [图片] 占位
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', lambda m: f'[图片: {m.group(1)}]' if m.group(1) else '[图片]', text)
    # 3. 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 4. 清理零宽字符和控制字符
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 5. 英文断词修复（保留原逻辑）
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    # 6. 合并多个空白行为单行
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # 7. 单换行转空格（保留原逻辑）
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 8. 压缩多余空白
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# 保留原函数名作为别名
clean_text_light = clean_text


# ================= MARKDOWN LOADER =================

def load_md(file_path: Path) -> List[Document]:
    """加载 Markdown 文件，支持多种编码"""
    log(f"开始加载 Markdown: {file_path.name}")
    results = []

    encodings = ['utf-8', 'gbk', 'utf-16', 'latin-1']
    content = None
    used_encoding = None

    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            used_encoding = encoding
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        log(f"[ERROR] 无法识别文件编码: {file_path}")
        return results

    if used_encoding != 'utf-8':
        log(f"[INFO] 文件 {file_path.name} 使用编码: {used_encoding}")

    # 清洗文本
    text = clean_text(content)

    if text:
        results.append(Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "doc_title": file_path.stem,
                "type": "md",
            }
        ))

    log(f"Markdown加载完成: {file_path.name}, doc数={len(results)}")
    return results


# ================= LOAD =================

def load_documents(path: str) -> List[Document]:
    """
    遍历目录，仅加载 .md 文件，并为每个文档补充相对路径 metadata。
    """
    root = Path(path).resolve()
    docs: List[Document] = []

    if not root.exists():
        log(f"文档目录不存在: {root}")
        return docs

    # 递归遍历所有文件
    for f in root.rglob("*"):
        try:
            if not f.is_file():
                continue

            # 仅处理 .md 文件
            if f.suffix.lower() != ".md":
                continue

            # 大文件预警：跳过超过 50MB 的文件
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
            if f.stat().st_size > MAX_FILE_SIZE:
                log(f"[WARNING] 文件过大已跳过: {f.name} ({f.stat().st_size / 1024 / 1024:.1f}MB)")
                continue

            # 计算相对路径
            rel_path = f.relative_to(root).as_posix()

            # 加载 markdown
            file_docs = load_md(f)

            # 补充 metadata
            for d in file_docs:
                if not d.metadata:
                    d.metadata = {}
                d.metadata.setdefault("source", str(f))
                d.metadata.setdefault("file_name", f.name)
                d.metadata.setdefault("doc_title", f.stem)
                d.metadata["rel_path"] = rel_path
                docs.append(d)

        except Exception as e:
            log(f"加载文件失败: {f}, 错误: {e}")

    log(f"总文档数: {len(docs)}")
    return docs


# ================= CHUNK ENRICH =================

def enrich_chunk_with_titles(doc: Document) -> Document:
    """
    将文档标题、相对路径等结构信息注入 chunk 前缀。
    """
    try:
        doc_title = doc.metadata.get("doc_title")
        rel_path = doc.metadata.get("rel_path")
        section_title = doc.metadata.get("section_title")

        prefix_parts = []
        if doc_title:
            prefix_parts.append(f"[文档标题] {doc_title}")
        if rel_path:
            prefix_parts.append(f"[路径] {rel_path}")
        if section_title:
            prefix_parts.append(f"[章节] {section_title}")

        prefix = "\n".join(prefix_parts)
        content = doc.page_content.strip()

        if prefix:
            content = prefix + "\n\n" + content

        return Document(
            page_content=content,
            metadata=doc.metadata
        )
    except Exception as e:
        log(f"chunk 标题增强失败: {e}")
        return doc


# ================= MARKDOWN STRUCTURE SPLIT =================

def split_by_markdown_headers(doc: Document, parent_chunk_size: int = 1500, parent_chunk_overlap: int = 200) -> List[Document]:
    """
    按 Markdown 标题（#、##、###）将文档切分为逻辑段落。

    每个段落继承原始 doc 的 metadata，并新增：
    - section_title: 当前段落的标题文本
    - header_path: 标题层级路径（如 "第三章 系统设计 > 3.1 硬件架构"）
    - header_level: 标题级别（1/2/3）

    如果文档无任何标题，返回原始文档列表（回退）。
    如果某段落过长（超过 parent_chunk_size * 2），递归切分。
    """
    text = doc.page_content
    if not text or not text.strip():
        return [doc]

    # 匹配 Markdown 标题行：# / ## / ###
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    # 如果无任何标题，回退返回原始文档
    if not matches:
        return [doc]

    sections = []
    # 维护标题层级栈，用于构建 header_path
    header_stack = {}  # level -> title

    for i, match in enumerate(matches):
        level = len(match.group(1))  # 1, 2, or 3
        title = match.group(2).strip()
        start = match.start()

        # 确定段落结束位置
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        # 更新标题栈：清除同级及以下层级
        header_stack[level] = title
        keys_to_remove = [k for k in header_stack if k > level]
        for k in keys_to_remove:
            del header_stack[k]

        # 构建 header_path
        sorted_levels = sorted(header_stack.keys())
        header_path = " > ".join(header_stack[lv] for lv in sorted_levels)

        # 构建段落 metadata
        section_metadata = dict(doc.metadata) if doc.metadata else {}
        section_metadata["section_title"] = title
        section_metadata["header_path"] = header_path
        section_metadata["header_level"] = level

        sections.append(Document(
            page_content=section_text,
            metadata=section_metadata
        ))

    # 处理标题之前的前言内容（如果有）
    if matches[0].start() > 0:
        preamble_text = text[:matches[0].start()].strip()
        if preamble_text:
            preamble_metadata = dict(doc.metadata) if doc.metadata else {}
            preamble_metadata["section_title"] = ""
            preamble_metadata["header_path"] = ""
            preamble_metadata["header_level"] = 0
            sections.insert(0, Document(
                page_content=preamble_text,
                metadata=preamble_metadata
            ))

    # 对过长段落进行递归切分
    max_size = parent_chunk_size * 2
    result = []
    for sec in sections:
        if len(sec.page_content) > max_size:
            # 使用 RecursiveCharacterTextSplitter 进行切分（overlap 从配置读取）
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parent_chunk_size,
                chunk_overlap=parent_chunk_overlap,
                separators=["\n\n", "\n", "。", ". ", " "]
            )
            sub_docs = sub_splitter.split_documents([sec])
            # 每个子文档继承该段落的 metadata
            for sd in sub_docs:
                sd.metadata.update(sec.metadata)
            result.extend(sub_docs)
        else:
            result.append(sec)

    return result


# ================= PARENT COLLECTION =================

def ensure_parent_collection(client, collection_name: str):
    """创建父块存储集合（使用最小维度向量占位，优化HNSW参数）"""
    if not client.collection_exists(collection_name):
        hnsw_config = HnswConfigDiff(
            m=32,
            ef_construct=200,
            full_scan_threshold=10000,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            hnsw_config=hnsw_config,
        )
        log(f"[INFO] 创建父块集合: {collection_name}（HNSW: m=32, ef_construct=200）")


def store_parent_chunks_batch(client, collection_name: str, parent_chunks_data: list, summary_map: dict = None):
    """批量存储父块到独立集合"""
    if summary_map is None:
        summary_map = {}
    points = []
    for item in parent_chunks_data:
        point_id = str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, f"parent|{item.get('source', 'unknown')}|{item['parent_id']}"))
        points.append(PointStruct(
            id=point_id,
            vector=[0.0, 0.0, 0.0, 0.0],
            payload={
                "parent_id": item["parent_id"],
                "parent_content": item["parent_content"],
                "source": item.get("source", ""),
                "doc_title": item.get("doc_title", ""),
                "section_title": item.get("section_title", ""),
                "header_path": item.get("header_path", ""),
                "parent_summary": summary_map.get(item["parent_id"], ""),
            }
        ))
    if points:
        # 分批 upsert，每批 64 个
        batch_size = 64
        for i in range(0, len(points), batch_size):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points[i:i+batch_size]
                )
            except Exception as e:
                log(f"[ERROR] 父块存储失败: {e}")
                raise
        log(f"[INFO] 已写入 {len(points)} 个父块到 {collection_name}")


# ================= CONTEXTUAL RETRIEVAL =================

def build_contextual_prefix(metadata: dict, parent_summary: str = "", injection_mode: str = "both") -> str:
    """为chunk构建上下文前缀，提升embedding质量（Contextual Retrieval）"""
    parts = []

    # 文档来源
    source = metadata.get("source", "")
    if source:
        doc_name = source.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in source else source.rsplit(".", 1)[0]
        parts.append(f"文档：{doc_name}")

    # 章节路径
    header_path = metadata.get("header_path", "")
    if header_path:
        parts.append(f"章节：{header_path}")

    # 段落标题（避免与 header_path 重复）
    section_title = metadata.get("section_title", "")
    if section_title and section_title not in header_path:
        parts.append(f"标题：{section_title}")

    # 摘要注入：如果有父块摘要且注入模式允许，追加简短摘要到前缀
    if parent_summary and injection_mode in ("both", "prefix_only"):
        short_summary = parent_summary[:100]
        parts.append(f"段落摘要：{short_summary}")

    if parts:
        return "【" + " > ".join(parts) + "】\n"
    return ""


# ==================== 摘要增强相关 ====================

SUMMARY_SYSTEM_PROMPT = """你是知识库内容摘要专家。给定一个来自学术论文或技术文档的文本段落，生成一个简洁准确的摘要。

摘要要求：
1. 保留核心概念、关键方法、重要结果或结论
2. 使用简洁学术语言，避免冗余
3. 保持原文意思准确，不添加额外推理
4. 去掉具体例子和参考文献编号
5. 长度控制在150-250字"""

SUMMARY_USER_PROMPT_TEMPLATE = """请为下面的文本段落生成摘要。

【文档】{doc_title}
【章节】{header_path}

【原文】
{parent_content}

直接输出摘要内容，不需要任何前缀或格式标记。"""


class SummaryCache:
    """摘要缓存管理，避免重复调用 vLLM"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, source: str, parent_id: str) -> str:
        """生成缓存键"""
        seed = f"{source}|{parent_id}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def get(self, source: str, parent_id: str):
        """获取缓存摘要，返回 str 或 None"""
        import json
        key = self._get_key(source, parent_id)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("summary")
        except Exception:
            return None

    def set(self, source: str, parent_id: str, summary: str):
        """保存摘要到缓存"""
        import json
        key = self._get_key(source, parent_id)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"summary": summary, "created_at": time.time()}, f, ensure_ascii=False)
        except Exception as e:
            log(f"保存摘要缓存失败: {e}")


def check_vllm_availability(cfg: dict) -> bool:
    """
    检测 vLLM 服务是否可用（发送健康检查请求）

    Args:
        cfg: 配置字典

    Returns:
        True 表示 vLLM 可用，False 表示不可用
    """
    base_url = cfg.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = cfg.get("VLLM_API_KEY", "lab-secret-key")

    # 尝试请求 /models 端点作为健康检查
    health_url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(health_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            debug_log("vLLM 服务健康检查通过")
            return True
        else:
            log(f"[WARNING] vLLM 健康检查返回非200状态码: {resp.status_code}")
            return False
    except requests.ConnectionError:
        log("[WARNING] vLLM 服务无法连接，可能未启动")
        return False
    except requests.Timeout:
        log("[WARNING] vLLM 健康检查超时，服务可能未就绪")
        return False
    except Exception as e:
        log(f"[WARNING] vLLM 健康检查异常: {e}")
        return False


def call_vllm_for_summary(parent_content: str, doc_title: str, header_path: str, cfg: dict):
    """
    调用 vLLM 生成父块摘要

    Args:
        parent_content: 父块文本内容
        doc_title: 文档标题
        header_path: 章节路径
        cfg: 配置字典

    Returns:
        摘要文本字符串，失败返回 None
    """
    base_url = cfg.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = cfg.get("VLLM_API_KEY", "lab-secret-key")
    model_name = cfg.get("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct")
    timeout = cfg.get("SUMMARY_VLLM_TIMEOUT", 30)
    max_retries = cfg.get("SUMMARY_VLLM_RETRIES", 3)
    max_tokens = cfg.get("SUMMARY_MAX_TOKENS", 300)

    url = f"{base_url}/chat/completions"
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        doc_title=doc_title or "未知",
        header_path=header_path or "未知",
        parent_content=parent_content[:2000]  # 防止过长
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.95,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                summary = data["choices"][0]["message"]["content"].strip()
                return summary
            elif resp.status_code in [429, 500, 502, 503]:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    log(f"vLLM 摘要调用失败 ({resp.status_code})，{wait_time}s 后重试")
                    time.sleep(wait_time)
                    continue
            else:
                log(f"vLLM 摘要调用返回错误: HTTP {resp.status_code}")
                return None
        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            log(f"vLLM 摘要调用超时，已重试 {max_retries} 次")
            return None
        except Exception as e:
            log(f"vLLM 摘要调用异常: {e}")
            return None
    return None


def generate_parent_summaries(parent_chunks_data: list, cfg: dict) -> dict:
    """
    并发为父块生成摘要

    Args:
        parent_chunks_data: 父块数据列表，每项为 dict 包含 parent_id, parent_content, doc_title, header_path, source
        cfg: 配置字典

    Returns:
        dict: {parent_id: summary_text} 的映射
    """
    if not cfg.get("ENABLE_SUMMARY_AUGMENTATION", True):
        log("摘要增强已禁用，跳过摘要生成")
        return {}

    # 前置检查：vLLM 服务是否可用
    if not check_vllm_availability(cfg):
        log("[WARNING] vLLM 服务不可用，摘要生成被跳过。入库将继续但不包含摘要增强。")
        log("[WARNING] 如需摘要增强，请先启动 vLLM 服务后重新入库，或设置 ENABLE_SUMMARY_AUGMENTATION=false 禁用此功能。")
        return {}

    max_workers = cfg.get("SUMMARY_MAX_WORKERS", 3)
    summary_map = {}

    # 初始化缓存
    cache = None
    if cfg.get("ENABLE_SUMMARY_CACHE", True):
        cache = SummaryCache(cfg.get("SUMMARY_CACHE_DIR", "./data/summary_cache"))

    def generate_one(item):
        parent_id = item["parent_id"]
        source = item.get("source", "")

        # 优先从缓存获取
        if cache:
            cached = cache.get(source, parent_id)
            if cached:
                return (parent_id, cached)

        # 调用 vLLM 生成
        summary = call_vllm_for_summary(
            parent_content=item["parent_content"],
            doc_title=item.get("doc_title", ""),
            header_path=item.get("header_path", ""),
            cfg=cfg
        )

        # 写入缓存
        if summary and cache:
            cache.set(source, parent_id, summary)

        return (parent_id, summary)

    log(f"开始为 {len(parent_chunks_data)} 个父块生成摘要（并发数={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_one, item): item for item in parent_chunks_data}
        completed = 0
        for future in as_completed(futures):
            try:
                parent_id, summary = future.result()
                if summary:
                    summary_map[parent_id] = summary
                completed += 1
                # 每10%输出进度
                if completed % max(1, len(parent_chunks_data) // 10) == 0:
                    log(f"摘要生成进度: {completed}/{len(parent_chunks_data)}")
            except Exception as e:
                log(f"摘要生成异常: {e}")

    success_count = len(summary_map)
    log(f"摘要生成完成: {success_count}/{len(parent_chunks_data)} 成功 ({100 * success_count // max(1, len(parent_chunks_data))}%)")
    return summary_map


# ================= CHUNK =================

def split_documents(docs: List[Document], cfg: dict):
    """
    小检索大召回（父子块）切分逻辑：
    1. 先切出较大的父块。
    2. 为每个父块分配 UUID，并将其完整文本保存备用。
    3. 将父块继续切分为小的子块。
    4. 将父块的 UUID 和完整文本写入子块的 metadata。
    """
    log("开始执行 父-子 (Small-to-Big) 两级文本切块")

    try:
        # 记录入库时间戳
        ingestion_time = int(time.time())

        # 第零层：Markdown 结构感知切分
        parent_chunk_size = cfg["PARENT_CHUNK_SIZE"]
        parent_chunk_overlap = cfg["PARENT_CHUNK_OVERLAP"]
        logical_docs = []
        for doc in docs:
            logical_docs.extend(split_by_markdown_headers(doc, parent_chunk_size, parent_chunk_overlap))
        log(f"Markdown 结构切分完成: {len(docs)} 个文档 -> {len(logical_docs)} 个逻辑段落")

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=cfg["PARENT_CHUNK_OVERLAP"],
            separators=["\n\n", "\n", "。", ". ", " "]
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["CHILD_CHUNK_SIZE"],
            chunk_overlap=cfg["CHILD_CHUNK_OVERLAP"],
            separators=["\n\n", "\n", "。", "！", "？", "，", ", ", " "]
        )

        final_child_chunks = []
        parent_chunks_to_store = []

        # 第一层：切分父文档
        parent_chunks = parent_splitter.split_documents(logical_docs)
        # 注意：不再对父块注入结构前缀，前缀统一在子块层面通过 build_contextual_prefix() 注入
        # 避免子块继承父块前缀导致重复（子块切分自父块文本，会携带父块中的前缀文本）

        # 第二层：收集父块信息并切分子文档
        for p_chunk in parent_chunks:
            parent_text = p_chunk.page_content
            # 基于父块内容哈希生成确定性 ID，避免文档修改后索引偏移导致旧子块指向错误父块
            source = p_chunk.metadata.get("source", "unknown")
            content_hash = hashlib.md5(parent_text.encode('utf-8')).hexdigest()
            parent_id = str(uuid_module.uuid5(
                uuid_module.NAMESPACE_URL,
                f"{source}|{content_hash}"
            ))

            # 收集父块数据，用于后续写入独立集合
            parent_chunks_to_store.append({
                "parent_id": parent_id,
                "parent_content": parent_text,
                "source": p_chunk.metadata.get("source", ""),
                "doc_title": p_chunk.metadata.get("doc_title", ""),
                "section_title": p_chunk.metadata.get("section_title", ""),
                "header_path": p_chunk.metadata.get("header_path", ""),
            })

        # 调用 vLLM 为父块生成摘要（数据增强）
        summary_map = generate_parent_summaries(parent_chunks_to_store, cfg)
        injection_mode = cfg.get("SUMMARY_INJECTION_MODE", "both")

        # 第三层：生成子块并注入摘要
        for p_chunk in parent_chunks:
            parent_text = p_chunk.page_content
            source = p_chunk.metadata.get("source", "unknown")
            content_hash = hashlib.md5(parent_text.encode('utf-8')).hexdigest()
            parent_id = str(uuid_module.uuid5(
                uuid_module.NAMESPACE_URL,
                f"{source}|{content_hash}"
            ))

            # 切分子块
            child_chunks = child_splitter.split_documents([p_chunk])

            # 获取当前父块的摘要
            parent_summary = summary_map.get(parent_id, "")

            for chunk_idx, c_chunk in enumerate(child_chunks):
                # 继承父块的所有 metadata，并追加父块相关信息
                c_metadata = c_chunk.metadata.copy()
                c_metadata.update({
                    "parent_id": parent_id,
                    "parent_content": parent_text,  # 冗余存储，供降级查询使用
                    "size": "small",
                    "chunk_index": chunk_idx,
                    "chunk_hash": hashlib.md5(c_chunk.page_content.encode()).hexdigest(),
                    "ingestion_time": ingestion_time,
                    "embedding_model": cfg["EMBEDDING_MODEL_NAME"],
                })

                # 将父块摘要存入子块 metadata（metadata_only 或 both 模式）
                if parent_summary and injection_mode in ("both", "metadata_only"):
                    c_metadata["parent_summary"] = parent_summary

                # Contextual Retrieval: 为子块注入上下文前缀，提升embedding检索精度
                prefix = build_contextual_prefix(c_metadata, parent_summary, injection_mode)
                child_content = prefix + c_chunk.page_content if prefix else c_chunk.page_content

                final_child_chunks.append(Document(
                    page_content=child_content,
                    metadata=c_metadata
                ))

        log(f"切块完成: 生成了 {len(parent_chunks)} 个父块，衍生出 {len(final_child_chunks)} 个子块")
        return final_child_chunks, parent_chunks_to_store, summary_map

    except Exception as e:
        log(f"文本切块失败: {e}")
        return [], [], {}


# ================= FILTER =================

def analyze_bad_chunk_reason(text: str, cfg: dict) -> str:
    """分析低质量 chunk 原因，适配实验室文档场景"""
    if not text or not text.strip():
        return "empty"
    text = text.strip()

    min_length = cfg.get("MIN_CHUNK_LENGTH", 120)
    if len(text) < min_length:
        return "too_short"

    # 有效字符检测：中文字 + 英文单词
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'[A-Za-z]{2,}', text)
    valid_count = len(chinese_chars) + len(english_words)

    if valid_count < 10:
        return "too_few_valid_chars"

    # 数字比例检测（允许公式/数据较多的论文内容）
    digits = len(re.findall(r'\d', text))
    if len(text) > 0 and digits / len(text) > 0.6:
        return "too_many_digits"

    # 特殊字符检测：排除中英文标点
    common_punct = re.sub(r'[\w\s\u4e00-\u9fff。，、；：？！""''（）【】《》\.\,\;\:\?\!\"\'\(\)\[\]\-\—\…]', '', text)
    if len(text) > 0 and len(common_punct) / len(text) > 0.25:
        return "too_many_special_chars"

    # 检测纯表头/纯引用
    lines = text.strip().split('\n')
    if all(line.strip().startswith('|') or line.strip().startswith('>') or line.strip() == '' for line in lines):
        if len(text) < 200:
            return "table_or_quote_only"

    return "good"


def filter_chunks(docs: List[Document], cfg: dict):
    """
    过滤低质量 chunk。
    """
    log("开始过滤低质量chunk")

    results = []
    reason_stats = {}

    try:
        for d in docs:
            reason = analyze_bad_chunk_reason(d.page_content, cfg)
            reason_stats[reason] = reason_stats.get(reason, 0) + 1
            if reason == "good":
                results.append(d)

        log(f"过滤前: {len(docs)}, 过滤后: {len(results)}")
        log(f"过滤统计: {reason_stats}")
        return results

    except Exception as e:
        log(f"过滤 chunk 失败: {e}")
        return docs


# ================= EMBEDDING =================

def _init_with_timeout(func, timeout=300):
    """带超时的初始化包装器（跨平台兼容，使用 threading 方案）"""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"初始化超时（{timeout}秒）")
    if error[0]:
        raise error[0]
    return result[0]


def build_embeddings(cfg):
    """
    初始化 embedding 模型（带超时保护）。
    """
    log("初始化 Embedding 模型")
    timeout = int(os.getenv("EMBEDDING_INIT_TIMEOUT", "300"))

    try:
        def _init_embedding():
            return HuggingFaceEmbeddings(
                model_name=cfg["EMBEDDING_MODEL_NAME"],
                model_kwargs={"device": cfg["EMBEDDING_DEVICE"]},
                encode_kwargs={"normalize_embeddings": True},
            )

        embeddings = _init_with_timeout(_init_embedding, timeout=timeout)
        log(f"Embedding 模型初始化成功（超时限制: {timeout}秒）")
        return embeddings
    except TimeoutError as e:
        log(f"[ERROR] Embedding 模型初始化超时: {e}")
        raise
    except Exception as e:
        log(f"[ERROR] 初始化 Embedding 失败: {e}")
        raise


# ================= QDRANT =================

def get_embedding_dimension(embeddings) -> int:
    """
    动态获取向量维度。
    """
    try:
        test_vec = embeddings.embed_query("测试文本")
        dim = len(test_vec)
        log(f"[INFO] Embedding 模型实际向量维度: {dim}")
        if dim not in [384, 768, 1024]:
            log(f"[WARNING] 向量维度 {dim} 不在常见范围内(384/768/1024)，请确认模型配置")
        return dim
    except Exception as e:
        log(f"获取向量维度失败: {e}")
        raise


def create_qdrant_client(cfg):
    """
    创建 Qdrant 客户端。
    """
    try:
        client = QdrantClient(
            host=cfg["QDRANT_HOST"],
            port=cfg["QDRANT_PORT"],
            timeout=30
        )
        log(f"Qdrant 客户端创建成功: {cfg['QDRANT_HOST']}:{cfg['QDRANT_PORT']}")
        return client
    except Exception as e:
        log(f"创建 Qdrant 客户端失败: {e}")
        raise


def prepare_collection(client: QdrantClient, cfg: dict, vector_size: int):
    """
    准备 Qdrant collection，包含维度一致性验证和优化的HNSW参数。
    """
    collection_name = cfg["QDRANT_COLLECTION_NAME"]
    recreate_flag = cfg.get("QDRANT_RECREATE_COLLECTION", False)

    existing = client.collection_exists(collection_name)

    if existing and not recreate_flag:
        # 验证维度一致性
        collection_info = client.get_collection(collection_name)
        existing_size = collection_info.config.params.vectors.size
        if existing_size != vector_size:
            raise ValueError(
                f"Collection '{collection_name}' 向量维度不匹配: "
                f"期望 {vector_size}, 实际 {existing_size}。"
                f"请设置 QDRANT_RECREATE_COLLECTION=true 重建集合。"
            )
        log(f"复用已有 Collection: {collection_name} (维度={existing_size})")
        # 增量入库时也确保文本索引存在
        _create_text_indexes(client, collection_name)
        return

    if existing and recreate_flag:
        client.delete_collection(collection_name)
        log(f"已删除旧 Collection: {collection_name}")

        # 同时重建父块集合
        parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
        if client.collection_exists(parent_collection):
            client.delete_collection(parent_collection)
            log(f"[INFO] 已删除旧的父块集合: {parent_collection}")

    # 创建新集合（使用优化的HNSW参数，提升学术知识库检索精度）
    hnsw_config = HnswConfigDiff(
        m=32,                # 默认16，增大提升召回（内存+30%）
        ef_construct=200,    # 默认100，增大提升索引质量
        full_scan_threshold=10000,  # 小于此数量时用全量扫描
    )

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=hnsw_config,
        )
    except Exception as e:
        log(f"[ERROR] 创建集合失败: {e}")
        raise
    log(f"已创建新 Collection: {collection_name} (维度={vector_size}, 距离=COSINE, HNSW: m=32, ef_construct=200)")

    # 创建文本索引（新建集合时）
    _create_text_indexes(client, collection_name)


def _create_text_indexes(client: QdrantClient, collection_name: str):
    """
    为集合创建 MatchText 所需的文本索引。
    索引字段包括：page_content、metadata.doc_title、metadata.section_title、
    metadata.file_name、metadata.rel_path。
    使用 try-except 包裹，因为增量入库时索引可能已存在。
    """
    text_index_params = TextIndexParams(
        type=TextIndexType.TEXT,
        tokenizer=TokenizerType.MULTILINGUAL,
        min_token_len=2,
        lowercase=True,
    )

    # 顶层字段
    top_level_fields = ["page_content"]
    # metadata 子字段
    metadata_fields = ["doc_title", "section_title", "file_name", "rel_path"]

    all_fields = top_level_fields + [f"metadata.{f}" for f in metadata_fields]

    for field_name in all_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=text_index_params,
            )
            log(f"[INFO] 文本索引创建成功: {collection_name}.{field_name}")
        except Exception as e:
            # 区分"索引已存在"和真正的错误
            error_msg = str(e).lower()
            if "already exists" in error_msg or "already_exists" in error_msg:
                debug_log(f"索引已存在，跳过: {field_name}")
            else:
                log(f"⚠️ 创建索引异常 [{field_name}]: {e}")


# ================= DETERMINISTIC ID =================

def generate_deterministic_id(source: str, content_hash: str, chunk_index: int) -> str:
    """基于源文件、内容哈希和块索引生成确定性 UUID，支持幂等入库"""
    seed = f"{source}|{content_hash}|{chunk_index}"
    return str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, seed))


# ================= SAVE =================

def save_to_qdrant(docs: List[Document], embeddings, cfg: dict, parent_chunks_data: list = None, summary_map: dict = None):
    """分批写入 Qdrant，优化大规模入库性能"""
    log("开始写入 Qdrant")

    if not docs:
        log("没有可写入的文档 chunk，跳过入库")
        return

    try:
        client = create_qdrant_client(cfg)
        vector_size = get_embedding_dimension(embeddings)
        prepare_collection(client, cfg, vector_size)

        # 确保父块集合在所有数据写入之前已创建，避免后续检索时父块展开查询失败
        parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
        try:
            ensure_parent_collection(client, parent_collection)
        except Exception as e:
            log(f"[ERROR] 父块集合创建失败: {e}，父块数据将无法存储")

        collection_name = cfg["QDRANT_COLLECTION_NAME"]

        # 显式指定 content_payload_key="page_content"，确保写入的文档内容字段名
        # 与 _create_text_indexes() 中创建文本索引的字段名、rag_agent.py 的 MatchText
        # 查询字段名保持一致，避免因不同版本 langchain_qdrant 默认值差异导致字段不匹配
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            content_payload_key="page_content",
        )

        # 为每个文档生成确定性 ID
        for doc in docs:
            meta = doc.metadata
            doc_id = generate_deterministic_id(
                meta.get("source", "unknown"),
                meta.get("parent_id", "unknown"),
                meta.get("chunk_index", 0)
            )
            meta["_qdrant_id"] = doc_id

        batch_size = cfg.get("INGEST_BATCH_SIZE", 64)
        total = len(docs)
        total_batches = (total + batch_size - 1) // batch_size

        log(f"开始写入 Qdrant: 共 {total} 个文档, 分 {total_batches} 批 (batch_size={batch_size})")

        failed_batches = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_docs = docs[start:end]

            # 提取确定性 ID，并校验长度一致性
            ids = [doc.metadata.get("_qdrant_id") for doc in batch_docs]
            # 过滤掉 None 值后检查长度是否与文档数匹配
            filtered_ids = [i for i in ids if i is not None]
            if len(filtered_ids) != len(batch_docs):
                log(f"[WARNING] 批次 {batch_idx+1}: IDs 数量({len(filtered_ids)})与文档数量({len(batch_docs)})不匹配，将使用自动生成 ID")
                ids = None
            else:
                ids = filtered_ids

            # 指数退避重试，最多3次
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(documents=batch_docs, ids=ids)
                    # 写入成功后清理临时字段
                    for doc in batch_docs:
                        doc.metadata.pop("_qdrant_id", None)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        log(f"  批次写入失败，{wait_time}s 后重试 ({attempt+1}/{max_retries-1}): {e}")
                        time.sleep(wait_time)
                    else:
                        log(f"  [ERROR] 批次写入最终失败: {e}")
                        failed_batches.append(batch_idx)

            log(f"  批次 {batch_idx + 1}/{total_batches} 完成 ({end}/{total})")

        if failed_batches:
            log(f"[WARNING] {len(failed_batches)} 个批次写入失败: {failed_batches}")

        log(f"Qdrant 写入完成: 共 {total} 个文档")

        # 存储父块到独立集合（父块集合已在子块写入前创建）
        if parent_chunks_data:
            parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
            store_parent_chunks_batch(client, parent_collection, parent_chunks_data, summary_map=summary_map or {})

    except Exception as e:
        log(f"写入 Qdrant 失败: {e}")
        raise


# ================= MAIN =================

def main():
    """
    主流程
    """
    log("===== 开始执行 ingest (Markdown Only) =====")

    try:
        cfg = load_config()

        # Qdrant 健康检查：在入库流程开始前验证 Qdrant 服务可用性
        qdrant_host = cfg["QDRANT_HOST"]
        qdrant_port = cfg["QDRANT_PORT"]
        try:
            health_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
            health_client.get_collections()
            log(f"Qdrant 健康检查通过: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            log(f"[ERROR] Qdrant 连接失败 ({qdrant_host}:{qdrant_port}): {e}")
            log("请检查 Qdrant 服务是否启动，以及 QDRANT_HOST/QDRANT_PORT 配置是否正确")
            sys.exit(1)

        docs = load_documents(cfg["DOCS_PATH"])
        if not docs:
            log("未加载到任何 .md 文档，请检查 DOCS_PATH 或文件后缀")
            return

        chunks, parent_chunks_data, summary_map = split_documents(docs, cfg)
        if not chunks:
            log("切块结果为空，任务结束")
            return

        chunks = filter_chunks(chunks, cfg)
        if not chunks:
            log("过滤后没有可用 chunk，任务结束")
            return

        embeddings = build_embeddings(cfg)
        save_to_qdrant(chunks, embeddings, cfg, parent_chunks_data=parent_chunks_data, summary_map=summary_map)

        log("===== ingest 完成 =====")

    except Exception as e:
        log(f"ingest 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
