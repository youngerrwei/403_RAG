# 入库任务采用严格退出码，并在成功覆盖后清理同源旧 point。
import os
import re
import sys
import json
import hashlib
import time
import unicodedata
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
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct

from langchain_qdrant import QdrantVectorStore

from .logger import get_logger
from .paths import PROJECT_ROOT
from . import lexicon


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
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    cfg = {
        "DOCS_PATH": os.getenv("DOCS_PATH", "/mnt/cpu_share"),
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
        "ENABLE_TABLE_SEMANTIC_ENRICHMENT": os.getenv("ENABLE_TABLE_SEMANTIC_ENRICHMENT", "true").lower() == "true",
        "TABLE_SEMANTIC_MAX_ROWS": int(os.getenv("TABLE_SEMANTIC_MAX_ROWS", "20")),
        # v3.3.0 检索规则词典路径（表格恢复模板/词表外置）
        "LEXICON_PATH": os.getenv("LEXICON_PATH", "config/retrieval_lexicon.json").strip(),
        # OCR 残缺目录页的点线含大量 U+FFFD，语义转写不得放大这类噪声
        "TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO": float(os.getenv("TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO", "0.15")),
        "TABLE_FLATTENED_LINE_MIN_CHARS": int(os.getenv("TABLE_FLATTENED_LINE_MIN_CHARS", "240")),
        "INGEST_BATCH_SIZE": int(os.getenv("INGEST_BATCH_SIZE", "64")),
        "EMBEDDING_INIT_TIMEOUT": int(os.getenv("EMBEDDING_INIT_TIMEOUT", "300")),
        # Markdown 结构保护与路径语义增强
        "ENABLE_MARKDOWN_STRUCTURE_GUARD": os.getenv("ENABLE_MARKDOWN_STRUCTURE_GUARD", "true").lower() == "true",
        "ENABLE_PATH_CONTEXT_PREFIX": os.getenv("ENABLE_PATH_CONTEXT_PREFIX", "true").lower() == "true",
        "CHUNK_FILTER_SPECIAL_CHAR_RATIO": float(os.getenv("CHUNK_FILTER_SPECIAL_CHAR_RATIO", "0.25")),
        "ENABLE_TABLE_FILTER_EXEMPTION": os.getenv("ENABLE_TABLE_FILTER_EXEMPTION", "true").lower() == "true",
        "INGEST_MAX_FAILED_FILES": int(os.getenv("INGEST_MAX_FAILED_FILES", "0")),
        "QDRANT_INDEX_FAIL_FAST": os.getenv("QDRANT_INDEX_FAIL_FAST", "true").lower() == "true",

        # vLLM 配置（供摘要调用使用）
        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "lab-secret-key"),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct"),
        "VLLM_ENABLE_THINKING": os.getenv("VLLM_ENABLE_THINKING", "false").lower() == "true",

        # 摘要增强配置
        "ENABLE_SUMMARY_AUGMENTATION": os.getenv("ENABLE_SUMMARY_AUGMENTATION", "true").lower() == "true",
        "SUMMARY_VLLM_TIMEOUT": int(os.getenv("SUMMARY_VLLM_TIMEOUT", "30")),
        "VLLM_INFERENCE_PROBE_TIMEOUT": int(os.getenv("VLLM_INFERENCE_PROBE_TIMEOUT", "15")),
        "SUMMARY_VLLM_RETRIES": int(os.getenv("SUMMARY_VLLM_RETRIES", "3")),
        "SUMMARY_MAX_WORKERS": int(os.getenv("SUMMARY_MAX_WORKERS", "3")),
        "SUMMARY_MAX_TOKENS": int(os.getenv("SUMMARY_MAX_TOKENS", "300")),
        "SUMMARY_INJECTION_MODE": os.getenv("SUMMARY_INJECTION_MODE", "both"),
        "ENABLE_SUMMARY_CACHE": os.getenv("ENABLE_SUMMARY_CACHE", "true").lower() == "true",
        "SUMMARY_CACHE_DIR": os.getenv("SUMMARY_CACHE_DIR", "./data/summary_cache"),
    }

    # 密钥只用于请求鉴权，禁止写入日志。
    safe_cfg = {key: value for key, value in cfg.items() if key != "VLLM_API_KEY"}
    log(f"配置加载完成: {safe_cfg}")
    return cfg


# ================= TEXT CLEAN =================

def _table_recovery_rules() -> dict:
    """表格恢复的词表与模板统一来自检索词典，代码只保留通用判定逻辑。"""
    return lexicon.load_lexicon()["table_recovery"]


def recover_flattened_variable_format_table(text: str) -> str:
    """修复已被 OCR/旧清洗流程压平成单行的变量格式表。"""
    rules = _table_recovery_rules()
    if lexicon.load_lexicon()["notation"]["recovery_marker"] in text:
        return text
    compact = re.sub(r"\s+", "", text)
    # OCR 对「正体+粗体」有多种写法，任一命中即可，避免措辞稍变就漏恢复
    if all(term in compact for term in rules["core_terms"]) and any(term in compact for term in rules["notation_terms"]):
        return f"{text.rstrip()}\n\n{rules['template']}"
    return text


def mark_suspect_flattened_tables(text: str, cfg: dict = None) -> str:
    """标记无法可靠还原的通用扁平表格；警告就近插入，保证与问题行落入同一块。"""
    lex = lexicon.load_lexicon()
    if lex["notation"]["recovery_marker"] in text or lex["flattened_table_marker"] in text:
        return text
    cfg = cfg or {}
    min_chars = max(1, int(cfg.get("TABLE_FLATTENED_LINE_MIN_CHARS", 240)))
    groups = lex["flattened_table_groups"]
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if len(line) >= min_chars and not line.lstrip().startswith("|"):
            hits = sum(1 for group in groups if any(term.lower() in lowered for term in group))
            if hits >= 3:
                warning = lex["flattened_table_warning"]
                # 紧跟问题行插入，切块后仍能与可疑内容一起被召回
                lines[idx + 1:idx + 1] = ["", warning]
                return "\n".join(lines)
    return text


def enrich_markdown_tables(text: str, cfg: dict = None) -> str:
    """为所有标准 Markdown 表追加有界逐行语义转写，显式保留列关系。"""
    cfg = cfg or {}
    if not bool(cfg.get("ENABLE_TABLE_SEMANTIC_ENRICHMENT", True)):
        return text
    max_rows = max(1, int(cfg.get("TABLE_SEMANTIC_MAX_ROWS", 20)))
    max_replacement_ratio = max(0.0, float(cfg.get("TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO", 0.15)))
    lines = text.splitlines()
    output = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            output.append(lines[index])
            index += 1
            continue
        end = index
        block = []
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            block.append(lines[end])
            end += 1
        output.extend(block)
        if len(block) >= 2:
            # 先保护转义竖线，避免把 \| 当成列分隔符
            rows = [
                [cell.strip() for cell in row.strip().strip("|").replace(r"\|", "\u0002").split("|")]
                for row in block
            ]
            rows = [[cell.replace("\u0002", "|") for cell in row] for row in rows]
            separator_index = next(
                (i for i, row in enumerate(rows)
                 if row and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in row)),
                None,
            )
            if separator_index is not None and separator_index >= 1 and len(rows[0]) >= 2:
                headers = rows[0]
                semantic_rows = []
                for row in rows[separator_index + 1:separator_index + 1 + max_rows]:
                    # OCR 残缺目录页的点线被替换成成串 U+FFFD；转写这类行只会把噪声
                    # 再复制一遍（实测放大约 3 倍），占用父块预算并污染向量化文本。
                    raw_row = "".join(row)
                    if raw_row and raw_row.count("\ufffd") / len(raw_row) > max_replacement_ratio:
                        continue
                    # 列数不等时按位置对齐已有列（表头继承），不再整行丢弃
                    pairs = [
                        f"{headers[i]}={cell}"
                        for i, cell in enumerate(row[:len(headers)])
                        if cell
                    ]
                    if pairs:
                        semantic_rows.append("；".join(pairs))
                if semantic_rows:
                    output.extend(["", "[表格结构化转写]", *[f"- {row}" for row in semantic_rows]])
        index = end
    return "\n".join(output)


# OCR 常见的 LaTeX 命令断字（如 "\b eg i n {fi g u re}"），仅在能还原为已知命令时合并
_BROKEN_LATEX_COMMANDS = (
    "begin", "end", "centering", "includegraphics", "subfigure", "caption",
    "label", "hfill", "vfill", "mathbf", "mathrm", "textbf", "frac",
    "section", "subsection", "documentclass", "usepackage", "noindent",
    "hspace", "vspace", "small", "large",
)


def _repair_broken_latex_commands(text: str) -> str:
    """合并 OCR 在 LaTeX 命令字母间插入的空格；只处理已知命令，避免误伤正文。"""
    for cmd in _BROKEN_LATEX_COMMANDS:
        pattern = r"\\" + r"\s*".join(re.escape(ch) for ch in cmd) + r"(?![A-Za-z])"
        try:
            # 用 lambda 作为替换值，避免 replacement 中的反斜杠被解释为转义
            text = re.sub(pattern, lambda _m, c=cmd: "\\" + c, text)
        except re.error as e:
            debug_log(f"LaTeX 断字修复跳过 {cmd}: {e}")
    return text


# 全角中文标点保持原样：NFKC 会把「，」「：」压成半角，破坏表格单元格与转写文案
_KEEP_FULLWIDTH_PUNCT = set("，。；：？！、（）【】《》「」『』〈〉…—～·")


def _normalize_compatibility_chars(text: str) -> str:
    """归一 CJK 兼容字符（⼯作→工作）、全角字母数字与连字，但保留中文标点。"""
    out = []
    for ch in text:
        if ch in _KEEP_FULLWIDTH_PUNCT:
            out.append(ch)
            continue
        out.append(unicodedata.normalize("NFKC", ch))
    return "".join(out)


_HTML_TAG_PATTERN = re.compile(
    r"</?(?:div|span|p|br|b|i|u|em|strong|table|thead|tbody|tr|td|th|img|a|sup|sub|font|center|h[1-6]|ul|ol|li|code|pre)\b[^>]*>",
    re.IGNORECASE,
)
_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")


def _clean_text_linewise(text: str) -> str:
    """
    逐行清洗：代码块、表格、公式与行内代码跳过破坏性正则。

    原实现对整篇文本执行 <[^>]+> 删除和 [ \\t]+ 压缩，会误删 $a<b>c$、<T> 泛型，
    并压平代码块缩进；这里改为只删已知 HTML 标签并保护结构化行。
    """
    out = []
    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            out.append(_ZERO_WIDTH_PATTERN.sub("", line))
            continue
        if in_fence:
            out.append(_ZERO_WIDTH_PATTERN.sub("", line))
            continue
        line = _ZERO_WIDTH_PATTERN.sub("", line)
        # 表格、公式、行内代码与缩进代码块保留空格结构，避免破坏列关系与泛型
        protected = (
            stripped.startswith("|")
            or "$" in line
            or "`" in line
            or line.startswith("    ")
        )
        if protected:
            out.append(line.rstrip())
            continue
        line = _HTML_TAG_PATTERN.sub("", line)
        out.append(re.sub(r"[ \t]+", " ", line).rstrip())
    return "\n".join(out)


def clean_text(text: str, cfg: dict = None) -> str:
    """增强文本清洗，适用于实验室论文/规范/教程类 Markdown（保持单参数兼容）。"""
    if not text:
        return ""
    cfg = cfg or {}
    # 1. HTML 注释：图片占位改为保留 token，避免图注信息整段丢失；其余注释移除
    text = re.sub(
        r'<!--.*?-->',
        lambda m: "[图片]" if "image" in m.group(0).lower() else "",
        text,
        flags=re.DOTALL,
    )
    # 2. 移除 Markdown 图片语法，保留 [图片] 占位
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', lambda m: f'[图片: {m.group(1)}]' if m.group(1) else '[图片]', text)
    # 3. 兼容字符归一：统一 CJK 兼容字符（⼯作→工作）、全角数字/字母与连字，保留中文标点
    text = _normalize_compatibility_chars(text)
    # 4. 清理零宽字符和控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 5. 英文断词修复与 OCR 拆分 LaTeX 命令还原
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
    text = _repair_broken_latex_commands(text)
    # 6. 逐行清洗：结构化行（表格/代码块/公式）不做破坏性替换
    text = _clean_text_linewise(text)
    # 7. 多个空白行压缩为一个空白行；禁止将单换行改为空格，否则表格会被压平。
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # 8. 兼容已经由 OCR 或旧版清洗压平的已知变量格式表。
    text = recover_flattened_variable_format_table(text)
    # 9. 无法确定恢复的通用扁平表格必须就近显式标记，避免静默产生错误答案。
    text = mark_suspect_flattened_tables(text, cfg)
    # 10. 对所有标准 Markdown 表增加可检索的逐行语义，不替换原表。
    text = enrich_markdown_tables(text, cfg)
    return text.strip()


# 保留原函数名作为别名
clean_text_light = clean_text


# ================= MARKDOWN LOADER =================

def _looks_like_mojibake(text: str) -> bool:
    """粗判解码结果是否为乱码，避免按错误编码静默入库。

    注意：候选编码均使用严格解码，解码成功后文本中的 U+FFFD 是源文档自身的
    OCR/转换残留（真实设备手册中大量存在），不能当作解码失败标志。
    """
    sample = text[:20000]
    if not sample:
        return False
    # UTF-8 多字节文本被按单字节解码时，拉丁扩展字符占比会异常高；
    # 正常中英文语料几乎不含这类字符，法文/德文变音字母占比也远低于阈值
    latin_chars = len(re.findall(r"[\u00c0-\u00ff]", sample))
    if latin_chars / len(sample) > 0.08:
        return True
    control = sum(
        1 for ch in sample
        if unicodedata.category(ch) in ("Cc", "Co", "Cn") and ch not in "\n\r\t"
    )
    return control / len(sample) > 0.02


def _decode_markdown(raw: bytes):
    """按 BOM 与候选编码解码，返回 (文本, 编码)；全部失败时返回 (None, None)。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        candidates = ["utf-8-sig"]
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ["utf-16"]
    else:
        # 不再用 latin-1 兜底：它会把任意字节解成"看似合法"的乱码
        candidates = ["utf-8", "gb18030", "big5"]
    for encoding in candidates:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
        if _looks_like_mojibake(text):
            continue
        return text, encoding
    return None, None


def load_md(file_path: Path, cfg: dict = None) -> List[Document]:
    """加载 Markdown 文件，支持 BOM 嗅探与乱码置信度校验。"""
    log(f"开始加载 Markdown: {file_path.name}")
    results = []

    try:
        raw = file_path.read_bytes()
    except OSError as e:
        log(f"[ERROR] 读取文件失败: {file_path}, 错误: {e}")
        return results

    content, used_encoding = _decode_markdown(raw)
    if content is None:
        log(f"[ERROR] 无法识别文件编码或解码结果疑似乱码: {file_path}")
        return results

    if used_encoding not in ('utf-8', 'utf-8-sig'):
        log(f"[INFO] 文件 {file_path.name} 使用编码: {used_encoding}")

    # 清洗文本（配置显式传入，避免表格相关开关绕过 cfg 直接读 env）
    text = clean_text(content, cfg)

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

def load_documents(path: str, cfg: dict = None) -> List[Document]:
    """
    遍历目录，仅加载 .md 文件，并为每个文档补充相对路径 metadata。

    单文件加载异常会被计数；超过 INGEST_MAX_FAILED_FILES 时中止入库，
    避免共享盘挂载/权限/编码异常导致知识库静默缺文档。
    """
    root = Path(path).resolve()
    docs: List[Document] = []
    cfg = cfg or {}
    max_failed = int(cfg.get("INGEST_MAX_FAILED_FILES", 0))
    failed_files = []

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

            # 加载 markdown（解码失败会返回空列表，计入失败数避免静默漏文档）
            file_docs = load_md(f, cfg)
            if not file_docs:
                failed_files.append(f"{f}: 解码失败或清洗后为空")
                log(f"[WARNING] 文档未产生有效内容: {f}")

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
            failed_files.append(f"{f}: {e}")
            log(f"加载文件失败: {f}, 错误: {e}")

    log(f"总文档数: {len(docs)}")
    if failed_files:
        log(f"[WARNING] Markdown 文件加载失败数: {len(failed_files)}")
        if len(failed_files) > max_failed:
            raise RuntimeError(
                f"有 {len(failed_files)} 个 Markdown 文件加载失败（允许上限 {max_failed}），"
                f"为避免知识库缺文档，本次入库中止。首个失败: {failed_files[0]}"
            )
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

# 切分阶段把表格/代码块内的换行替换为占位符，使其不会被递归切分器按行腰斩
_ATOMIC_NEWLINE = "\u0001"


def _protect_atomic_blocks(text: str) -> str:
    """把连续表格行与 fenced code 合并为不可分割单元，供 RecursiveCharacterTextSplitter 使用。"""
    lines = text.split("\n")
    out = []
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            end = index + 1
            while end < len(lines) and lines[end].lstrip()[:3] != marker:
                end += 1
            if end < len(lines):
                end += 1
            out.append(_ATOMIC_NEWLINE.join(lines[index:end]))
            index = end
            continue
        if stripped.startswith("|"):
            end = index
            while end < len(lines) and lines[end].lstrip().startswith("|"):
                end += 1
            out.append(_ATOMIC_NEWLINE.join(lines[index:end]))
            index = end
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out)


def _restore_atomic_blocks(text: str) -> str:
    """还原切分阶段的原子块占位符，保证最终文本仍是合法 Markdown。"""
    return text.replace(_ATOMIC_NEWLINE, "\n")


def split_by_markdown_headers(doc: Document, parent_chunk_size: int = 1500, parent_chunk_overlap: int = 200) -> List[Document]:
    """
    按 Markdown 标题（# ~ ####）将文档切分为逻辑段落。

    标题识别改用行级状态机，跳过 fenced code 内以 # 开头的注释行；
    尺寸切分统一交给 split_documents 的 parent_splitter，避免两级切分叠加导致
    overlap 漂移。保留 parent_chunk_size / parent_chunk_overlap 形参以兼容既有调用。
    """
    text = doc.page_content
    if not text or not text.strip():
        return [doc]

    lines = text.split("\n")
    headers = []
    in_fence = False
    fence_marker = ""
    for line_idx, line in enumerate(lines):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if match:
            headers.append((line_idx, len(match.group(1)), match.group(2).strip()))

    # 无任何标题时回退为整篇文档（短规范文档常见）
    if not headers:
        return [doc]

    sections = []
    header_stack = {}
    for i, (line_idx, level, title) in enumerate(headers):
        end_line = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        # 更新标题栈：清除同级以下层级
        header_stack[level] = title
        for key in [key for key in header_stack if key > level]:
            del header_stack[key]
        header_path = " > ".join(header_stack[lv] for lv in sorted(header_stack))

        section_text = "\n".join(lines[line_idx:end_line]).strip()
        if not section_text:
            continue
        section_metadata = dict(doc.metadata) if doc.metadata else {}
        section_metadata["section_title"] = title
        section_metadata["header_path"] = header_path
        section_metadata["header_level"] = level
        sections.append(Document(page_content=section_text, metadata=section_metadata))

    # 第一个标题之前的前言内容
    if headers[0][0] > 0:
        preamble_text = "\n".join(lines[:headers[0][0]]).strip()
        if preamble_text:
            preamble_metadata = dict(doc.metadata) if doc.metadata else {}
            preamble_metadata["section_title"] = ""
            preamble_metadata["header_path"] = ""
            preamble_metadata["header_level"] = 0
            sections.insert(0, Document(page_content=preamble_text, metadata=preamble_metadata))

    return sections or [doc]


# ================= PARENT COLLECTION =================

def ensure_parent_collection(client, collection_name: str, cfg: dict = None):
    """创建父块存储集合（使用最小维度向量占位），并补齐精确过滤所需的 keyword 索引。"""
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

    # 父块展开按 parent_id、陈旧清理按 source 精确过滤；补索引避免退化为全量 scroll
    keyword_index_params = rest.KeywordIndexParams(type=rest.KeywordIndexType.KEYWORD)
    for field_name in ("parent_id", "source"):
        _ensure_payload_index(client, collection_name, field_name, keyword_index_params)


def generate_parent_point_id(item: dict) -> str:
    """为父块生成稳定 ID，保证重复执行可幂等覆盖。"""
    seed = f"parent|{item.get('source', 'unknown')}|{item['parent_id']}"
    return str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, seed))


def store_parent_chunks_batch(client, collection_name: str, parent_chunks_data: list,
                              summary_map: dict = None, cfg: dict = None):
    """批量存储父块到独立集合，并保留路径/层级与出处标签供检索侧使用。"""
    if summary_map is None:
        summary_map = {}
    cfg = cfg or {}
    points = []
    for item in parent_chunks_data:
        point_id = generate_parent_point_id(item)
        rel_path = item.get("rel_path", "")
        header_path = item.get("header_path", "")
        # 轻量出处标签：父块原文不含任何文档信息，回答端据此标注来源
        source_label = " > ".join(
            part for part in (
                item.get("doc_title", ""),
                " > ".join(seg for seg in rel_path.split("/")[:-1] if seg),
                header_path,
            ) if part
        )
        points.append(PointStruct(
            id=point_id,
            vector=[0.0, 0.0, 0.0, 0.0],
            payload={
                "parent_id": item["parent_id"],
                "parent_content": item["parent_content"],
                "source": item.get("source", ""),
                "doc_title": item.get("doc_title", ""),
                "file_name": item.get("file_name", ""),
                "rel_path": rel_path,
                "section_title": item.get("section_title", ""),
                "header_path": header_path,
                "header_level": item.get("header_level", 0),
                "parent_ordinal": item.get("parent_ordinal", 0),
                "parent_source_label": source_label,
                "parent_summary": summary_map.get(item["parent_id"], ""),
            }
        ))
    if points:
        # 分批大小与子块写入共用同一配置
        batch_size = max(1, int(cfg.get("INGEST_BATCH_SIZE", 64)))
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

def _truncate_summary(summary: str, max_chars: int = 100) -> str:
    """在不超过 max_chars 的最后一个句读处截断摘要，避免半句话进入 Embedding。"""
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) <= max_chars:
        return summary
    window = summary[:max_chars]
    boundary = max(window.rfind(ch) for ch in "。；！？；.!?")
    if boundary >= max_chars // 2:
        return window[:boundary + 1]
    return window.rstrip()


def build_contextual_prefix(metadata: dict, parent_summary: str = "", injection_mode: str = "both", cfg: dict = None) -> str:
    """为 chunk 构建上下文前缀，提升 Embedding 质量（Contextual Retrieval）。"""
    cfg = cfg or {}
    parts = []

    # 文档来源
    source = metadata.get("source", "")
    if source:
        doc_name = source.rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in source else source.rsplit(".", 1)[0]
        parts.append(f"文档：{doc_name}")

    # 目录层级语义：小组 / 设备 / Manuals 进入向量，区分不同设备下的同名手册
    if bool(cfg.get("ENABLE_PATH_CONTEXT_PREFIX", True)):
        rel_path = metadata.get("rel_path", "")
        if rel_path:
            dir_segments = [seg for seg in rel_path.split("/")[:-1] if seg]
            if dir_segments:
                parts.append(f"路径：{' > '.join(dir_segments)}")

    # 章节路径
    header_path = metadata.get("header_path", "")
    if header_path:
        parts.append(f"章节：{header_path}")

    # 无章节路径的短文档才单独写标题，避免与 header_path 重复
    section_title = metadata.get("section_title", "")
    if section_title and not header_path:
        parts.append(f"标题：{section_title}")

    # 摘要注入：按句截断，避免截在句中；状态由 summary_status 写入 payload
    if parent_summary and injection_mode in ("both", "prefix_only"):
        parts.append(f"段落摘要：{_truncate_summary(parent_summary)}")

    if parts:
        return "【" + " > ".join(parts) + "】\n"
    return ""


# ==================== 摘要缓存文件路径 ====================

# 统一摘要缓存文件，key 为 parent_id（包含 content_hash，内容变化时自动失效）
SUMMARY_CACHE_FILE = str(PROJECT_ROOT / "data" / ".summary_cache.json")


def save_summary_cache(summary_map: dict, cache_file: str):
    """将摘要缓存持久化到 JSON 文件，确保后续步骤失败时摘要不丢失"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(summary_map, f, ensure_ascii=False, indent=2)
        log(f"摘要缓存已保存到: {cache_file}（共 {len(summary_map)} 条）")
    except Exception as e:
        log(f"[WARN] 保存摘要缓存失败: {e}")


def load_summary_cache(cache_file: str) -> dict:
    """从本地加载摘要缓存，支持断点续做"""
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log(f"从本地缓存加载了 {len(data)} 条摘要")
        # 丢弃旧版本可能写入的思考过程/整段回答污染，触发重新生成。
        cleaned = {}
        for key, value in data.items():
            text = str(value or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
            if "</think>" in text.lower():
                text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[-1]
            if any(marker in text for marker in ("知识库中未找到足够相关内容", "用户现在问的是", "上下文结果：")):
                text = ""
            if text.strip():
                cleaned[key] = text.strip()
        return cleaned
    except Exception as e:
        log(f"[WARN] 加载摘要缓存失败: {e}")
        return {}


# ==================== 摘要增强相关 ====================

SUMMARY_SYSTEM_PROMPT = """你是知识库内容摘要专家。给定一个来自学术论文或技术文档的文本段落，生成一个简洁准确的摘要。

摘要要求：
1. 保留核心概念、关键方法、重要结果或结论
2. 使用简洁学术语言，避免冗余
3. 保持原文意思准确，不添加额外推理
4. 去掉无关例子和参考文献编号；但如果表格规定变量、符号或公式格式，必须保留能区分正体、斜体、粗体、大小写的示例
5. 对表格按行保持“对象—规则”的对应关系，不得把不同单元格规则混合
6. 长度控制在150-250字"""

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
    检测 vLLM 控制面与真实生成能力是否可用。

    Args:
        cfg: 配置字典

    Returns:
        True 表示 vLLM 可用，False 表示不可用
    """
    base_url = cfg.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = cfg.get("VLLM_API_KEY", "lab-secret-key")

    # 先检查控制面，再用最小生成请求确认推理引擎确实可工作。
    health_url = f"{base_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(health_url, headers=headers, timeout=5)
        if resp.status_code != 200:
            log(f"[WARNING] vLLM 健康检查返回非200状态码: {resp.status_code}")
            return False

        probe_payload = {
            "model": cfg.get("VLLM_MODEL_NAME", "./models/Qwen3-8B-Instruct"),
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 1,
            "temperature": 0,
        }
        probe_timeout = cfg.get("VLLM_INFERENCE_PROBE_TIMEOUT", 15)
        probe_resp = requests.post(
            f"{base_url}/chat/completions",
            json=probe_payload,
            headers=headers,
            timeout=probe_timeout,
        )
        if probe_resp.status_code != 200:
            log(f"[WARNING] vLLM 真实生成探针返回 HTTP {probe_resp.status_code}")
            return False
        choices = probe_resp.json().get("choices")
        if not isinstance(choices, list) or not choices:
            log("[WARNING] vLLM 真实生成探针未返回 choices")
            return False
        debug_log("vLLM 控制面及真实生成探针均通过")
        return True
    except requests.ConnectionError:
        log("[WARNING] vLLM 服务无法连接，可能未启动")
        return False
    except requests.Timeout:
        log("[WARNING] vLLM 健康/真实生成探针超时，服务可能仅控制面可用")
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
        "chat_template_kwargs": {"enable_thinking": cfg.get("VLLM_ENABLE_THINKING", False)},
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
    并发为父块生成摘要，支持本地文件缓存断点续做。

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

    # === 从本地缓存文件加载已生成的摘要（支持断点续做） ===
    cached_summaries = load_summary_cache(SUMMARY_CACHE_FILE)
    summary_map = dict(cached_summaries)  # 以缓存为基础

    # 同时兼容旧的目录缓存（SummaryCache），逐步迁移
    old_cache = None
    if cfg.get("ENABLE_SUMMARY_CACHE", True):
        old_cache = SummaryCache(cfg.get("SUMMARY_CACHE_DIR", "./data/summary_cache"))

    # 过滤出需要生成摘要的父块（跳过已缓存的）
    needed_parents = [item for item in parent_chunks_data if item["parent_id"] not in summary_map]
    log(f"需要生成摘要: {len(needed_parents)} 个（已从本地缓存加载: {len(summary_map)} 个）")

    # 如果所有摘要都已缓存，直接返回
    if not needed_parents:
        log("所有父块摘要均已缓存，跳过 vLLM 调用")
        return summary_map

    def generate_one(item):
        parent_id = item["parent_id"]
        source = item.get("source", "")

        # 再次检查（防止并发重复），同时兼容旧目录缓存
        if old_cache:
            cached = old_cache.get(source, parent_id)
            if cached:
                return (parent_id, cached)

        # 调用 vLLM 生成
        summary = call_vllm_for_summary(
            parent_content=item["parent_content"],
            doc_title=item.get("doc_title", ""),
            header_path=item.get("header_path", ""),
            cfg=cfg
        )

        summary = str(summary or "").strip()
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.IGNORECASE | re.DOTALL).strip()
        if any(marker in summary for marker in ("知识库中未找到足够相关内容", "用户现在问的是", "上下文结果：")):
            summary = ""

        # 写入旧缓存目录（兼容）
        if summary and old_cache:
            old_cache.set(source, parent_id, summary)

        return (parent_id, summary)

    log(f"开始为 {len(needed_parents)} 个父块生成摘要（并发数={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_one, item): item for item in needed_parents}
        completed = 0
        for future in as_completed(futures):
            try:
                parent_id, summary = future.result()
                if summary:
                    summary_map[parent_id] = summary
                completed += 1
                # 每10%输出进度
                if completed % max(1, len(needed_parents) // 10) == 0:
                    log(f"摘要生成进度: {completed}/{len(needed_parents)}")
            except Exception as e:
                log(f"摘要生成异常: {e}")

    success_count = len(summary_map)
    total_count = len(parent_chunks_data)
    log(f"摘要生成完成: {success_count}/{total_count} 成功 ({100 * success_count // max(1, total_count)}%)")

    # === 摘要生成完成后立即持久化到本地，防止后续步骤（Embedding/Qdrant）失败导致结果丢失 ===
    save_summary_cache(summary_map, SUMMARY_CACHE_FILE)
    log("摘要已持久化到本地缓存文件，即使后续步骤失败也可复用")

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
        guard = bool(cfg.get("ENABLE_MARKDOWN_STRUCTURE_GUARD", True))

        # 第零层：Markdown 结构感知切分；表格/代码块标记为原子块避免按行腰斩
        parent_chunk_size = cfg["PARENT_CHUNK_SIZE"]
        parent_chunk_overlap = cfg["PARENT_CHUNK_OVERLAP"]
        logical_docs = []
        for doc in docs:
            for sec in split_by_markdown_headers(doc, parent_chunk_size, parent_chunk_overlap):
                if guard:
                    sec = Document(
                        page_content=_protect_atomic_blocks(sec.page_content),
                        metadata=sec.metadata,
                    )
                logical_docs.append(sec)
        log(f"Markdown 结构切分完成: {len(docs)} 个文档 -> {len(logical_docs)} 个逻辑段落")

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " "]
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["CHILD_CHUNK_SIZE"],
            chunk_overlap=cfg["CHILD_CHUNK_OVERLAP"],
            separators=["\n\n", "\n", "。", "！", "？", "，", ", ", " "]
        )

        # 第一层：父块尺寸切分（标题切分不再二次切分，避免 overlap 叠加）
        parent_chunks = parent_splitter.split_documents(logical_docs)
        if guard:
            for p_chunk in parent_chunks:
                p_chunk.page_content = _restore_atomic_blocks(p_chunk.page_content)

        # 父块序号参与 ID：设备手册常有重复章节，仅用内容哈希会让子块 ID 冲突覆盖
        parent_records = []
        for p_idx, p_chunk in enumerate(parent_chunks):
            parent_text = p_chunk.page_content
            source = p_chunk.metadata.get("source", "unknown")
            content_hash = hashlib.md5(parent_text.encode("utf-8")).hexdigest()
            parent_id = str(uuid_module.uuid5(
                uuid_module.NAMESPACE_URL,
                f"{source}|{p_idx}|{content_hash}",
            ))
            parent_records.append({
                "document": p_chunk,
                "parent_id": parent_id,
                "parent_text": parent_text,
                "source": source,
                "ordinal": p_idx,
            })

        # 父块 payload 补齐路径/层级字段，供按小组、设备过滤与时效管理
        parent_chunks_to_store = []
        for record in parent_records:
            meta = record["document"].metadata
            parent_chunks_to_store.append({
                "parent_id": record["parent_id"],
                "parent_content": record["parent_text"],
                "source": record["source"],
                "doc_title": meta.get("doc_title", ""),
                "file_name": meta.get("file_name", ""),
                "rel_path": meta.get("rel_path", ""),
                "section_title": meta.get("section_title", ""),
                "header_path": meta.get("header_path", ""),
                "header_level": meta.get("header_level", 0),
                "parent_ordinal": record["ordinal"],
            })

        # 调用 vLLM 为父块生成摘要（数据增强）
        summary_map = generate_parent_summaries(parent_chunks_to_store, cfg)
        injection_mode = cfg.get("SUMMARY_INJECTION_MODE", "both")

        # 第二层：生成子块并注入路径/章节前缀与摘要
        final_child_chunks = []
        for record in parent_records:
            parent_text = record["parent_text"]
            parent_id = record["parent_id"]
            p_chunk = record["document"]
            parent_summary = summary_map.get(parent_id, "")

            child_source = _protect_atomic_blocks(parent_text) if guard else parent_text
            child_chunks = child_splitter.split_documents([
                Document(page_content=child_source, metadata=p_chunk.metadata)
            ])

            for chunk_idx, c_chunk in enumerate(child_chunks):
                raw_content = _restore_atomic_blocks(c_chunk.page_content) if guard else c_chunk.page_content
                # 继承父块所有 metadata，并追加子块相关信息
                c_metadata = c_chunk.metadata.copy()
                c_metadata.update({
                    "parent_id": parent_id,
                    "parent_content": parent_text,  # 冗余存储，供父块查询降级使用
                    "size": "small",
                    "chunk_index": chunk_idx,
                    "chunk_hash": hashlib.md5(raw_content.encode("utf-8")).hexdigest(),
                    "ingestion_time": ingestion_time,
                    "embedding_model": cfg["EMBEDDING_MODEL_NAME"],
                    # 摘要可用性写入 payload，检索侧可判别降级状态
                    "summary_status": "ok" if parent_summary else "unavailable",
                })

                # 将父块摘要存入子块 metadata（metadata_only 或 both 模式）
                if parent_summary and injection_mode in ("both", "metadata_only"):
                    c_metadata["parent_summary"] = parent_summary

                # Contextual Retrieval：注入文档/路径/章节前缀，提升 embedding 检索精度
                prefix = build_contextual_prefix(c_metadata, parent_summary, injection_mode, cfg)
                child_content = prefix + raw_content if prefix else raw_content

                final_child_chunks.append(Document(
                    page_content=child_content,
                    metadata=c_metadata
                ))

        log(f"切块完成: 生成了 {len(parent_chunks)} 个父块，衍生出 {len(final_child_chunks)} 个子块")
        return final_child_chunks, parent_chunks_to_store, summary_map

    except Exception as e:
        log(f"文本切块失败: {e}")
        raise


# ================= FILTER =================

# 向量化前缀（build_contextual_prefix 生成）不参与质量判定，避免前缀稀释统计
_CONTEXT_PREFIX_PATTERN = re.compile(r"^【[^\n]*】\s*")
# Markdown / LaTeX / URL 中的结构性符号属于正常语料内容，不应计入「特殊字符」
_STRUCTURAL_CHARS = "。，、；：？！“”‘’（）【】《》.,;:?!\"'()[]-—…|$=^{}<>&%/~+#*:`"
_STRUCTURAL_PUNCT_PATTERN = re.compile(r"[\w\s\u4e00-\u9fff" + re.escape(_STRUCTURAL_CHARS) + r"]")


def _strip_context_prefix(text: str) -> str:
    """剥离上下文前缀，只对原始正文做质量判定。"""
    return _CONTEXT_PREFIX_PATTERN.sub("", text, count=1).strip()


def analyze_bad_chunk_reason(text: str, cfg: dict) -> str:
    """分析低质量 chunk 原因；判定对象为原始正文，并保护表格/代码块/型号参数。"""
    if not text or not text.strip():
        return "empty"
    text = _strip_context_prefix(text)
    if not text:
        return "empty"
    cfg = cfg or {}

    # 表格转写、变量格式恢复、代码围栏是语料中的高价值内容，豁免长度/字符占比过滤
    markers = lexicon.load_lexicon()["filter_exempt_markers"]
    if bool(cfg.get("ENABLE_TABLE_FILTER_EXEMPTION", True)) and (
        any(marker in text for marker in markers)
        or "```" in text
        or text.lstrip().startswith("|")
    ):
        return "good"

    min_length = cfg.get("MIN_CHUNK_LENGTH", 120)
    if len(text) < min_length:
        return "too_short"

    # 有效字符：中文字 + 英文词 + 设备型号/参数（如 MSO-X 3054A）
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'[A-Za-z]{2,}|[A-Za-z]?\d{2,}[A-Za-z0-9-]*', text)
    valid_count = len(chinese_chars) + len(english_words)
    if valid_count < 10:
        return "too_few_valid_chars"

    # 数字比例检测（允许公式/数据较多的论文内容）
    digits = len(re.findall(r'\d', text))
    if digits / len(text) > 0.6:
        return "too_many_digits"

    # 特殊字符比例：LaTeX 反斜杠与结构性符号已排除，阈值走配置
    special = _STRUCTURAL_PUNCT_PATTERN.sub("", text.replace("\\", ""))
    special_ratio = float(cfg.get("CHUNK_FILTER_SPECIAL_CHAR_RATIO", 0.25))
    if len(special) / len(text) > special_ratio:
        return "too_many_special_chars"

    # 检测纯表头/纯引用
    lines = text.split('\n')
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
    timeout = cfg["EMBEDDING_INIT_TIMEOUT"]

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
        _create_text_indexes(client, collection_name, cfg)
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
    _create_text_indexes(client, collection_name, cfg)


def _ensure_payload_index(client: QdrantClient, collection_name: str, field_name: str, field_schema) -> bool:
    """创建单个 payload 索引；已存在视为成功，其余异常记录日志并返回失败。"""
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        log(f"[INFO] 索引创建成功: {collection_name}.{field_name}")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "already_exists" in error_msg:
            debug_log(f"索引已存在，跳过: {collection_name}.{field_name}")
            return True
        log(f"[WARNING] 索引创建失败 [{collection_name}.{field_name}]: {e}")
        return False


def _verify_payload_indexes(client: QdrantClient, collection_name: str, fields: list, fail_fast: bool):
    """创建后读取 payload_schema 校验索引齐全，避免 MatchText 运行时才暴露问题。"""
    try:
        schema = client.get_collection(collection_name).payload_schema or {}
    except Exception as e:
        log(f"[WARNING] 无法读取 payload_schema 校验索引: {e}")
        return

    def is_indexed(field: str) -> bool:
        if field in schema:
            return True
        # 兼容不同 Qdrant 版本返回不带完整路径的字段名
        tail = field.rsplit(".", 1)[-1]
        return any(key == tail or key.endswith("." + tail) for key in schema)

    missing = [field for field in fields if not is_indexed(field)]
    if missing:
        message = f"集合 {collection_name} 缺少必要 payload 索引: {', '.join(missing)}"
        if fail_fast:
            raise RuntimeError(message + "；请确认 Qdrant 版本支持这些索引类型，或执行全量重建")
        log(f"[WARNING] {message}")
    else:
        log(f"[INFO] payload 索引校验通过: {collection_name}（{len(fields)} 个字段）")


def _create_text_indexes(client: QdrantClient, collection_name: str, cfg: dict = None):
    """
    为子块集合创建 MatchText 文本索引与增量覆盖所需的 keyword 索引。

    文本索引覆盖 page_content 及所有会被 MatchText 查询的 metadata 字段；
    keyword 索引用于 source / parent_id 精确过滤，避免 scroll 退化为全量扫描。
    创建结束后统一校验 payload_schema，缺失时按 QDRANT_INDEX_FAIL_FAST 决定是否中止。
    """
    cfg = cfg or {}
    fail_fast = bool(cfg.get("QDRANT_INDEX_FAIL_FAST", True))
    text_index_params = TextIndexParams(
        type=TextIndexType.TEXT,
        tokenizer=TokenizerType.MULTILINGUAL,
        min_token_len=2,
        lowercase=True,
    )
    keyword_index_params = rest.KeywordIndexParams(type=rest.KeywordIndexType.KEYWORD)

    # 顶层正文 + metadata 文本字段（新增 header_path 与 parent_summary，供路径/摘要检索）
    text_fields = [
        "page_content",
        "metadata.doc_title",
        "metadata.section_title",
        "metadata.file_name",
        "metadata.rel_path",
        "metadata.header_path",
        "metadata.parent_summary",
    ]
    # 精确匹配字段：collect_existing_ids_by_source 与父块展开依赖它们
    keyword_fields = ["metadata.source", "metadata.parent_id"]

    failed = []
    for field_name in text_fields:
        if not _ensure_payload_index(client, collection_name, field_name, text_index_params):
            failed.append(field_name)
    for field_name in keyword_fields:
        if not _ensure_payload_index(client, collection_name, field_name, keyword_index_params):
            failed.append(field_name)

    if failed:
        log(f"[WARNING] 以下索引未能创建: {', '.join(failed)}")
    _verify_payload_indexes(client, collection_name, text_fields + keyword_fields, fail_fast)


# ================= DETERMINISTIC ID =================

def generate_deterministic_id(source: str, content_hash: str, chunk_index: int) -> str:
    """基于源文件、内容哈希和块索引生成确定性 UUID，支持幂等入库"""
    seed = f"{source}|{content_hash}|{chunk_index}"
    return str(uuid_module.uuid5(uuid_module.NAMESPACE_URL, seed))


def collect_existing_ids_by_source(client, collection_name: str, sources: set, source_key: str) -> dict:
    """在覆盖写入前记录各源文件已有 point ID，供成功后清理陈旧块。"""
    existing = {source: set() for source in sources}
    if not sources or not client.collection_exists(collection_name):
        return existing

    for source in sources:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=rest.Filter(
                    must=[rest.FieldCondition(key=source_key, match=rest.MatchValue(value=source))]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            existing[source].update(str(point.id) for point in points)
            if offset is None:
                break
    return existing


def delete_stale_ids(client, collection_name: str, old_ids: dict, new_ids: dict):
    """仅在新子块和父块均成功写入后删除同源的旧 point。"""
    stale_ids = set()
    for source, ids in old_ids.items():
        stale_ids.update(ids - new_ids.get(source, set()))
    if not stale_ids:
        return

    client.delete(
        collection_name=collection_name,
        points_selector=rest.PointIdsList(points=sorted(stale_ids)),
        wait=True,
    )
    log(f"[INFO] 已从 {collection_name} 清理 {len(stale_ids)} 个陈旧 point")


# ================= SAVE =================

def save_to_qdrant(docs: List[Document], embeddings, cfg: dict, parent_chunks_data: list = None, summary_map: dict = None):
    """分批写入 Qdrant，优化大规模入库性能"""
    log("开始写入 Qdrant")

    if not docs:
        raise ValueError("没有可写入的文档 chunk")

    try:
        client = create_qdrant_client(cfg)
        vector_size = get_embedding_dimension(embeddings)
        prepare_collection(client, cfg, vector_size)

        # 确保父块集合在所有数据写入之前已创建，避免后续检索时父块展开查询失败
        parent_collection = cfg.get("QDRANT_PARENT_COLLECTION", "lab_knowledge_base_parents")
        ensure_parent_collection(client, parent_collection, cfg)

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

        # 为每个文档生成确定性 ID，并记录每个源文件本次应保留的 ID。
        child_ids_by_source = {}
        for doc in docs:
            meta = doc.metadata
            source = meta.get("source", "unknown")
            # 把已算出的 chunk_hash 纳入 ID：避免相同父块序号下内容未变却无法识别，
            # 也避免重复父文本（设备手册常见）导致子块 ID 冲突覆盖
            doc_id = generate_deterministic_id(
                source,
                f"{meta.get('parent_id', 'unknown')}:{meta.get('chunk_hash', '')}",
                meta.get("chunk_index", 0)
            )
            meta["_qdrant_id"] = doc_id
            child_ids_by_source.setdefault(source, set()).add(doc_id)

        parent_chunks_data = parent_chunks_data or []
        parent_ids_by_source = {}
        for item in parent_chunks_data:
            source = item.get("source", "")
            parent_ids_by_source.setdefault(source, set()).add(generate_parent_point_id(item))

        # 增量覆盖时先拍摄旧 ID；所有新数据写完后才执行清理，避免失败导致数据丢失。
        incoming_sources = set(child_ids_by_source) | set(parent_ids_by_source)
        if cfg.get("QDRANT_RECREATE_COLLECTION", False):
            old_child_ids = {}
            old_parent_ids = {}
        else:
            old_child_ids = collect_existing_ids_by_source(
                client, collection_name, incoming_sources, "metadata.source"
            )
            old_parent_ids = collect_existing_ids_by_source(
                client, parent_collection, incoming_sources, "source"
            )

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
                raise RuntimeError(
                    f"批次 {batch_idx + 1}: IDs 数量({len(filtered_ids)})"
                    f"与文档数量({len(batch_docs)})不匹配"
                )
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
            raise RuntimeError(f"{len(failed_batches)} 个子块批次写入失败: {failed_batches}")

        log(f"Qdrant 写入完成: 共 {total} 个文档")

        # 存储父块到独立集合（父块集合已在子块写入前创建）
        if parent_chunks_data:
            store_parent_chunks_batch(client, parent_collection, parent_chunks_data,
                                     summary_map=summary_map or {}, cfg=cfg)

        delete_stale_ids(client, collection_name, old_child_ids, child_ids_by_source)
        delete_stale_ids(client, parent_collection, old_parent_ids, parent_ids_by_source)

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

        docs = load_documents(cfg["DOCS_PATH"], cfg)
        if not docs:
            raise RuntimeError("未加载到任何 .md 文档，请检查 DOCS_PATH 或文件后缀")

        chunks, parent_chunks_data, summary_map = split_documents(docs, cfg)
        if not chunks:
            raise RuntimeError("切块结果为空")

        chunks = filter_chunks(chunks, cfg)
        if not chunks:
            raise RuntimeError("过滤后没有可用 chunk")

        # 按存活的 parent_id 同步裁剪父块与摘要，避免写入"有父无子"的孤儿点
        alive_parent_ids = {c.metadata.get("parent_id") for c in chunks}
        parent_chunks_data = [
            item for item in parent_chunks_data
            if item.get("parent_id") in alive_parent_ids
        ]
        summary_map = {
            pid: summary for pid, summary in (summary_map or {}).items()
            if pid in alive_parent_ids
        }
        debug_log(f"过滤后保留: 子块={len(chunks)}, 父块={len(parent_chunks_data)}")

        embeddings = build_embeddings(cfg)
        save_to_qdrant(chunks, embeddings, cfg, parent_chunks_data=parent_chunks_data, summary_map=summary_map)

        log("===== ingest 完成 =====")

    except Exception as e:
        log(f"ingest 执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
