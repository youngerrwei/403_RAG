# ingest_faiss.py

import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

import fitz
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ================= LOG =================

def log(msg: str):
    print(f"[INFO] {msg}")


# ================= CONFIG =================

def load_config():
    """
    加载环境变量配置。
    """
    load_dotenv()
    cfg = {
        "DOCS_PATH": os.getenv("DOCS_PATH", "./data"),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "900")),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", "200")),
        "VECTOR_STORE_PATH": os.getenv("VECTOR_STORE_PATH", "./vector_store/faiss_index"),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cuda"),
    }
    log(f"配置加载完成: {cfg}")
    return cfg


# ================= TEXT CLEAN =================

def clean_text_light(text: str) -> str:
    """
    轻量清洗文本：
    - 合并英文断词
    - 将单换行转为空格
    - 压缩多余空白
    """
    if not text:
        return ""

    # 处理英文单词跨行断开，如 "exam-\nple" -> "example"
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)

    # 单换行变空格，保留段落感弱一些，但有利于后续统一切块
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 压缩空白
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ================= TITLE / SECTION DETECTION =================

def extract_pdf_title(doc) -> Optional[str]:
    """
    从 PDF 首页中启发式提取文档标题。
    这不是完美方案，但对于大多数论文/实验文档足够实用。
    """
    try:
        first_page = doc[0].get_text("text")
        lines = [x.strip() for x in first_page.split("\n") if x.strip()]

        # 取首页前若干行作为候选
        candidates = []
        for line in lines[:12]:
            # 过滤太短/太长/看起来像页眉页脚的内容
            if 5 <= len(line) <= 200 and not is_header_footer(line):
                candidates.append(line)

        if candidates:
            return candidates[0]
    except Exception:
        pass
    return None


def detect_section_title(line: str) -> Optional[str]:
    """
    简单检测章节标题。
    可识别：
    - 全大写短句
    - 常见学术标题关键词
    - 编号标题，如 1. Introduction / 2 Method
    """
    line = line.strip()
    if not line:
        return None

    # 1) 全大写短句
    if len(line) < 100 and line.isupper():
        return line

    lower = line.lower()

    # 2) 常见章节关键词
    keywords = [
        "abstract", "introduction", "background", "related work",
        "system model", "problem formulation", "method", "methods",
        "algorithm", "approach", "experiment", "experiments",
        "results", "discussion", "conclusion", "references",
        "materials", "materials and methods"
    ]
    for k in keywords:
        if lower == k or lower.startswith(k) or f" {k}" in lower:
            return line

    # 3) 编号式标题
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z].{0,100}$", line):
        return line

    # 4) 罗马数字标题，如 "III. METHOD"
    if re.match(r"^[IVXLC]+\.\s+.{1,100}$", line):
        return line

    return None


# ================= HEADER FILTER =================

def is_header_footer(line: str) -> bool:
    """
    过滤常见页眉页脚。
    """
    line = line.strip().lower()
    if not line:
        return False

    patterns = [
        r"^ieee",
        r"^vol\.",
        r"^no\.",
        r"^pp\.",
        r"^\d{4}$",
        r"^page\s+\d+",
        r"^\d+\s*$",
    ]
    return any(re.search(p, line) for p in patterns)


# ================= PDF =================

def parse_pdf(file_path: Path) -> List[Document]:
    """
    解析 PDF：
    - 提取文档标题
    - 逐页解析
    - 识别章节标题并写入 metadata
    - 页面文本先不强行注入过多标题，后续 chunk 阶段统一注入
    """
    log(f"开始解析 PDF: {file_path.name}")

    docs = []
    pdf = fitz.open(str(file_path))
    doc_title = extract_pdf_title(pdf) or file_path.stem

    current_section = None

    for i, page in enumerate(pdf):
        text = page.get_text("text")

        lines = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            if is_header_footer(line):
                continue

            title = detect_section_title(line)
            if title:
                current_section = title
                lines.append(title)
                continue

            lines.append(line)

        cleaned = clean_text_light("\n".join(lines))
        if not cleaned:
            continue

        docs.append(Document(
            page_content=cleaned,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "doc_title": doc_title,
                "page": i + 1,
                "section": current_section,
                "type": "pdf"
            }
        ))

    log(f"PDF解析完成: {file_path.name}, 页数={len(pdf)}, 生成doc数={len(docs)}")
    return docs


# ================= DOCX =================

def load_docx(file_path: Path) -> List[Document]:
    """
    加载 DOCX。
    对 docx 暂不做复杂标题解析，先将文件名作为 doc_title。
    后续如需要可再基于段落样式做 heading 提取。
    """
    log(f"开始加载 DOCX: {file_path.name}")

    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()

    results = []
    for d in docs:
        text = clean_text_light(d.page_content)
        if text:
            results.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "doc_title": file_path.stem,
                    "page": None,
                    "section": None,
                    "type": "docx"
                }
            ))

    log(f"DOCX加载完成: {file_path.name}, doc数={len(results)}")
    return results


# ================= LOAD =================

def load_documents(path: str) -> List[Document]:
    """
    遍历目录，加载所有 PDF / DOCX。
    """
    root = Path(path)
    docs = []

    for f in root.rglob("*"):
        if f.suffix.lower() == ".pdf":
            docs.extend(parse_pdf(f))
        elif f.suffix.lower() == ".docx":
            docs.extend(load_docx(f))

    log(f"总文档数: {len(docs)}")
    return docs


# ================= CHUNK ENRICH =================

def enrich_chunk_with_titles(doc: Document) -> Document:
    """
    将标题等结构信息轻量注入到 chunk 文本前缀中。
    这是提升检索效果的关键步骤之一。
    """
    doc_title = doc.metadata.get("doc_title")
    section = doc.metadata.get("section")
    page = doc.metadata.get("page")

    prefix_parts = []
    if doc_title:
        prefix_parts.append(f"[文档标题] {doc_title}")
    if section:
        prefix_parts.append(f"[章节标题] {section}")
    if page:
        prefix_parts.append(f"[页码] {page}")

    prefix = "\n".join(prefix_parts)
    content = doc.page_content.strip()

    if prefix:
        content = prefix + "\n\n" + content

    return Document(
        page_content=content,
        metadata=doc.metadata
    )


# ================= CHUNK =================

def split_documents(docs: List[Document], chunk_size: int, overlap: int):
    """
    切块后统一给每个 chunk 注入标题/章节等结构信息。
    """
    log("开始文本切块")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    chunks = [enrich_chunk_with_titles(c) for c in chunks]

    log(f"切块完成: {len(chunks)}")
    return chunks


# ================= FILTER =================

def is_bad_chunk(text: str) -> bool:
    """
    更严格的低质量 chunk 过滤规则：
    1. 太短
    2. 有效文字太少
    3. 数字比例过高（疑似表格/编号残片）
    4. 特殊符号比例过高（疑似乱码/格式噪声）
    5. 重复字符/重复词过多
    """
    if not text:
        return True

    text = text.strip()
    if len(text) < 80:
        return True

    # 统计中英文有效字符
    valid_chars = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    valid_count = len(valid_chars)

    if valid_count < 30:
        return True

    digits = len(re.findall(r"\d", text))
    special_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))

    # 数字比例过高，通常是表格/编号/参数残片
    if valid_count > 0 and digits / valid_count > 1.2:
        return True

    # 特殊符号过多，通常是噪声
    if len(text) > 0 and special_chars / len(text) > 0.25:
        return True

    # 重复词过多：例如 "test test test test"
    words = text.split()
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    # 单字符重复过多：例如 "............." 或 "AAAAAAA"
    most_common_char_count = max(text.count(ch) for ch in set(text))
    if len(text) > 0 and most_common_char_count / len(text) > 0.35:
        return True

    return False


def analyze_bad_chunk_reason(text: str) -> str:
    if not text or not text.strip():
        return "empty"

    text = text.strip()
    if len(text) < 100:
        return "too_short"

    valid_chars = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    valid_count = len(valid_chars)
    if valid_count < 30:
        return "too_few_valid_chars"

    digits = len(re.findall(r"\d", text))
    special_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))

    if valid_count > 0 and digits / valid_count > 1.2:
        return "too_many_digits"

    if len(text) > 0 and special_chars / len(text) > 0.3:
        return "too_many_special_chars"

    words = text.split()
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return "too_repetitive_words"

    most_common_char_count = max(text.count(ch) for ch in set(text))
    if len(text) > 0 and most_common_char_count / len(text) > 0.35:
        return "too_repetitive_chars"

    return "good"


def filter_chunks(docs: List[Document]):
    log("开始过滤低质量chunk")

    results = []
    reason_stats = {}

    for d in docs:
        reason = analyze_bad_chunk_reason(d.page_content)
        reason_stats[reason] = reason_stats.get(reason, 0) + 1
        if reason == "good":
            results.append(d)

    log(f"过滤前: {len(docs)}, 过滤后: {len(results)}")
    log(f"过滤统计: {reason_stats}")
    return results


# ================= EMBEDDING =================

def build_embeddings(cfg):
    """
    初始化 embedding 模型。
    """
    log("初始化Embedding模型")

    return HuggingFaceEmbeddings(
        model_name=cfg["EMBEDDING_MODEL_NAME"],
        model_kwargs={"device": cfg["EMBEDDING_DEVICE"]},
        encode_kwargs={"normalize_embeddings": True},
    )


# ================= SAVE =================

def save_faiss(docs, embeddings, path: str):
    """
    构建并保存 FAISS。
    """
    log("开始构建FAISS")

    p = Path(path)

    if p.exists():
        log("删除旧索引")
        shutil.rmtree(p)

    p.mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(p))

    log(f"FAISS保存完成: {p}")
    log(f"最终chunk数: {len(docs)}")


# ================= MAIN =================

def main():
    log("===== 开始执行 ingest =====")

    cfg = load_config()

    docs = load_documents(cfg["DOCS_PATH"])

    chunks = split_documents(docs, cfg["CHUNK_SIZE"], cfg["CHUNK_OVERLAP"])

    chunks = filter_chunks(chunks)

    embeddings = build_embeddings(cfg)

    save_faiss(chunks, embeddings, cfg["VECTOR_STORE_PATH"])

    log("===== ingest 完成 =====")


if __name__ == "__main__":
    main()