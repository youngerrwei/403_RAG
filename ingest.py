# ingest.py

import os
import re
from pathlib import Path
from typing import List, Optional

import fitz
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_qdrant import QdrantVectorStore


# ================= LOG =================

def log(msg: str):
    """
    简单日志输出函数。
    """
    print(f"[INFO] {msg}")


# ================= CONFIG =================

def load_config():
    """
    加载环境变量配置。

    说明：
    - DOCS_PATH：知识库原始文件目录（PDF / DOCX）
    - CHUNK_SIZE / CHUNK_OVERLAP：文本切块参数
    - EMBEDDING_MODEL_NAME：向量模型名称
    - EMBEDDING_DEVICE：embedding 所使用的设备，如 cuda / cpu
    - QDRANT_HOST / QDRANT_PORT：远程 Qdrant 地址
    - QDRANT_COLLECTION_NAME：目标集合名称
    - QDRANT_RECREATE_COLLECTION：是否重建集合（true 会先删后建）
    """
    load_dotenv()

    cfg = {
        "DOCS_PATH": os.getenv("DOCS_PATH", "./data"),  # 这里保留数据集目录位置
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", "900")),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", "200")),
        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cuda"),

        "QDRANT_HOST": os.getenv("QDRANT_HOST", "172.18.216.71"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base"),
        "QDRANT_RECREATE_COLLECTION": os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true",
    }

    log(f"配置加载完成: {cfg}")
    return cfg


# ================= TEXT CLEAN =================

def clean_text_light(text: str) -> str:
    """
    轻量清洗文本：
    1. 合并英文断词
    2. 将单换行转为空格
    3. 压缩多余空白
    """
    if not text:
        return ""

    try:
        # 处理英文单词跨行断开，如 "exam-\nple" -> "example"
        text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)

        # 单换行变空格
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # 压缩空白
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    except Exception as e:
        log(f"文本清洗异常: {e}")
        return text.strip() if text else ""


# ================= TITLE / SECTION DETECTION =================
def detect_section_title(line: str) -> Optional[str]:
    """
    更严格、更保守的章节标题检测。
    目标：宁可漏判，也尽量避免把表格项/参数值误判为章节标题。
    """
    try:
        line = line.strip()
        if not line:
            return None

        # 1) 太长一般不是标题
        if len(line) > 80:
            return None

        # 2) 明显像参数/数值/规格描述的，直接排除
        # 如: 500 ps/div to 50 s/div
        #     5 ns
        #     200 mVpp from DC to 100 MHz
        #     5 GSa/s half channel interleaved, 2.5 GSa/s all channel
        if re.search(r"\d", line):
            # 含较多数字/单位，大概率不是章节标题
            unit_patterns = [
                r"\b(ns|us|ms|s|ps)\b",
                r"\b(hz|khz|mhz|ghz)\b",
                r"\b(v|mv|kv|vpp|mvpp)\b",
                r"\b(sa/s|gsa/s|msa/s|ksa/s)\b",
                r"\b(mpts|kpts|pts)\b",
                r"\b(bytes?)\b",
                r"\b(div|typical|maximum|minimum|channel|channels|pods)\b",
            ]
            lower = line.lower()
            if any(re.search(p, lower) for p in unit_patterns):
                return None

        # 3) 太短的全大写词，不当标题
        # 如 USB / XY / AM / FM / FSK
        if re.fullmatch(r"[A-Z][A-Z0-9:/\-]{0,8}", line):
            return None

        # 4) 冒号结尾通常是标签项，不是章节标题
        # 如 AM: / FM: / FSK:
        if re.search(r"[:：]$", line):
            return None

        # 5) 带太多标点或像一句话的，不当标题
        if re.search(r"[。！？；]$", line):
            return None
        if len(re.findall(r"[,:;()]", line)) > 2:
            return None

        lower = line.lower()

        # ===== 高置信英文标题 =====
        keywords = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "system model",
            "problem formulation",
            "method",
            "methods",
            "algorithm",
            "approach",
            "experiment",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "conclusions",
            "references",
            "materials",
            "materials and methods",
        ]

        # 6) 整行精确匹配章节名
        if lower in keywords:
            return line

        # 7) 编号章节标题，如 1 Introduction / 2.1 Methods
        for k in keywords:
            if re.fullmatch(rf"\d+(\.\d+)*\s+{re.escape(k)}", lower):
                return line

        # 8) 罗马数字章节标题，如 II. METHODS
        for k in keywords:
            if re.fullmatch(rf"[ivxlc]+\.\s+{re.escape(k)}", lower):
                return line

        # ===== 中文高置信标题 =====
        cn_keywords = [
            "摘要", "引言", "前言", "背景", "相关工作",
            "方法", "算法", "实验", "实验结果", "讨论",
            "结论", "参考文献"
        ]

        if line in cn_keywords:
            return line

        for k in cn_keywords:
            if re.fullmatch(rf"\d+(\.\d+)*\s*{re.escape(k)}", line):
                return line
            if re.fullmatch(rf"[一二三四五六七八九十]+\s*[、.．]\s*{re.escape(k)}", line):
                return line

    except Exception as e:
        log(f"章节标题识别异常: {e}")

    return None


# ================= HEADER FILTER =================

def is_header_footer(line: str) -> bool:
    """
    过滤常见页眉页脚。
    """
    try:
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
    except Exception:
        return False


# ================= PDF =================

def parse_pdf(file_path: Path) -> List[Document]:
    """
    解析 PDF：
    - 提取文档标题
    - 逐页解析
    - 不再识别章节标题
    """
    log(f"开始解析 PDF: {file_path.name}")

    docs = []

    try:
        pdf = fitz.open(str(file_path))
        doc_title = file_path.stem

        # 新增：相对路径（相对于 DOCS_PATH）
        # 注意：在这里看不到 cfg，只能记原始路径；真正相对路径在 load_documents 里统一处理
        for i, page in enumerate(pdf):
            text = page.get_text("text")

            lines = []
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue

                if is_header_footer(line):
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
                    "type": "pdf",
                    # 先占位，后面在 load_documents 再补真正的 rel_path
                    # "rel_path": 将在 load_documents 中统一设置
                }
            ))

        log(f"PDF解析完成: {file_path.name}, 页数={len(pdf)}, 生成doc数={len(docs)}")
        return docs

    except Exception as e:
        log(f"解析 PDF 失败: {file_path}, 错误: {e}")
        return docs


# ================= DOCX =================

def load_docx(file_path: Path) -> List[Document]:
    """
    加载 DOCX。
    """
    log(f"开始加载 DOCX: {file_path.name}")

    results = []

    try:
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()

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
                        "type": "docx",
                        # "rel_path": 将在 load_documents 中统一设置
                    }
                ))

        log(f"DOCX加载完成: {file_path.name}, doc数={len(results)}")
        return results

    except Exception as e:
        log(f"加载 DOCX 失败: {file_path}, 错误: {e}")
        return results


# ================= LOAD =================

def load_documents(path: str) -> List[Document]:
    """
    遍历目录，加载所有 PDF / DOCX，并为每个文档补充相对路径 metadata:
    - metadata["rel_path"] 例如: "设备操作指南/VLC小组/xxx.docx"
    """
    root = Path(path).resolve()
    docs: List[Document] = []

    if not root.exists():
        log(f"文档目录不存在: {root}")
        return docs

    for f in root.rglob("*"):
        try:
            if not f.is_file():
                continue

            if f.suffix.lower() not in [".pdf", ".docx"]:
                continue

            # 计算相对路径（统一使用正斜杠）
            rel_path = f.relative_to(root).as_posix()

            # 分别解析 pdf / docx
            if f.suffix.lower() == ".pdf":
                file_docs = parse_pdf(f)
            else:
                file_docs = load_docx(f)

            # 为每个 Document 补充 rel_path 到 metadata
            for d in file_docs:
                md = dict(d.metadata) if d.metadata else {}
                md.setdefault("source", str(f))
                md.setdefault("file_name", f.name)
                md.setdefault("doc_title", f.stem)
                md["rel_path"] = rel_path
                d.metadata = md
                docs.append(d)

        except Exception as e:
            log(f"加载文件失败: {f}, 错误: {e}")

    log(f"总文档数: {len(docs)}")
    return docs


# ================= CHUNK ENRICH =================

# ================= CHUNK ENRICH =================

def enrich_chunk_with_titles(doc: Document) -> Document:
    """
    将文档标题、页码、相对路径等结构信息注入 chunk 前缀。
    不再注入章节标题。
    """
    try:
        doc_title = doc.metadata.get("doc_title")
        page = doc.metadata.get("page")
        rel_path = doc.metadata.get("rel_path")  # 新增：相对路径

        prefix_parts = []
        if doc_title:
            prefix_parts.append(f"[文档标题] {doc_title}")
        if rel_path:
            prefix_parts.append(f"[路径] {rel_path}")
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
    except Exception as e:
        log(f"chunk 标题增强失败: {e}")
        return doc


# ================= CHUNK =================

def split_documents(docs: List[Document], chunk_size: int, overlap: int):
    """
    切块后给每个 chunk 注入标题/章节等结构信息。
    """
    log("开始文本切块")

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(docs)
        chunks = [enrich_chunk_with_titles(c) for c in chunks]

        log(f"切块完成: {len(chunks)}")
        return chunks

    except Exception as e:
        log(f"文本切块失败: {e}")
        return []


# ================= FILTER =================

def is_bad_chunk(text: str) -> bool:
    """
    判断是否为低质量 chunk。
    """
    if not text:
        return True

    text = text.strip()
    if len(text) < 80:
        return True

    valid_chars = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    valid_count = len(valid_chars)

    if valid_count < 30:
        return True

    digits = len(re.findall(r"\d", text))
    special_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))

    if valid_count > 0 and digits / valid_count > 1.2:
        return True

    if len(text) > 0 and special_chars / len(text) > 0.25:
        return True

    words = text.split()
    if len(words) >= 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    most_common_char_count = max(text.count(ch) for ch in set(text))
    if len(text) > 0 and most_common_char_count / len(text) > 0.35:
        return True

    return False


def analyze_bad_chunk_reason(text: str) -> str:
    """
    分析 chunk 被过滤的原因。
    """
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
    """
    过滤低质量 chunk。
    """
    log("开始过滤低质量chunk")

    results = []
    reason_stats = {}

    try:
        for d in docs:
            reason = analyze_bad_chunk_reason(d.page_content)
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

def build_embeddings(cfg):
    """
    初始化 embedding 模型。
    """
    log("初始化 Embedding 模型")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=cfg["EMBEDDING_MODEL_NAME"],
            model_kwargs={"device": cfg["EMBEDDING_DEVICE"]},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        log(f"初始化 Embedding 失败: {e}")
        raise


# ================= QDRANT =================

def get_embedding_dimension(embeddings) -> int:
    """
    通过对一段测试文本做 embedding，动态获取向量维度。
    这样可以避免手写维度出错。
    """
    try:
        test_vec = embeddings.embed_query("测试文本")
        dim = len(test_vec)
        log(f"检测到向量维度: {dim}")
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


def prepare_collection(client: QdrantClient, cfg, vector_size: int):
    """
    准备 Qdrant collection：
    - 若配置要求重建，则先删除再创建
    - 若不存在，则创建
    - 若已存在，则直接复用
    """
    collection_name = cfg["QDRANT_COLLECTION_NAME"]

    try:
        existing_collections = [c.name for c in client.get_collections().collections]

        if cfg["QDRANT_RECREATE_COLLECTION"] and collection_name in existing_collections:
            log(f"删除旧集合: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            existing_collections.remove(collection_name)

        if collection_name not in existing_collections:
            log(f"创建新集合: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            log(f"集合创建完成: {collection_name}")
        else:
            log(f"集合已存在，直接复用: {collection_name}")

    except Exception as e:
        log(f"准备 Qdrant collection 失败: {e}")
        raise


def save_to_qdrant(docs: List[Document], embeddings, cfg):
    """
    将文档写入 Qdrant。

    注意：
    - 与 FAISS 不同，Qdrant 是远程向量数据库
    - 这里将 chunk + metadata 一起写入 collection
    """
    log("开始写入 Qdrant")

    if not docs:
        log("没有可写入的文档 chunk，跳过入库")
        return

    try:
        client = create_qdrant_client(cfg)

        vector_size = get_embedding_dimension(embeddings)

        prepare_collection(client, cfg, vector_size)

        # 使用 LangChain 的 QdrantVectorStore 接口写入
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=cfg["QDRANT_COLLECTION_NAME"],
            embedding=embeddings,
        )

        vector_store.add_documents(documents=docs)

        log(f"Qdrant 写入完成，集合名: {cfg['QDRANT_COLLECTION_NAME']}")
        log(f"最终 chunk 数: {len(docs)}")

    except Exception as e:
        log(f"写入 Qdrant 失败: {e}")
        raise


# ================= MAIN =================

def main():
    """
    主流程：
    1. 加载配置
    2. 读取文档
    3. 文本切块
    4. 过滤低质量 chunk
    5. 初始化 embedding
    6. 写入 Qdrant
    """
    log("===== 开始执行 ingest =====")

    try:
        cfg = load_config()

        docs = load_documents(cfg["DOCS_PATH"])
        if not docs:
            log("未加载到任何文档，请检查 DOCS_PATH 是否正确")
            return

        chunks = split_documents(docs, cfg["CHUNK_SIZE"], cfg["CHUNK_OVERLAP"])
        if not chunks:
            log("切块结果为空，任务结束")
            return

        chunks = filter_chunks(chunks)
        if not chunks:
            log("过滤后没有可用 chunk，任务结束")
            return

        embeddings = build_embeddings(cfg)

        save_to_qdrant(chunks, embeddings, cfg)

        log("===== ingest 完成 =====")

    except Exception as e:
        log(f"ingest 执行失败: {e}")


if __name__ == "__main__":
    main()