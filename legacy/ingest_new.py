import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
# 【修改 1】移除 RecursiveCharacterTextSplitter，引入 SemanticChunker
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    - DOCS_PATH：知识库原始文件目录（仅限 Markdown）
    - SEMANTIC_THRESHOLD_TYPE / SEMANTIC_THRESHOLD：语义分割的阈值类型和数值
    - EMBEDDING_MODEL_NAME：向量模型名称
    - EMBEDDING_DEVICE：embedding 所使用的设备，如 cuda / cpu
    - QDRANT_HOST / QDRANT_PORT：远程 Qdrant 地址
    - QDRANT_COLLECTION_NAME：目标集合名称
    - QDRANT_RECREATE_COLLECTION：是否重建集合（true 会先删后建）
    """
    load_dotenv()

    cfg = {
        "DOCS_PATH": os.getenv("DOCS_PATH", "./data"),

        # 【修改 2】替换原有的 CHUNK_SIZE/OVERLAP，改为语义分割专属配置
        # 可选类型: percentile (百分位数), standard_deviation (标准差), interquartile (四分位距)
        "SEMANTIC_THRESHOLD_TYPE": os.getenv("SEMANTIC_THRESHOLD_TYPE", "percentile"),
        # 例如 80 表示：句子间距离大于 80% 的其他句子间距离时，进行切块断点
        "SEMANTIC_THRESHOLD": float(os.getenv("SEMANTIC_THRESHOLD", "85.0")),

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
        text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        log(f"文本清洗异常: {e}")
        return text.strip() if text else ""


# ================= MARKDOWN LOADER =================

def load_md(file_path: Path) -> List[Document]:
    """加载 Markdown 文件。"""
    log(f"开始加载 Markdown: {file_path.name}")
    results = []
    try:
        content = file_path.read_text(encoding="utf-8")
        text = clean_text_light(content)
        if text:
            results.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "doc_title": file_path.stem,
                    "page": None,
                    "type": "md",
                }
            ))
        log(f"Markdown加载完成: {file_path.name}, doc数={len(results)}")
        return results
    except Exception as e:
        log(f"加载 Markdown 失败: {file_path}, 错误: {e}")
        return results


# ================= LOAD =================

def load_documents(path: str) -> List[Document]:
    """遍历目录，加载 .md 文件"""
    root = Path(path).resolve()
    docs: List[Document] = []
    if not root.exists():
        log(f"文档目录不存在: {root}")
        return docs

    for f in root.rglob("*"):
        try:
            if not f.is_file() or f.suffix.lower() != ".md":
                continue
            rel_path = f.relative_to(root).as_posix()
            file_docs = load_md(f)
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

def enrich_chunk_with_titles(doc: Document) -> Document:
    """将文档标题、相对路径等结构信息注入 chunk 前缀。"""
    try:
        doc_title = doc.metadata.get("doc_title")
        rel_path = doc.metadata.get("rel_path")
        prefix_parts = []
        if doc_title:
            prefix_parts.append(f"[文档标题] {doc_title}")
        if rel_path:
            prefix_parts.append(f"[路径] {rel_path}")

        prefix = "\n".join(prefix_parts)
        content = doc.page_content.strip()

        if prefix:
            content = prefix + "\n\n" + content

        return Document(page_content=content, metadata=doc.metadata)
    except Exception as e:
        log(f"chunk 标题增强失败: {e}")
        return doc


# ================= CHUNK (SEMANTIC) =================

# 【修改 3】修改切块函数，接收 embeddings 模型，使用 SemanticChunker
def split_documents(docs: List[Document], embeddings, cfg):
    """
    使用双重切分：先进行超大块粗切（防止单文件过大导致 OOM），再进行语义分割。
    逐个处理并及时清理显存。
    """
    log("开始语义文本切块 (Semantic Chunking)...")
    final_chunks = []

    try:
        # 1. 预先粗切分：防止单个 Markdown 文件高达几万字，直接撑爆 SemanticChunker
        rough_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=0,
            separators=["\n\n", "\n"]
        )
        rough_docs = rough_splitter.split_documents(docs)
        log(f"预处理：将原始文档粗切分为 {len(rough_docs)} 个片段以控制显存...")

        # 2. 初始化语义切分器
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=cfg["SEMANTIC_THRESHOLD_TYPE"],
            breakpoint_threshold_amount=cfg["SEMANTIC_THRESHOLD"]
        )

        # 3. 逐个片段进行语义切分
        for i, rough_doc in enumerate(rough_docs):
            try:
                # 对单个片段进行语义切片
                sub_chunks = semantic_splitter.split_documents([rough_doc])

                # 标题增强并加入最终结果
                enriched_chunks = [enrich_chunk_with_titles(c) for c in sub_chunks]
                final_chunks.extend(enriched_chunks)

            except Exception as sub_e:
                log(f"切分第 {i + 1} 个片段时出错 (可能仍是OOM或文本异常): {sub_e}")

            finally:
                # 【关键】每处理完一个片段，强制清理无用的 GPU 显存缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        log(f"语义切块完成: 最终切割出 {len(final_chunks)} 个块")
        return final_chunks

    except Exception as e:
        log(f"语义文本切块整体失败: {e}")
        return []


# ================= FILTER =================

def analyze_bad_chunk_reason(text: str) -> str:
    if not text or not text.strip():
        return "empty"
    text = text.strip()
    if len(text) < 40:  # 稍微放宽对短句的限制，因为语义切块可能会产生较短的一句话
        return "too_short"

    valid_chars = re.findall(r"[A-Za-z\u4e00-\u9fff]", text)
    valid_count = len(valid_chars)
    if valid_count < 15:  # 稍微放宽
        return "too_few_valid_chars"

    digits = len(re.findall(r"\d", text))
    special_chars = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))

    if valid_count > 0 and digits / valid_count > 1.2:
        return "too_many_digits"

    if len(text) > 0 and special_chars / len(text) > 0.3:
        return "too_many_special_chars"

    return "good"


def filter_chunks(docs: List[Document]):
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
    log("初始化 Embedding 模型")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=cfg["EMBEDDING_MODEL_NAME"],
            model_kwargs={"device": cfg["EMBEDDING_DEVICE"]},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 4  # 【新增】强行降低批量大小，防止 OOM。如果显存还是不够，改成 2 甚至 1
            },
        )
        return embeddings
    except Exception as e:
        log(f"初始化 Embedding 失败: {e}")
        raise


# ================= QDRANT =================

def get_embedding_dimension(embeddings) -> int:
    try:
        test_vec = embeddings.embed_query("测试")
        dim = len(test_vec)
        log(f"检测到向量维度: {dim}")
        return dim
    except Exception as e:
        log(f"获取向量维度失败: {e}")
        raise


def create_qdrant_client(cfg):
    try:
        client = QdrantClient(host=cfg["QDRANT_HOST"], port=cfg["QDRANT_PORT"], timeout=30)
        log(f"Qdrant 客户端创建成功")
        return client
    except Exception as e:
        log(f"创建 Qdrant 客户端失败: {e}")
        raise


def prepare_collection(client: QdrantClient, cfg, vector_size: int):
    collection_name = cfg["QDRANT_COLLECTION_NAME"]
    try:
        existing = [c.name for c in client.get_collections().collections]
        if cfg["QDRANT_RECREATE_COLLECTION"] and collection_name in existing:
            log(f"删除旧集合: {collection_name}")
            client.delete_collection(collection_name=collection_name)
            existing.remove(collection_name)

        if collection_name not in existing:
            log(f"创建新集合: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            log(f"集合已存在，复用: {collection_name}")
    except Exception as e:
        log(f"准备 Qdrant collection 失败: {e}")
        raise


def save_to_qdrant(docs: List[Document], embeddings, cfg):
    log("开始写入 Qdrant")
    if not docs: return
    try:
        client = create_qdrant_client(cfg)
        vector_size = get_embedding_dimension(embeddings)
        prepare_collection(client, cfg, vector_size)
        vector_store = QdrantVectorStore(
            client=client, collection_name=cfg["QDRANT_COLLECTION_NAME"], embedding=embeddings,
        )
        vector_store.add_documents(documents=docs)
        log(f"Qdrant 写入完成")
    except Exception as e:
        log(f"写入 Qdrant 失败: {e}")
        raise


# ================= MAIN =================

def main():
    log("===== 开始执行 ingest (Semantic Splitting) =====")
    try:
        cfg = load_config()

        docs = load_documents(cfg["DOCS_PATH"])
        if not docs:
            log("未加载到任何文档")
            return

        # 【修改 4】语义分割依赖 Embedding 模型，所以需要提前初始化 Embedding 模型
        embeddings = build_embeddings(cfg)

        # 传入 embeddings 和配置进行语义切块
        chunks = split_documents(docs, embeddings, cfg)
        if not chunks:
            log("切块结果为空，任务结束")
            return

        chunks = filter_chunks(chunks)
        if not chunks:
            log("过滤后没有可用 chunk，任务结束")
            return

        save_to_qdrant(chunks, embeddings, cfg)
        log("===== ingest 完成 =====")

    except Exception as e:
        log(f"ingest 执行失败: {e}")


if __name__ == "__main__":
    main()