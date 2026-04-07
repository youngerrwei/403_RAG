# tools.py

import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_tool import rag_qa_tool

load_dotenv()

# 简单的全局缓存，避免每次工具调用都重新建模型和 client
_VECTORSTORE_CACHE: Optional[QdrantVectorStore] = None


def _load_vectorstore_for_tool() -> QdrantVectorStore:
    """
    给工具用的简化版 vectorstore，和 rag_core 保持一致配置。
    使用简单缓存，避免重复初始化。
    """
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE

    qdrant_host = os.getenv("QDRANT_HOST", "172.18.216.71")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base")

    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        timeout=30
    )

    vs = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    _VECTORSTORE_CACHE = vs
    return vs


def _search_files_by_keyword(keyword: str, limit: int = 100) -> List[Dict]:
    vectorstore = _load_vectorstore_for_tool()
    client = vectorstore.client
    collection_name = vectorstore.collection_name

    # 1. 使用 Qdrant 的 scroll 接口直接按元数据过滤 (更准确)
    # 这会查找元数据中包含 keyword 的记录，不涉及向量计算
    from qdrant_client.http import models as rest

    # 构造过滤条件：在 rel_path, file_name, doc_title 中模糊匹配 keyword
    search_filter = rest.Filter(
        should=[
            rest.FieldCondition(key="metadata.rel_path", match=rest.MatchText(text=keyword)),
            rest.FieldCondition(key="metadata.file_name", match=rest.MatchText(text=keyword)),
            rest.FieldCondition(key="metadata.doc_title", match=rest.MatchText(text=keyword)),
        ]
    )

    # 使用 scroll 获取所有匹配的记录（不受向量相似度排序限制）
    res, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=search_filter,
        limit=limit,
        with_payload=True
    )

    files: Dict[str, Dict] = {}

    # 如果 scroll 没找到足够的内容，再补充向量检索的结果 (保持兼容性)
    hits = res
    if len(hits) < 10:
        vector_hits = vectorstore.similarity_search(keyword, k=limit)
        # 将 vector_hits 转为统一格式合并处理...
        # (此处省略合并逻辑，优先处理 scroll 的结果)

    for hit in hits:
        # hit 可能是 ScoredPoint 或 Record，取决于调用方式
        md = hit.payload.get("metadata", hit.payload) if hasattr(hit, 'payload') else {}

        file_name = md.get("file_name", "")
        doc_title = md.get("doc_title", "")
        rel_path = md.get("rel_path", "")

        key = rel_path or f"{file_name}|{doc_title}"
        if key not in files:
            files[key] = {
                "file_name": file_name,
                "doc_title": doc_title,
                "rel_path": rel_path,
                "count": 0,
            }
        files[key]["count"] += 1

    return list(files.values())


@tool
def rag_qa(query: str) -> str:
    """
    通用问答工具：基于实验室知识库进行 RAG 检索并回答问题。
    适用于任何需要结合知识库内容的问题。
    使用时请给出尽量具体的问题。
    """
    return rag_qa_tool(query)


@tool
def list_group_files(keyword: str) -> str:
    """
    通用“文件/目录搜索”工具：
    根据任意关键词（例如 'VLC小组'、'LETO-3'、'设备操作指南'）列出相关的文件信息。

    返回一个可读的多行字符串，每行包含：
    - file_name: 文件名
    - doc_title: 文档标题
    - rel_path: 相对路径（可帮助判断目录/小组/项目归属）
    - chunks: 该文件在向量库中对应的 chunk 数量

    当你需要了解：
    - 某个小组有哪些文档/设备说明书（如 “VLC小组”）
    - 某个设备有哪些相关文档（如 “LETO-3 空间光调制器”）
    - 某个项目/实验相关的所有文档
    等场景时，请优先调用本工具。
    """
    files = _search_files_by_keyword(keyword)
    if not files:
        return f"未在知识库中找到与 {keyword} 相关的文件（基于近似检索）。"

    lines = []
    for f in files:
        line = (
            f"- file_name: {f['file_name']}, "
            f"doc_title: {f['doc_title']}, "
            f"rel_path: {f['rel_path']}, "
            f"chunks: {f['count']}"
        )
        lines.append(line)

    return "\n".join(lines)