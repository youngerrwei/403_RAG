# rag_core.py

import os
import re
import json
import traceback
from typing import List, Dict, Optional

from dotenv import load_dotenv
from difflib import SequenceMatcher

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# reranker
from sentence_transformers import CrossEncoder


# ─── 全局状态 ────────────────────────────────────────────────
chat_history: List[Dict[str, str]] = []
_runtime: Optional[dict] = None


# ─── 工具函数 ────────────────────────────────────────────────

def print_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ─── Token 估算与历史截断 ─────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    估算 token 数量。
    优先使用 tiktoken，没有则退化为简单近似。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, int(len(text) * 0.6))


def estimate_history_tokens(history: List[Dict[str, str]]) -> int:
    total = 0
    for msg in history:
        total += 4 + estimate_tokens(msg.get("content", ""))
    return total


def truncate_history(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """
    从后往前按轮保留历史对话，防止上下文过长。
    """
    if max_tokens <= 0:
        return []

    if estimate_history_tokens(history) <= max_tokens:
        return history

    pairs = []
    i = len(history) - 1
    while i >= 1:
        pair = [history[i - 1], history[i]]
        pairs.append(pair)
        i -= 2

    if i == 0:
        pairs.append([history[0]])

    kept = []
    accumulated_tokens = 0

    for pair in pairs:
        pair_tokens = sum(4 + estimate_tokens(m.get("content", "")) for m in pair)
        if accumulated_tokens + pair_tokens > max_tokens:
            break
        kept = pair + kept
        accumulated_tokens += pair_tokens

    dropped = len(history) - len(kept)
    if dropped > 0:
        print(f"[INFO] 历史记忆截断：丢弃最早 {dropped} 条消息，保留 {len(kept)} 条。")

    return kept


# ─── 配置加载 ────────────────────────────────────────────────

def load_config() -> dict:
    """
    加载配置。
    """
    load_dotenv()

    config = {
        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "VLLM_API_KEY": os.getenv("VLLM_API_KEY", ""),
        "VLLM_MODEL_NAME": os.getenv("VLLM_MODEL_NAME", ""),

        "EMBEDDING_MODEL_NAME": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        "EMBEDDING_DEVICE": os.getenv("EMBEDDING_DEVICE", "cpu"),

        # Qdrant 远程配置，替代 FAISS 本地路径
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "172.18.216.71"),
        "QDRANT_PORT": int(os.getenv("QDRANT_PORT", "6333")),
        "QDRANT_COLLECTION_NAME": os.getenv("QDRANT_COLLECTION_NAME", "lab_knowledge_base"),

        # 初始召回数（先尽量多取一些，后面再重排）
        "INITIAL_RETRIEVAL_K": int(os.getenv("INITIAL_RETRIEVAL_K", "20")),
        # 最终送给 LLM 的 topK
        "FINAL_TOP_K": int(os.getenv("FINAL_TOP_K", "5")),

        "MAX_LEN": int(os.getenv("MAX_LEN", "2048")),

        # reranker
        "RERANKER_MODEL_NAME": os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "RERANKER_DEVICE": os.getenv("RERANKER_DEVICE", "cpu"),
    }

    return config


# ─── Embedding / VectorStore / LLM / Reranker ───────────────

def build_embeddings(model_name: str, device: str):
    """
    初始化 embedding 模型。
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        print(f"[ERROR] Embedding 初始化失败: {e}")
        traceback.print_exc()
        raise


def load_vector_store(qdrant_host: str, qdrant_port: int, collection_name: str, embeddings):
    """
    连接远程 Qdrant 向量库。
    保持返回对象具备 similarity_search 接口，从而尽量不改后续逻辑。
    """
    try:
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            timeout=30
        )

        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            raise ValueError(f"Qdrant collection 不存在: {collection_name}")

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return vectorstore

    except Exception as e:
        print(f"[ERROR] Qdrant 向量库加载失败: {e}")
        traceback.print_exc()
        raise


def build_llm(base_url: str, api_key: str, model_name: str):
    """
    初始化对话模型（兼容本地 vLLM OpenAI 接口）。
    """
    try:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.1,
            max_tokens=1024,
            streaming=True,
            timeout=120,
        )
        return llm

    except Exception as e:
        print(f"[ERROR] LLM 初始化失败: {e}")
        traceback.print_exc()
        raise


def build_reranker(model_name: str, device: str):
    """
    初始化重排模型。
    推荐使用：BAAI/bge-reranker-v2-m3
    """
    try:
        reranker = CrossEncoder(model_name, device=device)
        return reranker
    except Exception as e:
        print(f"[ERROR] Reranker 初始化失败: {e}")
        traceback.print_exc()
        raise


# ─── 历史 / Prompt 工具 ──────────────────────────────────────

def format_chat_history(history: List[Dict[str, str]]) -> str:
    """
    将对话历史格式化为文本，便于问题改写与回答生成。
    """
    if not history:
        return "无历史对话。"

    lines = []
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def safe_parse_json(text: str) -> Optional[dict]:
    """
    尝试从 LLM 输出中提取 JSON。
    兼容模型输出 ```json ... ``` 的情况。
    """
    if not text:
        return None

    text = text.strip()

    # 去掉 markdown code block
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        return None


def rewrite_question_structured(llm, question: str, history: List[Dict[str, str]]) -> dict:
    """
    基于历史对话，将当前问题改写为：
    - standalone_question: 完整独立问题
    - keywords: 检索关键词
    - expanded_queries: 语义一致的不同表达

    若失败则退化到原问题。
    """
    try:
        if not history:
            return {
                "standalone_question": question,
                "keywords": [],
                "expanded_queries": [question]
            }

        prompt = ChatPromptTemplate.from_template("""
你是实验室知识库的“查询重写助手”，只负责把用户问题改写成更适合向量检索和重排的查询。
请根据历史对话，将当前问题改写为更适合知识库检索的结构化结果。

要求：
1. 不要回答问题，不要解释。
2. 如果当前问题已经完整清晰，保持原意，避免过度改写。
3. 如果存在代词、指代、省略、上文依赖，请补全为独立问题。
4. 保留专业术语、设备名、方法名、缩写、型号、参数名。
5. 如果历史对话对当前问题没有帮助，不要强行引用历史。
6. 输出 JSON，格式如下：
{{
  "standalone_question": "改写后的独立完整问题",
  "keywords": ["关键词1", "关键词2", "关键词3"],
  "expanded_queries": ["检索句1", "检索句2", "检索句3"]
}}

历史对话：
{chat_history}

当前问题：
{question}
""".strip())

        chain = prompt | llm
        result = chain.invoke({
            "chat_history": format_chat_history(history),
            "question": question
        })

        print_title("5. 查询重写原始输出")
        print(result.content)

        raw = result.content.strip()

        data = safe_parse_json(raw)

        if not data:
            return {
                "standalone_question": question,
                "keywords": [],
                "expanded_queries": [question]
            }

        standalone_question = data.get("standalone_question", question)
        keywords = data.get("keywords", [])
        expanded_queries = data.get("expanded_queries", [])

        if not isinstance(keywords, list):
            keywords = []
        if not isinstance(expanded_queries, list):
            expanded_queries = []

        queries = [standalone_question] + expanded_queries
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        queries = list(dict.fromkeys(queries))

        return {
            "standalone_question": standalone_question,
            "keywords": keywords[:6],
            "expanded_queries": queries[:3]
        }

    except Exception as e:
        print(f"[WARN] 问题改写失败，将使用原问题: {e}")
        return {
            "standalone_question": question,
            "keywords": [],
            "expanded_queries": [question]
        }


# ─── 检索 / 去重 / 重排 / 上下文构建 ─────────────────────────
def strip_structured_prefix(text: str) -> str:
    """
    去掉 chunk 中为了检索/上下文注入的结构前缀：
    [文档标题] xxx [页码] xxx
    """
    if not text:
        return ""

    cleaned = str(text).strip()

    # 兼容 prefix 被压成一行的情况
    # 例如：
    # [文档标题] xxx [页码] 14 正文...
    cleaned = re.sub(
        r"^\s*\[文档标题\]\s*.*?\s*\[页码\]\s*.*?(?=\s+[\u4e00-\u9fffA-Za-z0-9(（【\[])",
        "",
        cleaned,
        flags=re.S
    ).strip()

    # 如果上面的规则没完全命中，再做一次行首标签清洗
    cleaned = re.sub(r"^\s*\[文档标题\][^\n]*", "", cleaned).strip()
    cleaned = re.sub(r"^\s*\[页码\][^\n]*", "", cleaned).strip()

    # 清掉开头多余标点/空白
    cleaned = re.sub(r"^[\s:：\-—]+", "", cleaned).strip()

    return cleaned


def clean_retrieval_display_text(text: str) -> str:
    """
    清洗前端展示用的检索正文：
    1. 压缩重复
    2. 去掉结构前缀（文档标题/章节标题/页码）
    3. 规范空白
    """
    if not text:
        return ""

    text = compress_repeated_text(text)
    text = strip_structured_prefix(text)
    text = normalize_retrieval_text(text)
    return text


def normalize_retrieval_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compress_repeated_text(text: str) -> str:
    """
    压缩 A+A 型重复文本。
    """
    norm = normalize_retrieval_text(text)
    if not norm:
        return ""

    words = norm.split()
    n = len(words)
    if n < 20:
        return norm

    half = n // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:half * 2])

    if first and second:
        sim = SequenceMatcher(None, first, second).ratio()
        if sim >= 0.96:
            return first

    return norm


def dedup_retrieved_docs(docs, similarity_threshold: float = 0.95):
    """
    检索结果去重：
    1. 压缩单条内部重复
    2. 去除彼此高度相似的结果
    """
    unique_docs = []
    seen_exact = set()

    for doc in docs:
        file_name = doc.metadata.get("file_name", "")
        page = doc.metadata.get("page", "")
        content = compress_repeated_text(doc.page_content or "")
        doc.page_content = content

        exact_key = (file_name, page, content[:500])
        if exact_key in seen_exact:
            continue

        duplicated = False
        for kept in unique_docs:
            kept_content = normalize_retrieval_text(kept.page_content or "")
            curr_content = normalize_retrieval_text(content)

            sim = SequenceMatcher(None, kept_content[:800], curr_content[:800]).ratio()
            if sim >= similarity_threshold:
                duplicated = True
                break

        if duplicated:
            continue

        seen_exact.add(exact_key)
        unique_docs.append(doc)

    return unique_docs


def retrieve_docs_multi_query(vectorstore, queries: List[str], top_k_each: int = 8):
    """
    多 query 检索。
    将多个查询分别召回，再统一合并。
    """
    all_docs = []
    for q in queries:
        try:
            docs = vectorstore.similarity_search(q, k=top_k_each)
            all_docs.extend(docs)
        except Exception as e:
            print(f"[WARN] 某个查询检索失败，已跳过。query={q}, err={e}")
    return all_docs


def rerank_docs(reranker, query: str, docs, top_k: int = 5):
    """
    使用 CrossEncoder reranker 对召回结果重排。
    """
    if not docs:
        return []

    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked

    except Exception as e:
        print(f"[WARN] 重排失败，将退化为直接截断召回结果: {e}")
        return docs[:top_k]


def build_context(docs) -> str:
    """
    将最终检索片段拼装成上下文。
    对 LLM 保留结构信息，但正文使用清洗后的纯文本，避免重复头信息干扰。
    """
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        doc_title = doc.metadata.get("doc_title", "未知标题")
        page = doc.metadata.get("page", "未知页码")
        rel_path = doc.metadata.get("rel_path", "未知路径")

        content = clean_retrieval_display_text(doc.page_content or "")

        context_parts.append(
            f"[片段{i}]\n"
            f"[文档标题] {doc_title}\n"
            f"\n[路径] {rel_path}\n"
            f"[页码] {page}\n\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)


# ─── 生成回答 ────────────────────────────────────────────────

def answer_with_context_stream(llm, question: str, context: str, history: List[Dict[str, str]]):
    """
    基于检索上下文生成最终回答。
    """
    try:
        prompt = ChatPromptTemplate.from_template("""
你是实验室内部知识库问答助手。

请严格遵守以下规则：
1. 必须优先依据“检索到的上下文”回答问题。
2. 如果上下文没有直接下定义，但存在高度相关内容，可以基于上下文进行概括总结，并**明确表述**为“根据知识库内容可概括为”。
3. 如果知识库中没有足够信息，请**明确说明**：“知识库中未找到足够相关内容”。
4. 可以基于你的通用知识补充，但必须**明确标注**：“以下补充非知识库内容”。
5. 回答要准确、专业、清晰、简洁。
6. 回答最后提供最终采信的来源文档**完整主标题**。
7. 涉及数学符号或公式时，必须使用**标准 LaTeX 格式**输出：$公式$。

历史对话：
{chat_history}

检索到的上下文：
{context}

用户问题：
{question}
""".strip())

        chain = prompt | llm
        result = chain.stream({
            "chat_history": format_chat_history(history),
            "context": context if context.strip() else "无可用上下文",
            "question": question
        })

        return result

    except Exception as e:
        print(f"[ERROR] 生成回答失败: {e}")
        traceback.print_exc()
        raise


# ─── 运行时管理 ──────────────────────────────────────────────

def build_runtime():
    """
    初始化运行时组件：
    - embedding
    - vector store
    - llm
    - reranker
    """
    config = load_config()

    print_title("1. 初始化 Embedding")
    embeddings = build_embeddings(
        model_name=config["EMBEDDING_MODEL_NAME"],
        device=config["EMBEDDING_DEVICE"]
    )

    print_title("2. 连接远程 Qdrant 向量库")
    vectorstore = load_vector_store(
        qdrant_host=config["QDRANT_HOST"],
        qdrant_port=config["QDRANT_PORT"],
        collection_name=config["QDRANT_COLLECTION_NAME"],
        embeddings=embeddings
    )

    print_title("3. 初始化本地 vLLM")
    llm = build_llm(
        base_url=config["VLLM_BASE_URL"],
        api_key=config["VLLM_API_KEY"],
        model_name=config["VLLM_MODEL_NAME"]
    )

    print_title("4. 初始化 Reranker")
    reranker = build_reranker(
        model_name=config["RERANKER_MODEL_NAME"],
        device=config["RERANKER_DEVICE"]
    )

    print("[OK] RAG 运行时初始化成功。")
    return {
        "config": config,
        "vectorstore": vectorstore,
        "llm": llm,
        "reranker": reranker
    }


def get_runtime():
    """
    获取全局运行时；若未初始化则自动初始化。
    """
    global _runtime
    if _runtime is None:
        _runtime = build_runtime()
    return _runtime


def init_runtime():
    """
    可手动预热运行时。
    """
    get_runtime()


# ─── 核心问答入口 ─────────────────────────────────────────────

def ask_rag_stream(question: str):
    """
    执行一次 RAG 问答，流式返回：
    - metadata：检索结果
    - chunk：回答增量
    - done：结束
    """
    global chat_history

    try:
        runtime = get_runtime()
        config = runtime["config"]
        vectorstore = runtime["vectorstore"]
        llm = runtime["llm"]
        reranker = runtime["reranker"]

        # 先截断历史，控制上下文长度
        chat_history = truncate_history(chat_history, config["MAX_LEN"])

        # 1) 查询改写
        rewrite_result = rewrite_question_structured(
            llm=llm,
            question=question,
            history=chat_history
        )

        standalone_question = rewrite_result["standalone_question"]
        keywords = rewrite_result["keywords"]
        expanded_queries = rewrite_result["expanded_queries"]

        # 2) 组装多路检索 query
        queries = [question, standalone_question]
        queries.extend(expanded_queries)

        if keywords:
            keyword_query = " ".join([str(k).strip() for k in keywords if str(k).strip()])
            if keyword_query:
                queries.append(keyword_query)

        # 去重
        queries = [q.strip() for q in queries if q and q.strip()]
        queries = list(dict.fromkeys(queries))

        print_title("6. 检索查询")
        for idx, q in enumerate(queries, start=1):
            print(f"{idx}. {q}")

        # 3) 多 query 扩大召回
        top_k_each = max(4, config["INITIAL_RETRIEVAL_K"] // max(1, len(queries)))
        recalled_docs = retrieve_docs_multi_query(
            vectorstore=vectorstore,
            queries=queries,
            top_k_each=top_k_each
        )

        # 4) 去重
        recalled_docs = dedup_retrieved_docs(recalled_docs)

        print_title("7. 初始召回结果数量")
        print(f"去重后共召回 {len(recalled_docs)} 个片段。")

        # 5) 重排
        final_docs = rerank_docs(
            reranker=reranker,
            query=standalone_question,
            docs=recalled_docs,
            top_k=config["FINAL_TOP_K"]
        )

        print_title("8. 重排后结果数量")
        print(f"最终保留 {len(final_docs)} 个片段。")

        retrieval_summaries = []
        for i, doc in enumerate(final_docs, start=1):
            file_name = doc.metadata.get("file_name", "未知文件")
            page = doc.metadata.get("page", "未知页码")
            doc_title = doc.metadata.get("doc_title", "未知标题")
            rel_path = doc.metadata.get("rel_path", "未知路径")
            rerank_score = doc.metadata.get("rerank_score")

            raw_content = doc.page_content.strip()
            display_content = clean_retrieval_display_text(raw_content)
            preview = display_content[:220]

            # print(f"\n===== final doc {i} =====")
            # print(doc.metadata)
            # print(raw_content[:2000])
            # print("----- cleaned display content -----")
            # print(display_content[:1000])

            retrieval_summaries.append({
                "index": i,
                "file_name": file_name,
                "doc_title": doc_title,
                "page": page,
                "rel_path": rel_path,
                "rerank_score": rerank_score,
                "preview": preview,
                "content": display_content
            })

        context = build_context(final_docs)

        # 第一步：先把检索结果给前端
        yield {
            "type": "metadata",
            "rewritten_question": standalone_question,
            "keywords": keywords,
            "queries": queries,
            "retrievals": retrieval_summaries
        }

        # 第二步：流式生成答案
        stream = answer_with_context_stream(
            llm=llm,
            question=question,
            context=context,
            history=chat_history
        )

        full_answer = ""
        for chunk in stream:
            content = chunk.content
            if content:
                full_answer += content
                yield {
                    "type": "chunk",
                    "content": content
                }

        # 第三步：更新历史
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": full_answer})

        # 第四步：结束
        yield {
            "type": "done",
            "history": chat_history
        }

    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "content": str(e)}


def ask_rag(question: str):
    """
    非流式包装，便于本地测试。
    """
    full_answer = ""
    retrievals = []
    rewritten_question = question
    keywords = []
    queries = []

    for item in ask_rag_stream(question):
        if item["type"] == "metadata":
            retrievals = item["retrievals"]
            rewritten_question = item.get("rewritten_question", question)
            keywords = item.get("keywords", [])
            queries = item.get("queries", [])
        elif item["type"] == "chunk":
            full_answer += item["content"]
            print(item["content"], end="", flush=True)

    print()
    return {
        "answer": full_answer,
        "history": chat_history,
        "retrievals": retrievals,
        "rewritten_question": rewritten_question,
        "keywords": keywords,
        "queries": queries
    }


def clear_history() -> None:
    """
    清空历史对话。
    """
    global chat_history
    chat_history = []
    print("[INFO] 历史对话已清空。")


# ─── 本地测试 ────────────────────────────────────────────────

if __name__ == "__main__":
    print_title("RAG Core 本地测试")

    try:
        init_runtime()
        question = input("请输入测试问题：").strip()

        result = ask_rag(question)

        print_title("改写后的问题")
        print(result["rewritten_question"])

        print_title("关键词")
        print(result["keywords"])

        print_title("实际检索 queries")
        for q in result["queries"]:
            print("-", q)

        print_title("检索片段预览")
        for item in result["retrievals"]:
            print(item)

        print_title("模型回答")
        print(result["answer"])

    except Exception as e:
        print(f"[FATAL] 本地测试失败: {e}")
        traceback.print_exc()


def rag_qa_tool(question: str) -> str:
    """
    供 Agent 调用的简化 RAG 问答工具：
    - 只返回最终 answer 文本，不返回检索细节。
    - 内部复用 ask_rag（非流式版本）。
    """
    try:
        result = ask_rag(question)
        return result["answer"]
    except Exception as e:
        traceback.print_exc()
        return f"[RAG工具调用失败]: {e}"