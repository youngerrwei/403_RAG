"""ReAct Agent 请求级工具；全部复用主 RAG 运行时。"""

from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.tools import tool

from rag_agent import build_file_context, list_catalog_entries
from rag_tool import rag_qa_tool

_agent_username: ContextVar[str] = ContextVar("agent_username", default="legacy-agent")


@contextmanager
def agent_tool_context(username: str):
    """为当前 Agent 请求绑定用户，防止并发请求共享历史。"""
    token = _agent_username.set(username or "legacy-agent")
    try:
        yield
    finally:
        _agent_username.reset(token)


@tool
def rag_qa(query: str) -> str:
    """基于实验室知识库检索并回答问题。"""
    return rag_qa_tool(query, username=_agent_username.get())


@tool
def list_group_files(keyword: str) -> str:
    """根据任意关键词列出相关目录和文件。"""
    result = list_catalog_entries(keyword)
    text = build_file_context(result)
    return text or f"未找到与 {keyword} 相关的文件。"
