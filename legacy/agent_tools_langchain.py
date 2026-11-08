"""【已归档】v3.4.2 之前的 LangChain @tool 工具层（历史快照，禁止 import）。

归档原因（3.4.3 重构）：当时存在两条工具执行路径——
1) 本文件的 `@tool` + `ContextVar` 用户名绑定（仅非流式循环使用）；
2) `agent_entry.ask_agent_stream` 里内联的 `run_tool`（生产唯一路径，直接调
   `list_catalog_entries` / `ask_rag`）。
第二条使 ContextVar 绑定在生产链路上从未生效，且工具元数据分散在 docstring、
提示词、分派三处易漂移。现统一为 `src/lab_rag/agent_tools.py` 的 `TOOL_SPECS`
单一数据源，handler 显式接收 username。

本文件不在 Python 包路径内，仅作历史参考。
"""

# ---- 以下为归档时的原始代码（src/lab_rag/tools.py 全文）----
#
# """ReAct Agent 请求级工具；全部复用主 RAG 运行时。"""
#
# from contextlib import contextmanager
# from contextvars import ContextVar
# from langchain_core.tools import tool
#
# from .rag_agent import build_file_context, list_catalog_entries
# from .rag_tool import rag_qa_tool
#
# _agent_username: ContextVar[str] = ContextVar("agent_username", default="legacy-agent")
#
#
# @contextmanager
# def agent_tool_context(username: str):
#     """为当前 Agent 请求绑定用户，防止并发请求共享历史。"""
#     token = _agent_username.set(username or "legacy-agent")
#     try:
#         yield
#     finally:
#         _agent_username.reset(token)
#
#
# @tool
# def rag_qa(query: str) -> str:
#     """基于实验室知识库检索并回答问题。"""
#     return rag_qa_tool(query, username=_agent_username.get())
#
#
# @tool
# def list_group_files(keyword: str) -> str:
#     """根据任意关键词列出相关目录和文件。"""
#     result = list_catalog_entries(keyword)
#     text = build_file_context(result)
#     return text or f"未找到与 {keyword} 相关的文件。"
#
#
# ---- 以下为归档时的 rag_tool.py 中两个已删除的死辅助（全仓无引用）----
#
# def clear_history(username: str = "legacy-agent") -> None:
#     clear_user_history(username)
#
#
# def rag_qa_tool(question: str, username: str = "legacy-agent") -> str:
#     try:
#         return ask_rag(question, username=username).get("answer", "")
#     except Exception as exc:
#         _logger.error(f"Agent RAG 工具调用失败: {exc!r}")
#         return "知识库服务暂时不可用，请稍后重试。"
