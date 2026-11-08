"""Agent 的轻量兼容 RAG 适配层。

历史实现曾独立加载 Embedding、Reranker 和全局聊天历史。当前所有调用统一复用
``rag_agent`` 的运行时和用户级历史，保留旧函数名以兼容外部调用。
"""

from typing import Dict, Generator, Optional

from .rag_agent import ask_stream


def ask_rag_stream(question: str, username: str = "legacy-agent",
                   route_hint: Optional[dict] = None) -> Generator[Dict, None, None]:
    # Agent 内部构造的问题不应写入用户的正式对话历史。
    yield from ask_stream(question, username=username, persist_history=False,
                          route_hint=route_hint)


def ask_rag(question: str, username: str = "legacy-agent",
            route_hint: Optional[dict] = None) -> Dict:
    answer = ""
    final = {}
    for item in ask_rag_stream(question, username=username, route_hint=route_hint):
        if item.get("type") == "chunk":
            answer += item.get("content", "")
        elif item.get("type") == "final":
            final = dict(item)
            answer = item.get("content") or answer
        elif item.get("type") == "error":
            raise RuntimeError("RAG 服务暂时不可用")
    final.setdefault("answer", answer)
    final.setdefault("content", answer)
    final.setdefault("history", [])
    return final

