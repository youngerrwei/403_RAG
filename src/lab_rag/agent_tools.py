"""Agent 工具注册表：提示词、校验、分派共用的单一数据源。

此前工具元数据分散在三处（LangChain @tool 的 docstring、提示词里的自然语言描述、
agent_entry 里的硬编码分派），容易漂移。这里集中定义 ToolSpec，由它：
- 生成 REACT_SYSTEM_PROMPT 的工具段落；
- 校验模型输出的 Action 工具名；
- 通过 dispatch_tool 提供唯一执行路径（返回结构化结果，含耗时与可追溯 id）。

handler 统一显式接收 username，不再依赖 ContextVar 的隐式绑定。
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rag_agent import build_file_context, list_catalog_entries
from .rag_tool import ask_rag


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_name: str
    description: str
    bullets: Tuple[str, ...]
    role: str  # "catalog"（目录定位）或 "content"（正文检索），供编排器判断复合意图
    emits_catalog: bool
    handler: Callable[[str, str, Optional[dict]], Tuple[str, Any]]


def _handle_rag_qa(arg: str, username: str,
                   route_hint: Optional[dict]) -> Tuple[str, Any]:
    rag_result = ask_rag(arg, username=username, route_hint=route_hint)
    return rag_result.get("answer", ""), rag_result.get("file_result")


def _handle_list_group_files(arg: str, username: str,
                             route_hint: Optional[dict]) -> Tuple[str, Any]:
    result = list_catalog_entries(arg)
    text = build_file_context(result)
    return (text or f"未找到与 {arg} 相关的文件。"), result


TOOL_SPECS: Tuple[ToolSpec, ...] = (
    ToolSpec(
        name="rag_qa",
        arg_name="query",
        description="通用问答工具：基于知识库正文进行检索并回答问题。",
        bullets=(
            "用于具体技术问题、操作步骤、名词定义、原理说明、格式/要求/规范/步骤/方法、内容总结与对比。",
            "可在拿到文件清单后，针对其中具体文件或设备构造更聚焦的问题再次调用。",
        ),
        role="content",
        emits_catalog=True,
        handler=_handle_rag_qa,
    ),
    ToolSpec(
        name="list_group_files",
        arg_name="keyword",
        description="文件/目录搜索工具：按关键词列出相关文件信息。",
        bullets=(
            "关键词可以是小组名、设备名或型号、文档类型、日期、章节名、`Manuals` 等子目录名。",
            "返回 file_name、doc_title、rel_path、chunks 数量，用于了解有哪些文档及其所在目录层级。",
            "仅用于用户明确要求列出或定位文件、目录、路径、设备清单的场景。",
        ),
        role="catalog",
        emits_catalog=True,
        handler=_handle_list_group_files,
    ),
)

TOOL_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
CONTENT_TOOL_NAME: str = next(spec.name for spec in TOOL_SPECS if spec.role == "content")


def tool_role(name: str) -> str:
    """供编排器按角色判断，避免把工具名硬编码进控制流。"""
    spec = TOOL_BY_NAME.get(name)
    return spec.role if spec else "unknown"


def tool_names() -> List[str]:
    return [spec.name for spec in TOOL_SPECS]


def render_tool_prompt_section() -> str:
    """由注册表生成提示词的工具段落，保证「提示词描述」与「可执行工具」不漂移。"""
    lines = ["## 你可以使用以下工具（每轮只能调用一个）", ""]
    for index, spec in enumerate(TOOL_SPECS, 1):
        lines.append(f"{index}) {spec.name}({spec.arg_name}: str) -> str")
        lines.append(f"   - {spec.description}")
        for bullet in spec.bullets:
            lines.append(f"   - {bullet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def dispatch_tool(name: str, arg: str, username: str,
                  route_hint: Optional[dict] = None,
                  tool_call_id: str = "") -> Dict[str, Any]:
    """Agent 工具调用的唯一执行路径。

    返回结构化结果：observation（完整原文）、catalog_result、ok、elapsed_s、
    tool_call_id。任何异常都降级为「无可用观察」，不向模型抛栈。
    """
    spec = TOOL_BY_NAME.get(name)
    if spec is None:
        return {
            "ok": False,
            "observation": (f"[工具错误] 未找到名为 {name} 的工具。"
                            f"可用工具：{', '.join(tool_names())}"),
            "catalog_result": None,
            "elapsed_s": 0.0,
            "tool_call_id": tool_call_id,
        }

    started = time.perf_counter()
    try:
        observation, catalog_result = spec.handler(arg, username, route_hint)
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "observation": str(observation or ""),
            "catalog_result": catalog_result,
            "elapsed_s": round(elapsed, 3),
            "tool_call_id": tool_call_id,
        }
    except Exception as exc:  # 工具失败不终止整个 Agent 轮次
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "observation": "[工具执行异常] 服务暂时不可用，请稍后重试。",
            "catalog_result": None,
            "elapsed_s": round(elapsed, 3),
            "tool_call_id": tool_call_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
