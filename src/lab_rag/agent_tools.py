"""Agent 工具注册表：提示词、校验、分派共用的单一数据源。

此前工具元数据分散在三处（LangChain @tool 的 docstring、提示词里的自然语言描述、
agent_entry 里的硬编码分派），容易漂移。这里集中定义 ToolSpec，由它：
- 生成 REACT_SYSTEM_PROMPT 的工具段落；
- 校验模型输出的 Action 工具名；
- 通过 dispatch_tool 提供唯一执行路径（返回结构化结果，含耗时与可追溯 id）。

handler 统一显式接收 username，不再依赖 ContextVar 的隐式绑定。
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rag_agent import build_file_context, list_catalog_entries, load_config, _runtime
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
                   route_hint: Optional[dict]) -> Tuple[str, Any, Optional[str]]:
    rag_result = ask_rag(arg, username=username, route_hint=route_hint)
    coverage = rag_result.get("coverage") or {}
    # 透传底层覆盖度级别，供编排器判断「检索是否获得充分依据」（无依据时不作为作答依据）
    coverage_level = coverage.get("level")
    return (rag_result.get("answer", ""), rag_result.get("file_result"), coverage_level)


def _handle_list_group_files(arg: str, username: str,
                             route_hint: Optional[dict]) -> Tuple[str, Any]:
    result = list_catalog_entries(arg)
    text = build_file_context(result)
    return (text or f"未找到与 {arg} 相关的文件。"), result


def _resolve_kb_roots(cfg: Dict) -> List[str]:
    """解析知识库根目录（与 list_catalog_entries 共用 KNOWLEDGE_BASE_ROOT / DOCS_PATH）。"""
    roots = []
    for key in ("KNOWLEDGE_BASE_ROOT", "DOCS_PATH"):
        value = str(cfg.get(key, "") or "").strip()
        if value and value not in roots:
            roots.append(value)
    return roots


def _safe_join(root: str, rel_path: str) -> Optional[str]:
    """将 rel_path 安全拼接到 root，阻止路径穿越；返回绝对路径或 None。"""
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, rel_path))
    if target == root_abs or target.startswith(root_abs + os.sep):
        return target
    return None


def _read_markdown_text(path: str, max_chars: int) -> Optional[str]:
    """读取文本文档并返回（超长截断）；失败返回 None。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + (
            f"\n\n[文档已截断，仅显示前 {max_chars} 字符；"
            f"如需更多请换更聚焦的子目录/章节再次 read_file]")
    return text


def _handle_read_file(arg: str, username: str,
                     route_hint: Optional[dict]) -> Tuple[str, Any]:
    """按 rel_path 读取知识库文档/目录的完整正文。

    文件：返回其完整正文（受 READ_FILE_MAX_CHARS 截断）。
    目录：拼接其下全部 .md/.txt（受数量与字符预算约束），便于一次性获取某设备的全部手册。
    未找到或路径非法时抛 ValueError，由 dispatch_tool 降级为「工具失败」以触发重试恢复。
    """
    config = _runtime["config"] if _runtime is not None else load_config()
    max_chars = max(2000, int(config.get("READ_FILE_MAX_CHARS", 6000)))
    rel_path = str(arg or "").strip().strip('"').strip("'").strip()
    if not rel_path:
        raise ValueError("read_file 需要 rel_path 参数（来自 list_group_files 返回的路径）。")
    roots = _resolve_kb_roots(config)
    target = None
    for root in roots:
        cand = _safe_join(root, rel_path)
        if cand and os.path.exists(cand):
            target = cand
            break
    if not target:
        raise ValueError(
            f"未找到路径 {rel_path} 对应的文档或目录；请用 list_group_files 重新定位后再 read_file。")

    if os.path.isfile(target):
        text = _read_markdown_text(target, max_chars)
        if text is None:
            raise ValueError(f"读取文件失败：{rel_path}")
        return (f"[文档] {rel_path}\n" + text, None)

    # 目录：拼接其下全部文本文档（受预算与数量上限约束）
    md_files = []
    for dp, _dirs, fns in os.walk(target):
        for fn in fns:
            if fn.lower().endswith((".md", ".markdown", ".txt")):
                md_files.append(os.path.join(dp, fn))
    md_files.sort()
    md_files = md_files[:10]
    if not md_files:
        raise ValueError(f"目录 {rel_path} 下未发现可读取的文档。")
    per_file = max(1000, max_chars // len(md_files))
    parts = [f"[目录] {rel_path}（共读取 {len(md_files)} 个文档）"]
    for fp in md_files:
        sub = _read_markdown_text(fp, per_file)
        if sub is None:
            continue
        parts.append(f"\n--- 文件: {os.path.relpath(fp, target)} ---\n" + sub)
    return ("\n".join(parts), None)


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
        name="read_file",
        arg_name="rel_path",
        description="文档阅读工具：按 list_group_files / 目录枚举返回的 rel_path 读取知识库中某文档（或某设备/目录）的完整正文。",
        bullets=(
            "参数 rel_path：来自文件清单的路径，如 设备操作指南/VLC小组/.../Manual.md，或某个设备目录。",
            "读取文件返回其完整正文（超长会截断并标注）；读取目录则返回该目录下全部手册的拼接正文，便于一次性获取某设备的完整用法。",
            "需要某一设备/文档的具体用法、步骤、参数时，优先用 list_group_files 定位路径再用 read_file 精读，而非仅用 rag_qa 全局检索；"
            "rag_qa 更适合跨多文档综合检索或用户未指明具体文档的场景。",
        ),
        role="content",
        emits_catalog=False,
        handler=_handle_read_file,
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
            "coverage_level": None,
            "elapsed_s": 0.0,
            "tool_call_id": tool_call_id,
        }

    started = time.perf_counter()
    try:
        handler_out = spec.handler(arg, username, route_hint)
        # 正文工具返回三元组（含覆盖度级别），目录工具返回二元组
        if len(handler_out) == 3:
            observation, catalog_result, coverage_level = handler_out
        else:
            observation, catalog_result = handler_out
            coverage_level = None
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "observation": str(observation or ""),
            "catalog_result": catalog_result,
            "coverage_level": coverage_level,
            "elapsed_s": round(elapsed, 3),
            "tool_call_id": tool_call_id,
        }
    except Exception as exc:  # 工具失败不终止整个 Agent 轮次
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "observation": (f"[工具执行异常] {type(exc).__name__}: {exc}"
                           f"（请换更聚焦的参数重试，或用 list_group_files 重新定位）"),
            "catalog_result": None,
            "elapsed_s": round(elapsed, 3),
            "tool_call_id": tool_call_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
