# agent_entry.py：可选 ReAct 编排层，不持有独立模型实例。

import time
import traceback
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .logger import get_logger
from .paths import PROJECT_ROOT
from .rag_agent import (
    append_user_chat_history,
    build_catalog_answer_section,
    build_file_context,
    get_runtime,
    list_catalog_entries,
    rule_based_route,
    sanitize_generated_answer,
)
from .rag_tool import ask_rag
from .tools import rag_qa, list_group_files, agent_tool_context


load_dotenv(PROJECT_ROOT / ".env")
_logger = get_logger("agent")


def debug_log(*args):
    _logger.debug(" ".join(str(arg) for arg in args))


def build_qwen_llm() -> ChatOpenAI:
    """兼容旧入口，返回主 RAG 运行时中的同一个 LLM 实例。"""
    return get_runtime()["llm"]


TOOLS = {
    "rag_qa": rag_qa,
    "list_group_files": list_group_files,
}


REACT_SYSTEM_PROMPT = """你是实验室内部知识库助手，采用 ReAct 思维链 + 工具调用方式工作。

你可以使用以下工具：

1) rag_qa(query: str) -> str
   - 通用问答工具：基于实验室知识库进行 RAG 检索并回答问题。
   - 当用户提出具体技术问题、操作步骤、名词定义、原理说明、设备用法等时，应优先调用此工具。
   - 你也可以在拿到某些文件列表后，构造针对这些文件的具体问题，并用 rag_qa 进一步深入阅读和总结。

2) list_group_files(keyword: str) -> str
   - 通用“文件/目录搜索”工具：根据任意关键词列出相关的文件信息。
   - 关键词可以是：小组名称（如“VLC小组”）、设备/仪器名称（如“LETO-3”）、项目名称、实验名称、文档类型（如“设备操作指南”）等。
   - 返回内容中包含 file_name、doc_title、rel_path、chunks 数量，可用于了解有哪些相关文档、它们大致位于哪个目录/小组/项目下。
   - 当用户的问题属于“有哪些文档/有哪些设备/有哪些资料/某个主题下有哪些文件”等清单或目录类查询时，应优先调用本工具。

使用规范（ReAct 格式）：
- 你需要在回答前先“思考”(Thought)，再决定是否调用工具(Action)。
- 如果需要调用工具，请严格按照以下格式单独输出一行：
  Action: 工具名[参数值]

  例如：
  Action: list_group_files["VLC小组"]
  或：
  Action: rag_qa["根据知识库说明 HOLOEYE LETO-3 空间光调制器的主要参数和用途"]

- 工具执行完毕后，我会返回一段以 "Observation:" 开头的内容，包含工具的返回结果。
  你需要基于 Observation 继续 Thought/Action，或在信息足够时给出最终 Answer。

- 当你已经获得足够信息可以直接回答用户问题时，请输出：
  Answer: 你的最终回答内容

重要约束：
- 回答时必须优先依据工具返回结果，不要凭空编造。
- “格式是什么/怎么写/要求/规范/步骤/方法”等是在查询文档内容，必须直接调用 rag_qa；不能先调用 list_group_files 猜测模板文件。
- list_group_files 仅用于用户明确要求列出或定位文件、目录、路径、设备清单的情况。
- 如果目录工具未找到结果，不代表知识库正文没有答案；应使用用户原始问题调用 rag_qa，禁止改为输出通用模板或知识库外常识。
- 未获得工具返回的知识库依据时，不得输出“通用规范”“通用模板”或虚构“主要参考”。
- 当你在使用工具后进行总结时，请明确使用类似表述：“根据知识库内容可概括为”。
- 对于“有哪些/清单/列表”等问题：
  * 应在 Observation 中提供的所有文件信息基础上，尽量汇总并去重相关实体（如设备名称、文档名称、工具名称）。
  * 如用户还问“怎么用/如何使用/使用方法/内容是什么”，应在列出文档后，继续调用 rag_qa，基于相关文档总结，而不是停留在文件列表层面。
  * 不要仅依据单一文档片段作答，而应综合多个相关文件。
- 回答末尾请给出你主要参考的文档标题或路径（如果 Observation 或 rag_qa 结果中有相关信息）。"""


def parse_action(text: str) -> Dict[str, Any]:
    """
    从模型输出中解析 Action 行，形如：
    Action: tool_name["arg string"]
    返回: {"tool_name": ..., "arg": ...}
    若未解析到则返回 {}。
    """
    # 找到最后一行包含 "Action:" 的行
    lines = text.strip().splitlines()
    action_line = ""
    for line in reversed(lines):
        if line.strip().startswith("Action:"):
            action_line = line.strip()
            break

    if not action_line:
        return {}

    # 例：Action: list_group_files["VLC小组"]
    m = re.match(r"Action:\s*([a-zA-Z0-9_]+)\s*\[(.*)\]\s*$", action_line)
    if not m:
        return {}

    tool_name = m.group(1)
    raw_arg = m.group(2).strip()

    # 去掉首尾引号
    if (raw_arg.startswith('"') and raw_arg.endswith('"')) or (
        raw_arg.startswith("'") and raw_arg.endswith("'")
    ):
        raw_arg = raw_arg[1:-1]

    return {"tool_name": tool_name, "arg": raw_arg}


def parse_answer(text: str) -> str:
    """
    从模型输出中解析最终 Answer:
    形如: Answer: xxx
    如果没有显式 Answer: 则返回全文。
    """
    match = re.search(r"(?:^|\n)\s*Answer:\s*([\s\S]*)$", text.strip())
    if match:
        return match.group(1).strip()
    # 没有显式 Answer，则直接返回原文
    return text.strip()


def expected_agent_tool(route: Optional[str], used_list_tool: bool, used_rag_tool: bool) -> Optional[str]:
    """根据通用路由语义和已完成步骤返回下一项必需工具；None 表示可以回答。"""
    if route == "rag_search":
        return None if used_rag_tool else "rag_qa"
    if route == "file_list":
        return None if used_list_tool else "list_group_files"
    if route == "hybrid":
        if not used_list_tool:
            return "list_group_files"
        return None if used_rag_tool else "rag_qa"
    return None


def _initial_messages(question: str, chat_history: Optional[List[Dict[str, str]]] = None):
    messages: List[Dict[str, str]] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    for item in (chat_history or [])[-20:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def run_react_once(
    llm: ChatOpenAI,
    question: str,
    max_steps: int = 5,
    debug: bool = True,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    手写一个简单 ReAct 循环：
    - 系统提示 + 用户问题 → LLM 输出 Thought/Action/Answer
    - 解析 Action，如果有工具调用则执行，并把 Observation 追加到对话中，再次让 LLM 推理
    - 最多 max_steps 次工具调用
    """
    messages = _initial_messages(question, chat_history)
    initial_route = rule_based_route(question)
    route_name = initial_route.get("route") if initial_route else None
    used_list_tool = False
    used_rag_tool = False

    observation_text = ""
    for step in range(1, max_steps + 1):
        if debug:
            debug_log(f"ReAct Step {step}")

        # 调用模型
        if debug:
            debug_log(f"准备调用 LLM, step={step}, message_count={len(messages)}")
            total_chars = sum(len(m.get("content", "")) for m in messages)
            debug_log(f"messages 总字符数: {total_chars}")
            for i, m in enumerate(messages[-4:], 1):  # 只打印最后几条，避免刷屏
                debug_log(f"最近消息{i}: role={m['role']}, chars={len(m['content'])}")

        llm_start = time.perf_counter()
        resp = llm.invoke(messages)
        llm_cost = time.perf_counter() - llm_start

        content = resp.content if hasattr(resp, "content") else str(resp)

        if debug:
            debug_log(f"LLM 调用完成, step={step}, 耗时={llm_cost:.3f}s, 输出字符数={len(content)}")
            debug_log("LLM 输出：", content)

        # 尝试解析 Answer（如果模型已经给出最终回答）
        if "Answer:" in content:
            required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
            if required_tool:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"尚未取得回答所需的知识库依据。请先调用 {required_tool}，不得直接作答。",
                })
                continue
            final_answer = parse_answer(content)
            if debug:
                debug_log("解析到最终 Answer，结束循环")
            return final_answer

        # 解析 Action
        action = parse_action(content)
        if not action:
            required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
            if required_tool:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"输出缺少可执行的 Action。请由你调用 {required_tool} 后再回答。",
                })
                continue
            if route_name:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "必需工具调用已经完成。请严格使用 Answer: 开头输出有知识库依据的最终回答。",
                })
                continue
            # 路由未规定必需工具时，兼容模型的直接回答。
            if debug:
                debug_log("未解析到 Action，直接将本次输出当作回答")
            return content.strip()

        tool_name = action["tool_name"]
        arg = action["arg"]

        required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
        if required_tool and tool_name != required_tool:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"该调用不符合当前通用工具契约：下一步必须调用 {required_tool}。"
                    "请保留用户原始问题的内容意图并重新选择工具。"
                ),
            })
            continue
        if route_name and required_tool is None:
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "必需工具调用已经完成，不要继续调用工具。请使用 Answer: 输出最终回答。",
            })
            continue

        if debug:
            debug_log(f"解析到 Action: tool={tool_name}, arg={arg}")

        tool = TOOLS.get(tool_name)
        if tool is None:
            observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
        else:
            tool_start = time.perf_counter()
            try:
                # langchain_core.tools.Tool 对象：用 .invoke 调用
                # 若你的版本不支持 .invoke，可以用 .run 或直接 .func 看实际类型
                tool_input_key = list(tool.args.keys())[0]  # 第一个参数名
                tool_input = {tool_input_key: arg}
                with agent_tool_context(username):
                    obs = tool.invoke(tool_input)
                tool_cost = time.perf_counter() - tool_start
                observation_text = str(obs)
                used_list_tool = used_list_tool or tool_name == "list_group_files"
                used_rag_tool = used_rag_tool or tool_name == "rag_qa"

                if debug:
                    debug_log(
                        f"工具调用完成: tool={tool_name}, 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")
            except Exception as e:
                tool_cost = time.perf_counter() - tool_start
                observation_text = "[工具执行异常] 服务暂时不可用，请稍后重试。"
                if debug:
                    debug_log(f"工具调用异常: tool={tool_name}, 耗时={tool_cost:.3f}s, error={e}")

        if debug:
            debug_log("Observation:", observation_text)

        # 将本轮 LLM 输出 + Observation 追加到对话历史，再继续下一轮
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation_text}\n请继续推理，若可以则给出 Answer。"})


    # 超出 max_steps 仍无 Answer，则返回最后一次 observation + 提示
    return f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}"


def ask_agent(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    应用侧统一入口：ReAct LLM + 工具。
    """
    llm = build_qwen_llm()
    answer = run_react_once(
        llm,
        question,
        max_steps=5,
        debug=True,
        username=username,
        chat_history=chat_history,
    )
    append_user_chat_history(username, "user", question)
    append_user_chat_history(username, "assistant", answer)
    return answer


def ask_agent_stream(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """
    流式 ReAct Agent：
    - type=step_llm：每一步 LLM 的输出（Thought/Action 文本）
    - type=step_tool：每次工具调用及其Observation
    - type=final：最终 Answer 文本
    - type=error：错误信息
    """
    initial_route = rule_based_route(question)
    route_name = initial_route.get("route") if initial_route else None

    observation_text = ""
    used_list_tool = False
    used_rag_tool = False
    latest_catalog_result = None

    def finalize_answer(value: str) -> str:
        answer = sanitize_generated_answer(value)
        if latest_catalog_result and "## 目录 / 文件检索结果" not in answer:
            answer = f"{build_catalog_answer_section(latest_catalog_result)}\n\n## 文档内容说明\n\n{answer}".strip()
        return answer

    try:
        llm = build_qwen_llm()
        messages = _initial_messages(question, chat_history)
        max_steps = 4

        for step in range(1, max_steps + 1):
            # 1) 调 LLM
            debug_log(
                f"[stream] step={step} 开始, messages={len(messages)}, total_chars={sum(len(m.get('content', '')) for m in messages)}")

            llm_start = time.perf_counter()
            resp = llm.invoke(messages)
            llm_cost = time.perf_counter() - llm_start

            content = resp.content if hasattr(resp, "content") else str(resp)
            debug_log(f"[stream] step={step} LLM耗时={llm_cost:.3f}s, 输出字符数={len(content)}")

            # 把本轮 LLM 输出先流出去
            yield {
                "type": "step_llm",
                "step": step,
                "content": content,
            }

            # 2) 判断是否是最终 Answer
            if "Answer:" in content:
                required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
                if required_tool:
                    # 模型必须自行完成所需工具调用；这里只拒绝无依据的过早回答，不伪造 Action。
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"尚未取得回答所需的知识库依据。请先调用 {required_tool}，不得直接作答。"
                            "工具参数必须保留用户原始问题的意图。"
                        )
                    })
                    continue

                final_answer = finalize_answer(parse_answer(content))
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {
                    "type": "final",
                    "content": final_answer,
                    "file_result": latest_catalog_result,
                }
                return

            # 3) 解析 Action
            action = parse_action(content)
            if not action:
                required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
                if required_tool:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"输出缺少可执行的 Action。请由你调用 {required_tool} 后再回答。",
                    })
                    continue
                if route_name:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "必需工具调用已经完成。请严格使用 Answer: 开头输出有知识库依据的最终回答。",
                    })
                    continue
                # 路由未规定必需工具时，兼容模型的直接回答。
                final_answer = finalize_answer(content)
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {
                    "type": "final",
                    "content": final_answer,
                    "file_result": latest_catalog_result,
                }
                return

            tool_name = action["tool_name"]
            arg = action["arg"]

            required_tool = expected_agent_tool(route_name, used_list_tool, used_rag_tool)
            if required_tool and tool_name != required_tool:
                debug_log(
                    f"[stream] 拒绝不符合工具契约的调用: route={route_name}, "
                    f"proposed={tool_name}, required={required_tool}"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"该调用不符合当前通用工具契约：下一步必须调用 {required_tool}。"
                        "请由你重新输出正确的 Action；系统不会替你调用或改写工具。"
                    ),
                })
                continue
            if route_name and required_tool is None:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "必需工具调用已经完成，不要继续调用工具。请使用 Answer: 输出最终回答。",
                })
                continue

            tool = TOOLS.get(tool_name)
            catalog_result = None
            if tool is None:
                observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
            else:
                try:
                    tool_start = time.perf_counter()
                    if tool_name == "list_group_files":
                        catalog_result = list_catalog_entries(arg)
                        observation_text = build_file_context(catalog_result)
                    elif tool_name == "rag_qa":
                        rag_result = ask_rag(arg, username=username)
                        observation_text = rag_result.get("answer", "")
                        catalog_result = rag_result.get("file_result")
                    else:
                        tool_input_key = list(tool.args.keys())[0]
                        tool_input = {tool_input_key: arg}
                        with agent_tool_context(username):
                            obs = tool.invoke(tool_input)
                        observation_text = str(obs)
                    tool_cost = time.perf_counter() - tool_start
                    debug_log(
                        f"[stream] step={step} tool={tool_name} 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")

                    if tool_name == "list_group_files":
                        used_list_tool = True
                    if tool_name == "rag_qa":
                        used_rag_tool = True
                    if catalog_result:
                        latest_catalog_result = catalog_result

                except Exception as e:
                    _logger.error(f"Agent 工具执行异常: tool={tool_name}, error={e!r}")
                    observation_text = "[工具执行异常] 服务暂时不可用，请稍后重试。"

            # 4) 把本次工具调用结果也流出去
            yield {
                "type": "step_tool",
                "step": step,
                "tool": tool_name,
                "arg": arg,
                "observation": observation_text,
            }
            if catalog_result:
                yield {
                    "type": "tool",
                    "tool_name": "list_catalog_entries",
                    "content": catalog_result,
                }

            # 5) 历史追加，进入下一轮
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Observation: {observation_text}\n"
                    "请继续思考并决定下一步：如果信息尚不足以回答，继续选择合适的工具调用；"
                    "如果信息已经足够，并且（仅当原始问题同时要求清单和内容说明时）已在目录结果基础上调用过 rag_qa，"
                    "则可以给出 Answer。"
                )
            })

        # 超出最大步数未得到 Answer
        final_answer = finalize_answer(f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}")
        append_user_chat_history(username, "user", question)
        append_user_chat_history(username, "assistant", final_answer)
        yield {
            "type": "final",
            "content": final_answer,
            "file_result": latest_catalog_result,
        }

    except Exception as e:
        _logger.error(f"ReAct 执行异常: {e!r}")
        _logger.debug(traceback.format_exc())
        yield {
            "type": "error",
            "content": "服务暂时不可用，请稍后重试。",
        }


if __name__ == "__main__":
    q = input("请输入问题（例如：VLC小组有哪些设备？）：").strip()
    if not q:
        q = "VLC小组有哪些设备？"
    ans = ask_agent(q)
    _logger.info("=== Agent 最终回答 ===")
    _logger.info(ans)
