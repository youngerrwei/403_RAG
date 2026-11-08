"""【已归档】v3.4.2 之前的非流式 ReAct 实现与 CLI（历史快照，禁止 import）。

归档原因（3.4.3 重构）：当时 `run_react_once` 与 `ask_agent_stream` 各自实现了一套
ReAct 循环，导致步数预算、重复抑制、上下文压缩、收敛轮、错误降级出现分叉。
现已收敛为 `src/lab_rag/agent_entry.py` 中的单一编排器，非流式入口改为消费同一
事件流的薄适配器；LangChain @tool 层见 `legacy/agent_tools_langchain.py`。

本文件不在 Python 包路径内，且带相对导入，仅作历史参考，不应被任何生产代码加载。
"""

# ---- 以下为归档时的原始代码（保持原样，仅注释掉无法解析的包内相对导入）----
#
# def run_react_once(
#     llm, question, max_steps=None, debug=True,
#     username="legacy-agent", chat_history=None,
# ) -> str:
#     """手写 ReAct 循环：LLM 输出 Thought/Action/Answer，解析 Action 执行工具，
#     把 Observation 追加到对话再次推理，最多 max_steps 次工具调用。"""
#     messages = _initial_messages(question, chat_history)
#     cfg = _agent_config()
#     if max_steps is None:
#         max_steps = max(1, int(cfg.get("AGENT_LEGACY_MAX_STEPS", 5)))
#     max_idle = max(0, int(cfg.get("AGENT_MAX_IDLE_STEPS", 2)))
#     llm_budget = max_steps + max_idle + 1
#     guard_repeat = bool(cfg.get("AGENT_REPEAT_ACTION_GUARD", True))
#     seen_actions = set()
#     observation_indices = []
#     tool_log = []
#     tool_calls = idle_steps = llm_calls = 0
#     ...
#     while tool_calls < max_steps and llm_calls < llm_budget:
#         resp = llm.invoke(messages)
#         content = resp.content if hasattr(resp, "content") else str(resp)
#         if "Answer:" in content:
#             return parse_answer(content)
#         action = parse_action(content)
#         if not action:
#             return content.strip()
#         tool_name, arg = action["tool_name"], action["arg"]
#         if guard_repeat and (tool_name, arg) in seen_actions:
#             ...
#             continue
#         seen_actions.add((tool_name, arg))
#         tool = TOOLS.get(tool_name)
#         tool_input_key = list(tool.args.keys())[0]
#         with agent_tool_context(username):
#             observation_text = str(tool.invoke({tool_input_key: arg}))
#         messages.append({"role": "assistant", "content": content})
#         append_observation(observation_text, "请继续推理，若可以则给出 Answer。")
#     # 收敛轮 + _agent_fallback_answer 兜底
#     return _agent_fallback_answer(question, tool_log)
#
#
# def ask_agent(question, username="legacy-agent", chat_history=None) -> str:
#     route = rule_based_route(question)
#     if route and route.get("route") == "rag_search":
#         answer = ask_rag(question, username=username).get("answer", "")
#     else:
#         answer = run_react_once(build_qwen_llm(), question, debug=True,
#                                 username=username, chat_history=chat_history)
#     append_user_chat_history(username, "user", question)
#     append_user_chat_history(username, "assistant", answer)
#     return answer
#
#
# if __name__ == "__main__":
#     q = input("请输入问题（例如：VLC小组有哪些设备？）：").strip()
#     ans = ask_agent(q or "VLC小组有哪些设备？")
#     print(ans)
