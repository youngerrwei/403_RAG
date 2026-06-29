#!/usr/bin/env python3
"""
test_html.py — 验证 templates/index.html 前端页面的完整性和正确性
仅使用 Python 标准库（html.parser, re, os），无需启动服务器即可运行。
"""

import os
import re
from html.parser import HTMLParser


# ═══════════════════════════════════════════════════
# 辅助：简易 HTML 解析器，提取标签 / 属性 / 文本
# ═══════════════════════════════════════════════════
class SimpleHTMLAnalyzer(HTMLParser):
    """收集所有标签、id、class 以及 <style>/<script> 文本内容"""

    def __init__(self):
        super().__init__()
        self.tags = []           # [(tag, attrs_dict), ...]
        self.ids = set()         # 所有 id 属性值
        self.classes = set()     # 所有 class 中的单个类名
        self.tag_stack = []
        self.style_text = ""
        self.script_text = ""
        self._current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tags.append((tag, attrs_dict))
        self._current_tag = tag
        self.tag_stack.append(tag)
        if "id" in attrs_dict:
            self.ids.add(attrs_dict["id"])
        if "class" in attrs_dict:
            for cls in attrs_dict["class"].split():
                self.classes.add(cls)

    def handle_endtag(self, tag):
        self._current_tag = None
        if self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()

    def handle_data(self, data):
        if self.tag_stack:
            current = self.tag_stack[-1]
            if current == "style":
                self.style_text += data
            elif current == "script":
                self.script_text += data

    def has_tag_with_id(self, tag, id_val):
        return any(t == tag and a.get("id") == id_val for t, a in self.tags)

    def has_tag_with_attr(self, tag, attr, value):
        return any(t == tag and a.get(attr) == value for t, a in self.tags)

    def has_id(self, id_val):
        return id_val in self.ids

    def has_class(self, cls):
        return cls in self.classes


# ═══════════════════════════════════════════════════
# 测试框架
# ═══════════════════════════════════════════════════
class TestResult:
    def __init__(self):
        self.results = []

    def check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append((status, name, detail))
        icon = "✅" if passed else "❌"
        msg = f"  {icon} [{status}] {name}"
        if detail:
            msg += f"  —  {detail}"
        print(msg)

    def summary(self):
        total = len(self.results)
        passed = sum(1 for s, _, _ in self.results if s == "PASS")
        failed = total - passed
        print("\n" + "=" * 60)
        print(f"  测试总结: {passed}/{total} 通过", end="")
        if failed:
            print(f", {failed} 项失败")
        else:
            print("  —  全部通过! 🎉")
        print("=" * 60)
        return failed == 0


# ═══════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════
def main():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if not os.path.isfile(html_path):
        print(f"❌ 文件不存在: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 解析 HTML
    analyzer = SimpleHTMLAnalyzer()
    parse_ok = True
    try:
        analyzer.feed(html_content)
    except Exception as e:
        parse_ok = False
        parse_error = str(e)

    T = TestResult()

    # ═══════════════════════════════════════════════
    # 1. HTML 结构检查
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  1. HTML 结构检查")
    print("=" * 60)

    T.check(
        "HTML 文件可被正确解析 (html.parser)",
        parse_ok,
        "" if parse_ok else f"解析错误: {parse_error}",
    )

    # 问答输入区域
    T.check(
        "问答输入区域 — textarea#question 存在",
        analyzer.has_tag_with_id("textarea", "question"),
    )
    T.check(
        "问答输入区域 — 提交按钮#submitBtn 存在",
        analyzer.has_tag_with_id("button", "submitBtn"),
    )

    # 答案展示区域
    T.check(
        "答案展示区域 — div#answer 存在",
        analyzer.has_id("answer"),
    )
    T.check(
        "答案展示区域 — 拥有 answer-box class",
        analyzer.has_class("answer-box") or "answer-box" in html_content,
    )

    # 历史对话区域
    T.check(
        "历史对话区域 — div#history 存在",
        analyzer.has_id("history"),
    )
    T.check(
        "历史对话区域 — 拥有 chat-list class",
        analyzer.has_class("chat-list") or "chat-list" in html_content,
    )
    T.check(
        "历史对话区域 — card-history 卡片容器存在",
        analyzer.has_class("card-history") or "card-history" in html_content,
    )

    # Agent 模式切换
    T.check(
        "Agent 模式切换 — input#agentModeToggle checkbox 存在",
        analyzer.has_tag_with_id("input", "agentModeToggle"),
    )
    # 验证它确实是 checkbox
    is_checkbox = any(
        t == "input" and a.get("id") == "agentModeToggle" and a.get("type") == "checkbox"
        for t, a in analyzer.tags
    )
    T.check(
        "Agent 模式切换 — type=checkbox",
        is_checkbox,
    )

    # ═══════════════════════════════════════════════
    # 2. JavaScript 函数/变量检查（正则匹配）
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  2. JavaScript 函数/变量检查")
    print("=" * 60)

    js_text = analyzer.script_text  # 所有 <script> 标签中的文本

    # currentSourceMap 变量
    T.check(
        "currentSourceMap 变量声明存在",
        bool(re.search(r'\blet\s+currentSourceMap\b|\bvar\s+currentSourceMap\b|\bconst\s+currentSourceMap\b', js_text)),
        "搜索 let/var/const currentSourceMap",
    )

    # renderSourceLinks 函数
    T.check(
        "renderSourceLinks 函数定义存在",
        bool(re.search(r'\bfunction\s+renderSourceLinks\s*\(', js_text)),
        "搜索 function renderSourceLinks(",
    )

    # HISTORY_STATE 对象（含 SYNCED/MODIFIED/SAVING）
    has_history_state = bool(re.search(r'\bHISTORY_STATE\s*=\s*\{', js_text))
    has_synced = "SYNCED" in js_text
    has_modified = "MODIFIED" in js_text
    has_saving = "SAVING" in js_text
    T.check(
        "HISTORY_STATE 对象定义存在",
        has_history_state,
        "搜索 HISTORY_STATE = {",
    )
    T.check(
        "HISTORY_STATE 包含 SYNCED/MODIFIED/SAVING",
        has_synced and has_modified and has_saving,
        f"SYNCED={has_synced}, MODIFIED={has_modified}, SAVING={has_saving}",
    )

    # updateHistoryLabel 函数
    T.check(
        "updateHistoryLabel 函数定义存在",
        bool(re.search(r'\bfunction\s+updateHistoryLabel\s*\(', js_text)),
        "搜索 function updateHistoryLabel(",
    )

    # historyState 变量
    T.check(
        "historyState 变量存在",
        bool(re.search(r'\blet\s+historyState\b|\bvar\s+historyState\b|\bconst\s+historyState\b', js_text)),
        "搜索 let/var/const historyState",
    )

    # escapeHtml 函数
    T.check(
        "escapeHtml 函数存在",
        bool(re.search(r'\bfunction\s+escapeHtml\s*\(', js_text)),
        "搜索 function escapeHtml(",
    )

    # SSE: step_llm 处理
    T.check(
        "SSE 事件处理 — step_llm 类型存在",
        bool(re.search(r'["\']step_llm["\']', js_text)),
        '搜索 "step_llm" 或 \'step_llm\'',
    )

    # SSE: step_tool 处理
    T.check(
        "SSE 事件处理 — step_tool 类型存在",
        bool(re.search(r'["\']step_tool["\']', js_text)),
        '搜索 "step_tool" 或 \'step_tool\'',
    )

    # fetch body 中包含 use_agent
    T.check(
        "fetch 请求 body 包含 use_agent 参数",
        bool(re.search(r'use_agent', js_text)),
        "搜索 use_agent",
    )

    # ═══════════════════════════════════════════════
    # 3. 数据格式检查
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  3. 数据格式检查")
    print("=" * 60)

    # rerank_score 格式化（toFixed + 百分比）
    has_toFixed = "toFixed" in js_text
    has_percent = bool(re.search(r"rerank_score.*toFixed|toFixed.*%", js_text))
    T.check(
        "rerank_score 格式化逻辑 — toFixed 调用存在",
        has_toFixed,
        "搜索 toFixed",
    )
    T.check(
        "rerank_score 百分比格式化 — rerank_score 与百分比/toFixed 关联",
        has_percent,
        "搜索 rerank_score 附近的 toFixed 或 % 符号",
    )

    # source_map tooltip 渲染
    has_source_link = "source-link" in html_content
    has_tooltip_title = bool(re.search(r'title=.*doc_title', js_text))
    T.check(
        "source_map tooltip — source-link class 存在",
        has_source_link,
        "搜索 source-link",
    )
    T.check(
        "source_map tooltip — 含 doc_title 的 title 属性渲染",
        has_tooltip_title,
        "搜索 title=...doc_title",
    )

    # ═══════════════════════════════════════════════
    # 4. CSS / 样式检查
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  4. CSS / 样式检查")
    print("=" * 60)

    css_text = analyzer.style_text  # <style> 内的文本
    full_text = html_content        # 内联样式也需要检查

    # Agent 步骤折叠面板样式（agent-step 或 details 相关）
    has_agent_step_class = "agent-step" in full_text
    has_details_tag = "<details" in full_text
    T.check(
        "Agent 步骤折叠面板 — agent-step class 使用存在",
        has_agent_step_class,
        "搜索 agent-step",
    )
    T.check(
        "Agent 步骤折叠面板 — <details> 标签存在",
        has_details_tag,
        "搜索 <details",
    )

    # source-link 样式
    has_source_link_style = bool(
        re.search(r'\.source-link|source-link', css_text)
        or re.search(r'class=["\']source-link["\']', full_text)
        or re.search(r"class=\"source-link\"", full_text)
    )
    # 也检查内联样式中的 source-link
    has_source_link_inline = bool(re.search(r'class="source-link"', full_text))
    T.check(
        "source-link 样式/class 存在",
        has_source_link_style or has_source_link_inline,
        "搜索 CSS 或内联 .source-link",
    )

    # ═══════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════
    all_passed = T.summary()
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
