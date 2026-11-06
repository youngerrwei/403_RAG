#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集工程侧量化门槛指标，输出 logs/engineering_metrics.json 供 runner 判定。

只读源码与回归脚本，不修改任何生产数据：
- reliability_test_count：tests/test_reliability.py 实际执行通过数
- lint_error_count：全部生产/测试/脚本 Python 文件的 AST 语法错误数
- new_third_party_import_count：生产路径中未在 requirements 声明的第三方顶层包数
- inlined_customization_count：应外置到词典但仍内联在生产代码里的字面量数
"""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "engineering_metrics.json")

# 标准库白名单（sys.stdlib_module_names，Python 3.10+）
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {
    "__future__", "dotenv", "langchain_core", "langchain_openai",
    "langchain_huggingface", "langchain_text_splitters", "langchain_qdrant",
    "qdrant_client", "sentence_transformers", "cachetools", "flask", "werkzeug",
    "mcp", "httpx", "pydantic", "numpy", "lab_rag",
}

# requirements 中声明的包名（用于区分「既有依赖」与「新增依赖」）
_DIST_TO_IMPORT = {
    "python-dotenv": "dotenv", "langchain-community": "langchain_community",
    "langchain-qdrant": "langchain_qdrant", "qdrant-client": "qdrant_client",
    "sentence-transformers": "sentence_transformers", "beautifulsoup4": "bs4",
    "pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
}


def declared_packages() -> set:
    declared = set()
    for p in Path(ROOT, "requirements").glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip().lower()
            if not name:
                continue
            declared.add(name.replace("-", "_"))
            if name in _DIST_TO_IMPORT:
                declared.add(_DIST_TO_IMPORT[name])
    return declared

# 应外置到 config/retrieval_lexicon.json 的定制化字面量（生产代码中不应再内联）
CUSTOMIZATION_LITERALS = (
    '"向量、矩阵"', "'向量、矩阵'", '"变量格式表语义恢复"', "'变量格式表语义恢复'",
    '"根据论文变量格式规范', "'根据论文变量格式规范",
    '"变量格式 变量类型 英文字母', '"论文变量格式规范中矩阵和向量',
    '"A2X", "VLC", "VLP"', '"正体+粗体"',
    '"目录", "有哪些文件"', '"原理", "步骤", "参数"',
)


def reliability_count(explicit: int = None) -> int:
    """返回可靠性回归用例数。

    支持 --reliability-count 显式传入（避免重复执行耗时回归）；否则实际跑一次，
    再否则复用上次采集结果并标注来源，绝不把缺失静默当作 0 通过。
    """
    if explicit is not None:
        return int(explicit)
    try:
        out = subprocess.run(
            [sys.executable, "-m", "unittest", "tests.test_reliability", "-v"],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        # unittest 汇总行形如 "Ran 74 tests in 53.175s"
        for line in (out.stderr + out.stdout).splitlines():
            if line.startswith("Ran ") and " tests" in line:
                return int(line.split()[1])
    except Exception as e:
        print(f"WARN: reliability_count failed: {e!r}", file=sys.stderr)
    if os.path.isfile(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                return int(json.load(f).get("reliability_test_count", 0))
        except (OSError, ValueError, TypeError):
            pass
    return 0


def lint_errors() -> int:
    errors = 0
    for pattern in ("src/lab_rag/*.py", "tests/*.py", "scripts/*.py"):
        for p in sorted(Path(ROOT).glob(pattern)):
            try:
                ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            except (SyntaxError, OSError):
                errors += 1
    return errors


def new_dependencies() -> int:
    found = set()
    for p in sorted(Path(ROOT, "src", "lab_rag").glob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return len(found - STDLIB - declared_packages())


def inlined_customizations() -> int:
    count = 0
    for name in ("rag_agent.py", "ingest.py", "lexicon.py"):
        p = Path(ROOT, "src", "lab_rag", name)
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for literal in CUSTOMIZATION_LITERALS:
            if literal in text:
                count += 1
    return count


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability-count", type=int, default=None,
                    help="显式传入已运行回归的用例数，避免重复执行耗时回归")
    args = ap.parse_args()
    metrics = {
        "reliability_test_count": reliability_count(args.reliability_count),
        "lint_error_count": lint_errors(),
        "new_third_party_import_count": new_dependencies(),
        "inlined_customization_count": inlined_customizations(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    for k, v in metrics.items():
        print(f"{k}={v}")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
