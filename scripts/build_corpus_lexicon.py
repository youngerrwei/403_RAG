#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从语料 rel_path 离线生成「小组别名 ↔ 设备型号 ↔ 手册类型」派生词典。

只读 DOCS_PATH，不依赖 Qdrant / vLLM / 模型。生成 data/corpus_lexicon.json：
- group_aliases: 小组缩写 → 真实目录名（含目录名自身的别名）
- device_models: 从设备目录名与文件名提取的字母数字混合型号
- manual_types: 手册类型词（Manuals / Datasheet / Specifications / User Guide ...）

词典是派生数据，可随时重建；指纹取全部 md 相对路径的 SHA-256，便于判断是否过期。
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "corpus_lexicon.json")

_GUIDE_DIR = "设备操作指南"
# 型号形态：字母数字混合（N9322C / EDU36311A / USRP-2944 / SVA1000X / MSO-X 3054A）
_MODEL_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\d[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b"
    r"|\b\d+[A-Za-z][A-Za-z0-9]*\b"
)
# 手册类型：英文通用词 + 中文常见后缀
_MANUAL_TYPE_WORDS = (
    "Manuals", "Manual", "Datasheet", "Data Sheet", "Specifications", "Specification",
    "User Guide", "Quick Guide", "Quick Reference", "Reference Guide", "Programming Guide",
    "Service Manual", "Getting Started", "Application Note", "User Manual", "Release Notes",
    "用户指南", "技术手册", "数据手册", "产品手册", "编程指南", "快速入门", "服务手册", "规格",
)
# 目录名里不该当型号的词（语言/通用词）
_MODEL_STOPWORDS = {
    "cn", "en", "ch", "pdf", "md", "v1", "v2", "v3", "ver", "www", "findic",
    "basic", "quick", "user", "guide", "data", "sheet", "datasheet", "manual",
    "series", "specifications", "application", "framework", "getting", "started",
    "ni", "labview", "matlab", "windows", "linux", "mac", "os", "sdk", "rio",
}


def build(docs_root: str) -> dict:
    root = Path(docs_root)
    md_files = sorted(root.rglob("*.md"))
    rel_paths = [str(p.relative_to(root)) for p in md_files]
    fingerprint = hashlib.sha256("\n".join(rel_paths).encode("utf-8")).hexdigest()

    group_aliases: dict[str, str] = {}
    device_models: Counter = Counter()
    manual_types: Counter = Counter()
    document_terms: Counter = Counter()

    guide = root / _GUIDE_DIR
    if guide.is_dir():
        for group_dir in sorted(p for p in guide.iterdir() if p.is_dir()):
            name = group_dir.name
            # 「A2X小组」→ 别名 A2X；同时登记全名，避免大小写/后缀差异
            short = re.sub(r"(小组|课题组|实验室)$", "", name).strip()
            if short and re.fullmatch(r"[A-Za-z0-9_-]{2,16}", short):
                group_aliases[short.upper()] = name
            group_aliases[name] = name

    for rel in rel_paths:
        parts = rel.split(os.sep)
        # 设备目录名（结构：设备操作指南/小组/设备名/...）
        device_dir = ""
        if len(parts) >= 3 and parts[0] == _GUIDE_DIR:
            device_dir = parts[2]
        for source in (device_dir, Path(rel).stem):
            for raw in _MODEL_RE.findall(source or ""):
                token = raw.strip("-_ ")
                low = token.lower()
                if len(token) < 3 or len(token) > 20 or low in _MODEL_STOPWORDS or token.isdigit():
                    continue
                # 文件名拼接出的复合 token（APD210-Manual）与纯单位（523nm）不作型号
                if any(bad in low for bad in ("manual", "sheet", "guide", "spec", "datasheet")):
                    continue
                if re.fullmatch(r"\d+(nm|mm|ghz|mhz|khz|db|v|a|w)", low):
                    continue
                if not re.search(r"\d", token) or not re.search(r"[A-Za-z]", token):
                    continue
                device_models[token] += 1
        for word in _MANUAL_TYPE_WORDS:
            if word.lower() in rel.lower():
                manual_types[word] += 1
        # 文档词元索引：从目录名与文件名提取中文片段（2~6 字）与英文词，
        # 用于判断「问题实体是否在语料中出现过」，作为域外问题的语料侧证据。
        names = parts[:-1] + [Path(rel).stem]
        for name in names:
            for seg in re.findall(r"[\u4e00-\u9fff]{2,6}", name):
                document_terms[seg] += 1
            for word in re.findall(r"[A-Za-z]{3,}", name.lower()):
                if word not in _MODEL_STOPWORDS:
                    document_terms[word] += 1

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_root": str(root),
        "docs_root_fingerprint": fingerprint,
        "n_markdown_files": len(md_files),
        "group_aliases": dict(sorted(group_aliases.items())),
        "device_models": [m for m, _ in sorted(device_models.items(), key=lambda x: (-x[1], x[0]))],
        "manual_types": [m for m, _ in sorted(manual_types.items(), key=lambda x: (-x[1], x[0]))],
        "document_terms": [m for m, _ in sorted(document_terms.items(), key=lambda x: (-x[1], x[0]))],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.getenv("DOCS_PATH", "/mnt/cpu_share"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="只校验现有词典指纹是否过期")
    args = ap.parse_args()

    if not Path(args.docs).is_dir():
        print(f"ERROR: DOCS_PATH 不存在: {args.docs}")
        return 2

    current = build(args.docs)
    if args.check:
        if not Path(args.out).is_file():
            print("STALE: 词典不存在")
            return 1
        existing = json.loads(Path(args.out).read_text(encoding="utf-8"))
        ok = existing.get("docs_root_fingerprint") == current["docs_root_fingerprint"]
        print(("FRESH" if ok else "STALE") + f": n_md={current['n_markdown_files']}")
        return 0 if ok else 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    print(f"written: {args.out}")
    print(f"groups={len(current['group_aliases'])} device_models={len(current['device_models'])} "
          f"manual_types={len(current['manual_types'])} "
          f"document_terms={len(current['document_terms'])} n_md={current['n_markdown_files']}")
    print("groups:", current["group_aliases"])
    print("top_models:", current["device_models"][:15])
    return 0


if __name__ == "__main__":
    sys.exit(main())
