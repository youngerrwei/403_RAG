"""不依赖模型的文档结构保护：只保留原文，不按主题补写知识。"""
import re
from html.parser import HTMLParser


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.depth = 0
        self.invalid = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            self.invalid |= self.depth > 1
        elif tag == "tr":
            if self.row is not None:
                self.invalid = True
            self.row = []
        elif tag in {"td", "th"}:
            if self.row is None or self.cell is not None:
                self.invalid = True
            attrs = dict(attrs)
            self.cell = {"text": [], "header": tag == "th",
                         "rowspan": attrs.get("rowspan", "1"), "colspan": attrs.get("colspan", "1")}
        elif tag in {"br", "p", "div"} and self.cell is not None:
            self.cell["text"].append(" ")
        elif self.cell is not None:
            marker = {"b": "**", "strong": "**", "i": "*", "em": "*", "sup": "^{", "sub": "_{"}.get(tag)
            if marker:
                self.cell["text"].append(marker)
            elif tag == "img":
                alt = dict(attrs).get("alt", "")
                self.cell["text"].append(f"[图片: {alt}]" if alt else "[图片]")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            if self.row is not None:
                self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.cell is not None:
                self.invalid = True
            self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            self.depth -= 1
        elif self.cell is not None:
            marker = {"b": "**", "strong": "**", "i": "*", "em": "*", "sup": "}", "sub": "}"}.get(tag)
            if marker:
                self.cell["text"].append(marker)


def preserve_html_tables(text: str) -> str:
    """普通表转 Markdown；合并/异常表保留单元格及跨度，不猜测表头关系。"""
    def convert(match):
        raw = match.group(0)
        parser = _TableParser()
        try:
            parser.feed(raw)
            parser.close()
        except Exception:
            return "\n[表格结构疑似丢失] HTML 表无法解析，需核对原文。\n" + raw
        rows = parser.rows
        if not rows or parser.invalid or parser.depth or parser.row is not None or parser.cell is not None:
            # 无法可靠分配行列时只标记，不自动编造对应关系。
            return "\n[表格结构疑似丢失] HTML 表结构不完整，需核对原文。\n" + raw
        def cell_text(cell):
            value = re.sub(r"\s+", " ", "".join(cell["text"])).strip()
            return value.replace("|", r"\|")
        merged = any(c["rowspan"] != "1" or c["colspan"] != "1" for row in rows for c in row)
        if merged or len({len(row) for row in rows}) != 1:
            lines = ["[表格结构化记录] 合并或不规则表：以下为原始单元格顺序与跨度，不代表展开后的列号。"]
            for row_num, row in enumerate(rows, 1):
                for cell_num, cell in enumerate(row, 1):
                    lines.append(f"- 行{row_num} 单元格{cell_num}（rowspan={cell['rowspan']}, colspan={cell['colspan']}）：{cell_text(cell)}")
            return "\n\n" + "\n".join(lines) + "\n\n"
        width = len(rows[0])
        if not width:
            return "\n"
        # 只有显式 th 才提升为表头；全 td 的首行也作为原始数据保留。
        if all(cell["header"] for cell in rows[0]):
            header, data = [cell_text(c) for c in rows[0]], rows[1:]
        else:
            header, data = [f"列{i}" for i in range(1, width + 1)], rows
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines.extend("| " + " | ".join(cell_text(c) for c in row) + " |" for row in data)
        return "\n\n" + "\n".join(lines) + "\n\n"

    return re.sub(r"<table\b[^>]*>[\s\S]*?</table\s*>", convert, text, flags=re.IGNORECASE)


def split_table_row(line: str) -> list:
    """识别未转义且不在行内代码中的竖线，保留空单元格。"""
    line = line.strip()
    parts, start, index, ticks = [], 0, 0, 0
    while index < len(line):
        if line[index] == "\\":
            index += 2
            continue
        if line[index] == "`":
            end = index
            while end < len(line) and line[end] == "`":
                end += 1
            count = end - index
            if not ticks:
                ticks = count
            elif ticks == count:
                ticks = 0
            index = end
            continue
        if line[index] == "|" and not ticks:
            parts.append(line[start:index].strip())
            start = index + 1
        index += 1
    parts.append(line[start:].strip())
    if line.startswith("|") and parts and not parts[0]:
        parts.pop(0)
    if start == len(line) and line.endswith("|") and parts:
        parts.pop()
    return parts


def markdown_tables(text: str) -> list:
    """返回规则 Markdown 表的原文位置与行；不推断损坏表的缺失列。"""
    lines = text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    tables, index, fence_char, fence_size = [], 0, "", 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            token = fence.group(1)
            if not fence_char:
                fence_char, fence_size = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_size and not line[fence.end():].strip():
                fence_char, fence_size = "", 0
        if fence_char or fence or index + 1 >= len(lines):
            index += 1
            continue
        header = split_table_row(line)
        separator = split_table_row(lines[index + 1])
        if (len(header) < 2 or len(separator) != len(header)
                or not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separator)):
            index += 1
            continue
        end, rows, valid = index + 2, [], True
        while end < len(lines) and lines[end].strip() and "|" in lines[end]:
            cells = split_table_row(lines[end])
            if len(cells) < 2:
                break
            valid &= len(cells) == len(header)
            rows.append(lines[end].rstrip("\r\n"))
            end += 1
        if rows and valid:
            tables.append({
                "start": offsets[index], "end": offsets[end] if end < len(lines) else len(text),
                "header": [line.rstrip("\r\n"), lines[index + 1].rstrip("\r\n")],
                "rows": rows,
            })
        index = end
    return tables


def split_table_rows(header: list, rows: list, chunk_size: int) -> list:
    """重复原表头、按整行分块；超长单行保持原样，由调用方记录警告。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    prefix = "\n".join(header)
    chunks, current = [], []
    length = len(prefix)
    for row in rows:
        if current and length + len(row) + 1 > chunk_size:
            chunks.append(prefix + "\n" + "\n".join(current))
            current, length = [], len(prefix)
        current.append(row)
        length += len(row) + 1
    if current:
        chunks.append(prefix + "\n" + "\n".join(current))
    return chunks


def split_markdown_text(text: str, chunk_size: int, split_plain) -> list:
    """普通正文沿用既有切分器；表格不经过字符切分，周围正文不丢弃。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    tables = markdown_tables(text)
    if not tables:
        return split_plain(text)
    pieces, cursor = [], 0
    for table in tables:
        prose = text[cursor:table["start"]].strip()
        if prose:
            pieces.extend(split_plain(prose))
        pieces.extend(split_table_rows(table["header"], table["rows"], chunk_size))
        cursor = table["end"]
    if text[cursor:].strip():
        pieces.extend(split_plain(text[cursor:].strip()))
    # 能容纳时保留表格旁的说明；不为跨表重叠制造重复数据行。
    chunks = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 2 + len(piece) <= chunk_size:
            chunks[-1] += "\n\n" + piece
        else:
            chunks.append(piece)
    return chunks


def markdown_header_matches(text: str) -> list:
    """识别全部六级标题，跳过围栏代码，包括未闭合的围栏。"""
    matches = []
    fence_char, fence_size, offset = "", 0, 0
    for line in text.splitlines(keepends=True):
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            token = fence.group(1)
            if not fence_char:
                fence_char, fence_size = token[0], len(token)
            elif token[0] == fence_char and len(token) >= fence_size and not line[fence.end():].strip():
                fence_char, fence_size = "", 0
        elif not fence_char:
            match = re.compile(r"(#{1,6})\s+([^\n]+)").match(text, offset)
            if match:
                matches.append(match)
        offset += len(line)
    return matches


def strip_ingest_prefix(text: str) -> str:
    """仅删除入库生成的开头标签，不能让长路径/摘要掩盖空正文。"""
    text = str(text or "").strip()
    text = re.sub(r"\A【文档：[\s\S]*?】\s*", "", text)
    return re.sub(r"\A(?:\[(?:文档标题|路径|章节|页码)\][^\n]*\n\s*)+", "", text).strip()


def has_structured_evidence(text: str) -> bool:
    """短条款和数据表可包含完整事实；纯标题/图片/目录不因此放行。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for table in markdown_tables(text):
        if any(re.search(r"[\w\u4e00-\u9fff]", cell)
               for row in table["rows"] for cell in split_table_row(row)):
            return True
    body = [line for line in lines if not re.match(r"^#{1,6}\s", line)]
    body = [line for line in body if not re.fullmatch(r"(?:\[图片[^\]]*\]|!\[[^\]]*\]\([^)]*\))", line)]
    return any(re.search(r"[。；！？.!?;]$", line) and re.search(r"[\u4e00-\u9fffA-Za-z]", line)
               and not re.search(r"\.{3,}|https?://|www\.", line, re.I) for line in body)


def evidence_window(text: str, anchor: str, budget: int) -> str:
    """围绕实际命中正文取连续窗口，禁止把命中句截成无关的父块开头。"""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    anchor = strip_ingest_prefix(anchor)
    # 表格证据只输出原表头和完整数据行；预算不足时不截取半行或错配列。
    tables = markdown_tables(text)
    anchor_tables = markdown_tables(anchor)
    position = text.find(anchor) if anchor else -1
    for table in tables:
        # 同一行文本可能出现在不同参数表中，必须先确认表头一致。
        compatible = [candidate for candidate in anchor_tables
                      if [split_table_row(line) for line in candidate["header"]]
                      == [split_table_row(line) for line in table["header"]]]
        anchor_rows = [row for candidate in compatible for row in candidate["rows"]]
        matching = [i for i, row in enumerate(table["rows"]) if row in anchor_rows]
        if not matching and anchor and not anchor_tables:
            matching = [i for i, row in enumerate(table["rows"]) if anchor in row]
        touches = position >= 0 and table["start"] <= position < table["end"]
        if not matching and not touches and not (not anchor and table["start"] == 0):
            continue
        row_index = matching[0] if matching else 0
        prefix = "\n".join(table["header"])
        selected = [table["rows"][row_index]]
        if len(prefix) + 1 + len(selected[0]) > budget:
            return ""
        # 先保留命中行，再连续向后扩展，不拼接不相邻行造成假象。
        length = len(prefix) + 1 + len(selected[0])
        for row in table["rows"][row_index + 1:]:
            if length + 1 + len(row) > budget:
                break
            selected.append(row)
            length += 1 + len(row)
        return prefix + "\n" + "\n".join(selected)
    if position < 0:
        # 格式变换导致定位失败时优先保留真实命中子块，不冒认父块开头。
        if anchor and anchor_tables:
            return evidence_window(anchor, "", budget)
        start, target = 0, anchor or text
        if anchor:
            tables = markdown_tables(anchor)
    else:
        before = max(0, (budget - len(anchor)) // 2)
        start = max(0, position - before)
        start = min(start, max(0, len(text) - budget))
        target = text
    end = start + budget
    # 正文窗口碰到表格边界时收缩，不夹带丢表头或不完整的数据行。
    for table in tables:
        if table["start"] < start < table["end"]:
            start = table["end"]
        if table["start"] < end < table["end"]:
            end = table["start"]
    return target[start:end].strip()
