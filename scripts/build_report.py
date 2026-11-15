"""Build the team's report PDF from the Markdown parts: reports/report.pdf.

    .venv\\Scripts\\python.exe scripts\\build_report.py                 # parts from project.yaml (below)
    .venv\\Scripts\\python.exe scripts\\build_report.py --parts e3 --out $env:TEMP\\e3.pdf   # one part, page check
    .venv\\Scripts\\python.exe scripts\\build_report.py --keep-tex      # also write reports/build/report.tex

Parts, always assembled in this order, each starting on a new page (reports/README.md):
  core      reports/report_core.md                                   limit 6 pages
  e1..e4    reports/extensions/e1_validation_protocol.md, e2_learning_curve.md,
            e3_leakage.md, e4_feature_stability.md (the owned ones)    limit 2 pages each
  contrib   reports/contribution_table.md                            limit 1 page
Without --parts: core, the extensions in project.yaml report.extensions_owned, contrib. An empty or
missing list means the owners are not decided yet: all four extensions are included, with a warning.

Layout: XeLaTeX, Calibri 11 pt (LaTeX's 10.95 pt), A4, 2.54 cm margins, single line spacing, block
paragraphs 4 pt apart (+-1 pt as TeX fills a page), widow and orphan lines avoided, page numbers at
the foot. Tables in 10 pt (9 pt and narrower column gaps only
if a table does not fit otherwise). Figures at the text width unless the Markdown gives a width.
The first heading of each part is its title. One word space after a full stop, as in Word
("et al. (2008)" is not spaced like a sentence end); a justified line squeezes its word spaces by
at most 1/5 of a space. Line breaks in running text: none next to an operator or relation between
spaces (p = 0.05, 5 x 20, A - B, 7 / 8 / 9, P(>= 0.999) stay together) and none before % or a
middle dot used as a separator (20 %); table cells are not affected. A paragraph that ends with a
colon, or a short bold or italic line used as a heading, stays on the page of the paragraph, list,
table or figure after it. A paragraph made of hard-broken lines (a reference list) is set ragged
right, one line per paragraph, so that a page break never splits a wrapped reference.

Markdown subset (GitHub-style): ATX headings (#, ##, ###, ...); paragraphs (a line ending in two
spaces or a backslash breaks the line; in the paragraph right under a part's title every line break
is kept, for the team / author lines); **bold**, *italic*, `code`, [links](https://...), bare URLs,
backslash escapes, HTML entities, <br>, <sup>, <sub>; images on a line of their own, with a width hint
in one of three forms:
    ![alt](../figures/x.png "width=65%")     ![alt](../figures/x.png){width=65%}
    <img src="../figures/x.png" width="65%">
(the first form looks clean on GitHub too); GitHub pipe tables; bullet and numbered lists (nested);
block quotes; horizontal rules; fenced code blocks. HTML comments are dropped. Straight quotes
become curly quotes; -- and --- become en and em dashes. Unicode text (+-, x, minus, approx, >=,
dashes, Greek letters, superscripts) is set in Calibri as written.
Not supported: reference-style links, footnotes, math ($...$ is printed as is), indented code blocks,
other HTML (tags are dropped with a warning), images inside running text (only their alt text shows).

Tables: column widths follow the dashes of the separator row when they differ and the table source
is wider than 72 characters (|---|-| = 3:1, as pandoc does; a column never gets narrower than its
longest word or number); otherwise they are fitted to the content. A table with a "Table N" caption
paragraph directly below or above it (and at most 15 body rows) is kept on one page together with
the caption, as is a figure with its "Figure N" caption; other tables may break across pages
between rows (header repeated, at least two rows on each page).

XeLaTeX can fail on non-ASCII paths (the project folder is not ASCII), so the document is compiled
in an ASCII temp folder (--tmp-dir, else $SSC_TEX_TMP, the system temp folder, <drive>:\\tmp) with
copies of the figures, twice; the PDF is then copied to --out. --keep-tex also writes the .tex and
the XeLaTeX log to reports/build/ (not tracked by git; compile it there with the figures in
reports/figures/).

Prints the page count of every part, measured while typesetting, and warns when a part is over its
limit, when TODO is left in a part, when an extension's headings are not exactly Question / Protocol /
Results / Conclusion, about missing figures or glyphs and lines that stick out into the margin.
Exit code 0 (also with warnings), 1 if XeLaTeX fails or the PDF cannot be written, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import unquote

import _common as C
from ssc import paths

ROOT = paths.ROOT


@dataclass(frozen=True)
class PartSpec:
    key: str
    path: str        # relative to the project root
    limit: int       # page limit
    label: str


PARTS = {
    "core": PartSpec("core", "reports/report_core.md", 6, "core report"),
    "e1": PartSpec("e1", "reports/extensions/e1_validation_protocol.md", 2, "extension E1"),
    "e2": PartSpec("e2", "reports/extensions/e2_learning_curve.md", 2, "extension E2"),
    "e3": PartSpec("e3", "reports/extensions/e3_leakage.md", 2, "extension E3"),
    "e4": PartSpec("e4", "reports/extensions/e4_feature_stability.md", 2, "extension E4"),
    "contrib": PartSpec("contrib", "reports/contribution_table.md", 1, "contribution table"),
}
ORDER = list(PARTS)
EXTENSIONS = ("e1", "e2", "e3", "e4")
EXT_HEADINGS = ["Question", "Protocol", "Results", "Conclusion"]

DEFAULT_OUT = ROOT / "reports" / "report.pdf"
BUILD_DIR = ROOT / "reports" / "build"

# Page geometry in TeX points (A4 210 mm, 2.54 cm margins): used only to choose table column widths.
TEXT_WIDTH_PT = (210.0 - 2 * 25.4) / 25.4 * 72.27
TABLE_STEPS = ((r"\small", 10.0, 6.0), (r"\small", 10.0, 4.0), (r"\footnotesize", 9.0, 4.0),
               (r"\footnotesize", 9.0, 3.0))     # (font size, size in pt, \tabcolsep in pt), tried in order
LONG_TABLE_ROWS = 15     # a captioned table is kept whole (with its caption) up to this many body rows
WIDE_SOURCE = 72         # pandoc's rule: dash widths count only for tables wider than this in the source
MAXDIMEN_PT = 16000.0    # \pagegoal of an empty page is \maxdimen = 16383.99998pt


# ---- reading the Markdown ---------------------------------------------------------------------------

_FENCE_OR_COMMENT = re.compile(r"(^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\2[`~]*[ \t]*$)|([ \t]*<!--.*?-->)",
                               re.S | re.M)


def normalise(text: str) -> str:
    """Unix newlines, no BOM, tabs expanded to 4 columns."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.expandtabs(4) for line in text.split("\n"))


def strip_comments(text: str, warnings: list[str] | None = None) -> str:
    """Drop HTML comments (<!-- ... -->, also over several lines) outside fenced code blocks."""
    out = _FENCE_OR_COMMENT.sub(lambda m: m.group(1) if m.group(1) else "", text)
    # an unclosed comment hides the rest of the file (as on GitHub)
    k = -1
    for m in re.finditer(r"<!--", out):
        before = out[:m.start()]
        if before.count("```") % 2 == 0 and before.count("~~~") % 2 == 0:
            k = m.start()
            break
    if k >= 0:
        if warnings is not None:
            line = out[:k].count("\n") + 1
            warnings.append(f"unclosed HTML comment at line {line}: the rest of the file is not printed")
        out = out[:k]
    return out


# ---- blocks -------------------------------------------------------------------------------------------

@dataclass
class Block:
    kind: str                       # heading para table image list quote hr code group
    level: int = 0                  # heading level
    text: str = ""                  # heading text, code block content
    lines: list = field(default_factory=list)      # paragraph lines
    hard_breaks: bool = False       # paragraph: keep every line break (title block)
    children: list = field(default_factory=list)   # quote/group: blocks; list: one block list per item
    ordered: bool = False
    start: int = 1
    delim: str = "-"
    tight: bool = True
    header: list = field(default_factory=list)     # table
    aligns: list = field(default_factory=list)
    dashes: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    wide: bool = False
    nodes: list = field(default_factory=list)      # image: the inline image node


_ATX = re.compile(r"^ {0,3}(#{1,6})(?=[ \t]|$)(.*)$")
_HR = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_QUOTE = re.compile(r"^ {0,3}> ?")
_FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_LIST = re.compile(r"^( *)([-+*]|(\d{1,9})([.)]))([ \t]+|$)")
_DELIM_CELL = re.compile(r"^:?-+:?$")
_PIPE = re.compile(r"(?<!\\)\|")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def heading_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", text)       # optional closing #s
    return text.strip()


def list_marker(line: str):
    m = _LIST.match(line)
    if not m:
        return None
    indent, marker = len(m.group(1)), m.group(2)
    content = line[m.end():]
    spaces = len(m.group(5)) if content.strip() else 1
    if spaces > 4:
        spaces = 1
    ordered = m.group(3) is not None
    return {"indent": indent, "marker": marker, "ordered": ordered,
            "number": int(m.group(3)) if ordered else 0,
            "delim": m.group(4) if ordered else marker,
            "col": indent + len(marker) + spaces, "empty": not content.strip()}


def _can_interrupt(lm) -> bool:
    """CommonMark: a list item may interrupt a paragraph only if it is not empty and, if ordered,
    starts with 1 (so a wrapped line such as '2003) with ...' stays in the paragraph)."""
    return not lm["empty"] and (not lm["ordered"] or lm["number"] == 1)


def split_row(line: str) -> list[str]:
    """Cells of a pipe-table row; '\\|' is a literal pipe (kept escaped for the inline parser)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, i = [], [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("\\|")
            i += 2
            continue
        if s[i] == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(s[i])
        i += 1
    cells.append("".join(cur).strip())
    return cells


def delimiter_cells(line: str) -> list[str] | None:
    if "-" not in line or not _PIPE.search(line):
        return None
    cells = [c.replace(" ", "") for c in split_row(line)]
    if cells and all(_DELIM_CELL.match(c) for c in cells):
        return cells
    return None


def is_table_start(lines: list[str], i: int) -> bool:
    return (i + 1 < len(lines) and bool(_PIPE.search(lines[i]))
            and delimiter_cells(lines[i]) is None and delimiter_cells(lines[i + 1]) is not None)


def starts_block(line: str) -> bool:
    """Lines that end a paragraph / lazy continuation (tables need a look-ahead and are checked apart)."""
    if _ATX.match(line) or _HR.match(line) or _QUOTE.match(line) or _FENCE_OPEN.match(line):
        return True
    lm = list_marker(line)
    return bool(lm and lm["indent"] <= 3 and _can_interrupt(lm))


def parse_table(lines: list[str], i: int, warnings: list[str]) -> tuple[Block, int]:
    header = split_row(lines[i])
    delims = delimiter_cells(lines[i + 1]) or []
    n = len(header)
    if len(delims) != n:
        warnings.append(f"table header has {n} cells but its separator row {len(delims)}: "
                        f"'{lines[i].strip()[:50]}'")
        delims = (delims + ["---"] * n)[:n]
    aligns, dashes = [], []
    for d in delims:
        left, right = d.startswith(":"), d.endswith(":")
        aligns.append("center" if left and right else "right" if right else "left")
        dashes.append(max(1, len(d.strip(":"))))
    raw = [lines[i], lines[i + 1]]
    rows = []
    i += 2
    while i < len(lines) and lines[i].strip() and _PIPE.search(lines[i]) and not starts_block(lines[i]):
        rows.append((split_row(lines[i]) + [""] * n)[:n])
        raw.append(lines[i])
        i += 1
    wide = any(len(r.rstrip()) > WIDE_SOURCE for r in raw)
    return Block("table", header=header, aligns=aligns, dashes=dashes, rows=rows, wide=wide), i


def parse_list(lines: list[str], i: int, warnings: list[str]) -> tuple[Block, int]:
    first = list_marker(lines[i])
    ordered, delim = first["ordered"], first["delim"]
    n = len(lines)
    items: list[list[str]] = []
    tight = True
    while i < n:
        lm = list_marker(lines[i])
        if not lm or lm["ordered"] != ordered or lm["delim"] != delim:
            break
        col, nest_at = lm["col"], min(lm["col"], lm["indent"] + 2)
        item = [lines[i][lm["indent"] + len(lm["marker"]):].strip()]
        i += 1
        while i < n:
            line = lines[i]
            if not line.strip():
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and _indent(lines[j]) >= nest_at:
                    item.extend([""] * (j - i))
                    tight = False
                    i = j
                    continue
                break
            ind = _indent(line)
            if ind >= nest_at:
                item.append(line[min(ind, col):])
                i += 1
                continue
            nm = list_marker(line)
            if nm and ((nm["ordered"] == ordered and nm["delim"] == delim) or _can_interrupt(nm)):
                break
            if starts_block(line) or is_table_start(lines, i):
                break
            if item and item[-1].strip():          # lazy continuation of the item's paragraph
                item.append(line.strip())
                i += 1
                continue
            break
        while item and not item[-1].strip():
            item.pop()
        items.append(item)
        if i < n and not lines[i].strip():          # blank line(s), then maybe the next item
            j = i
            while j < n and not lines[j].strip():
                j += 1
            nm = list_marker(lines[j]) if j < n else None
            if nm and nm["ordered"] == ordered and nm["delim"] == delim and nm["indent"] <= first["indent"] + 3:
                tight = False
                i = j
                continue
            break
    children = [parse_blocks(item, warnings) for item in items]
    return Block("list", ordered=ordered, start=first["number"] if ordered else 1, delim=delim,
                 tight=tight, children=children), i


def parse_blocks(lines: list[str], warnings: list[str]) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append(Block("para", lines=list(para)))
            para.clear()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
            continue
        m = _FENCE_OPEN.match(line)
        if m and not (m.group(2)[0] == "`" and "`" in m.group(3)):
            flush()
            fence, indent, lang = m.group(2), len(m.group(1)), m.group(3).strip()
            close = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
            body = []
            i += 1
            while i < n and not close.match(lines[i]):
                body.append(lines[i][min(indent, _indent(lines[i])):])
                i += 1
            i += 1
            blocks.append(Block("code", text="\n".join(body), lines=[lang]))
            continue
        if para and _SETEXT.match(line):
            level = 1 if line.strip()[0] == "=" else 2
            blocks.append(Block("heading", level=level, text=" ".join(s.strip() for s in para)))
            para.clear()
            i += 1
            continue
        m = _ATX.match(line)
        if m:
            flush()
            blocks.append(Block("heading", level=len(m.group(1)), text=heading_text(m.group(2))))
            i += 1
            continue
        if _HR.match(line):
            flush()
            blocks.append(Block("hr"))
            i += 1
            continue
        if _QUOTE.match(line):
            flush()
            quoted = []
            while i < n:
                if _QUOTE.match(lines[i]):
                    quoted.append(_QUOTE.sub("", lines[i], count=1))
                elif lines[i].strip() and quoted and quoted[-1].strip() and not starts_block(lines[i]):
                    quoted.append(lines[i])            # lazy continuation
                else:
                    break
                i += 1
            blocks.append(Block("quote", children=parse_blocks(quoted, warnings)))
            continue
        if is_table_start(lines, i):
            flush()
            table, i = parse_table(lines, i, warnings)
            blocks.append(table)
            continue
        lm = list_marker(line)
        if lm and lm["indent"] <= 3 and (not para or _can_interrupt(lm)):
            flush()
            lst, i = parse_list(lines, i, warnings)
            blocks.append(lst)
            continue
        para.append(line)
        i += 1
    flush()
    return blocks


# ---- inline -------------------------------------------------------------------------------------------
# Nodes: ["text", s] ["code", s] ["soft"] ["br"] ["emph", kids] ["strong", kids] ["sup", kids]
# ["sub", kids] ["link", kids, url] ["url", target, shown] ["image", alt, src, attrs]

ASCII_PUNCT = frozenset(string.punctuation)
_AUTOLINK = re.compile(r"<([A-Za-z][A-Za-z0-9+.\-]{1,31}:[^\s<>]*)>")
_EMAIL = re.compile(r"<([A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)>")
_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:\s[^<>]*?)?)\s*(/?)>")
_ATTR = re.compile(r"""([A-Za-z][\w:-]*)\s*=\s*("[^"]*"|'[^']*'|[^\s"'>}]+)""")
_ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_BARE_URL = re.compile(r"(?:https?://|www\.)[^\s<]+")
HTML_TAGS = {"a", "b", "big", "br", "center", "code", "del", "details", "div", "em", "font", "h1", "h2", "h3",
             "h4", "h5", "h6", "hr", "i", "img", "kbd", "li", "ol", "p", "s", "small", "span", "strike",
             "strong", "sub", "summary", "sup", "table", "tbody", "td", "th", "thead", "tr", "u", "ul"}
FORMAT_TAGS = {"b", "big", "br", "code", "del", "em", "i", "kbd", "s", "small", "strike", "strong", "sub", "sup",
               "u"}                         # recognised only without attributes


class Delim:
    """A run of * or _ while emphasis is being matched (CommonMark's delimiter run)."""
    __slots__ = ("ch", "count", "orig", "can_open", "can_close")

    def __init__(self, ch: str, count: int, can_open: bool, can_close: bool):
        self.ch, self.count, self.orig = ch, count, count
        self.can_open, self.can_close = can_open, can_close


def _is_space(ch: str) -> bool:
    return ch == "" or ch.isspace()


def _is_punct(ch: str) -> bool:
    return bool(ch) and (ch in ASCII_PUNCT or unicodedata.category(ch)[0] in "PS")


def _backtick_close(s: str, j: int, run: int) -> int:
    k, n = j, len(s)
    while k < n:
        if s[k] == "`":
            e = k
            while e < n and s[e] == "`":
                e += 1
            if e - k == run:
                return k
            k = e
        else:
            k += 1
    return -1


def _bracket_close(s: str, i: int) -> int:
    depth, k, n = 0, i, len(s)
    while k < n:
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == "`":
            e = k
            while e < n and s[e] == "`":
                e += 1
            close = _backtick_close(s, e, e - k)
            k = close + (e - k) if close >= 0 else e
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def _link_tail(s: str, k: int):
    """Parse '(dest "title")' starting at s[k] == '('. -> (dest, title, end) or None."""
    n = len(s)
    k += 1
    while k < n and s[k] in " \t\n":
        k += 1
    if k < n and s[k] == "<":
        e = s.find(">", k + 1)
        if e < 0:
            return None
        dest, k = s[k + 1:e], e + 1
    else:
        start, depth = k, 0
        while k < n:
            c = s[k]
            if c == "\\" and k + 1 < n:
                k += 2
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    break
                depth -= 1
            elif c.isspace():
                break
            k += 1
        dest = s[start:k]
    while k < n and s[k] in " \t\n":
        k += 1
    title = ""
    if k < n and s[k] in "\"'(":
        close = {"\"": "\"", "'": "'", "(": ")"}[s[k]]
        e = k + 1
        while e < n and s[e] != close:
            e += 2 if s[e] == "\\" else 1
        if e >= n:
            return None
        title, k = s[k + 1:e], e + 1
        while k < n and s[k] in " \t\n":
            k += 1
    if k < n and s[k] == ")":
        return re.sub(r"\\([%s])" % re.escape(string.punctuation), r"\1", dest), title, k + 1
    return None


def _link(s: str, i: int):
    """'[text](dest "title")' at s[i] == '['. -> (text, dest, title, end) or None."""
    close = _bracket_close(s, i)
    if close < 0 or close + 1 >= len(s) or s[close + 1] != "(":
        return None
    tail = _link_tail(s, close + 1)
    if tail is None:
        return None
    dest, title, end = tail
    return s[i + 1:close], dest, title, end


def _attr_block(s: str, k: int) -> tuple[dict, int]:
    """'{width=60% height=5cm}' right after an image -> attributes."""
    if k < len(s) and s[k] == "{":
        e = s.find("}", k)
        if e > 0:
            return {m.group(1).lower(): m.group(2).strip("\"'") for m in _ATTR.finditer(s[k + 1:e])}, e + 1
    return {}, k


def _smart_quote(s: str, i: int) -> str:
    k = i - 1
    while k >= 0 and s[k] in "*_":
        k -= 1
    prev = s[k] if k >= 0 else ""
    k = i + 1
    while k < len(s) and s[k] in "*_":
        k += 1
    nxt = s[k] if k < len(s) else ""
    opening = (prev == "" or prev.isspace() or prev in "([{<-/\u2013\u2014\u201c\u2018") and not _is_space(nxt)
    if s[i] == '"':
        return "\u201c" if opening else "\u201d"
    if prev.isalnum():
        return "\u2019"                        # apostrophe: don't, team's
    return "\u2018" if opening else "\u2019"


def _trim_url(url: str) -> str:
    while url and url[-1] in "?!.,:;*_~'\"":
        url = url[:-1]
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url


def parse_inline(s: str, warnings: list[str] | None = None, hard_breaks: bool = False) -> list:
    """Markdown inline text -> node list (see the comment above Delim)."""
    warnings = warnings if warnings is not None else []
    nodes: list = []
    buf: list[str] = []

    def flush():
        if buf:
            nodes.append(["text", "".join(buf)])
            buf.clear()

    def warn(msg: str):
        if msg not in warnings:
            warnings.append(msg)

    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt == "\n":
                flush()
                nodes.append(["br"])
                i += 2
                while i < n and s[i] in " \t":
                    i += 1
                continue
            if nxt and nxt in ASCII_PUNCT:
                buf.append(nxt)
                i += 2
                continue
            buf.append("\\")
            i += 1
            continue
        if ch == "`":
            j = i
            while j < n and s[j] == "`":
                j += 1
            close = _backtick_close(s, j, j - i)
            if close >= 0:
                flush()
                code = s[j:close].replace("\n", " ")
                if len(code) >= 2 and code[0] == " " and code[-1] == " " and code.strip():
                    code = code[1:-1]
                nodes.append(["code", code])
                i = close + (j - i)
                continue
            buf.append(s[i:j])
            i = j
            continue
        if ch == "!" and s.startswith("[", i + 1):
            link = _link(s, i + 1)
            if link:
                alt, dest, title, end = link
                attrs, end = _attr_block(s, end)
                if title:
                    attrs.setdefault("title", title)
                flush()
                nodes.append(["image", alt, dest, attrs])
                i = end
                continue
        if ch == "[":
            link = _link(s, i)
            if link:
                text, dest, _title, end = link
                flush()
                nodes.append(["link", parse_inline(text, warnings), dest])
                i = end
                continue
        if ch == "<":
            m = _AUTOLINK.match(s, i)
            if m:
                flush()
                nodes.append(["url", m.group(1), m.group(1)])
                i = m.end()
                continue
            m = _EMAIL.match(s, i)
            if m:
                flush()
                nodes.append(["url", "mailto:" + m.group(1), m.group(1)])
                i = m.end()
                continue
            m = _TAG.match(s, i)
            if m and m.group(2).lower() in FORMAT_TAGS and (m.group(3) or "").strip():
                m = None                       # '<b and c>' in running text is not a tag
            if m and m.group(2).lower() in HTML_TAGS:
                closing, tag = m.group(1) == "/", m.group(2).lower()
                attrs = {a.group(1).lower(): a.group(2).strip("\"'") for a in _ATTR.finditer(m.group(3) or "")}
                i = m.end()
                if tag == "br":
                    flush()
                    nodes.append(["br"])
                    continue
                if tag == "img" and not closing:
                    flush()
                    nodes.append(["image", attrs.get("alt", ""), attrs.get("src", ""), attrs])
                    continue
                kind = {"sup": "sup", "sub": "sub", "b": "strong", "strong": "strong", "i": "emph",
                        "em": "emph", "code": "code"}.get(tag)
                if kind and not closing:
                    end_tag = re.compile(r"</%s\s*>" % tag, re.I).search(s, i)
                    if end_tag:
                        inner = s[i:end_tag.start()]
                        flush()
                        if kind == "code":
                            nodes.append(["code", html.unescape(inner)])
                        else:
                            nodes.append([kind, parse_inline(inner, warnings)])
                        i = end_tag.end()
                        continue
                warn(f"HTML tag <{'/' if closing else ''}{tag}> dropped (not supported)")
                continue
        if ch == "&":
            m = _ENTITY.match(s, i)
            if m:
                value = html.unescape(m.group(0))
                if value != m.group(0):
                    buf.append(value)
                    i = m.end()
                    continue
        if ch in "*_":
            j = i
            while j < n and s[j] == ch:
                j += 1
            before = s[i - 1] if i > 0 else ""
            after = s[j] if j < n else ""
            left = not _is_space(after) and (not _is_punct(after) or _is_space(before) or _is_punct(before))
            right = not _is_space(before) and (not _is_punct(before) or _is_space(after) or _is_punct(after))
            if ch == "*":
                can_open, can_close = left, right
            else:
                can_open = left and (not right or _is_punct(before))
                can_close = right and (not left or _is_punct(after))
            flush()
            nodes.append(Delim(ch, j - i, can_open, can_close))
            i = j
            continue
        if ch == "\n":
            text = "".join(buf)
            kept = text.rstrip(" ")
            hard = hard_breaks or len(text) - len(kept) >= 2
            buf.clear()
            if kept:
                buf.append(kept)
            flush()
            nodes.append(["br"] if hard else ["soft"])
            i += 1
            while i < n and s[i] in " \t":
                i += 1
            continue
        if ch in "hw" and (i == 0 or s[i - 1] in " \t\n(*_~\"'"):
            m = _BARE_URL.match(s, i)
            if m:
                url = _trim_url(m.group(0))
                if url.startswith("http") or "." in url[4:]:
                    flush()
                    nodes.append(["url", url if url.startswith("http") else "http://" + url, url])
                    i += len(url)
                    continue
        if ch in "\"'":
            buf.append(_smart_quote(s, i))
            i += 1
            continue
        if ch == "-" and s.startswith("--", i):
            j = i
            while j < n and s[j] == "-":
                j += 1
            buf.append({2: "\u2013", 3: "\u2014"}.get(j - i, "-" * (j - i)))
            i = j
            continue
        buf.append(ch)
        i += 1
    flush()
    return process_emphasis(nodes)


def _finalize(nodes: list) -> list:
    """Unmatched delimiter runs become literal text."""
    out = []
    for nd in nodes:
        if isinstance(nd, Delim):
            if nd.count:
                out.append(["text", nd.ch * nd.count])
        else:
            out.append(nd)
    return out


def process_emphasis(nodes: list) -> list:
    """CommonMark's emphasis matching (closer looks back for the nearest compatible opener)."""
    i = 0
    while i < len(nodes):
        closer = nodes[i]
        if not (isinstance(closer, Delim) and closer.can_close and closer.count):
            i += 1
            continue
        at = None
        for j in range(i - 1, -1, -1):
            op = nodes[j]
            if isinstance(op, Delim) and op.ch == closer.ch and op.can_open and op.count:
                rule_of_3 = ((op.can_close or closer.can_open) and (op.orig + closer.orig) % 3 == 0
                             and not (op.orig % 3 == 0 and closer.orig % 3 == 0))
                if not rule_of_3:
                    at = j
                    break
        if at is None:
            i += 1
            continue
        op = nodes[at]
        use = 2 if op.count >= 2 and closer.count >= 2 else 1
        op.count -= use
        closer.count -= use
        nodes[at + 1:i] = [["strong" if use == 2 else "emph", _finalize(nodes[at + 1:i])]]
        i = at + 2
        if op.count == 0:
            del nodes[at]
            i -= 1
        if closer.count == 0:
            del nodes[i]
    return _finalize(nodes)


def plain_text(nodes: list) -> str:
    out = []
    for nd in nodes:
        k = nd[0]
        if k in ("text", "code"):
            out.append(nd[1])
        elif k in ("soft", "br"):
            out.append(" ")
        elif k in ("emph", "strong", "sup", "sub", "link"):
            out.append(plain_text(nd[1]))
        elif k == "url":
            out.append(nd[2])
        elif k == "image":
            out.append(nd[1])
    return "".join(out)


# ---- LaTeX output of inline nodes --------------------------------------------------------------------

TEXT_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
            "%": r"\%", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}", "_": r"\_",
            "[": "{[}", "]": "{]}", "*": "{*}", "\u00a0": "~", "\u00ad": r"\-", "\u200b": r"\allowbreak{}"}
PDF_ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "%": r"\%", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}", "_": r"\_"}


def _char(ch: str, table: dict) -> str:
    e = table.get(ch)
    if e is not None:
        return e
    return "" if unicodedata.category(ch) == "Cc" else ch


# Ties (no line break) in running text: on both sides of an operator or relation that stands between
# spaces ('p = 0.05', '5 × 20', 'A − B', '7 / 8', 'P(≥ 0.999)'), before '%' ('20 %') and before a
# separator ('·', a spaced dash), so that no line starts or ends with one. A signed number ('−0.03',
# '+0.063') may still start a line. A run of tied words is kept below TIE_MAX characters.
TIE_OPS = frozenset("=≈≠≤≥<>±∓×÷−+/~∼→←↔⇒∝≡")
TIE_SEPARATORS = frozenset("·–—-")
TIE_MAX = 40


def _tie_after(tok: str) -> bool:
    return tok[-1] in TIE_OPS or tok == "-"


def _tie_before(tok: str) -> bool:
    core = tok.rstrip(".,;:!?)]}")
    return tok[0] in "%‰" or (len(core) == 1 and (core in TIE_OPS or core in TIE_SEPARATORS))


def tex_text(s: str, ties: bool = False) -> str:
    """Escape running text. Long tokens (run ids, file names) may break after / and _. With ties,
    the spaces next to operators and before % become non-breaking (see TIE_OPS)."""
    out = []
    toks = [t for t in re.split(r"([ \t\n]+)", s) if t]
    run = 0                                       # characters since the last breakable space
    for i, tok in enumerate(toks):
        if tok.strip(" \t\n") == "":
            prev = toks[i - 1] if i > 0 else ""
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            tie = ties and ((prev and _tie_after(prev)) or (nxt and _tie_before(nxt)))
            if tie and run + 1 + len(nxt) <= TIE_MAX:
                out.append("~")
                run += 1
            else:
                out.append(" ")
                run = 0
            continue
        run += len(tok)
        breakable = len(tok) >= 16
        for k, ch in enumerate(tok):
            out.append(_char(ch, TEXT_ESC))
            if breakable and ch in "/_" and 0 < k < len(tok) - 1:
                out.append(r"\allowbreak{}")
    return "".join(out)


def tex_code(s: str) -> str:
    """Inline code in the mono font; long paths may break after / \\ _ - = ; and some dots."""
    out = []
    for k, ch in enumerate(s):
        out.append(_char(ch, TEXT_ESC))
        nxt = s[k + 1] if k + 1 < len(s) else ""
        if not k or not nxt or nxt.isspace():
            continue
        if ch in "/\\_=;" or (ch == "-" and nxt != "-"):
            out.append(r"\allowbreak{}")
        elif ch in ".," and not (s[k - 1].isdigit() and nxt.isdigit()):
            if ch == "." and re.match(r"[A-Za-z0-9]{1,4}(?:$|[\s);:,])", s[k + 1:]):
                continue                            # keep a file extension (.md, .csv, .yaml) with its name
            out.append(r"\allowbreak{}")
    return r"\texttt{" + "".join(out) + "}"


def tex_url(url: str) -> str:
    out = []
    for ch in url:
        if ch in "%#":
            out.append("\\" + ch)
        elif ch in "\\{}^ \"<>|`" or ord(ch) < 32:
            out.append("%%%02X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def pdf_string(s: str) -> str:
    """Text for PDF bookmarks and metadata."""
    return "".join(_char(ch, PDF_ESC) for ch in re.sub(r"\s+", " ", s).strip())


def _merge_text(nodes: list) -> list:
    """Adjacent text and soft line-break nodes -> one text node (a soft break is a space), so that
    tex_text sees the words on both sides of every space."""
    out: list = []
    for nd in nodes:
        if nd[0] in ("text", "soft"):
            s = nd[1] if nd[0] == "text" else " "
            if out and out[-1][0] == "text":
                out[-1] = ["text", out[-1][1] + s]
            else:
                out.append(["text", s])
        else:
            out.append(nd)
    return out


def render_inline(nodes: list, allow_newline: bool = True, ties: bool = True) -> str:
    """Inline nodes -> LaTeX. ties=False in table cells, whose widths are computed word by word."""
    out = []
    prev = None
    for nd in _merge_text(nodes):
        k = nd[0]
        if k == "text":
            out.append(tex_text(nd[1], ties))
        elif k == "code":
            out.append(tex_code(nd[1]))
        elif k == "br":
            if allow_newline and out and prev != "br":
                out.append(r"\newline{}")
            elif not allow_newline:
                out.append(" ")
        elif k in ("emph", "strong", "sup", "sub"):
            cmd = {"emph": r"\emph", "strong": r"\textbf", "sup": r"\textsuperscript", "sub": r"\textsubscript"}[k]
            out.append(cmd + "{" + render_inline(nd[1], allow_newline, ties) + "}")
        elif k == "link":
            text = render_inline(nd[1], allow_newline, ties)
            if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", nd[2]):
                out.append(r"\href{" + tex_url(nd[2]) + "}{" + text + "}")
            else:
                out.append(text)          # relative link or #anchor: means nothing in the PDF
        elif k == "url":
            if nd[1] == nd[2]:
                out.append(r"\url{" + tex_url(nd[1]) + "}")
            else:
                out.append(r"\href{" + tex_url(nd[1]) + r"}{\nolinkurl{" + tex_url(nd[2]) + "}}")
        elif k == "image":
            out.append(tex_text("[" + (nd[1] or "image") + "]"))
        prev = k
    text = "".join(out)
    while text.endswith(r"\newline{}"):
        text = text[:-len(r"\newline{}")]
    return text


# ---- font metrics (table column widths) ---------------------------------------------------------------

def find_font_file(name: str) -> Path | None:
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    if os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts")
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    return None


class FontMetrics:
    """Advance widths of Calibri regular and bold (from the font files, via fontTools); 0.5 em per
    character when they cannot be read. Only used to choose table column widths."""
    MONO_EM = 0.521        # Consolas (0.55 em) scaled to Calibri's x-height, as in the preamble

    def __init__(self, load: bool = True):
        self.widths: dict[str, dict[int, float]] = {}
        if not load:
            return
        try:
            from fontTools.ttLib import TTFont
            for style, fname in (("r", "calibri.ttf"), ("b", "calibrib.ttf")):
                p = find_font_file(fname)
                if p is None:
                    continue
                font = TTFont(str(p), lazy=True)
                upem = font["head"].unitsPerEm
                hmtx = font["hmtx"]
                self.widths[style] = {cp: hmtx[g][0] / upem for cp, g in font.getBestCmap().items()}
                font.close()
        except Exception:                      # no fontTools, unreadable font: use the estimate
            self.widths = {}

    def width(self, text: str, size: float, style: str = "r") -> float:
        if style == "m":
            return len(text) * self.MONO_EM * size
        table = self.widths.get(style) or self.widths.get("r")
        if not table:
            return len(text) * 0.5 * size * (1.06 if style == "b" else 1.0)
        return sum(table.get(ord(ch), 0.5) for ch in text) * size


def _runs(nodes: list, bold: bool = False, out: list | None = None) -> list[tuple[str, str]]:
    """Inline nodes -> (text, style) runs; style r(egular), b(old) or m(ono)."""
    out = [] if out is None else out
    for nd in nodes:
        k = nd[0]
        if k == "text":
            out.append((nd[1], "b" if bold else "r"))
        elif k == "code":
            out.append((nd[1], "m"))
        elif k in ("soft", "br"):
            out.append((" ", "r"))
        elif k == "strong":
            _runs(nd[1], True, out)
        elif k in ("emph", "sup", "sub", "link"):
            _runs(nd[1], bold, out)
        elif k == "url":
            out.append((nd[2], "r"))
        elif k == "image":
            out.append(("[" + nd[1] + "]", "r"))
    return out


def cell_extent(nodes: list, metrics: FontMetrics, size: float) -> tuple[float, float]:
    """(natural width on one line, narrowest width without breaking a word) in pt. Numbers and
    other cells without letters (e.g. '0.056 +- 0.020') are not broken at all."""
    runs = _runs(nodes)
    total = sum(metrics.width(t, size, st) for t, st in runs)
    plain = "".join(t for t, _ in runs)
    if len(plain) <= 24 and not any(ch.isalpha() for ch in plain):
        return total, total
    longest = cur = 0.0
    for t, st in runs:
        for k, piece in enumerate(t.split(" ")):
            if k:
                longest, cur = max(longest, cur), 0.0
            cur += metrics.width(piece, size, st)
    return total, max(longest, cur)


@dataclass
class TableLayout:
    mode: str                     # "natural" (l/c/r columns) or "p" (wrapping columns, full width)
    fractions: list               # p: share of the width available for content, per column
    size_cmd: str
    tabcolsep: float
    fits: bool = True


def _floor_alloc(minw: list[float], weights: list[float], content: float) -> list[float]:
    """Widths proportional to the weights, but none below its minimum (water filling)."""
    n = len(minw)
    fixed: dict[int, float] = {}
    while True:
        free = [j for j in range(n) if j not in fixed]
        rest = content - sum(fixed.values())
        wsum = sum(weights[j] for j in free) or 1.0
        widths = {j: rest * weights[j] / wsum for j in free}
        low = [j for j in free if widths[j] < minw[j]]
        if not low:
            widths.update(fixed)
            return [widths[j] for j in range(n)]
        for j in low:
            fixed[j] = minw[j]


def layout_table(cells: list[list[list]], dashes: list[int], wide: bool, avail: float,
                 metrics: FontMetrics) -> TableLayout:
    """cells: rows (header first) of inline node lists."""
    n = len(dashes)
    hinted = wide and len(set(dashes)) > 1
    for size_cmd, size, sep in TABLE_STEPS:
        ext = [[cell_extent(c, metrics, size) for c in row] for row in cells]
        maxw = [max(r[j][0] for r in ext) for j in range(n)]
        minw = [max(r[j][1] for r in ext) for j in range(n)]
        content = avail - 2 * n * sep
        if hinted:
            if sum(minw) <= content:
                widths = _floor_alloc(minw, [float(d) for d in dashes], content)
                return TableLayout("p", [w / content for w in widths], size_cmd, sep)
            continue
        if sum(maxw) <= content:
            return TableLayout("natural", [], size_cmd, sep)
        if sum(minw) <= content:
            span = sum(maxw) - sum(minw)
            widths = [lo + (hi - lo) * (content - sum(minw)) / span for lo, hi in zip(minw, maxw)]
            return TableLayout("p", [w / content for w in widths], size_cmd, sep)
    total = sum(minw) or 1.0
    return TableLayout("p", [w / total for w in minw], size_cmd, sep, fits=False)


# ---- LaTeX output of blocks ---------------------------------------------------------------------------

class ImageStore:
    """Figures used by the document -> their names in the compile folder (figures/<ascii name>)."""

    def __init__(self):
        self.files: dict[Path, str] = {}

    def add(self, src: Path) -> str:
        src = src.resolve()
        if src in self.files:
            return self.files[src]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", src.stem).strip("-") or "figure"
        ext = src.suffix.lower()
        name, k = f"figures/{stem}{ext}", 2
        while name in self.files.values():
            name, k = f"figures/{stem}-{k}{ext}", k + 1
        self.files[src] = name
        return name

    def copy_to(self, folder: Path) -> None:
        for src, name in self.files.items():
            dest = folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


@dataclass
class Ctx:
    base: Path                    # folder of the Markdown file (relative image paths)
    images: ImageStore
    metrics: FontMetrics
    warnings: list
    avail: float = TEXT_WIDTH_PT  # line width in pt (for table layout)
    nested: bool = False          # inside a list or quote: headings become bold lines
    list_depth: int = 0           # LaTeX allows 4 levels of itemize / enumerate


_CAPTION = re.compile(r"^[\s*_]*(Table|Figure|Fig\.)\s*\d", re.I)
_WIDTH_HINT = re.compile(r"width\s*=\s*([0-9.]+\s*(?:%|cm|mm|in|pt|bp|px)?)", re.I)


def tex_length(value: str | None) -> str | None:
    """'65%', '0.65', '12cm', '400px' -> a LaTeX length (fractions of the line width are capped at 1)."""
    if not value:
        return None
    v = value.strip().strip("\"'").lower().replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%", v)
    if m:
        return f"{min(float(m.group(1)) / 100, 1.0):.3f}\\linewidth"
    m = re.fullmatch(r"(\d*\.\d+|1(?:\.0*)?|0)", v)
    if m:
        return f"{min(float(v), 1.0):.3f}\\linewidth"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(cm|mm|in|pt|bp)", v)
    if m:
        return v
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:px)?", v)
    if m:                                      # pixels at 96 dpi, as in a browser
        return f"{min(float(m.group(1)) * 0.75 / TEXT_WIDTH_PT, 1.0):.3f}\\linewidth"
    return None


def caption_kind(block: Block) -> str | None:
    if block.kind == "para" and block.lines:
        m = _CAPTION.match(block.lines[0])
        if m:
            return "table" if m.group(1).lower() == "table" else "image"
    return None


def _keep_whole(block: Block) -> str | None:
    if block.kind == "table" and len(block.rows) <= LONG_TABLE_ROWS:
        return "table"
    if block.kind == "image":
        return "image"
    return None


def prepare_blocks(blocks: list[Block], warnings: list[str]) -> list[Block]:
    """Paragraphs that are only an image become image blocks; short tables and figures are grouped
    with an adjacent 'Table N' / 'Figure N' caption paragraph so that they stay on one page."""
    out: list[Block] = []
    for b in blocks:
        if b.kind == "para":
            nodes = parse_inline("\n".join(line.strip() for line in b.lines), warnings)
            real = [nd for nd in nodes if not (nd[0] in ("soft", "br") or (nd[0] == "text" and not nd[1].strip()))]
            if len(real) == 1 and real[0][0] == "image":
                out.append(Block("image", nodes=real[0]))
                continue
            if not real:
                continue                     # only dropped HTML tags
        out.append(b)
    grouped: list[Block] = []
    i = 0
    while i < len(out):
        b = out[i]
        kind = _keep_whole(b)
        if kind and i + 1 < len(out) and caption_kind(out[i + 1]) == kind:
            grouped.append(Block("group", children=[b, out[i + 1]]))
            i += 2
            continue
        if kind and grouped and grouped[-1].kind == "para" and caption_kind(grouped[-1]) == kind:
            grouped.append(Block("group", children=[grouped.pop(), b]))
            i += 1
            continue
        grouped.append(b)
        i += 1
    return grouped


def render_heading(b: Block, ctx: Ctx) -> str:
    nodes = parse_inline(b.text, ctx.warnings)
    text = render_inline(nodes, allow_newline=False)
    if ctx.nested:
        return r"\par\textbf{" + text + r"}\par" + "\n"
    cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}.get(b.level, "paragraph")
    return "\\%s{\\texorpdfstring{%s}{%s}}\n" % (cmd, text, pdf_string(plain_text(nodes)))


def _has_image(nodes: list) -> bool:
    return any(nd[0] == "image" or (nd[0] in ("emph", "strong", "sup", "sub", "link") and _has_image(nd[1]))
               for nd in nodes)


def render_para(b: Block, ctx: Ctx) -> str:
    text = "\n".join(line.lstrip() for line in b.lines).rstrip()
    nodes = parse_inline(text, ctx.warnings, hard_breaks=b.hard_breaks)
    if _has_image(nodes):
        msg = "an image inside running text is printed as its alt text; put it on a line of its own"
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)
    if sum(nd[0] == "br" for nd in nodes) >= 2:
        # A block of hard-broken lines (a reference list): one paragraph per line, no space between
        # them, so that a page break never falls inside an entry that wraps (widow/club control), and
        # ragged right, as justified text stretches a line that wraps before a long word with wide gaps.
        segments: list[list] = [[]]
        for nd in nodes:
            if nd[0] == "br":
                segments.append([])
            else:
                segments[-1].append(nd)
        lines = [t for t in (render_inline(seg).strip() for seg in segments) if t]
        if len(lines) >= 2:
            return ("{\\raggedright " + lines[0] + "\\par\\setlength{\\parskip}{0pt}\n"
                    + "\\par\n".join(lines[1:]) + "\\par}\n")
    return render_inline(nodes).strip() + "\n"


def render_image(b: Block, ctx: Ctx) -> str:
    _, alt, src, attrs = b.nodes
    shown = src or "(no src)"
    if not src or re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*://", src):
        ctx.warnings.append(f"figure {shown}: only local image files can be included")
        return r"\mdmissing{" + tex_text(shown) + "}\n"
    path = Path(unquote(src))
    path = path if path.is_absolute() else ctx.base / path
    if not path.is_file():
        ctx.warnings.append(f"figure {src} not found ({C.rel(path)})")
        return r"\mdmissing{" + tex_text(src) + "}\n"
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf"):
        ctx.warnings.append(f"figure {src}: XeLaTeX takes PNG, JPEG or PDF, not {path.suffix}")
        return r"\mdmissing{" + tex_text(src) + "}\n"
    hint = attrs.get("width")
    if not hint:
        m = _WIDTH_HINT.search(attrs.get("title", "") or "")
        hint = m.group(1) if m else None
    width = tex_length(hint)
    if hint and width is None:
        ctx.warnings.append(f"figure {src}: width hint {hint!r} not understood (use e.g. 65%)")
    height = tex_length(attrs.get("height")) if attrs.get("height") else None
    if height and height.endswith(r"\linewidth"):
        height = height.replace(r"\linewidth", r"\textheight")
    width = width or r"\linewidth"
    height = height or r"0.6\textheight"          # cap for very tall images; keeps the aspect ratio
    opts = f"width={width},height={height},keepaspectratio"
    return r"{\centering\includegraphics[" + opts + "]{" + ctx.images.add(path) + r"}\par}" + "\n"


def render_table(b: Block, ctx: Ctx, keep_whole: bool) -> str:
    n = len(b.header)
    cells = [[parse_inline(c, ctx.warnings) for c in b.header]]
    cells += [[parse_inline(c, ctx.warnings) for c in row] for row in b.rows]
    lay = layout_table(cells, b.dashes, b.wide, ctx.avail, ctx.metrics)
    if not lay.fits:
        ctx.warnings.append(f"table '{plain_text(cells[0][0])[:30]}...' is too wide even at 9 pt; "
                            "some cells stick out (shorten them or split the table)")
    if lay.mode == "natural":
        spec = "".join({"right": "r", "center": "c"}.get(a, "l") for a in b.aligns)
    else:
        total = 9990                               # a hair of slack against rounding
        ks = [max(1, int(f * total)) for f in lay.fractions]
        spec = ""
        for a, k in zip(b.aligns, ks):
            ragged = {"right": r"\raggedleft", "center": r"\centering"}.get(a, r"\raggedright")
            spec += (r">{" + ragged + r"\arraybackslash}p{\dimexpr(\linewidth-" + str(2 * n)
                     + r"\tabcolsep)*" + str(k) + "/10000" + r"\relax}")
    newline_ok = lay.mode == "p"

    def row(cs: list) -> str:
        return " & ".join(render_inline(c, allow_newline=newline_ok, ties=False) for c in cs)

    setup = lay.size_cmd + (r"\setlength{\tabcolsep}{%gpt}" % lay.tabcolsep if lay.tabcolsep != 6.0 else "")
    head = row(cells[0]) + r" \\"
    body = [row(c) for c in cells[1:]]
    if keep_whole or ctx.nested:               # with its caption / inside a list: one unbreakable block
        lines = [r"{\centering" + setup, r"\begin{tabular}{" + spec + "}", r"\toprule", head, r"\midrule"]
        lines += [r + r" \\" for r in body]
        lines += [r"\bottomrule", r"\end{tabular}\par}"]
    else:
        lines = ["{" + setup, r"\begin{longtable}[c]{" + spec + "}", r"\toprule", head, r"\midrule",
                 r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot",
                 r"\bottomrule", r"\endlastfoot"]
        for k, r in enumerate(body):              # at least two rows at the top and bottom of each page
            lines.append(r + (r" \\*" if k < 1 or k >= len(body) - 2 else r" \\"))
        lines += [r"\end{longtable}\par}"]
    return "\n".join(lines) + "\n"


def render_list(b: Block, ctx: Ctx, keep_start: bool = False) -> str:
    """keep_start: no page break between the paragraph before the list and its first item."""
    if ctx.list_depth >= 4:                    # deeper levels: plain lines with a marker
        msg = "lists nested deeper than 4 levels are printed as plain lines"
        if msg not in ctx.warnings:
            ctx.warnings.append(msg)
        out = []
        for k, item in enumerate(b.children):
            marker = f"{b.start + k}{b.delim} " if b.ordered else "– "
            out.append(r"\par " + marker + render_blocks(item, ctx).strip())
        return "\n".join(out) + r"\par" + "\n"
    env = "enumerate" if b.ordered else "itemize"
    opts = []
    if b.ordered and b.start != 1:
        opts.append(f"start={b.start}")
    if b.ordered and b.delim == ")":
        opts.append(r"label=\arabic*)")
    if not b.tight:
        opts.append("itemsep=3pt,parsep=3pt")
    if keep_start:
        opts.append("beginpenalty=10000")
    inner = replace(ctx, nested=True, avail=ctx.avail - (21.9 if b.ordered else 16.4), list_depth=ctx.list_depth + 1)
    out = ["\\begin{%s}%s" % (env, "[" + ",".join(opts) + "]" if opts else "")]
    for item in b.children:
        body = render_blocks(item, inner)
        out.append(r"\item " + body.strip() if body.strip() else r"\item\mbox{}")
    out.append("\\end{%s}" % env)
    return "\n".join(out) + "\n"


def render_block(b: Block, ctx: Ctx) -> str:
    if b.kind == "heading":
        return render_heading(b, ctx)
    if b.kind == "para":
        return render_para(b, ctx)
    if b.kind == "image":
        return render_image(b, ctx)
    if b.kind == "table":
        return render_table(b, ctx, keep_whole=False)
    if b.kind == "list":
        return render_list(b, ctx)
    if b.kind == "quote":
        inner = replace(ctx, nested=True, avail=ctx.avail - 54.8)
        return "\\begin{quote}\n" + render_blocks(b.children, inner) + "\\end{quote}\n"
    if b.kind == "hr":
        return r"\par\noindent\hfil\rule{0.5\linewidth}{0.4pt}\hfil\par" + "\n"
    if b.kind == "code":
        body = b.text.rstrip("\n").replace(r"\end{Verbatim}", r"\end {Verbatim}")
        return "\\begin{Verbatim}[fontsize=\\small]\n" + body + "\n\\end{Verbatim}\n"
    if b.kind == "group":
        inner = []
        for child in b.children:
            inner.append(render_table(child, ctx, keep_whole=True) if child.kind == "table"
                         else render_block(child, ctx))
        # [b]: the box's baseline is that of its last line, so the next paragraph keeps its normal distance
        return ("\\par\\noindent\\begin{minipage}[b]{\\linewidth}\\setlength{\\parskip}{4pt plus 1pt minus 1pt}"
                "\\fussy\\setlength{\\emergencystretch}{2em}\n" + "\n".join(inner) + "\\end{minipage}\\par\n")
    return ""


def leads_in(b: Block) -> bool:
    """A paragraph that introduces the next block: it ends with a colon ('... was pre-registered:'),
    or it is a short bold or italic line used as a heading ('**Negative results.**')."""
    if b.kind != "para":
        return False
    nodes = parse_inline("\n".join(line.strip() for line in b.lines).strip())
    text = plain_text(nodes).strip()
    if text.endswith(":"):
        return True
    real = [nd for nd in nodes if not (nd[0] in ("soft", "br") or (nd[0] == "text" and not nd[1].strip()))]
    return len(real) == 1 and real[0][0] in ("strong", "emph") and len(text) <= 80


def render_blocks(blocks: list[Block], ctx: Ctx) -> str:
    """A lead-in paragraph (leads_in) stays on the page of the paragraph, list, table or figure after
    it: no page break before the first item of a list, \\nopagebreak before the others."""
    prepared = prepare_blocks(blocks, ctx.warnings)
    out = []
    for k, b in enumerate(prepared):
        lead = k > 0 and leads_in(prepared[k - 1])
        tex = render_list(b, ctx, keep_start=lead) if b.kind == "list" else render_block(b, ctx)
        nxt = prepared[k + 1] if k + 1 < len(prepared) else None
        if nxt is not None and nxt.kind in ("para", "group", "image", "table") and leads_in(b):
            tex = tex.rstrip("\n") + "\n\\par\\nopagebreak\n"
        out.append(tex)
    return "\n".join(out)


# ---- one part -----------------------------------------------------------------------------------------

@dataclass
class PartResult:
    key: str
    source: Path
    title: str                    # plain text
    latex: str
    warnings: list
    todo: int
    headings: list                # level-2 headings, plain text


def convert_markdown(text: str, base: Path, key: str = "part", images: ImageStore | None = None,
                     metrics: FontMetrics | None = None, source: Path | None = None) -> PartResult:
    """Markdown text of one part -> LaTeX body (title block from the first heading)."""
    warnings: list[str] = []
    text = strip_comments(normalise(text), warnings)
    todo = text.count("TODO")
    blocks = parse_blocks(text.split("\n"), warnings)
    ctx = Ctx(base=base, images=images or ImageStore(), metrics=metrics or FontMetrics(load=False),
              warnings=warnings)
    first = next((k for k, b in enumerate(blocks) if b.kind == "heading"), None)
    headings = [plain_text(parse_inline(b.text)).strip() for b in blocks if b.kind == "heading" and b.level == 2]
    out: list[str] = []
    title = ""
    anchor = "mdpart." + re.sub(r"[^A-Za-z0-9]+", "-", key)
    if first is None:
        warnings.append("no heading: the part has no title")
        out.append(r"\pdfbookmark[0]{" + pdf_string(PARTS[key].label if key in PARTS else key) + "}{" + anchor + "}")
        body = blocks
    else:
        out.append(render_blocks(blocks[:first], ctx))
        nodes = parse_inline(blocks[first].text, warnings)
        title = plain_text(nodes).strip()
        out.append(r"\mdtitle{" + render_inline(nodes, allow_newline=False) + "}{" + pdf_string(title) + "}{"
                   + anchor + "}")
        body = blocks[first + 1:]
        if body and body[0].kind == "para":
            body[0].hard_breaks = True             # team / author lines keep their line breaks
    out.append(render_blocks(body, ctx))
    return PartResult(key=key, source=source or base, title=title, latex="\n".join(s for s in out if s),
                      warnings=warnings, todo=todo, headings=headings)


def convert(text: str, base: Path | None = None) -> tuple[str, list[str]]:
    """Convenience for tests: Markdown -> (LaTeX body, warnings)."""
    res = convert_markdown(text, base or Path.cwd())
    return res.latex, res.warnings


def check_part(res: PartResult) -> list[str]:
    """Content checks of reports/README.md that can be automated."""
    notes = []
    if res.key in EXTENSIONS:
        got = [h.rstrip(" .:") for h in res.headings]
        if got != EXT_HEADINGS:
            notes.append(f"headings are {' / '.join(got) or '(none)'}; expected exactly "
                         f"{' / '.join(EXT_HEADINGS)}")
    return notes


# ---- document, XeLaTeX, page counts -------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
% Generated by scripts/build_report.py from the Markdown parts in reports/. Do not edit this file:
% edit the Markdown and rebuild.
\usepackage[a4paper,margin=2.54cm]{geometry}
\usepackage{fontspec}
% WordSpace: a justified line may squeeze its word spaces by 1/5 of a space, not 1/3 (XeTeX's
% default, down to 0.15 em in Calibri, which looks crushed); Word does not squeeze them at all.
\IfFontExistsTF{Calibri}{\setmainfont{Calibri}[Ligatures=TeXOff,WordSpace={1,1,0.6}]%
  \typeout{MDFONT main=Calibri}}{%
  \IfFontExistsTF{Carlito}{\setmainfont{Carlito}[Ligatures=TeXOff,WordSpace={1,1,0.6}]%
    \typeout{MDFONT main=Carlito}}{\typeout{MDFONT main=LatinModern}}}
\IfFontExistsTF{Consolas}{\setmonofont{Consolas}[Scale=MatchLowercase]\typeout{MDFONT mono=Consolas}}{%
  \typeout{MDFONT mono=LatinModernMono}}
% Copied or searched PDF text = the Markdown text: '-' stays U+002D and the middle dot U+00B7 (without
% this, Calibri's shared glyphs map them back to U+2010 and U+2219), and words hyphenated at a line end
% come out whole.
\ifdefined\XeTeXgenerateactualtext\XeTeXgenerateactualtext=1\fi
\usepackage{graphicx}
\usepackage{array,longtable,booktabs}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor={blue!50!black},
  bookmarksopen=true,bookmarksopenlevel=0,pdftitle={@@TITLE@@},pdfauthor={@@AUTHOR@@},
  pdfcreator={scripts/build\_report.py (XeLaTeX)},pdflang={en}}
\urlstyle{same}
\graphicspath{{./}{../}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{2em}
\frenchspacing
\clubpenalty=10000
\widowpenalty=10000
\raggedbottom
\setcounter{secnumdepth}{-1}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{10pt plus 3pt minus 2pt}{2pt plus 1pt}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{8pt plus 2pt minus 2pt}{1pt plus 1pt}
\titleformat{\subsubsection}{\normalsize\bfseries\itshape}{}{0pt}{}
\titlespacing*{\subsubsection}{0pt}{6pt plus 2pt minus 1pt}{1pt}
\titleformat{\paragraph}[runin]{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\paragraph}{0pt}{4pt plus 1pt}{0.5em}
\setlist{topsep=2pt,partopsep=0pt,itemsep=1pt,parsep=0pt}
\setlist[itemize]{leftmargin=1.5em}
\setlist[itemize,1]{label=\textbullet}
\setlist[itemize,2]{label=\textendash}
\setlist[itemize,3]{label=\textopenbullet}
\setlist[itemize,4]{label=\textperiodcentered}
\setlist[enumerate]{leftmargin=2em,label=\arabic*.}
\setlength{\LTpre}{4pt}
\setlength{\LTpost}{4pt}
\newwrite\mdpages
\immediate\openout\mdpages=\jobname.pgs
% Page marks for build_report.py. The \penalty0 is a legal page break, so TeX settles the page (moves
% material that does not fit to the next page) before \thepage and \pagetotal are read.
\newcommand{\mdpartstart}[1]{\par\immediate\write\mdpages{start #1 \thepage}}
\newcommand{\mdpartend}[1]{\par\penalty0
  \immediate\write\mdpages{end #1 \thepage\space\the\pagetotal\space\the\pagegoal}}
\newcommand{\mdtitle}[3]{\pdfbookmark[0]{#2}{#3}{\Large\bfseries\raggedright #1\par}\vspace{2pt}}
\newcommand{\mdmissing}[1]{{\centering\fbox{\parbox{0.8\linewidth}{\centering\small missing figure: #1}}\par}}
\begin{document}
"""


def assemble(results: list[PartResult], title: str, author: str) -> tuple[str, dict[str, tuple[int, int]]]:
    """-> (.tex source, {part key: (first line, last line)} for mapping log messages to parts)."""
    tex = PREAMBLE.replace("@@TITLE@@", pdf_string(title)).replace("@@AUTHOR@@", pdf_string(author))
    lines = tex.count("\n") + 1
    ranges = {}
    chunks = [tex]
    for k, res in enumerate(results):
        chunk = ("\\clearpage\n" if k else "") + "\\mdpartstart{%s}\n%s\n\\mdpartend{%s}\n" % (
            res.key, res.latex.strip(), res.key)
        ranges[res.key] = (lines, lines + chunk.count("\n"))
        lines += chunk.count("\n")
        chunks.append(chunk)
    chunks.append("\\end{document}\n")
    return "".join(chunks), ranges


@dataclass
class PageCount:
    start: int
    end: int
    pages: int
    used: float                   # pages up to the filled share of the last one, e.g. 1.94


def parse_page_marks(text: str) -> dict[str, PageCount]:
    """Lines 'start <key> <page>' and 'end <key> <page> <pagetotal> <pagegoal>' -> page counts. A part
    that ends exactly at the foot of a page leaves an empty page (\\pagegoal = \\maxdimen) behind."""
    starts: dict[str, int] = {}
    out: dict[str, PageCount] = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "start":
            starts[f[1]] = int(f[2])
        elif len(f) >= 5 and f[0] == "end":
            page, total, goal = int(f[2]), float(f[3].rstrip("pt")), float(f[4].rstrip("pt"))
            if goal >= MAXDIMEN_PT or total <= 0:
                end, share = page - 1, 1.0
            else:
                end, share = page, min(total / goal, 1.0)
            start = starts.get(f[1], end)
            end = max(end, start)
            out[f[1]] = PageCount(start, end, end - start + 1, round(end - start + share, 2))
    return out


def find_xelatex(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    if os.environ.get("SSC_XELATEX") and Path(os.environ["SSC_XELATEX"]).is_file():
        return os.environ["SSC_XELATEX"]
    found = shutil.which("xelatex")
    if found:
        return found
    for drive in ("C:", "D:"):
        for year in range(2030, 2019, -1):
            for base in (rf"{drive}\texlive\{year}", rf"{drive}\Program Files\texlive\{year}"):
                for sub in ("windows", "win32", "win64"):
                    p = Path(base) / "bin" / sub / "xelatex.exe"
                    if p.is_file():
                        return str(p)
    return None


def _ascii(p: Path) -> bool:
    try:
        str(p).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_tmp_base(explicit: str | None = None) -> Path | None:
    """A writable folder with an ASCII path for the XeLaTeX run."""
    if explicit:
        cands = [Path(explicit)]
    else:
        cands = [Path(p) for p in (os.environ.get("SSC_TEX_TMP"), tempfile.gettempdir()) if p]
        cands += [Path(ROOT.anchor) / "tmp", Path("C:/tmp")]
    for c in cands:
        try:
            c = c.resolve()
            if not _ascii(c):
                continue
            c.mkdir(parents=True, exist_ok=True)
            probe = Path(tempfile.mkdtemp(prefix="probe_", dir=c))
            probe.rmdir()
            return c
        except OSError:
            continue
    return None


@dataclass
class CompileResult:
    ok: bool
    work: Path
    log: str
    marks: str


def run_xelatex(tex: str, images: ImageStore, xelatex: str, tmp_base: Path, runs: int = 2) -> CompileResult:
    work = Path(tempfile.mkdtemp(prefix="build_report_", dir=tmp_base))
    images.copy_to(work)
    (work / "report.tex").write_text(tex, encoding="utf-8")
    env = dict(os.environ, max_print_line="10000", error_line="254", half_error_line="238")
    log = ""
    for _ in range(runs):
        try:
            r = subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                                "report.tex"], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, timeout=600)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -1
        log_path = work / "report.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if code != 0 or not (work / "report.pdf").exists():
            return CompileResult(False, work, log, "")
    marks_path = work / "report.pgs"
    marks = marks_path.read_text(encoding="utf-8", errors="replace") if marks_path.exists() else ""
    return CompileResult(True, work, log, marks)


def _part_of(line: int, ranges: dict[str, tuple[int, int]]) -> str:
    for key, (a, b) in ranges.items():
        if a <= line <= b:
            return key
    return "preamble"


def log_findings(log: str, ranges: dict[str, tuple[int, int]]) -> list[str]:
    notes = []
    fonts = dict(re.findall(r"MDFONT (\w+)=(\w+)", log))
    if fonts.get("main", "Calibri") != "Calibri":
        notes.append(f"Calibri not found; the text font is {fonts['main']}, so page counts differ from the "
                     "team's layout")
    missing = sorted(set(re.findall(r"Missing character: There is no (.+?) \(U\+([0-9A-Fa-f]+)\)", log)))
    if missing:
        notes.append("characters missing from the font (printed as blanks): "
                     + ", ".join(f"U+{cp.upper()}" for _, cp in missing))
    over: dict[str, list[float]] = {}
    for amount, a in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\) .*?lines? (\d+)", log):
        if float(amount) >= 1.0:
            over.setdefault(_part_of(int(a), ranges), []).append(float(amount))
    for key, amounts in over.items():
        notes.append(f"{key}: {len(amounts)} line(s) stick out into the margin by up to {max(amounts):.1f} pt "
                     "(long words, paths or table cells; see --keep-tex)")
    for amount, a in re.findall(r"Overfull \\vbox \(([\d.]+)pt too high\) .*?lines? (\d+)", log):
        if float(amount) >= 1.0:
            notes.append(f"{_part_of(int(a), ranges)}: a block is {float(amount):.0f} pt taller than the page")
    return notes


def log_errors(log: str, tex: str) -> list[str]:
    lines = log.splitlines()
    tex_lines = tex.split("\n")
    out = []
    for k, line in enumerate(lines):
        m = re.match(r"^(?:\./)?report\.tex:(\d+): (.*)$", line)
        if m or line.startswith("! "):
            out.append(line.strip())
            out += [s.strip() for s in lines[k + 1:k + 3] if s.strip()]
            if m:
                n = int(m.group(1))
                if 0 < n <= len(tex_lines):
                    out.append(f"tex line {n}: {tex_lines[n - 1][:150]}")
            if len(out) > 12:
                break
    return out or [s for s in lines[-15:] if s.strip()]


# ---- parts and project.yaml ---------------------------------------------------------------------------

def normalise_ext(value) -> str | None:
    """'e1', 'E1', 1, 'e1_validation_protocol' -> 'e1'; anything else -> None."""
    s = str(value).strip().lower()
    if re.fullmatch(r"[1-4]", s):
        s = "e" + s
    m = re.match(r"e([1-4])(?:$|[_\W])", s)
    return f"e{m.group(1)}" if m else None


def parts_from_project(project: dict) -> tuple[list[str], list[str]]:
    """Default parts from project.yaml report.extensions_owned -> (keys, warnings)."""
    warnings = []
    report = project.get("report") if isinstance(project.get("report"), dict) else {}
    owned = report.get("extensions_owned")
    if isinstance(owned, (str, int)):
        owned = [owned]
    if not owned:
        warnings.append("project.yaml report.extensions_owned is empty (owners not decided yet): all four "
                        "extensions are included; list only the owned ones, e.g. [e1, e3]")
        return ["core", *EXTENSIONS, "contrib"], warnings
    exts = []
    for v in owned:
        k = normalise_ext(v)
        if k is None:
            raise ValueError(f"project.yaml report.extensions_owned: {v!r} is not an extension (use e1..e4)")
        if k not in exts:
            exts.append(k)
    members = (project.get("team") or {}).get("members") or []
    declared = {normalise_ext(m.get("extension")) for m in members if isinstance(m, dict)} - {None}
    if declared and declared != set(exts):
        warnings.append(f"report.extensions_owned {sorted(exts)} differs from the extensions of "
                        f"team.members {sorted(declared)} in project.yaml")
    return sorted(["core", *exts, "contrib"], key=ORDER.index), warnings


def parse_parts_arg(text: str) -> list[str]:
    keys = []
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        t = tok.lower()
        k = t if t in PARTS else "contrib" if t.startswith("contrib") else normalise_ext(t)
        if k not in PARTS:
            raise ValueError(f"unknown part {tok!r}; choose from {', '.join(ORDER)}")
        if k not in keys:
            keys.append(k)
    if not keys:
        raise ValueError("--parts is empty")
    return sorted(keys, key=ORDER.index)


def team_name(project: dict) -> str:
    name = str((project.get("team") or {}).get("name") or "").strip()
    return "" if not name or "TODO" in name.upper() else name


def console(s: str) -> str:
    """ASCII for the console (Windows consoles may be GBK or cp1252)."""
    table = {"\u2013": "-", "\u2014": "--", "\u2212": "-", "\u00d7": "x", "\u00b1": "+/-", "\u2018": "'",
             "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ", "\u2264": "<=",
             "\u2265": ">=", "\u2248": "~"}
    s = "".join(table.get(ch, ch) for ch in s)
    return s.encode("ascii", "backslashreplace").decode("ascii")


# ---- main ---------------------------------------------------------------------------------------------

@dataclass
class BuildResult:
    ok: bool
    pdf: Path | None
    parts: list                   # PartResult
    pages: dict                   # key -> PageCount
    total_pages: int
    warnings: list                # strings, already prefixed with the part key where it applies
    work: Path | None = None


def build(sources: list[tuple[str, Path]], out: Path, *, xelatex: str, tmp_base: Path,
          title: str = "", author: str = "", keep_tex: Path | None = None, runs: int = 2,
          metrics: FontMetrics | None = None) -> BuildResult:
    metrics = metrics or FontMetrics()
    images = ImageStore()
    results = []
    warnings: list[str] = []
    for key, path in sources:
        res = convert_markdown(path.read_text(encoding="utf-8-sig"), path.parent, key, images, metrics, path)
        results.append(res)
        warnings += [f"{key}: {w}" for w in res.warnings + check_part(res)]
    doc_title = title or next((r.title for r in results if r.key == "core"), "") or (results[0].title if results else "")
    tex, ranges = assemble(results, doc_title, author)
    if keep_tex is not None:
        keep_tex.parent.mkdir(parents=True, exist_ok=True)
        keep_tex.write_text(tex, encoding="utf-8")
    comp = run_xelatex(tex, images, xelatex, tmp_base, runs)
    if keep_tex is not None and comp.log:
        keep_tex.with_suffix(".log").write_text(comp.log, encoding="utf-8")
    if not comp.ok:
        return BuildResult(False, None, results, {}, 0, warnings + log_errors(comp.log, tex), comp.work)
    pages = parse_page_marks(comp.marks)
    m = re.search(r"Output written on report\.pdf \((\d+) pages?", comp.log)
    total = int(m.group(1)) if m else max((p.end for p in pages.values()), default=0)
    warnings += log_findings(comp.log, ranges)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(comp.work / "report.pdf", out)
    except OSError as e:
        warnings.append(f"cannot write {out} ({e.strerror or e}; open in a PDF viewer?); the PDF is in {comp.work}")
        return BuildResult(False, None, results, pages, total, warnings, comp.work)
    shutil.rmtree(comp.work, ignore_errors=True)
    return BuildResult(True, out, results, pages, total, warnings, None)


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", help="comma-separated parts in any order: core,e1,e2,e3,e4,contrib "
                                    "(default: core + project.yaml report.extensions_owned + contrib)")
    ap.add_argument("--out", help=r"output PDF (default reports\report.pdf)")
    ap.add_argument("--keep-tex", action="store_true",
                    help="also write the .tex and the XeLaTeX log to reports/build/")
    ap.add_argument("--tmp-dir", help="ASCII folder for the XeLaTeX run (default: $SSC_TEX_TMP, the system "
                                      r"temp folder or <drive>:\tmp)")
    ap.add_argument("--xelatex", help="path of xelatex.exe (default: $SSC_XELATEX, then PATH)")
    args = ap.parse_args()

    project = C.load_project()
    notes: list[str] = []
    try:
        if args.parts:
            keys = parse_parts_arg(args.parts)
        else:
            keys, notes = parts_from_project(project)
    except ValueError as e:
        C.die(str(e), 2)
    sources = [(k, ROOT / PARTS[k].path) for k in keys]
    missing = [C.rel(p) for _, p in sources if not p.is_file()]
    if missing:
        C.die(f"missing part file(s): {', '.join(missing)}", 2)
    out = Path(args.out).resolve() if args.out else DEFAULT_OUT
    xelatex = find_xelatex(args.xelatex)
    if not xelatex:
        C.die(f"--xelatex {args.xelatex} is not a file" if args.xelatex else
              "xelatex not found: install TeX Live (or MiKTeX) or pass --xelatex <path>", 2)
    tmp_base = ascii_tmp_base(args.tmp_dir)
    if tmp_base is None:
        C.die(f"--tmp-dir must be a writable folder with an ASCII path (XeLaTeX), not {console(args.tmp_dir)}"
              if args.tmp_dir else
              "no writable folder with an ASCII path for XeLaTeX: pass --tmp-dir C:\\tmp (or set $SSC_TEX_TMP)", 2)

    keep = BUILD_DIR / (out.stem + ".tex") if args.keep_tex else None
    print(f"Building {C.rel(out)} from {', '.join(keys)} (XeLaTeX, Calibri 11 pt, A4, 2.54 cm margins)")
    res = build(sources, out, xelatex=xelatex, tmp_base=tmp_base, author=team_name(project), keep_tex=keep)
    if not res.ok and res.pdf is None and not res.pages:
        for line in res.warnings:
            print(console(line))
        if keep:
            print(f"LaTeX source and log: {C.rel(keep)}, {C.rel(keep.with_suffix('.log'))}")
        C.die(f"XeLaTeX failed; its files are in {res.work}", 1)

    print(f"\n{'part':<8} {'pages':>5} {'used':>5} {'limit':>5}  {'pp.':<7} title")
    over = []
    for r in res.parts:
        pc = res.pages.get(r.key)
        spec = PARTS.get(r.key)
        limit = spec.limit if spec else 0
        if pc is None:
            print(f"{r.key:<8} {'?':>5}")
            continue
        flag = "  OVER" if limit and pc.pages > limit else ""
        if flag:
            over.append(f"{r.key} has {pc.pages} pages (limit {limit}; {pc.used:.2f} pages used)")
        pp = f"{pc.start}-{pc.end}" if pc.end > pc.start else str(pc.start)
        print(f"{r.key:<8} {pc.pages:>5} {pc.used:>5.2f} {limit:>5}  {pp:<7} {console(r.title)[:60]}{flag}")
    print(f"total    {res.total_pages:>5}")

    warnings = notes + over + res.warnings
    todo = [f"{r.key} {r.todo}" for r in res.parts if r.todo]
    if todo:
        warnings.append("TODO left: " + ", ".join(todo))
    try:
        full = parts_from_project(project)[0]
    except ValueError:
        full = keys
    if args.parts and out == DEFAULT_OUT and keys != full:
        warnings.append(f"{C.rel(out)} now holds only {', '.join(keys)}; rebuild without --parts before packaging")
    elif out != DEFAULT_OUT and (ROOT / "reports") in out.parents and "reflections" not in out.parts:
        warnings.append(f"{C.rel(out)} is under reports/: package_deliverables.py will put it into the zip")
    for w in warnings:
        print("WARNING: " + console(w))
    if keep:
        print(f"LaTeX source: {C.rel(keep)} (log: {C.rel(keep.with_suffix('.log'))})")
    if not res.ok:
        C.die(f"the PDF was not written to {C.rel(out)}", 1)
    print(f"Wrote {C.rel(out)} ({res.total_pages} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
