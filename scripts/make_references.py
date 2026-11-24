"""Write reports/references.md: one complete, alphabetical reference list for the whole report.

    .venv\\Scripts\\python.exe scripts\\make_references.py            # write reports/references.md
    .venv\\Scripts\\python.exe scripts\\make_references.py --check    # check only, write nothing

The entries come from registry/references.md (BibTeX inside Markdown): every entry with status "used"
(course rule 4: everything used must be cited in the report) and every other entry that a report part
cites. The parts read for citations are the core report and the four extensions; build_report.py
places the list after the extensions (part "refs").

Checks, printed as warnings: an author-year citation in a part that matches no registry entry, and a
listed entry that no part cites (an academic reference list should contain cited works only).
With --check the exit code is 1 if there is any warning.

Style (author-year, complete entries): surnames and initials of up to six authors, then "et al.";
year; title as in the registry; journal or book in italics, volume in italics with the issue in
parentheses, pages; "In <proceedings or book> (pp. ...)" for papers in proceedings or collections;
edition and publisher for books; "Technical report" and institution for reports; the arXiv identifier
or URL for preprints and software; the DOI where the registry has one. Sorted by the first author's
surname; with the same first author, single-author works first, then by year.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import _common as C
from ssc import paths

ROOT = paths.ROOT
REGISTRY = ROOT / "registry" / "references.md"
OUT = ROOT / "reports" / "references.md"
PARTS = [ROOT / "reports" / "report_core.md"] + [
    ROOT / "reports" / "extensions" / f for f in
    ("e1_validation_protocol.md", "e2_learning_curve.md", "e3_leakage.md", "e4_feature_stability.md")]
MAX_AUTHORS = 6
BS = "\\"
ACCENTS = {('"', "u"): "ü", ('"', "o"): "ö", ('"', "a"): "ä", ('"', "U"): "Ü", ('"', "O"): "Ö",
           ("'", "e"): "é", ("'", "a"): "á", ("'", "o"): "ó", ("`", "e"): "è", ("^", "o"): "ô", ("~", "n"): "ñ"}


# ---------------------------------------------------------------- reading the registry
def parse_bibtex(text: str) -> list[dict]:
    """Entries of a BibTeX text that may sit inside Markdown; '% status: ...' comments are kept as 'status'."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        typ, key = m.group(1).lower(), m.group(2)
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        raw = text[m.end():i - 1]
        status = re.search(r"status:\s*([\w-]+)", text[m.start():i])
        body = re.sub(r"(?<!\\)%[^\n]*", "", raw)                      # drop BibTeX comments
        fields, j = {}, 0
        field_re = re.compile(r"\s*,?\s*(\w+)\s*=\s*")
        while True:
            fm = field_re.search(body, j)
            if not fm:
                break
            name, k = fm.group(1).lower(), fm.end()
            if k < len(body) and body[k] == "{":
                d, s = 1, k + 1
                while d and s < len(body):
                    d += {"{": 1, "}": -1}.get(body[s], 0)
                    s += 1
                val, j = body[k + 1:s - 1], s
            elif k < len(body) and body[k] == '"':
                s = body.index('"', k + 1)
                val, j = body[k + 1:s], s + 1
            else:
                end = re.compile(r"[,\n}]").search(body, k)
                s = end.start() if end else len(body)
                val, j = body[k:s].strip(), s
            fields[name] = re.sub(r"\s+", " ", val).strip()
        entries.append({"type": typ, "key": key, "status": status.group(1) if status else "", **fields})
    return entries


def clean(s: str) -> str:
    """BibTeX markup -> plain Unicode text (accents, en dashes, URLs, protected capitals)."""
    s = s.replace("$b_2$", "b<sub>2</sub>").replace("$" + BS + "sqrt{b_1}$", "√b<sub>1</sub>")
    s = s.replace("$" + BS + "widehat{R}$", "R̂")
    s = re.sub(re.escape(BS) + r"([\"'`^~])\{?([A-Za-z])\}?",
               lambda m: ACCENTS.get((m.group(1), m.group(2)), m.group(2)), s)
    s = re.sub(re.escape(BS) + r"url\{([^}]*)\}", r"\1", s)
    s = s.replace(BS + "&", "&").replace("--", "–").replace("~", " ")
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def author_list(field: str) -> tuple[list[str], bool]:
    names = [n.strip() for n in re.split(r"\s+and\s+", clean(field)) if n.strip()]
    others = bool(names) and names[-1].lower() == "others"
    return (names[:-1] if others else names), others


def surname_key(name: str) -> str:
    """'D'Agostino, R.' -> 'dagostino'; 'Bühlmann' -> 'buhlmann' (for sorting and matching citations)."""
    s = name.split(",")[0]
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


# ---------------------------------------------------------------- formatting
def fmt_authors(field: str) -> str:
    names, others = author_list(field)
    if len(names) > MAX_AUTHORS:
        names, others = names[:MAX_AUTHORS], True
    if others:
        return ", ".join(names) + ", et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, & {names[1]}"
    return ", ".join(names[:-1]) + f", & {names[-1]}"


def _stop(s: str) -> str:
    return s if s.endswith((".", "?", "!")) else s + "."


MINOR = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "as", "at", "by", "in", "of", "off",
         "on", "per", "to", "up", "via", "vs"}


def title_case(title: str) -> str:
    """One capitalisation for every title (the registry mixes sentence and title case): the first word, the word
    after a colon and every word that is not a short minor word start with a capital; words that already carry an
    inner capital, a digit or markup (LIBSVM, NumPy, K-Fold, b<sub>2</sub>) are left as they are."""
    words, out, start = title.split(" "), [], True
    for w in words:
        core = w.lstrip("(\"'‘“")
        lead = w[:len(w) - len(core)]
        if not core or "<" in core or any(c.isdigit() for c in core) or any(c.isupper() for c in core[1:]):
            out.append(w)
        elif not start and core.lower().rstrip(".,:;?!") in MINOR:
            out.append(lead + core.lower())
        else:
            out.append(lead + core[0].upper() + core[1:])
        start = w.endswith((":", ".", "?", "!"))
    return " ".join(out)


def _link(url: str) -> str:
    """A URL as a Markdown link, so that the hard line break after it is not taken into the URL."""
    return f"[{url}]({url})"


def format_entry(e: dict) -> str:
    t = e["type"]
    head = f"{fmt_authors(e.get('author', ''))} ({e.get('year', 'n.d.')})."
    title = title_case(clean(e.get("title", "")))
    vol, no, pages = e.get("volume"), e.get("number"), clean(e.get("pages", ""))
    if t == "article":
        s = f"{head} {_stop(title)} *{clean(e.get('journal', ''))}*"
        if vol:
            s += f", *{vol}*" + (f"({no})" if no else "")
        if pages:
            s += f", {pages}"
        s += "."
    elif t in ("inproceedings", "incollection"):
        s = f"{head} {_stop(title)} In *{clean(e.get('booktitle', ''))}*" + (f" (pp. {pages})" if pages else "") + "."
        if e.get("publisher"):
            s += f" {clean(e['publisher'])}."
    elif t == "book":
        s = f"{head} *{title}*" + (f" ({clean(e['edition'])} ed.)" if e.get("edition") else "") + "."
        if e.get("publisher"):
            s += f" {clean(e['publisher'])}."
    elif t == "techreport":
        s = f"{head} *{title}* (Technical report). {clean(e.get('institution', ''))}."
    else:                                                               # misc: preprints, software
        s = f"{head} *{title}*."
        how = clean(e.get("howpublished", ""))
        if how.startswith("http"):
            s += " " + _link(how)
        elif how:
            s += f" {_stop(how)}"
    if e.get("doi"):
        s += " " + _link(f"https://doi.org/{clean(e['doi'])}")
    return s


def sort_key(e: dict):
    names, others = author_list(e.get("author", ""))
    return (surname_key(names[0]) if names else e["key"], len(names) > 1 or others, e.get("year", ""), e["key"])


# ---------------------------------------------------------------- citations in the parts
NAME = r"[A-Z][A-Za-z'’À-ſ-]+"
CITE = re.compile(rf"\b({NAME})(?:\s+et\s+al\.)?(?:\s*(?:&|and)\s*{NAME}(?:\s{NAME})?)?(?:,\s*|\s+\()((?:19|20)\d{{2}})\b")


def citations(text: str) -> set[tuple[str, str]]:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"</?(?:sub|sup)>|[*_`]|&nbsp;", " ", text).replace(" ", " ")
    return {(surname_key(m.group(1)), m.group(2)) for m in CITE.finditer(text)}


def select(entries: list[dict], cited: set[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    index: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        names, _ = author_list(e.get("author", ""))
        if names and e.get("year"):
            index.setdefault((surname_key(names[0]), e["year"]), []).append(e)
    warnings = [f"citation {s} {y} matches no entry of registry/references.md"
                for s, y in sorted(cited) if (s, y) not in index]
    cited_keys = {e["key"] for c in cited for e in index.get(c, [])}
    chosen = [e for e in entries if e["status"] == "used" or e["key"] in cited_keys]
    for e in chosen:
        if e["key"] not in cited_keys:
            warnings.append(f"{e['key']} is listed (status used) but cited in no part")
    return sorted(chosen, key=sort_key), warnings


def render(entries: list[dict]) -> str:
    lines = [format_entry(e) for e in entries]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d")
    return ("<!-- Generated by scripts/make_references.py from registry/references.md (" + stamp + "); do not edit by\n"
            "hand: change the registry or the citations in the parts and run the script again. One list for the whole\n"
            "document; the core report and the extensions cite into it (author-year). -->\n"
            "# References\n\n<!-- small -->\n" + "\\\n".join(lines) + "\n<!-- /small -->\n")


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check only; write nothing")
    args = ap.parse_args()
    entries = parse_bibtex(REGISTRY.read_text(encoding="utf-8"))
    cited: set[tuple[str, str]] = set()
    for p in PARTS:
        cited |= citations(p.read_text(encoding="utf-8"))
    chosen, warnings = select(entries, cited)
    for w in warnings:
        print("WARNING: " + w)
    print(f"{len(chosen)} entries ({sum(e['status'] == 'used' for e in chosen)} with status used); "
          f"{len(cited)} distinct citations in {len(PARTS)} parts")
    if args.check:
        return 1 if warnings else 0
    OUT.write_text(render(chosen), encoding="utf-8")
    print(f"Wrote {C.rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
