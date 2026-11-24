"""Numbered citations and the reference list of the report (IEEE style), from registry/references.md.

    .venv\\Scripts\\python.exe scripts\\make_references.py            # check the citations, print the list
    .venv\\Scripts\\python.exe scripts\\make_references.py --out reports\\build\\references_preview.md

The report parts cite works by registry key, in pandoc syntax: [@hanley1982], several in one bracket
[@xgboost; @lightgbm], with a locator [@hastie2009, Sec. 7.10.2], or narratively ("Nadeau and Bengio
[@nadeau2003]"). scripts/build_report.py calls resolve() on the parts it builds, in document order:
each work gets a number at its first citation, every [@...] becomes [1], [3, 7], [2–4] or [5, Sec. 7.10.2],
and the part "refs" (the References after the extensions) lists the cited works in that order.

This script checks the citations of the core report and the four extensions without building: a key
that is not in the registry is an error (exit code 1); an entry with status "used" that no part cites is
a warning (course rule 4: everything used must be cited in the report).

Entry style (IEEE): initials and surnames of up to six authors, else the first author and "et al.";
"Title," in quotes; journal or proceedings in italics; vol., no., pp.; year; books with edition and
publisher; technical reports with institution; preprints with their arXiv identifier; software with
its URL; the DOI where the registry has one. Titles use one capitalisation (title case).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry" / "references.md"
PARTS = [ROOT / "reports" / "report_core.md"] + [
    ROOT / "reports" / "extensions" / f for f in
    ("e1_validation_protocol.md", "e2_learning_curve.md", "e3_leakage.md", "e4_feature_stability.md")]
MAX_AUTHORS = 6
BS = "\\"
ACCENTS = {('"', "u"): "ü", ('"', "o"): "ö", ('"', "a"): "ä", ('"', "U"): "Ü", ('"', "O"): "Ö",
           ("'", "e"): "é", ("'", "a"): "á", ("'", "o"): "ó", ("`", "e"): "è", ("^", "o"): "ô", ("~", "n"): "ñ"}
CITE_GROUP = re.compile(r"\[(@[^\[\]]+)\]")
SKIP = re.compile(r"<!--.*?-->|```.*?```|`[^`\n]*`", re.S)            # comments and code keep their text


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
        status = re.search(r"status:\s*([\w-]+)", text[m.start():i])
        body = re.sub(r"(?<!\\)%[^\n]*", "", text[m.end():i - 1])       # drop BibTeX comments
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


def load_registry(path: Path = REGISTRY) -> dict[str, dict]:
    return {e["key"]: e for e in parse_bibtex(path.read_text(encoding="utf-8"))}


def clean(s: str) -> str:
    """BibTeX markup -> plain Unicode text (accents, en dashes, URLs, protected capitals, three formulas)."""
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
    """'D'Agostino, R.' -> 'dagostino'; 'Bühlmann' -> 'buhlmann'."""
    s = unicodedata.normalize("NFKD", name.split(",")[0]).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


# ---------------------------------------------------------------- formatting (IEEE)
MINOR = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "as", "at", "by", "in", "of", "off",
         "on", "per", "to", "up", "via", "vs"}


def title_case(title: str) -> str:
    """One capitalisation for every title (the registry mixes sentence and title case): the first word, the word
    after a colon and every word that is not a short minor word start with a capital; words that already carry an
    inner capital, a digit or markup (LIBSVM, NumPy, K-Fold, b<sub>2</sub>) are left as they are."""
    out, start = [], True
    for w in title.split(" "):
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


def ieee_name(name: str) -> str:
    """'Pedregosa, F.' -> 'F. Pedregosa'; 'van der Walt, S. J.' -> 'S. J. van der Walt'."""
    if "," not in name:
        return name
    surname, given = (p.strip() for p in name.split(",", 1))
    return f"{given} {surname}" if given else surname


def ieee_authors(field: str) -> str:
    names, others = author_list(field)
    names = [ieee_name(n) for n in names]
    if not names:
        return ""
    if others or len(names) > MAX_AUTHORS:
        return f"{names[0]} et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# Standard IEEE / ISO 4 abbreviations of the journals and proceedings in the registry (names not listed stay whole).
VENUE = {
    "ACM Transactions on Intelligent Systems and Technology": "ACM Trans. Intell. Syst. Technol.",
    "Advances in Neural Information Processing Systems": "Adv. Neural Inf. Process. Syst.",
    "BMC Medical Informatics and Decision Making": "BMC Med. Inform. Decis. Mak.",
    "Bayesian Analysis": "Bayesian Anal.",
    "Computing in Science & Engineering": "Comput. Sci. Eng.",
    "Electronic Journal of Statistics": "Electron. J. Statist.",
    "ICML 2013 Workshop on Challenges in Representation Learning": "ICML Workshop Challenges Represent. Learn.",
    "IEEE Transactions on Information Theory": "IEEE Trans. Inf. Theory",
    "International Conference on Machine Learning": "Proc. Int. Conf. Mach. Learn.",
    "Journal of Machine Learning Research": "J. Mach. Learn. Res.",
    "Journal of Multivariate Analysis": "J. Multivariate Anal.",
    "Journal of Open Source Software": "J. Open Source Softw.",
    "Journal of the American Statistical Association": "J. Amer. Statist. Assoc.",
    "Journal of the Royal Statistical Society: Series B": "J. Roy. Statist. Soc. B",
    "Machine Learning": "Mach. Learn.",
    "Molecular Biology and Evolution": "Mol. Biol. Evol.",
    "Nature Methods": "Nat. Methods",
    "Proceedings of the 14th International Joint Conference on Artificial Intelligence":
        "Proc. 14th Int. Joint Conf. Artif. Intell.",
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining":
        "Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.",
    "Proceedings of the 33rd Annual Meeting of the Association for Computational Linguistics":
        "Proc. 33rd Annu. Meeting Assoc. Comput. Linguist.",
    "Proceedings of the 9th Python in Science Conference": "Proc. 9th Python Sci. Conf.",
    "Proceedings of the National Academy of Sciences": "Proc. Nat. Acad. Sci.",
    "Statistical Science": "Statist. Sci.",
    "The Annals of Mathematical Statistics": "Ann. Math. Statist.",
    "The Annals of Statistics": "Ann. Statist.",
}


def venue(name: str) -> str:
    name = clean(name)
    if name in VENUE:
        return VENUE[name]
    m = re.match(r"(.*?) (\d+)$", name)                                # "Advances in ... Systems 30"
    return f"{VENUE[m.group(1)]} {m.group(2)}" if m and m.group(1) in VENUE else name


def _pages(p: str) -> str:
    p = clean(p)
    return f"pp. {p}" if "–" in p else f"Art. no. {p}"


def format_entry(e: dict) -> str:
    """One reference in IEEE style (Markdown: italics with *...*)."""
    t, year = e["type"], e.get("year", "n.d.")
    au = ieee_authors(e.get("author", ""))
    title = title_case(clean(e.get("title", "")))
    q = f"“{title},”"                                       # “Title,”
    if t == "article":
        bits = [f"*{venue(e.get('journal', ''))}*"]
        if e.get("volume"):
            bits.append(f"vol. {e['volume']}")
        if e.get("number"):
            bits.append(f"no. {clean(e['number'])}")
        if e.get("pages"):
            bits.append(_pages(e["pages"]))
        s = f"{au}, {q} {', '.join(bits)}, {year}."
    elif t == "inproceedings":
        s = f"{au}, {q} in *{venue(e.get('booktitle', ''))}*, {year}" + (
            f", {_pages(e['pages'])}" if e.get("pages") else "") + "."
    elif t == "incollection":
        pub = f" {clean(e['publisher'])}," if e.get("publisher") else ""
        s = f"{au}, {q} in *{venue(e.get('booktitle', ''))}*.{pub} {year}" + (
            f", {_pages(e['pages'])}" if e.get("pages") else "") + "."
    elif t == "book":
        ed = f", {clean(e['edition'])} ed." if e.get("edition") else "."
        pub = f" {clean(e['publisher'])}," if e.get("publisher") else ""
        s = f"{au}, *{title}*{ed}{pub} {year}."
    elif t == "techreport":
        s = f"{au}, {q} {clean(e.get('institution', ''))}, Tech. Rep., {year}."
    else:                                                               # misc: preprints, software, documents
        how = clean(e.get("howpublished", ""))
        if how.startswith("http"):
            s = f"{au}, {q} {year}. [Online]. Available: [{how}]({how})"
        elif how:
            s = f"{au}, {q} {how}, {year}."
        else:
            s = f"{au}, {q} {year}."
    if e.get("doi"):
        s += f" doi: {clean(e['doi'])}."
    return s


# ---------------------------------------------------------------- citations
def _items(group: str) -> list[tuple[str, str]]:
    """'@a; @b, Sec. 7' -> [('a', ''), ('b', 'Sec. 7')]."""
    out = []
    for it in group.split(";"):
        it = it.strip()
        m = re.match(r"@([\w:.-]+)\s*(?:,\s*(.*))?$", it)
        if m:
            out.append((m.group(1), (m.group(2) or "").strip()))
    return out


def _compress(nums: list[int]) -> str:
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


def _visible_spans(text: str):
    """(start, end) of the stretches outside comments and code."""
    i = 0
    for m in SKIP.finditer(text):
        yield i, m.start()
        i = m.end()
    yield i, len(text)


def cited_keys(texts: list[str]) -> list[str]:
    """Keys in the order of their first citation across the texts."""
    order: list[str] = []
    for text in texts:
        for a, b in _visible_spans(text):
            for m in CITE_GROUP.finditer(text, a, b):
                for key, _ in _items(m.group(1)):
                    if key not in order:
                        order.append(key)
    return order


def resolve(texts: list[str], registry: dict[str, dict]) -> tuple[list[str], list[str], list[str]]:
    """Replace every [@...] citation in the texts by its number. -> (new texts, numbered keys, warnings)."""
    order = [k for k in cited_keys(texts) if k in registry]
    unknown = [k for k in cited_keys(texts) if k not in registry]
    number = {k: i + 1 for i, k in enumerate(order)}
    warnings = [f"citation key @{k} is not in registry/references.md" for k in unknown]

    def repl(m: re.Match) -> str:
        items = _items(m.group(1))
        if not items:
            return m.group(0)
        if any(loc for _, loc in items):
            return ", ".join(f"[{number.get(k, '?')}" + (f", {loc}" if loc else "") + "]" for k, loc in items)
        nums = [number[k] for k, _ in items if k in number]
        return "[" + (_compress(nums) if nums else "?") + "]"

    out = []
    for text in texts:
        pieces, last = [], 0
        for a, b in _visible_spans(text):
            pieces.append(text[last:a])
            seg = CITE_GROUP.sub(repl, text[a:b])
            pieces.append(re.sub(r" (?=\[[\d?])", "&nbsp;", seg))       # no line break before a citation
            last = b
        pieces.append(text[last:])
        out.append("".join(pieces))
    return out, order, warnings


def render_list(order: list[str], registry: dict[str, dict]) -> str:
    """Markdown of the reference list part: numbered entries, one per hard-broken line, in small type."""
    lines = [f"[{i}] {format_entry(registry[k])}" for i, k in enumerate(order, 1)]
    body = "\\\n".join(lines) if lines else "No works are cited."
    return "# References\n\n<!-- small -->\n" + body + "\n<!-- /small -->\n"


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import _common as C
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the reference list (Markdown) to this file")
    args = ap.parse_args()
    registry = load_registry()
    texts = [p.read_text(encoding="utf-8") for p in PARTS]
    _, order, problems = resolve(texts, registry)
    for w in problems:
        print("ERROR: " + w)
    uncited = [k for k, e in registry.items() if e["status"] == "used" and k not in order]
    for k in uncited:
        print(f"WARNING: {k} has status used but no part cites it (course rule 4)")
    print(f"{len(order)} works cited in {len(PARTS)} parts (core and E1-E4, in this order); "
          f"{sum(registry[k]['status'] != 'used' for k in order)} of them not yet marked used in the registry")
    if args.out:
        Path(args.out).write_text(render_list(order, registry), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
