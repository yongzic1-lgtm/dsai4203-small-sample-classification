"""scripts/build_report.py: Markdown -> LaTeX converter, part selection, page marks.

The converter tests run on small Markdown snippets and need no TeX installation. One test
compiles a two-part document with XeLaTeX (one part uses every supported construct, the other
runs over a page) and checks the page counts; it is skipped when xelatex or an ASCII temp
folder is missing.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import pytest

from ssc import paths

# 1 x 1 white PNG
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC")


@pytest.fixture
def B(monkeypatch):
    monkeypatch.syspath_prepend(str(paths.ROOT / "scripts"))
    import build_report
    return build_report


def body(B, md: str, base: Path | None = None) -> str:
    latex, _ = B.convert(md, base)
    return latex


# ---------------------------------------------------------------- inline text
def test_latex_specials_are_escaped(B):
    out = body(B, r"50% & $5 #1 a_b ~x ^y {z} back\\slash S###")
    for s in (r"50\%", r"\&", r"\$5", r"\#1", r"a\_b", r"\textasciitilde{}x", r"\textasciicircum{}y",
              r"\{z\}", r"back\textbackslash{}slash", r"S\#\#\#"):
        assert s in out


def test_brackets_and_asterisks_cannot_become_latex_syntax(B):
    out = body(B, r"[0.881, 0.916] and m\*(k)")
    assert "{[}0.881, 0.916{]}" in out          # never an optional argument after \\ or \item
    assert "m{*}(k)" in out and r"\emph" not in out


def test_bold_italic_and_intraword_underscores(B):
    out = body(B, "**bold** and *it* and ***both*** and _u_; k_feat, reg_param and Σ_i x_i stay plain")
    assert r"\textbf{bold}" in out and r"\emph{it}" in out and r"\emph{u}" in out
    assert r"\emph{\textbf{both}}" in out
    assert r"k\_feat" in out and r"reg\_param" in out and r"x\_i" in out
    assert out.count(r"\emph") == 3


def test_emphasis_over_a_wrapped_line(B):
    out = body(B, "*Table 1. A caption that is\nwrapped in the source.*")
    assert r"\emph{Table 1. A caption that is wrapped in the source.}" in out


def test_code_spans(B):
    out = body(B, "`reg_param` and ``a`b`` and `50%` and `r307_e2_comparison.md`")
    assert r"\texttt{reg\_\allowbreak{}param}" in out
    assert r"\texttt{a`b}" in out and r"\texttt{50\%}" in out
    assert r"comparison.md}" in out                       # no break before a file extension


def test_links_and_urls(B):
    out = body(B, "[text](https://x.org/a_b#c) [notes](../registry/journal.md) see https://example.com/x.")
    assert r"\href{https://x.org/a_b\#c}{text}" in out
    assert "notes" in out and "journal.md" not in out          # relative link: text only
    assert r"\url{https://example.com/x}." in out


def test_unicode_passes_through_and_quotes_are_curled(B):
    text = "± × − ≈ ≥ ≤ – — α Σ χ² Φ √ 10⁻⁴"
    assert text in body(B, text).replace("~", " ")       # ~: no line break next to an operator
    out = body(B, "\"quoted\" and it's -- and ---")
    assert "\u201cquoted\u201d and it\u2019s~\u2013 and~\u2014" in out       # a line never starts with a dash


def test_no_line_break_next_to_operators_and_before_percent(B):
    out = body(B, "p = 0.05, P(≥ 0.999), (≈ 395), 20 % and 5 × 20; A − B = −0.0044; 2 / 3 / 5 folds; "
                  "setup · uploads; a gap of −0.03 and +0.06")
    for s in ("p~=~0.05", "P(≥~0.999)", "(≈~395)", r"20~\%", "5~×~20", "A~−~B~=~−0.0044", "2~/~3~/~5 folds",
              "setup~· uploads"):
        assert s in out, s
    assert "of −0.03 and +0.06" in out                    # a signed number may start a line
    assert "k~=~7" in body(B, "with k\n= 7 here")          # also over a source line break
    assert r"f~=~\texttt{flip" in body(B, "rate f = `flip_y`")
    tied = body(B, " + ".join(str(i) for i in range(1, 30))).strip().split("\n")[-1]
    assert "~" in tied and " " in tied                    # a run of tied words stops at TIE_MAX characters
    assert "~" not in body(B, "| p = 0.05 | 20 % |\n|---|---|\n| a = 1 | b |")   # table cells: widths per word


def test_html_inline_tags(B):
    out, warnings = B.convert("x<sup>2</sup>, H<sub>2</sub>O, a<br>b, <span>s</span>, if a<b and c>d")
    assert r"x\textsuperscript{2}" in out and r"H\textsubscript{2}O" in out and r"a\newline{}b" in out
    assert "if a<b and c>d" in out                           # not a tag
    assert any("<span>" in w for w in warnings)


# ---------------------------------------------------------------- comments, title block, headings
def test_html_comments_are_dropped(B):
    out = body(B, "# T\n\na <!-- hidden --> b\n<!--\nmulti\nline\n-->\nc\n\n```\n<!-- kept in code -->\n```")
    assert "hidden" not in out and "multi" not in out
    assert "a b" in out and "\nc" in out
    assert "<!-- kept in code -->" in out


def test_unclosed_comment_warns(B):
    out, warnings = B.convert("# T\n\na\n<!-- never closed\nb")
    assert "never" not in out and any("unclosed HTML comment" in w for w in warnings)


def test_title_block_from_first_heading(B):
    out = body(B, "# My Title\n\n**Team: X** (Kaggle: Y)\nMembers: A, B\n\n## Question\n\npara\nwrapped  \nhard")
    assert out.startswith(r"\mdtitle{My Title}{My Title}")
    assert r"\textbf{Team: X} (Kaggle: Y)\newline{}Members: A, B" in out    # title block keeps its lines
    assert r"\section{\texorpdfstring{Question}{Question}}" in out
    assert r"para wrapped\newline{}hard" in out                              # elsewhere: soft breaks


def test_extension_heading_check(B):
    res = B.convert_markdown("# E\n\n## Question\n\n## Protocol\n\n## Results\n\n## Conclusion\n", Path("."), "e2")
    assert B.check_part(res) == []
    res = B.convert_markdown("# E\n\n## Question\n\n## Method\n\n## Results\n", Path("."), "e2")
    assert "expected exactly" in B.check_part(res)[0]


# ---------------------------------------------------------------- lists, quotes, rules
def test_nested_lists_and_numbering(B):
    out = body(B, "- a\n  1. one\n  2. two\n- b\n\n3. three\n4. four")
    assert out.index(r"\begin{itemize}") < out.index(r"\begin{enumerate}") < out.index(r"\end{itemize}")
    assert r"\begin{enumerate}[start=3]" in out


def test_wrapped_line_starting_with_a_number_stays_in_the_paragraph(B):
    out = body(B, "corrected paired t-tests, Nadeau & Bengio\n2003) with all 150 features")
    assert r"\begin{enumerate}" not in out and r"Bengio 2003) with" in out


def test_block_quote_and_rule(B):
    out = body(B, "> quoted\n> more\n\n---\n\nafter")
    assert "\\begin{quote}\nquoted more\n\\end{quote}" in out and r"\rule{0.5\linewidth}" in out


def test_lead_in_paragraph_stays_with_the_next_block(B):
    out = body(B, "Every upload was pre-registered:\n\n- a\n- b\n\nPlain paragraph.\n\n1. c")
    assert r"\begin{itemize}[beginpenalty=10000]" in out          # no page break after the colon
    assert "\\begin{enumerate}\n" in out                           # an ordinary paragraph before it
    out = body(B, "**Negative results.**\n\nText after.\n\n*A long italic line that is a sentence and not a "
                  "heading, as it runs on for well over eighty characters.*\n\nNext.")
    assert "\\textbf{Negative results.}\n\\par\\nopagebreak\n" in out
    assert out.count("nopagebreak") == 1


def test_line_block_is_set_one_ragged_paragraph_per_line(B):
    out = body(B, "Intro.\n\nChen, T. (2016). XGBoost. *Proc. KDD*.\\\nEfron, B. (1993). *Bootstrap*.\\\n"
                  "Yarowsky, D. (1995). Word senses.")
    assert ("{\\raggedright Chen, T. (2016). XGBoost. \\emph{Proc. KDD}.\\par\\setlength{\\parskip}{0pt}\n"
            "Efron, B. (1993). \\emph{Bootstrap}.\\par\nYarowsky, D. (1995). Word senses.\\par}") in out
    assert r"\newline" not in out
    assert r"one\newline{}two" in body(B, "one  \ntwo")                  # a single line break stays as it was


def test_preamble_spacing_and_pdf_text(B):
    assert r"\frenchspacing" in B.PREAMBLE                               # no wide space after "et al."
    assert "WordSpace={1,1,0.6}" in B.PREAMBLE
    assert r"\XeTeXgenerateactualtext=1" in B.PREAMBLE                   # copied text keeps '-' and words


# ---------------------------------------------------------------- tables
def test_small_table_natural_width_with_alignment(B):
    out = body(B, "| a | b |\n|---|--:|\n| 1 | 2 |")
    assert r"\begin{longtable}[c]{lr}" in out and r"\toprule" in out and r"\bottomrule" in out


def test_escaped_pipe_stays_in_the_cell(B):
    out = body(B, "| a \\| b | c |\n|---|---|\n| x | y |")
    assert "a | b & c" in out


def test_dash_widths_with_a_floor_for_long_numbers(B):
    metrics = B.FontMetrics(load=False)
    short = [[B.parse_inline(c) for c in row] for row in (["Scheme", "Value"], ["x", "0.1"])]
    lay = B.layout_table(short, [3, 1], True, B.TEXT_WIDTH_PT, metrics)
    assert lay.mode == "p" and lay.fractions == pytest.approx([0.75, 0.25])     # |---|-| = 3:1
    cells = [[B.parse_inline(c) for c in row] for row in (["Scheme", "Value"], ["x", "0.0061 [0.0051, 0.0076]"])]
    lay = B.layout_table(cells, [40, 1], True, B.TEXT_WIDTH_PT, metrics)
    content = B.TEXT_WIDTH_PT - 2 * 2 * lay.tabcolsep
    assert lay.fractions[1] * content >= B.cell_extent(cells[1][1], metrics, 10.0)[1] - 1e-6   # not broken
    lay = B.layout_table(cells, [3, 1], False, B.TEXT_WIDTH_PT, metrics)       # short source: no hint
    assert lay.mode == "natural"


def test_wide_table_without_hints_is_fitted_to_content(B):
    long = "a fairly long cell text that has to wrap onto several lines " * 3
    md = "| key | text |\n|---|---|\n| k1 | " + long + " |"
    out = body(B, md)
    assert r"\raggedright\arraybackslash}p{\dimexpr(\linewidth-4\tabcolsep)" in out
    lay = B.layout_table([[B.parse_inline("key"), B.parse_inline(long)]], [3, 3], True, B.TEXT_WIDTH_PT,
                         B.FontMetrics(load=False))
    assert lay.fractions[1] > 0.8


def test_captioned_table_is_kept_whole_and_long_tables_break(B):
    rows = "\n".join(f"| r{i} | {i} |" for i in range(20))
    out = body(B, "| a | b |\n|---|---|\n| 1 | 2 |\n\n*Table 1. Caption.*")
    assert r"\begin{minipage}" in out and r"\begin{tabular}" in out and "longtable" not in out
    out = body(B, "| a | b |\n|---|---|\n" + rows)
    assert r"\begin{longtable}" in out and r"\endhead" in out and r"\\*" in out


# ---------------------------------------------------------------- images
def test_images_relative_to_the_markdown_file(B, tmp_path):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "x.png").write_bytes(PNG)
    (tmp_path / "extensions").mkdir()
    images = B.ImageStore()
    res = B.convert_markdown("# T\n\n![alt](figures/x.png)", tmp_path, "core", images)
    assert r"\includegraphics[width=\linewidth,height=0.6\textheight,keepaspectratio]{figures/x.png}" in res.latex
    res = B.convert_markdown("# T\n\n![alt](../figures/x.png \"width=65%\")", tmp_path / "extensions", "e1", images)
    assert r"width=0.650\linewidth" in res.latex and "{figures/x.png}" in res.latex
    assert list(images.files.values()) == ["figures/x.png"]              # one copy for both parts
    out = body(B, "# T\n\n![alt](figures/x.png){width=50%}\n\n<img src=\"figures/x.png\" width=\"40%\">", tmp_path)
    assert r"width=0.500\linewidth" in out and r"width=0.400\linewidth" in out


def test_figure_and_caption_are_grouped(B, tmp_path):
    (tmp_path / "x.png").write_bytes(PNG)
    out = body(B, "# T\n\n![a](x.png)\n\n*Figure 1. Caption.*", tmp_path)
    assert out.index(r"\begin{minipage}") < out.index("includegraphics") < out.index("Figure 1") \
        < out.index(r"\end{minipage}")


def test_missing_image_gives_a_placeholder_and_a_warning(B, tmp_path):
    out, warnings = B.convert("# T\n\n![a](figures/nope.png)", tmp_path)
    assert r"\mdmissing{figures/" in out and "nope.png}" in out and "includegraphics" not in out
    assert any("not found" in w for w in warnings)


def test_width_hints(B):
    assert B.tex_length("65%") == r"0.650\linewidth"
    assert B.tex_length("0.5") == r"0.500\linewidth"
    assert B.tex_length("120%") == r"1.000\linewidth"
    assert B.tex_length("8cm") == "8cm"
    assert B.tex_length("wide") is None


# ---------------------------------------------------------------- parts, page marks
def test_parts_from_project_yaml(B):
    keys, warnings = B.parts_from_project({})
    assert keys == ["core", "e1", "e2", "e3", "e4", "contrib"] and warnings
    keys, warnings = B.parts_from_project({"report": {"extensions_owned": []}})
    assert keys == ["core", "e1", "e2", "e3", "e4", "contrib"] and warnings
    keys, warnings = B.parts_from_project({"report": {"extensions_owned": ["E3", "e1_validation_protocol"]}})
    assert keys == ["core", "e1", "e3", "contrib"] and not warnings
    with pytest.raises(ValueError):
        B.parts_from_project({"report": {"extensions_owned": ["e5"]}})
    assert B.parse_parts_arg("contrib, core,e2") == ["core", "e2", "contrib"]
    with pytest.raises(ValueError):
        B.parse_parts_arg("core,appendix")


def test_page_marks(B):
    marks = ("start core 1\nend core 2 350.0pt 700.0pt\n"
             "start e1 3\nend e1 5 0.0pt 16383.99998pt\n")        # e1 ended at the foot of page 4
    pages = B.parse_page_marks(marks)
    assert (pages["core"].start, pages["core"].end, pages["core"].pages, pages["core"].used) == (1, 2, 2, 1.5)
    assert (pages["e1"].end, pages["e1"].pages, pages["e1"].used) == (4, 2, 2.0)


# ---------------------------------------------------------------- XeLaTeX (skipped without TeX)
def _ascii_tmp() -> Path | None:
    for p in (os.environ.get("SSC_TEX_TMP"), tempfile.gettempdir()):
        if p:
            try:
                p.encode("ascii")
                return Path(p)
            except UnicodeEncodeError:
                continue
    return None


# every supported construct once; the build must compile without a LaTeX error
TORTURE = ("# Torture: *every* construct & `chars` #1\n\n**Team: T_1** (Kaggle: K)\nMembers: A & B, 100%\n\n"
           "<!-- comment\nover lines -->\n\n## 1. Inline\n\n"
           "Specials 50% & $5 #1 a_b ~x ^y {z} back\\\\slash \\* m\\*(k) S### [0.8, 0.9] "
           "[link](https://e.com/a_b?x=1&y=2#f) <https://e.org/~u/%20x> https://www.kaggle.com/c/x. www.e.com "
           "<someone@e.com> &amp; &lt; &copy; &#8805; \"dq\" 'sq' it's -- --- ± × − ≈ ≥ ≤ α Σ χ² 10⁻⁴ ü.\n"
           "**bold *it* bold** ***both*** _u_ __s__ snake_case x_i `C:\\tmp\\x_y.csv` ``a ` b`` `{}#%$&_^~\\` "
           "x<sup>2</sup> H<sub>2</sub>O a<br>b <b>b</b> <code>c_d</code> <span>s</span> a<b and c>d hard  \n"
           "break\\\nagain ![inline](figures/x.png) end.\n\nSetext\n------\n\n### Sub `code`\n\n#### Four\n\n"
           "##### Five\n\n###### Six\n\ntext\n\n## 2. Blocks\n\n> quote **b**\n>\n> - q1\n> - q2\nlazy\n\n---\n\n"
           "1. one\n2. two\n   - nested\n     1. three\n        - four\n          - five\n            - six\n"
           "3. three\n\n   loose paragraph\n\n7) seven\n8) eight\n\n+ plus\n* star\n\n"
           "```python\ndef f(x):  # % \\ {}\n    return x  <!-- kept -->\n```\n\n~~~\ntilde ~~~\n~~~\n\n"
           "| left | center | right | default |\n|:-----|:------:|------:|---------|\n"
           "| a \\| pipe | **b** | `c_d` | [l](https://x.y) |\n| [0.1, 0.2] | *it* | 1.0 | ü ± |\n\n"
           "*Table 1. Caption with `code` and 5%.*\n\n"
           "| k | a long header that wraps | v |\n|---|---|---|\n| 1 | " + "word " * 40 + "| 0.5 |\n\n"
           "| n | value |\n|---|---:|\n" + "".join(f"| {i} | {i / 10:.1f} |\n" for i in range(1, 30)) + "\n"
           "![Figure alt](figures/x.png \"width=30%\")\n\n*Figure 1. Caption.*\n\n![Missing](figures/missing.png)\n\n"
           "<img src=\"figures/x.png\" width=\"20%\">\n\n<div align=\"center\">\n\nIn a div.\n\n</div>\n\n"
           "Lead-in with a colon:\n\n- kept with this line\n- second\n\n**Pseudo heading.**\n\n"
           "After it: p = 0.05, P(≥ 0.999), 20 %, A − B = −0.01 and 2 / 3 / 5 · end.\n\n"
           "Ref A. (2001). Title. *J.*, 1–2.\\\nRef B. (2002). Title.\\\nRef C. (2003). Title.\n\nLast line.\n")


def test_compile_two_parts_and_count_pages(B, tmp_path):
    xelatex, tmp_base = B.find_xelatex(), _ascii_tmp()
    if not xelatex or tmp_base is None:
        pytest.skip("xelatex or an ASCII temp folder not available")
    reports = tmp_path / "reports"
    (reports / "figures").mkdir(parents=True)
    (reports / "extensions").mkdir()
    (reports / "figures" / "x.png").write_bytes(PNG)
    (reports / "report_core.md").write_text(TORTURE, encoding="utf-8")
    (reports / "extensions" / "e1_validation_protocol.md").write_text(
        "# Extension E1: test\n\n## Question\n\n![f](../figures/x.png \"width=30%\")\n\n*Figure 1. F.*\n\n"
        + "\n\n".join(["Filler paragraph. " * 60] * 7), encoding="utf-8")
    out = tmp_path / "out.pdf"
    res = B.build([("core", reports / "report_core.md"), ("e1", reports / "extensions" / "e1_validation_protocol.md")],
                  out, xelatex=xelatex, tmp_base=tmp_base, runs=1, metrics=B.FontMetrics(load=False))
    assert res.ok, res.warnings
    assert out.exists() and out.read_bytes().startswith(b"%PDF")
    core, e1 = res.pages["core"], res.pages["e1"]
    assert core.start == 1 and core.pages >= 2
    assert e1.start == core.end + 1 and e1.pages >= 2                    # each part on a new page; filler > 1 page
    assert res.total_pages == e1.end
    assert any("missing.png not found" in w for w in res.warnings)
    assert any("deeper than 4 levels" in w for w in res.warnings)
    assert not [w for w in res.warnings if "characters missing" in w or "stick out" in w]
