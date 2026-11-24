# reports/: assembling the final document

The team submits **one document** (inside the team zip) made of these parts, in this order, each starting on a new page:

| Part | Source file | Limit |
|---|---|---|
| Core report | `reports/report_core.md` | ≤ 6 pages |
| Extensions (one per owner, headed with the owner's name) | `reports/extensions/e1_validation_protocol.md`, `e2_learning_curve.md`, `e3_leakage.md`, `e4_feature_stability.md` (only the owned ones) | ≤ 2 pages each |
| References (one complete list for all parts) | `reports/references.md`, written by `scripts\make_references.py` from `registry/references.md` | no limit (the brief's 6-page core report lists only its five sections) |
| Contribution table | `reports/contribution_table.md` | 1 page |

Individual reflections (`reports/reflections/`) are **not** part of this document: each member exports their own PDF and uploads it to Canvas separately (`scripts\package_deliverables.py` leaves `reports/reflections/` out of the zip).

Figures and tables come only from scripts: `reports/figures/` and `reports/tables/` (prefixes `core_`, `e1_`..`e4_`, `lb_noise_`). Write image links relative to the Markdown file: `figures/<name>.png` in `report_core.md`, `../figures/<name>.png` in `reports/extensions/*.md`.

**Final output**: `reports\report.pdf`, written by `scripts\build_report.py` (next section). `package_deliverables.py` picks up the PDFs under `reports/` (excluding reflections) and treats `reports/report.pdf` as the main one.

## Building the PDF: `scripts\build_report.py`

Run from the project root. The project folder is currently `D:\polyu_class\新建文件夹`, so the interpreter is `D:\polyu_class\新建文件夹\.venv\Scripts\python.exe`; the relative form below works wherever the folder lives.

```powershell
.venv\Scripts\python.exe scripts\make_references.py           # after any change of citations or of registry/references.md
.venv\Scripts\python.exe scripts\build_report.py [--parts core,e1,e2,e3,e4,refs,contrib] [--out reports\report.pdf] [--keep-tex]
```

- **What it does**: assembles the parts in document order (core report, the chosen extensions, contribution table), each starting on a new page; compiles them with XeLaTeX in the agreed layout (Times New Roman 11 pt for body, headings, tables and captions; A4, 2.54 cm margins, single line spacing); writes `reports\report.pdf`; and **prints the page count of every part**, to compare with the limits below.
- **`--parts`**: a comma-separated list in document order. Default: `core`, the extensions listed in `project.yaml` under `report.extensions_owned` (for example `[e1, e3]`), then `contrib`. If that key is not set, all four extensions are included and the script prints a warning. Set it as soon as the owners are known, so that drafts nobody owns stay out of the document.
- **`--out`**: output file, default `reports\report.pdf`. For a trial build of a single part (for example `--parts e3`), write outside `reports/` (for example `--out $env:TEMP\e3_check.pdf`): `package_deliverables.py` picks up every PDF under `reports/`.
- **`--keep-tex`**: keep the generated LaTeX source, to inspect layout problems.
- **References**: the parts cite author–year only. `scripts\make_references.py` writes `reports/references.md`: every registry entry with status "used", plus any other entry that a part cites, as complete entries in alphabetical order. It warns about a citation without an entry and about a listed entry that no part cites (`--check` only checks). Do not edit `reports/references.md` by hand.
- **Smaller type for a reference list**: put `<!-- small -->` and `<!-- /small -->` on lines of their own around the blocks (for example the paragraphs under References); they are set in 10 pt instead of 11 pt. Like any HTML comment, the markers do not show on GitHub. Use them only for references, not to squeeze body text under a page limit.
- **Needs**: TeX Live with `xelatex` (TeX Live 2024 is installed on the team laptop) and the Times New Roman font (Windows; without it the script falls back to TeX Gyre Termes and warns, as page counts then differ). pandoc is not needed.

The Markdown files stay the source of truth: change them, then rebuild; never edit the PDF by hand, so that the archive matches the PDF. If the script cannot be used, the fallback is the manual route: paste each part into Word, Google Docs (shared only with team members; rule 5) or a private Overleaf project, use the same layout, start each part on a new page, and count pages part by part.

## Page-limit and content checks (before every export)

- [ ] Core report ≤ 6 pages (title block, tables, figures, AI disclosure and references counted). Each extension ≤ 2 pages. Contribution table = 1 page. Read the per-part page counts printed by `scripts\build_report.py`, then look at the PDF itself (a part that ends with a near-empty page still counts that page).
- [ ] Readable font size (the build uses 11 pt) and figures legible at 100 % zoom; do not shrink fonts or figures to fit.
- [ ] Each extension: the owner's name under the title; headings exactly Question / Protocol / Results / Conclusion; at least one table or figure in Results.
- [ ] No TODO left:
  ```powershell
  Select-String -Path reports\report_core.md, reports\contribution_table.md, reports\extensions\*.md -Pattern TODO
  ```
- [ ] Every number traces to a run_id / sub_id (captions or text); no `sandbox/` results.
- [ ] References: `scripts\make_references.py --check` reports no warning, then `reports/references.md` is regenerated; AI disclosure from `registry/ai_usage.md`.
- [ ] `.venv\Scripts\python.exe scripts\package_deliverables.py` (dry run) lists `reports/report.pdf`.
