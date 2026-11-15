"""Build the team's final submission zip: deliverables/<team.name>.zip.

    .venv\\Scripts\\python.exe scripts\\package_deliverables.py             # dry run (default): list + checklist
    .venv\\Scripts\\python.exe scripts\\package_deliverables.py --verbose   # ... and every file
    .venv\\Scripts\\python.exe scripts\\package_deliverables.py --build     # write the zip

Contents (inside a top folder named after the team):
  video/      exactly one .mp4/.mov/.mkv, <= 300 MB (all members, faces shown, <= 5 min)
  reports/    the report PDF(s) (core report <= 6 pages + extensions + contribution table);
              reports/reflections/ is NOT included: each member uploads their own reflection
  code        src/ scripts/ extensions/ configs/ notebooks/ tests/ research/ colab/ docs/
              README.md pyproject.toml requirements.txt requirements-lock.txt project.yaml
              data/raw/SHA256SUMS.txt (checksums only, no competition data)
  records     registry/ (experiment + submission history), experiments/ without *.npy
              (config, scores and notes of every run; --no-experiments leaves it out)
Never included: the competition data (data/ except SHA256SUMS.txt), .venv/, sandbox/,
deliverables/, logs/, __pycache__, experiments/**/*.npy, archives (*.zip, *.tar.gz), and
any kaggle.json / access_token* / credentials.json / .env / *.env file.
E-mail addresses: the zip copy of project.yaml has them replaced by <removed> (the Google accounts
in compute.accounts are only notes; colab/remote.py uses the profile names pro1 / pro2). Any other
included text file that contains an address is flagged as CHECK in the dry run.

--build refuses while project.yaml team.name is still TODO.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import _common as C
from ssc import paths
from ssc.tracking import read_registry

ROOT = paths.ROOT
# research/ and colab/ produced archived runs (e.g. r001_*), docs/SETUP.md is the setup the README cites
CODE_DIRS = ("src", "scripts", "extensions", "configs", "notebooks", "tests", "research", "colab", "docs")
RECORD_DIRS = ("registry", "experiments")
TOP_FILES = ("README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "project.yaml",
             "data/raw/SHA256SUMS.txt")
SKIP_DIRS = {"data", ".venv", "venv", "sandbox", "deliverables", "logs", "__pycache__", ".ipynb_checkpoints",
             ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
SECRETS = {"kaggle.json", "access_token", "access_token.txt", "credentials.json", ".env"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".part", ".zip", ".gz", ".tgz"}
VIDEO_EXT = {".mp4", ".mov", ".mkv"}
MAX_VIDEO_BYTES = 300 * 1000 * 1000      # 300 MB (decimal, the stricter reading)
LARGE_FILE_BYTES = 20 * 1000 * 1000
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
REDACT_FILES = {"project.yaml"}          # zipped with every e-mail address replaced
EMAIL_OK = {"noreply@anthropic.com"}     # AI attribution address, not personal data
EMAIL_OK_DOMAINS = ("example.com", "example.org", "example.net", "e.com")   # made-up addresses in tests
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".csv", ".txt", ".toml", ".tex", ".ipynb", ".sh", ".ps1"}


def mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def redact_emails(text: str) -> tuple[str, int]:
    """Replace every e-mail address in text by <removed>; return the new text and the count."""
    return EMAIL_RE.subn("<removed>", text)


def emails_in(path: Path) -> list[str]:
    """Personal-looking e-mail addresses in a text file (empty for binary or large files)."""
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
        return []
    found = EMAIL_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
    return sorted({a for a in found
                   if a.lower() not in EMAIL_OK and not a.lower().split("@")[1].endswith(EMAIL_OK_DOMAINS)})


def excluded(rel: Path) -> str | None:
    """Reason a project-relative path is left out, or None to include it."""
    parts = rel.parts
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts[:-1]):
        return "excluded folder"
    if rel.name in SECRETS or rel.name.endswith(".env") or rel.name.startswith((".env.", "access_token")):
        return "SECRET"
    if rel.suffix in SKIP_SUFFIXES:
        return "build/temp file"
    if parts[0] == "experiments" and rel.suffix == ".npy":
        return "experiments/**/*.npy"
    return None


def walk(folder: Path, secrets: list[Path]) -> list[Path]:
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        why = excluded(rel)
        if why == "SECRET":
            secrets.append(rel)
        elif why is None:
            out.append(rel)
    return out


def collect(with_experiments: bool):
    """-> (files {category: [rel paths]}, checklist rows, secrets found)."""
    secrets: list[Path] = []
    files: dict[str, list[Path]] = {}
    checks: list[tuple[str, str, str]] = []

    videos = sorted(p.relative_to(ROOT) for p in (ROOT / "video").rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT) if (ROOT / "video").exists() else []
    files["video"] = videos
    if not videos:
        checks.append(("MISSING", "video", "no .mp4/.mov/.mkv under video/"))
    elif len(videos) > 1:
        checks.append(("PROBLEM", "video", f"{len(videos)} video files; keep exactly one: "
                                           + ", ".join(v.as_posix() for v in videos)))
    else:
        size = (ROOT / videos[0]).stat().st_size
        status = "OK" if size <= MAX_VIDEO_BYTES else "PROBLEM"
        checks.append((status, "video", f"{videos[0].as_posix()} ({mb(size)}; limit 300 MB, <= 5 min, all "
                                        "members presenting with faces shown)"))

    reports = [p for p in walk(ROOT / "reports", secrets)
               if p.suffix.lower() == ".pdf" and "reflections" not in p.parts]
    files["report"] = reports
    if not reports:
        checks.append(("MISSING", "report PDF", "no PDF under reports/ (e.g. reports/report.pdf)"))
    else:
        main = "reports/report.pdf" if (ROOT / "reports/report.pdf").exists() else reports[0].as_posix()
        checks.append(("OK", "report PDF", f"{len(reports)} PDF(s), main: {main} (core <= 6 pages, "
                                           "extensions <= 2 pages each)"))
    contrib = [p for p in (ROOT / "reports").rglob("*contribution*") if p.is_file()] \
        if (ROOT / "reports").exists() else []
    names = ", ".join(p.relative_to(ROOT).as_posix() for p in contrib)
    if any(p.suffix.lower() == ".pdf" for p in contrib):
        checks.append(("OK", "contribution table", names))
    elif contrib:
        checks.append(("CHECK", "contribution table", f"{names} found, but only PDFs are zipped: the 1-page table "
                                                      "must be a page of the report PDF (or a separate PDF)"))
    else:
        checks.append(("CHECK", "contribution table", "no reports/*contribution* file: make sure the 1-page "
                                                      "table (every section/extension -> member) is in the PDF"))

    code = [Path(f) for f in TOP_FILES if (ROOT / f).is_file()]
    for d in CODE_DIRS:
        code += walk(ROOT / d, secrets)
    files["code"] = code
    missing_top = [f for f in TOP_FILES if not (ROOT / f).is_file()]
    checks.append(("MISSING" if missing_top else "OK", "code",
                   f"missing: {', '.join(missing_top)}" if missing_top else
                   f"{len(code)} files (enough to reproduce the final submission?)"))

    records = []
    for d in RECORD_DIRS:
        if d == "experiments" and not with_experiments:
            continue
        records += walk(ROOT / d, secrets)
    files["records"] = records
    subs = read_registry(ROOT / "registry" / "submissions.csv")
    n_sub = len(subs)
    n_pub = int((subs["public_lb"] != "").sum()) if n_sub else 0
    n_final = int((subs["selected_final"] == "yes").sum()) if n_sub else 0
    checks.append(("OK" if n_sub and n_pub else "CHECK", "submission history",
                   f"registry/submissions.csv: {n_sub} submissions, {n_pub} with public score, {n_final} final"))

    flagged, redacted = [], 0
    for rel in (r for cat in ("code", "records") for r in files[cat]):
        found = emails_in(ROOT / rel)
        if rel.as_posix() in REDACT_FILES:
            redacted += len(found)
        elif found:
            flagged.append(f"{rel.as_posix()} ({len(found)})")
    checks.append(("CHECK" if flagged else "OK", "e-mail addresses",
                   (f"in {', '.join(flagged[:5])}{' ...' if len(flagged) > 5 else ''}; remove them before --build. "
                    if flagged else "none in the zipped files except ")
                   + f"project.yaml ({redacted}), whose zip copy has them replaced by <removed>"))
    return files, checks, secrets


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="list contents and checklist only (default)")
    mode.add_argument("--build", action="store_true", help="write deliverables/<team>.zip")
    ap.add_argument("--no-experiments", action="store_true", help="leave experiments/ out of the zip")
    ap.add_argument("--verbose", action="store_true", help="list every file")
    args = ap.parse_args()

    project = C.load_project()
    team = str((project.get("team") or {}).get("name") or "TODO")
    team_ok = "TODO" not in team.upper()
    safe = re.sub(r"[^A-Za-z0-9 ._-]+", "_", team).strip(" .") or "team"
    zip_path = ROOT / "deliverables" / f"{safe}.zip"

    files, checks, secrets = collect(not args.no_experiments)
    checks.insert(0, ("OK" if team_ok else "MISSING", "team name",
                      f"{team!r} -> {C.rel(zip_path)}" if team_ok else "project.yaml team.name is still TODO"))
    todo_lines = [ln.strip() for ln in paths.PROJECT_YAML.read_text(encoding="utf-8").splitlines()
                  if "TODO" in ln.split("#")[0]]
    if todo_lines:
        checks.append(("CHECK", "project.yaml", f"{len(todo_lines)} TODO value(s) left (team members, Kaggle names)"))

    print(f"Deliverables for team {team!r}: {C.rel(zip_path)}  ({'BUILD' if args.build else 'dry run'})\n")
    total = 0
    for cat, rels in files.items():
        sizes = [(ROOT / r).stat().st_size for r in rels]
        total += sum(sizes)
        print(f"[{cat}] {len(rels)} file(s), {mb(sum(sizes))}")
        by_top: dict[str, list[int]] = {}
        for r, s in zip(rels, sizes):
            by_top.setdefault(r.parts[0] if len(r.parts) > 1 else r.name, []).append(s)
        if args.verbose or cat in ("video", "report"):
            for r, s in zip(rels, sizes):
                print(f"    {mb(s):>10}  {r.as_posix()}")
        else:
            for top, ss in by_top.items():
                print(f"    {mb(sum(ss)):>10}  {top}{'/' if (ROOT / top).is_dir() else ''}  ({len(ss)} files)")
    print(f"\nTotal (uncompressed): {mb(total)}")
    for rel in secrets:
        print(f"EXCLUDED SECRET: {rel.as_posix()} (never zip or commit credentials)")
    big = [r for rels in (files["code"], files["records"]) for r in rels
           if (ROOT / r).stat().st_size > LARGE_FILE_BYTES]
    for r in big:
        checks.append(("CHECK", "large file", f"{r.as_posix()} is {mb((ROOT / r).stat().st_size)}: data copy?"))

    print("\nChecklist:")
    for status, item, detail in checks:
        print(f"  [{status:<7}] {item}: {detail}")
    print("  [REMIND ] reflections: each member uploads an individual reflection PDF to Canvas separately "
          "(not in this zip; missing reflection = 0 for that member's extension)")
    print("  [REMIND ] report: cite every library/tutorial/method used; disclose generative-AI use (tool, purpose)")

    incomplete = any(s in ("MISSING", "PROBLEM") for s, _, _ in checks)
    if not args.build:
        print("\nDry run: nothing written. Use --build to create the zip.")
        return 0
    if not team_ok:
        C.die("refusing to build: set team.name in project.yaml first (the zip is named after the team)", 2)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cat, rels in files.items():
            for r in rels:
                ctype = zipfile.ZIP_STORED if cat == "video" else zipfile.ZIP_DEFLATED
                if r.as_posix() in REDACT_FILES:
                    text, _ = redact_emails((ROOT / r).read_text(encoding="utf-8"))
                    zf.writestr(f"{safe}/{r.as_posix()}", text.encode("utf-8"), compress_type=ctype)
                else:
                    zf.write(ROOT / r, arcname=f"{safe}/{r.as_posix()}", compress_type=ctype)
    with zipfile.ZipFile(tmp) as zf:
        bad = zf.testzip()
        n = len(zf.namelist())
    if bad:
        C.die(f"zip verification failed at {bad}; {tmp} left for inspection", 1)
    tmp.replace(zip_path)
    print(f"\nWrote {zip_path} ({n} files, {mb(zip_path.stat().st_size)})")
    if incomplete:
        C.banner("The zip is INCOMPLETE (see MISSING/PROBLEM above). Rebuild before uploading to Canvas.", "!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
