"""Read-only integrity check of the real archive (docs/ARCHIVE_RULES.md weekly checklist).

    .venv\\Scripts\\python.exe scripts\\archive_check.py            # report problems, exit 1 if any
    .venv\\Scripts\\python.exe scripts\\archive_check.py --snapshot # ...and write backups/snapshot_<JST>.zip

Checks
  data        data/raw checksums (ssc.data.verify)
  runs        every experiments/<run_id>/ has config.yaml, meta.json, summary.json, notes.md;
              status done; a written Conclusion in notes.md
  registry    registry/experiments.csv rows <-> run folders (both directions, no duplicates)
  provenance  meta.json compute.code_bundle exists in logs/code_bundles/ (Colab runs)
  jobs        registry/compute_jobs.csv: logs/colab/<job>.log exists, listed runs exist
  submissions registry/submissions.csv <-> submissions/*.csv, sha256 prefix matches, run exists
  reports     research/reports/*.md and reports/extensions/*.md present; figures/tables they cite exist
The snapshot contains everything needed to rebuild the archive except the competition data and
the environment: registry/, experiments/, submissions/, logs/, reports/, research/, src/, scripts/,
extensions/, configs/, colab/, docs/, tests/, project.yaml, pyproject.toml, requirements*.txt, README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssc import data, paths  # noqa: E402
from ssc.data import sha256  # noqa: E402
from ssc.tracking import ARCHIVE_TZ, read_registry  # noqa: E402

REQUIRED = ("config.yaml", "meta.json", "summary.json", "notes.md")
SNAPSHOT_ITEMS = ["registry", "experiments", "submissions", "logs", "reports", "research", "src", "scripts",
                  "extensions", "configs", "colab", "docs", "tests", "notebooks", "video", "project.yaml",
                  "pyproject.toml", "requirements.txt", "requirements-lock.txt", "README.md", ".gitignore",
                  "data/raw/SHA256SUMS.txt"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def conclusion_written(notes: str) -> bool:
    m = re.search(r"## Conclusion[^\n]*\n(.*)", notes, flags=re.S)
    return bool(m and m.group(1).strip())


def check(root: Path) -> tuple[list[str], list[str], dict]:
    problems, warnings, stats = [], [], {}

    # data
    for p in data.verify("raw"):
        problems.append(f"data: {p}")

    # runs
    exp_dir = root / "experiments"
    runs = sorted(d for d in exp_dir.iterdir() if d.is_dir()) if exp_dir.exists() else []
    stats["runs"] = len(runs)
    no_concl, bundles_needed = [], {}
    for d in runs:
        missing = [f for f in REQUIRED if not (d / f).exists()]
        if missing:
            problems.append(f"run {d.name}: missing {missing}")
            continue
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if summ.get("status") != "done":
            warnings.append(f"run {d.name}: status {summ.get('status')!r}")
        if not conclusion_written((d / "notes.md").read_text(encoding="utf-8")):
            no_concl.append(d.name)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        b = (meta.get("compute") or {}).get("code_bundle")
        if b and not b.startswith("unsynced"):
            bundles_needed.setdefault(b, []).append(d.name)
    stats["runs_without_conclusion"] = len(no_concl)
    for name in no_concl:
        warnings.append(f"run {name}: no Conclusion written in notes.md")

    # registry <-> folders
    reg = read_registry(root / "registry" / "experiments.csv")
    ids = list(reg["run_id"]) if not reg.empty else []
    dup = sorted({i for i in ids if ids.count(i) > 1})
    for i in dup:
        problems.append(f"registry: run_id {i} appears {ids.count(i)} times in experiments.csv")
    folder_ids = {d.name for d in runs}
    for i in sorted(set(ids) - folder_ids):
        problems.append(f"registry: row {i} has no run folder")
    for i in sorted(folder_ids - set(ids)):
        problems.append(f"registry: run folder {i} has no row in experiments.csv")
    stats["registry_rows"] = len(ids)

    # provenance: the bundles are kept only on the owner's machine (not in git, D016: they contain an old
    # project.yaml with personal addresses), so a clone without any bundle gets one warning, not one problem per bundle
    bundle_dir = root / "logs" / "code_bundles"
    if bundles_needed and not any(bundle_dir.glob("*.tar.gz")):
        warnings.append(f"provenance: no code bundles in logs/code_bundles/ ({len(bundles_needed)} referenced); "
                        "they are not in git (D016) - ask the owner for a snapshot if you need them")
    else:
        for b, rs in sorted(bundles_needed.items()):
            if not (bundle_dir / f"{b}.tar.gz").exists():
                problems.append(f"provenance: code bundle {b} (used by {len(rs)} runs, e.g. {rs[0]}) is not archived")
    stats["code_bundles_referenced"] = len(bundles_needed)

    # jobs
    jobs = read_registry(root / "registry" / "compute_jobs.csv")
    for row in jobs.to_dict("records") if not jobs.empty else []:
        if not (root / "logs" / "colab" / f"{row['job_id']}.log").exists():
            warnings.append(f"job {row['job_id']}: no log in logs/colab/")
        for r in filter(None, (row.get("runs") or "").split(";")):
            if r not in folder_ids:
                problems.append(f"job {row['job_id']}: lists run {r} which has no folder")
    stats["jobs"] = 0 if jobs.empty else len(jobs)

    # submissions
    subs = read_registry(root / "registry" / "submissions.csv")
    for row in subs.to_dict("records") if not subs.empty else []:
        f = root / row["file"]
        if not f.exists():
            problems.append(f"submission {row['sub_id']}: file {row['file']} missing")
        elif sha256(f)[:16] != row["sha256"]:
            problems.append(f"submission {row['sub_id']}: sha256 mismatch (file changed after registration)")
        if row["run_id"] not in folder_ids:
            problems.append(f"submission {row['sub_id']}: run {row['run_id']} has no folder")
        if row.get("submitted_at_utc") and not row.get("public_lb"):
            warnings.append(f"submission {row['sub_id']}: uploaded but public_lb not recorded")
    stats["submissions"] = 0 if subs.empty else len(subs)

    # reports: cited figures/tables exist
    docs = list((root / "research" / "reports").glob("*.md")) + list((root / "reports" / "extensions").glob("*.md"))
    stats["report_docs"] = len(docs)
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        for ref in sorted(set(re.findall(r"(?:reports/)?(figures|tables)/([A-Za-z0-9_.\-]+\.(?:png|csv|md))", text))):
            if not (root / "reports" / ref[0] / ref[1]).exists():
                warnings.append(f"{doc.relative_to(root).as_posix()}: cites {ref[0]}/{ref[1]} which does not exist")
    return problems, warnings, stats


def snapshot(root: Path) -> Path:
    out_dir = root / "backups"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"snapshot_{datetime.now(ARCHIVE_TZ):%Y%m%d-%H%M%S}.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in SNAPSHOT_ITEMS:
            p = root / item
            if not p.exists():
                continue
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                rel = f.relative_to(root)
                if SKIP_PARTS & set(rel.parts) or f.suffix == ".pyc" or any(s.endswith(".egg-info") for s in rel.parts):
                    continue
                z.write(f, rel.as_posix())
                n += 1
    print(f"snapshot: {out.relative_to(root)}  ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", action="store_true", help="also write backups/snapshot_<JST>.zip")
    ap.add_argument("--quiet-warnings", action="store_true", help="print only the warning count")
    a = ap.parse_args(argv)
    root = paths.archive_root("raw")
    problems, warnings, stats = check(root)
    print("archive:", root)
    print("stats  :", ", ".join(f"{k}={v}" for k, v in stats.items()))
    for p in problems:
        print("PROBLEM:", p)
    if a.quiet_warnings:
        print(f"warnings: {len(warnings)}")
    else:
        for w in warnings:
            print("warning:", w)
    print(f"result : {len(problems)} problem(s), {len(warnings)} warning(s)")
    if a.snapshot:
        snapshot(root)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
