"""Turn an archived run into a Kaggle submission file and pre-register it.

    .venv\\Scripts\\python.exe scripts\\make_submission.py --run 20261022-101500_baseline_l2_logreg ^
        --desc "L2 logreg C=0.05, all features" --expected-public 0.80

--expected-public is REQUIRED: write down the public score you expect BEFORE
uploading (pre-registration). The registry then shows, for every submission,
how far the leaderboard was from what we believed, which is the material for
the report's submission history and reflection question (4).

Calls ssc.submission.create: submissions/S###_<run_id>.csv (validated against
sample_submission.csv) + one row in registry/submissions.csv. It does NOT upload.
Before writing it checks that test.csv is the file the run predicted and that its
ids equal sample_submission's ids in the same order.

--dry-run runs every check and prints what would be created (sub_id, file name,
sha256 prefix, score transform), but writes nothing.

Team rule: at most 3 uploads per Kaggle day (project.yaml competition.team_daily_limit,
default 3; Kaggle itself allows 5). When that many uploads are already recorded for
the current Kaggle day (UTC), the script stops; --force-limit overrides.

With --data synthetic the file is also scored against the hidden synthetic
labels (simulated public/private AUC, recorded with notes "simulated"), so the
whole flow can be rehearsed without using a real Kaggle submission.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

import _common as C
from ssc import paths, submission, synthetic
from ssc.tracking import load_run


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id (or a unique prefix) with a test_pred.npy")
    ap.add_argument("--desc", required=True, help="what this submission is, in a few words")
    ap.add_argument("--expected-public", required=True, type=float,
                    help="REQUIRED pre-registration: the public LB AUC you expect, written before uploading")
    ap.add_argument("--author", help="member id (default: $SSC_AUTHOR, else the run's author)")
    ap.add_argument("--notes", default="", help="free text stored in the registry row")
    ap.add_argument("--dry-run", action="store_true", help="check and print what would be created; write nothing")
    ap.add_argument("--force-limit", action="store_true",
                    help="create the file even though the team's daily upload limit is already used up")
    C.add_data_arg(ap)
    args = ap.parse_args()
    source = paths.resolve_source(args.data)

    if not 0.0 <= args.expected_public <= 1.0:
        C.die("--expected-public is an AUC and must be in [0, 1]", 2)
    if not args.desc.strip():
        C.die("--desc must not be empty", 2)
    run_id = C.resolve_run_id(args.run, source)
    run = load_run(run_id, source)
    C.check_run_source(run, source)
    summ = run.get("summary", {})
    C.print_archive(source)
    if summ.get("status") != "done":
        C.die(f"run {run_id} has status {summ.get('status')!r}; only finished runs can be submitted", 2)
    if "test_pred" not in run:
        C.die(f"run {run_id} has no test predictions (test_pred: none); re-run it with test_pred "
              "fold_mean or refit", 2)
    if summ.get("official_cv") is not True:
        C.banner([f"WARNING: {run_id} was not validated on the official CV",
                  "(official_cv is not true in summary.json). Its CV score is not comparable",
                  "with the other submissions in the registry."])

    # team rule: at most team_daily_limit() uploads per Kaggle day (UTC); checked BEFORE anything is written
    used, limit = submission.submissions_today(source), submission.team_daily_limit()
    now = datetime.now(timezone.utc)
    reset_jst = (datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
                 .astimezone(timezone(timedelta(hours=9))))
    day = (f"{used} upload{'' if used == 1 else 's'} recorded for Kaggle day {now.date().isoformat()} (UTC); team limit {limit} "
           f"(Kaggle allows {C.daily_limit()}); next reset {reset_jst:%Y-%m-%d %H:%M} JST")
    if used >= limit and not args.force_limit:
        if args.dry_run:
            C.banner([f"DRY RUN: the real command would stop here: {day}.",
                      "Pre-register after the reset, or pass --force-limit."], "!")
            return 1
        C.die(f"team daily limit reached: {day}. Create the submission after the reset, or pass "
              "--force-limit if the team agreed to an extra upload today.", 1)

    # author: --author > $SSC_AUTHOR > the run's author > project.yaml default_author (ssc.submission.create)
    try:
        row = submission.create(run_id, args.desc.strip(), expected_public=args.expected_public,
                                author=args.author, source=source, notes=args.notes, dry_run=args.dry_run)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        C.die(f"could not create the submission: {e}", 2)
    sub_id = row["sub_id"]
    path = paths.archive_root(source) / row["file"]
    print("\nDRY RUN: nothing written" if args.dry_run else "")
    print(f"sub_id          : {sub_id}{' (would be)' if args.dry_run else ''}")
    print(f"file            : {path}{' (not written)' if args.dry_run else ''}")
    print(f"sha256 (prefix) : {row['sha256']}   score transform: {row['score_transform']}")
    print(f"cv auc_mean     : {C.fnum(row['cv_auc_mean'])} +/- {C.fnum(row['cv_auc_std_repeat'])} (std across repeats)")
    print(f"expected public : {C.fnum(args.expected_public)}  (pre-registered)")
    print("checks          : test.csv = the run's test file, ids = sample_submission ids (same order), "
          "file format valid")
    print(f"uploads today   : {day}")
    if used >= limit:  # only reachable with --force-limit
        C.banner(f"Team daily limit already used ({used}/{limit}); continuing because of --force-limit.", "!")
    if args.dry_run:
        print("\nNothing was written. Run the same command without --dry-run to create the submission.")
        return 0

    if source == "synthetic":
        scores = synthetic.score_submission(pd.read_csv(path))
        notes = "simulated" if not args.notes else f"{args.notes}; simulated"
        submission.record(sub_id, source=source, public_lb=f"{scores['public_auc']:.5f}",
                          private_lb=f"{scores['private_auc']:.5f}", notes=notes)
        cv_mean = C.to_float(row["cv_auc_mean"])
        vs_cv = "" if cv_mean is None else f", public - cv {scores['public_auc'] - cv_mean:+.4f}"
        print(f"\nSimulated LB (synthetic hidden labels, {scores['n_public']} public rows):")
        print(f"  public  {scores['public_auc']:.4f}   (public - expected "
              f"{scores['public_auc'] - args.expected_public:+.4f}{vs_cv})")
        print(f"  private {scores['private_auc']:.4f}   (private - public {scores['private_auc'] - scores['public_auc']:+.4f})")
        print("Recorded in the sandbox registry with notes 'simulated'. Nothing is uploaded for synthetic data.")
        print("\nNext (rehearsal): " + C.hint("record_lb.py", f"--sub {sub_id} --submitted-at now --data synthetic"))
        return 0

    msg = f"{sub_id} {run_id} {args.desc.strip()}"
    slug = (C.load_project().get("competition") or {}).get("kaggle_slug", submission.COMPETITION)
    print("\nNext steps (nothing has been uploaded yet):")
    print(f"  1. upload ONE of two ways, with the description: {msg}")
    print(f"     a) web page: https://www.kaggle.com/competitions/{slug}/submit  (file above), then")
    print("        " + C.hint("record_lb.py", f"--sub {sub_id} --submitted-at now --public <PUBLIC_SCORE>"))
    print("     b) CLI: " + C.hint("kaggle_submit.py", f"--sub {sub_id}") + "   (records submitted_at), then")
    print("        " + C.hint("record_lb.py", f"--sub {sub_id} --public <PUBLIC_SCORE>"))
    print("  2. after the deadline: " + C.hint("record_lb.py", f"--sub {sub_id} --private <PRIVATE_SCORE>"))
    print("     final selection:    " + C.hint("record_lb.py", f"--sub {sub_id} --final yes"))
    return 0


if __name__ == "__main__":
    sys.exit(C.run_main(main))
