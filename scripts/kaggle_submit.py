"""Upload a registered submission file to Kaggle with the official CLI.

    .venv\\Scripts\\python.exe scripts\\kaggle_submit.py --sub S003            # asks for confirmation
    .venv\\Scripts\\python.exe scripts\\kaggle_submit.py --sub S003 --dry-run  # show, never upload

Order of checks: (1) real data only; (2) Kaggle credentials present (only file
existence / env var names are checked, token contents are never read); (3) the
row exists in registry/submissions.csv, was not uploaded before, and its file
still matches the recorded sha256 and the sample_submission layout; (4) today's
uploads (UTC day) are below Kaggle's daily limit of 5 and the team limit of 3
(project.yaml competition.team_daily_limit; --force-limit overrides the team
limit only). Then it shows what will be
uploaded and asks you to type "yes" (unless --yes) before running

    kaggle competitions submit -c small-sample-classification-challenge -f <file> -m "<sub_id> <run_id> <desc>"

On success submitted_at_utc is recorded; afterwards record the public score with
scripts/record_lb.py. Credential setup: docs/SETUP.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import _common as C
from ssc import paths, submission
from ssc.data import sha256
from ssc.tracking import read_registry

EXIT_NO_CREDENTIALS = 2


def credential_sources() -> list[str]:
    """Where the Kaggle CLI would find credentials (existence only; nothing is read)."""
    found = []
    if os.environ.get("KAGGLE_API_TOKEN"):
        found.append("env KAGGLE_API_TOKEN")
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("env KAGGLE_USERNAME + KAGGLE_KEY")
    home = Path(os.path.expanduser("~")) / ".kaggle"
    config_dir = Path(os.environ["KAGGLE_CONFIG_DIR"]) if os.environ.get("KAGGLE_CONFIG_DIR") else home
    candidates = [config_dir / "kaggle.json", home / "access_token", home / "access_token.txt",
                  home / "credentials.json"]  # credentials.json = OAuth login via `kaggle auth login`
    found += [str(p) for p in dict.fromkeys(candidates) if p.is_file()]
    return found


def print_setup_help() -> None:
    sys.stdout.flush()  # keep stdout and stderr in order when both go to one log / pipe
    print("No Kaggle credentials found, so nothing was sent to Kaggle.", file=sys.stderr)
    print("Set up ONE of the following (details: docs/SETUP.md), then run this command again:", file=sys.stderr)
    print("  - API token: kaggle.com > Settings > API > Create New Token, save the token as", file=sys.stderr)
    print("    %USERPROFILE%\\.kaggle\\access_token   (or set the env var KAGGLE_API_TOKEN)", file=sys.stderr)
    print("  - legacy key: %USERPROFILE%\\.kaggle\\kaggle.json   (or env KAGGLE_USERNAME + KAGGLE_KEY)", file=sys.stderr)
    print("Never put the token inside the project folder or in git.", file=sys.stderr)
    print("Alternative without the CLI: upload the file on the competition web page and then run", file=sys.stderr)
    print("  " + C.hint("record_lb.py", "--sub S### --submitted-at now --public <SCORE>"), file=sys.stderr)


def kaggle_command() -> list[str]:
    exe = shutil.which("kaggle", path=str(Path(sys.executable).parent)) or shutil.which("kaggle")
    if not exe:
        C.die("kaggle CLI not found (it is installed in .venv; run this with .venv\\Scripts\\python.exe)", 1)
    return [exe]


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", required=True, help="submission id from make_submission.py, e.g. S003")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded; never call Kaggle")
    ap.add_argument("--force-limit", action="store_true",
                    help="allow an upload beyond the team limit of 3 per Kaggle day (never beyond Kaggle's 5)")
    C.add_data_arg(ap)
    args = ap.parse_args()
    source = paths.resolve_source(args.data)
    sub_id = args.sub.strip().upper()

    if source != "raw":
        C.die("only real-data submissions can be uploaded (--data raw). Synthetic submissions are scored "
              "locally by make_submission.py --data synthetic.", 2)

    creds = credential_sources()
    if not creds and not args.dry_run:
        print_setup_help()
        return EXIT_NO_CREDENTIALS

    reg = read_registry(paths.submissions_csv(source))
    if reg.empty or sub_id not in set(reg["sub_id"]):
        C.die(f"{sub_id} not found in {paths.submissions_csv(source)} (create it with make_submission.py)", 2)
    row = reg.loc[reg["sub_id"] == sub_id].iloc[0].to_dict()
    if row.get("submitted_at_utc"):
        C.die(f"{sub_id} was already uploaded at {row['submitted_at_utc']}; uploading it again wastes a daily slot.", 2)
    path = paths.archive_root(source) / row["file"]
    if not path.exists():
        C.die(f"submission file {path} is missing", 2)
    if sha256(path)[:16] != row["sha256"]:
        C.die(f"{path} changed after it was registered (sha256 mismatch); create a new submission instead", 2)
    problems = submission.validate_file(path, source)
    if problems:
        C.die(f"{path} is not a valid submission: {problems}", 2)

    used, limit = submission.submissions_today(source), C.daily_limit()
    if used >= limit:
        C.die(f"daily limit reached: {used}/{limit} uploads recorded for today's Kaggle day (UTC). "
              "The day resets at 00:00 UTC = 09:00 JST (08:00 HKT).", 1)
    # team rule (D008): at most team_daily_limit() uploads per Kaggle day, same check as make_submission.py
    team_limit = submission.team_daily_limit()
    if used >= team_limit and not args.force_limit:
        C.die(f"team daily limit reached: {used}/{team_limit} uploads recorded for today's Kaggle day (UTC) "
              f"(Kaggle allows {limit}). Upload after the reset (09:00 JST), or pass --force-limit if the team "
              "agreed to an extra upload today.", 1)

    slug = (C.load_project().get("competition") or {}).get("kaggle_slug", submission.COMPETITION)
    message = f"{sub_id} {row['run_id']} {row['description']}"[:500]
    cmd = kaggle_command() + ["competitions", "submit", "-c", slug, "-f", str(path), "-m", message]
    print("About to upload to Kaggle:")
    print(f"  competition     : {slug}")
    print(f"  sub_id / run_id : {sub_id} / {row['run_id']}")
    print(f"  file            : {path}  ({path.stat().st_size / 1e6:.2f} MB, sha256 {row['sha256']})")
    print(f"  message         : {message}")
    print(f"  cv auc_mean     : {C.fnum(row.get('cv_auc_mean'))} +/- {C.fnum(row.get('cv_auc_std_repeat'))}")
    print(f"  expected public : {C.fnum(row.get('expected_public'))}")
    print(f"  uploads today   : {used} / {limit} (this would be number {used + 1})")
    print(f"  credentials     : {', '.join(creds) if creds else 'NOT FOUND'}")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded. Command that would run:")
        print("  kaggle " + " ".join(f'"{a}"' if " " in a else a for a in cmd[1:]))
        if not creds:
            print()
            print_setup_help()
            return EXIT_NO_CREDENTIALS
        return 0

    if not args.yes:
        try:
            answer = input("Type 'yes' to upload (uses one of today's submissions): ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "yes":
            print("Aborted: nothing uploaded.")
            return 1

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        C.die(f"could not run the kaggle CLI: {e}", 1)
    output = (proc.stdout or "") + (proc.stderr or "")
    print(output.strip())
    if proc.returncode == 0 and "successfully submitted" in output.lower():
        when = submission.utc_now_iso()
        try:
            submission.record(sub_id, source=source, submitted_at_utc=when)
        except Exception as e:  # noqa: BLE001 - the upload happened; never lose that fact silently
            C.banner([f"UPLOADED, but recording failed: {type(e).__name__}: {str(e).splitlines()[0][:150]}",
                      "Record it NOW (it counts towards today's limit), after fixing the cause:",
                      "  " + C.hint("record_lb.py", f"--sub {sub_id} --submitted-at {when}")], "!")
            return 1
        print(f"\nRecorded {sub_id} submitted_at_utc = {when}")
        print("Next: read the public score on Kaggle (My Submissions), then")
        print("  " + C.hint("record_lb.py", f"--sub {sub_id} --public <PUBLIC_SCORE>"))
        return 0
    if proc.returncode == 0:
        print(f"\nUNCLEAR: the CLI exited normally but did not confirm the upload. Check "
              f"https://www.kaggle.com/competitions/{slug}/submissions ; if {sub_id} is listed there run")
        print("  " + C.hint("record_lb.py", f"--sub {sub_id} --submitted-at now"))
        return 1
    print(f"\nUpload FAILED (exit code {proc.returncode}); nothing was recorded.")
    return 1


if __name__ == "__main__":
    sys.exit(C.run_main(main))
