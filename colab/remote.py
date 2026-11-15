"""Run project scripts on a Google Colab VM instead of this laptop.

The official Colab CLI (PyPI: google-colab-cli) supports Linux/macOS only, so it
is installed inside WSL (Ubuntu-24.04) and driven from here through `wsl.exe`.

    .venv\\Scripts\\python.exe colab\\remote.py doctor
    .venv\\Scripts\\python.exe colab\\remote.py run -- scripts/run_experiment.py --config configs/baseline_l1_logreg.yaml
    .venv\\Scripts\\python.exe colab\\remote.py run --high-mem -- extensions/e1_validation_protocol/run.py --n-jobs -1
    .venv\\Scripts\\python.exe colab\\remote.py down

What `run` does
  1. ensures a session exists (default name "ssc"), then bootstraps it once:
     a Python 3.11 venv at /content/venv with colab/requirements-colab.txt
     (same versions as requirements-lock.txt) and the ssc package (editable);
  2. uploads the code bundle (src, scripts, extensions, research, configs, colab, project.yaml,
     pyproject.toml) and, if missing or changed, the data (data/raw, checksummed); large
     research data files (*.npz / *.npy over 1 MB under research/) travel separately as
     "research data", uploaded only when they change, so the code bundle stays small;
  3. starts the command detached on the VM with SSC_ARCHIVE_ROOT set to a fresh
     job folder, and passes the local git commit, author and compute info as env vars;
  4. streams the log over one long-lived connection (reconnecting from the last byte if it
     drops) until the command exits (Ctrl-C detaches; the job keeps running and
     `fetch JOB_ID` collects it later); an unresponsive session is recreated before launch;
  5. downloads the job archive and merges it into the LOCAL archive: run folders into
     experiments/, rows into registry/experiments.csv, report outputs into reports/,
     the log into logs/colab/<job>.log, and one row into registry/compute_jobs.csv.

Submissions are never created on the VM; make_submission.py stays local.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssc import paths  # noqa: E402
from ssc.tracking import EXPERIMENT_COLUMNS, append_csv_row, now_iso, read_registry  # noqa: E402

WSL_DISTRO = "Ubuntu-24.04"
DEFAULT_ACCOUNT = "pro1"                  # the Google account logged in under the normal WSL $HOME
DEFAULT_SESSION = "ssc"
REMOTE_ML = "/content/ml"
REMOTE_JOBS = "/content/jobs"
REMOTE_VENV = "/content/venv"
REMOTE_PY = f"{REMOTE_VENV}/bin/python"
CODE_ITEMS = ["src", "scripts", "extensions", "research", "configs", "colab", "project.yaml", "pyproject.toml"]
DATA_FILES = ["train.csv", "test.csv", "sample_submission.csv", "SHA256SUMS.txt"]
# Packed replica arrays (e.g. research/r935_r933_pooled.npz, 3.5 MB) are data, not code: they are left out of
# the code bundle (which is re-uploaded and kept in logs/code_bundles on every code change) and synced as
# "research data" instead, into the same place on the VM, so scripts that read them run unchanged.
RESEARCH_DATA_SUFFIXES = (".npz", ".npy")
RESEARCH_DATA_MIN_BYTES = 1_000_000
JOB_COLUMNS = ["job_id", "created_at", "finished_at", "account", "session", "hardware", "data_source",
               "command", "exit_code", "wall_seconds", "runs", "git_commit", "git_dirty", "author", "status"]

# Several Google accounts can be used side by side: every account other than the default
# keeps its own Colab CLI state (token, sessions) under $HOME/.colab-accounts/<name> in WSL.
ACCOUNT = DEFAULT_ACCOUNT


def tmp_dir(session: str) -> Path:
    d = paths.SANDBOX / "colab_tmp" / f"{ACCOUNT}_{session}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- WSL / CLI bridge
def wsl_path(p: Path) -> str:
    p = Path(p).resolve()
    drive, rest = p.drive.rstrip(":").lower(), p.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _colab_prefix() -> str:
    if ACCOUNT == DEFAULT_ACCOUNT:
        return '"$HOME/.local/bin/colab"'
    return (f'C="$HOME/.local/bin/colab"; A="$HOME/.colab-accounts/{ACCOUNT}"; mkdir -p "$A"; '
            f'HOME="$A" "$C"')


def colab(*args: str, timeout: float = 600, check: bool = True, retries: int = 0) -> str:
    """Run `colab <args>` inside WSL (as the selected account) and return its stdout (utf-8)."""
    cmd = ["wsl", "-d", WSL_DISTRO, "-e", "bash", "-lc",  # -e: no outer shell re-expanding $VARS
           " ".join([_colab_prefix()] + [_sh_quote(a) for a in args])]
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt < retries:
                print(f"[remote] colab {args[0]} timed out after {timeout:.0f}s; retrying ({attempt + 1}/{retries})")
                continue
            raise
        out = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
        if r.returncode == 0:
            return out
        transient = any(s in err or s in out for s in ("Connection was lost", "ConnectionResetError", "503", "502"))
        if transient and attempt < retries:
            print(f"[remote] colab {args[0]}: transient error; retrying ({attempt + 1}/{retries})")
            time.sleep(3 * (attempt + 1))
            continue
        if check:
            tail = "\n".join((err or out).strip().splitlines()[-12:])
            raise RuntimeError(f"colab {' '.join(args[:2])} failed (exit {r.returncode}):\n{tail}")
        return out
    return ""


def _sh_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def exec_py(session: str, code: str, timeout: float = 30, retries: int = 4) -> str:
    """Execute a short Python cell in the session's kernel; returns printed stdout.

    Cells here are tiny (launch / poll / marker reads), so a call that takes much longer
    than `timeout` is a hung websocket, not work: kill it and retry (launch is idempotent)."""
    f = tmp_dir(session) / f"cell_{hashlib.sha1(code.encode()).hexdigest()[:10]}.py"
    f.write_text(code, encoding="utf-8")
    return colab("exec", "-s", session, "-f", wsl_path(f), "--timeout", str(timeout),
                 timeout=timeout + 45, retries=retries)


def marked(out: str, tag: str) -> str:
    """Extract the payload printed between <<tag>> markers by a remote cell."""
    a, b = f"<<{tag}>>", f"<</{tag}>>"
    if a not in out or b not in out:
        raise RuntimeError(f"remote output lacks {tag} marker:\n{out[-1500:]}")
    return out.split(a, 1)[1].split(b, 1)[0]


# --------------------------------------------------------------------------- sessions
def active_sessions() -> dict[str, str]:
    out = colab("sessions", timeout=90, retries=1)
    found = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            found[line[1:line.index("]")]] = line
    return found


def ensure_session(session: str, gpu: str | None, high_mem: bool) -> str:
    sessions = active_sessions()
    if session in sessions:
        return sessions[session]
    args = ["new", "-s", session]
    if gpu:
        args += ["--gpu", gpu]
    if high_mem:
        args.append("--high-mem")
    print(f"[remote] creating Colab session '{session}' ({'high-mem ' if high_mem else ''}{gpu or 'CPU'}) ...")
    colab(*args, timeout=600)
    time.sleep(5)  # the kernel websocket is sometimes not ready immediately after creation
    return active_sessions().get(session, "")


def hardware_label(session_line: str) -> str:
    parts = [p.strip() for p in session_line.split("|")[1:]]
    return " ".join(p.split(":", 1)[-1].strip() for p in parts) or "unknown"


# --------------------------------------------------------------------------- remote shell jobs
# Every `colab exec` opens a new websocket, and on a degraded runtime new connections
# can hang. So the design minimises connections: one cell launches a job, ONE long-lived
# cell streams its log (reconnecting from the last byte offset if the socket drops), and
# all sync markers are read in a single cell.
LAUNCH = r'''
import json, os, subprocess
jd = {jd!r}
os.makedirs(jd, exist_ok=True)
if os.path.exists(os.path.join(jd, "job.json")):
    print("<<LAUNCH>>exists<</LAUNCH>>")
else:
    env = dict(os.environ, **{env!r})
    env["JD"] = jd
    with open(os.path.join(jd, "job.json"), "w") as f:
        json.dump({meta!r}, f)
    script = 'cd "$WORKDIR" && "$@" > "$JD/log.txt" 2>&1; echo $? > "$JD/exit_code"'
    p = subprocess.Popen(["bash", "-c", script, "job"] + {cmd!r}, env=env, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("<<LAUNCH>>started pid=%d<</LAUNCH>>" % p.pid)
'''

STREAM = r'''
import os, sys, time
jd, off, T = {jd!r}, {off}, {max_secs}
log, ecf = os.path.join(jd, "log.txt"), os.path.join(jd, "exit_code")
t0 = last = time.time()
ec = ""
while True:
    done = os.path.exists(ecf)
    if os.path.exists(log):
        with open(log, "rb") as f:
            f.seek(off)
            data = f.read()
        if data:
            off += len(data)
            sys.stdout.write(data.decode("utf-8", "replace").replace("<<", "< <"))
            sys.stdout.flush()
            last = time.time()
    if done:
        ec = open(ecf).read().strip()
        break
    if time.time() - t0 > T:
        break
    if time.time() - last > 20:
        print("<<HB>>", flush=True)
        last = time.time()
    time.sleep(2)
print("\n<<OFF>>%d<</OFF>>" % off)
print("<<EC>>" + ec + "<</EC>>")
'''


def colab_stream(args: list[str], timeout: float) -> tuple[int, str, str]:
    """Like colab() but echoes stdout lines live (minus protocol markers)."""
    cmd = ["wsl", "-d", WSL_DISTRO, "-e", "bash", "-lc",  # -e: no outer shell re-expanding $VARS
           " ".join([_colab_prefix()] + [_sh_quote(a) for a in args])]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(timeout, p.kill)
    timer.start()
    out_lines = []
    try:
        for raw in p.stdout:
            line = raw.decode("utf-8", "replace")
            out_lines.append(line)
            if "<<" not in line:
                sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
                sys.stdout.flush()
        p.wait()
    finally:
        timer.cancel()
    err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
    return p.returncode, "".join(out_lines), err


def launch(session: str, job_dir: str, cmd: list[str], env: dict, meta: dict, workdir: str = REMOTE_ML) -> str:
    env = {**env, "WORKDIR": workdir}
    out = exec_py(session, LAUNCH.format(jd=job_dir, env=env, meta=meta, cmd=cmd), timeout=30, retries=6)
    return marked(out, "LAUNCH")


def follow(session: str, job_dir: str, quiet: bool = False, cell_secs: int = 540,
           max_disconnect: float = 1800) -> int:
    """Stream the remote log until the job writes its exit code; survives dropped sockets."""
    off, failing_since = 0, None
    while True:
        code = STREAM.format(jd=job_dir, off=off, max_secs=cell_secs)
        f = tmp_dir(session) / f"stream_{hashlib.sha1(code.encode()).hexdigest()[:10]}.py"
        f.write_text(code, encoding="utf-8")
        args = ["exec", "-s", session, "-f", wsl_path(f), "--timeout", str(cell_secs + 60)]
        if quiet:
            try:
                out, rc = colab(*args, timeout=cell_secs + 120, check=False), 0
            except subprocess.TimeoutExpired:
                out, rc = "", 1
        else:
            rc, out, _ = colab_stream(args, timeout=cell_secs + 120)
        if "<<OFF>>" in out and "<</EC>>" in out:
            failing_since = None
            off = int(marked(out, "OFF"))
            ec = marked(out, "EC").strip()
            if ec:
                return int(ec)
            continue
        failing_since = failing_since or time.time()
        down = time.time() - failing_since
        if down > max_disconnect:
            raise RuntimeError(f"lost contact with the VM for {down / 60:.0f} min; the job may still be running "
                               f"(collect it later with `remote.py fetch`)")
        print(f"[remote] log stream dropped (rc={rc}); reconnecting in 10 s (offline {down:.0f}s)")
        time.sleep(10)


def run_remote_shell(session: str, name: str, shell: str, quiet: bool = True) -> int:
    """Run a bash snippet on the VM as a detached job and wait for it."""
    jd = f"{REMOTE_JOBS}/_{name}_{datetime.now():%Y%m%d-%H%M%S}"
    launch(session, jd, ["bash", "-c", shell], {}, {"kind": name}, workdir="/content")
    ec = follow(session, jd, quiet=quiet)
    if ec != 0:
        code = f'print("<<L>>" + open({jd + "/log.txt"!r}, errors="replace").read()[-3000:].replace("<<", "< <") + "<</L>>")'
        log = marked(exec_py(session, code), "L")
        raise RuntimeError(f"remote step '{name}' failed (exit {ec}):\n{log}")
    return ec


# --------------------------------------------------------------------------- sync
def research_data_files() -> list[Path]:
    """Large binary data files under research/ that travel as research data, not in the code bundle."""
    base = ROOT / "research"
    if not base.exists():
        return []
    return sorted(f for f in base.rglob("*") if f.is_file() and f.suffix.lower() in RESEARCH_DATA_SUFFIXES
                  and f.stat().st_size > RESEARCH_DATA_MIN_BYTES and "__pycache__" not in f.parts)


def _tar_bytes(items: list[tuple[Path, str]], exclude: frozenset[str] = frozenset()) -> bytes:
    """Deterministic .tar.gz of `items` (path, arcname); `exclude` = arcnames of files to leave out."""
    buf = io.BytesIO()

    def skip(ti: tarfile.TarInfo):
        parts = Path(ti.name).parts
        if any(p in ("__pycache__", ".pytest_cache", ".ipynb_checkpoints") or p.endswith(".egg-info") for p in parts) \
                or ti.name.endswith(".pyc") or ti.name in exclude:
            return None
        ti.mtime = 0  # deterministic bundle -> stable hash
        return ti
    # gzip header mtime=0 as well, otherwise every bundle hashes differently and is re-uploaded
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for src, arc in items:
                if src.exists():
                    tar.add(src, arcname=arc, filter=skip)
    return buf.getvalue()


SYNC_MARKERS = ("code", "data", "env", "rdata")  # <REMOTE_ML>/.sync_<name>, written by prepare() on the VM


def remote_markers(session: str, retries: int = 6) -> dict:
    code = ("import json, os\nm = {}\n"
            f"for n in {SYNC_MARKERS!r}:\n    p = {REMOTE_ML!r} + '/.sync_' + n\n"
            "    m[n] = open(p).read().strip() if os.path.exists(p) else ''\n"
            "print('<<M>>' + json.dumps(m) + '<</M>>')")
    return json.loads(marked(exec_py(session, code, retries=retries), "M"))


def code_bundle() -> tuple[bytes, list[Path]]:
    """The code bundle (CODE_ITEMS without the research data files) and the research data files."""
    big = research_data_files()
    exclude = frozenset(f.relative_to(ROOT).as_posix() for f in big)
    return _tar_bytes([(ROOT / it, it) for it in CODE_ITEMS], exclude=exclude), big


def vm_markers(session: str) -> dict:
    """--no-sync: nothing is uploaded, so record what the VM already has, i.e. the .sync_* markers
    the last prepare() left there (code bundle, env, data, research data hashes)."""
    try:
        have = remote_markers(session, retries=2)
    except (subprocess.TimeoutExpired, RuntimeError) as e:
        print(f"[remote] --no-sync: could not read the VM's sync markers ({type(e).__name__}); "
              "the run is recorded as unsynced")
        return {}
    have = {k: v for k, v in have.items() if v}
    local = hashlib.sha256(code_bundle()[0]).hexdigest()[:16]
    print(f"[remote] --no-sync: VM has code bundle {have.get('code', '(none)')}, env {have.get('env', '(none)')}"
          + ("" if have.get("code") in (None, local) else f"; the local code differs (bundle {local}), not uploaded"))
    return have


def prepare(session: str, source: str = "raw", with_data: bool = True) -> dict:
    """Make the VM match the laptop: code bundle, data, research data, Python env. One marker
    read, at most three uploads, and one remote shell step that extracts and (re)builds what changed.

    The exact code bundle is also kept in <archive>/logs/code_bundles/<hash>.tar.gz and its
    hash goes into every run's meta.json (compute.code_bundle): runs stay reproducible
    even before the first git commit (D005). Research data files (RESEARCH_DATA_SUFFIXES over
    RESEARCH_DATA_MIN_BYTES under research/) are not in the bundle: their sha256 list is kept
    as logs/code_bundles/rdata_<hash>.json and the hash goes into compute.research_data.
    Returns the bundle/env/data/research-data hashes."""
    code_blob, big = code_bundle()
    want = {"code": hashlib.sha256(code_blob).hexdigest()[:16],
            "env": hashlib.sha256((ROOT / "colab" / "requirements-colab.txt").read_bytes()).hexdigest()[:16]}
    keep = paths.archive_root(source) / "logs" / "code_bundles" / f"{want['code']}.tar.gz"
    if not keep.exists():
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_bytes(code_blob)
    rdata_blob = _tar_bytes([(f, f.relative_to(ROOT).as_posix()) for f in big]) if big else b""
    want["rdata"] = hashlib.sha256(rdata_blob).hexdigest()[:16] if big else ""
    if big:
        manifest = keep.parent / f"rdata_{want['rdata']}.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({f.relative_to(ROOT).as_posix(): {
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size} for f in big},
                indent=1), encoding="utf-8")
    data_blob = b""
    if with_data:
        missing = [f for f in DATA_FILES if not (paths.DATA_RAW / f).exists()]
        if missing:
            raise FileNotFoundError(f"data/raw is missing {missing}; see docs/SETUP.md")
        data_blob = _tar_bytes([(paths.DATA_RAW / f, f"data/raw/{f}") for f in DATA_FILES])
        want["data"] = hashlib.sha256(data_blob).hexdigest()[:16]
    have = remote_markers(session)
    steps = []
    clean = " ".join(_sh_quote(f"{REMOTE_ML}/{c}") for c in CODE_ITEMS if not c.endswith((".yaml", ".toml")))
    rdata = "/content/rdata.tar.gz"  # the research data now on the VM (re-extracted after every code update)
    for name, blob in (("code", code_blob), ("data", data_blob), ("rdata", rdata_blob)):
        if name in want and have.get(name) != want[name]:
            remote = f"/content/{name}_new.tar.gz" if name == "rdata" else f"/content/{name}.tar.gz"
            if blob:
                local = tmp_dir(session) / f"{name}.tar.gz"
                local.write_bytes(blob)
                label = "research data " + ", ".join(f.name for f in big) if name == "rdata" else name
                print(f"[remote] uploading {label} ({len(blob) / 1e6:.1f} MB) ...")
                colab("upload", "-s", session, wsl_path(local), remote, timeout=900, retries=3)
            if name == "rdata":  # drop the previous research data files, then unpack the new set (if any)
                new = f"mv {remote} {rdata}; tar -xzf {rdata} -C {REMOTE_ML}; " if blob else f"rm -f {rdata}; "
                steps.append(f"if [ -f {rdata} ]; then (cd {REMOTE_ML} && tar -tzf {rdata} | xargs -r rm -f); fi; "
                             f"{new}echo {want[name]} > {REMOTE_ML}/.sync_{name}")
                continue
            rm = f"rm -rf {clean}; " if name == "code" else ""
            back = f"if [ -f {rdata} ]; then tar -xzf {rdata} -C {REMOTE_ML}; fi; " if name == "code" else ""
            steps.append(f"mkdir -p {REMOTE_ML}; {rm}tar -xzf /content/{name}.tar.gz -C {REMOTE_ML}; {back}"
                         f"echo {want[name]} > {REMOTE_ML}/.sync_{name}")
    if have.get("env") != want["env"] or have.get("code") != want["code"]:
        if have.get("env") != want["env"]:
            print("[remote] building the VM environment (Python 3.11 venv + pinned packages, ~1-2 min) ...")
        steps.append(f"""command -v uv >/dev/null || pip install -q uv
[ -x {REMOTE_PY} ] || uv venv {REMOTE_VENV} --python 3.11 -q
[ "$(cat {REMOTE_ML}/.sync_env 2>/dev/null)" = "{want['env']}" ] || uv pip install --python {REMOTE_PY} -q -r {REMOTE_ML}/colab/requirements-colab.txt
uv pip install --python {REMOTE_PY} -q --no-deps -e {REMOTE_ML}
{REMOTE_PY} -c "import ssc, sklearn, numpy, pandas; print('env ok', sklearn.__version__, numpy.__version__, pandas.__version__)"
echo {want['env']} > {REMOTE_ML}/.sync_env""")
    if steps:
        run_remote_shell(session, "prepare", "set -e\n" + "\n".join(steps))
    else:
        print("[remote] VM is up to date")
    return want


def synthetic_on_vm(session: str) -> None:
    """Regenerate data/synthetic on the VM (deterministic) instead of uploading 56 MB, and
    check it matches the laptop copy (content hash; pandas writes CRLF on Windows)."""
    local = paths.DATA_SYNTHETIC / "train.csv"
    want = hashlib.sha256(local.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if local.exists() else ""
    run_remote_shell(session, "synthetic", f"""set -e
cd {REMOTE_ML}
[ -f data/synthetic/train.csv ] || {REMOTE_PY} -c "from ssc import synthetic; synthetic.write()"
{REMOTE_PY} -c "import hashlib; b=open('data/synthetic/train.csv','rb').read().replace(b'\\r\\n', b'\\n'); open('data/synthetic/.hash','w').write(hashlib.sha256(b).hexdigest())" """)
    got = marked(exec_py(session, f'print("<<H>>" + open({REMOTE_ML + "/data/synthetic/.hash"!r}).read() + "<</H>>")'), "H")
    if want and got != want:
        raise RuntimeError("synthetic data on the VM differs from local data/synthetic (different numpy?)")


def prepare_with_heal(a, source: str) -> tuple[str, dict]:
    """ensure_session + prepare; if the runtime is unresponsive before any job started,
    recreate the session once (a fresh VM fixes hung websockets)."""
    for attempt in (1, 2):
        line = ensure_session(a.session, a.gpu, a.high_mem)
        if a.no_sync:  # no upload and never a new VM (it would have no code); just record the VM's markers
            return line, vm_markers(a.session)
        try:
            want = prepare(a.session, source)
            if source == "synthetic":
                synthetic_on_vm(a.session)
            return line, want
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            if attempt == 2 or getattr(a, "no_heal", False):
                raise
            print(f"[remote] session '{a.session}' looks unhealthy ({type(e).__name__}); recreating it ...")
            colab("stop", "-s", a.session, timeout=180, check=False, retries=1)
            time.sleep(5)
    return "", {}


# --------------------------------------------------------------------------- merge results
def project_timezone() -> str:
    """IANA timezone of the team's laptops (project.yaml `timezone`, default Asia/Tokyo)."""
    try:
        import yaml
        cfg = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8")) or {}
        return str(cfg.get("timezone") or "Asia/Tokyo")
    except (OSError, ImportError):
        return "Asia/Tokyo"


def git_state() -> tuple[str, str]:
    def g(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = g("rev-parse", "--short=10", "HEAD") or "no-commit"
    status = g("status", "--porcelain")
    return commit, ("" if status is None else str(bool(status)))


def merge_job(job_id: str, tar_path: Path, data_source: str) -> list[str]:
    """Merge a downloaded job archive into the local archive; returns run ids."""
    stage = paths.SANDBOX / "colab_jobs" / job_id
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(stage, filter="data")
    jd = stage / job_id
    arch = jd / "archive"
    runs: list[str] = []
    log_text = (jd / "log.txt").read_text(encoding="utf-8", errors="replace") if (jd / "log.txt").exists() else ""

    for run_dir in sorted((arch / "experiments").glob("*")) if (arch / "experiments").exists() else []:
        if not run_dir.is_dir():
            continue
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
        source = meta.get("data_source", data_source)
        dest = paths.experiments_dir(source) / run_dir.name
        if dest.exists():
            print(f"[remote] WARNING {dest} already exists locally; kept local copy")
            continue
        paths.ensure_archive(source)
        shutil.copytree(run_dir, dest)
        (dest / "colab_job.log").write_text(log_text, encoding="utf-8")
        runs.append(run_dir.name)

    reg = read_registry(arch / "registry" / "experiments.csv")
    for row in reg.to_dict("records") if not reg.empty else []:
        if row.get("run_id") in runs:
            append_csv_row(paths.experiments_csv(row.get("data_source") or data_source), EXPERIMENT_COLUMNS, row)
    if (arch / "registry" / "submissions.csv").exists():
        print("[remote] WARNING the job wrote submissions on the VM; ignored (create submissions locally)")

    for sub in ("figures", "tables"):
        src = arch / "reports" / sub
        if src.exists():
            dst = paths.figures_dir(data_source) if sub == "figures" else paths.tables_dir(data_source)
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file():
                    target = dst / f.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, target)

    logs = paths.archive_root(data_source) / "logs" / "colab"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{job_id}.log").write_text(log_text, encoding="utf-8")
    return runs


def data_source_of(cmd: list[str]) -> str:
    for i, a in enumerate(cmd):
        if a == "--data" and i + 1 < len(cmd):
            return cmd[i + 1]
        if a.startswith("--data="):
            return a.split("=", 1)[1]
    return "raw"


def fetch(session: str, job_id: str, data_source: str) -> tuple[list[str], int | None]:
    jd = f"{REMOTE_JOBS}/{job_id}"
    run_remote_shell(session, "pack", f"set -e; cd {REMOTE_JOBS}; tar -czf {jd}.tar.gz {job_id}")
    local = tmp_dir(session) / f"{job_id}.tar.gz"
    colab("download", "-s", session, f"{jd}.tar.gz", wsl_path(local), timeout=900, retries=1)
    runs = merge_job(job_id, local, data_source)
    ec_file = paths.SANDBOX / "colab_jobs" / job_id / job_id / "exit_code"
    ec = int(ec_file.read_text().strip()) if ec_file.exists() else None
    return runs, ec


# --------------------------------------------------------------------------- commands
def cmd_doctor(a) -> int:
    print("[doctor] WSL distro:", WSL_DISTRO)
    print("[doctor] colab version:", colab("version", timeout=60).strip())
    print(colab("usage", timeout=90).strip())
    print("[doctor] sessions:\n" + (colab("sessions", timeout=90).strip() or "(none)"))
    return 0


def cmd_up(a) -> int:
    a.no_sync = False
    line, _ = prepare_with_heal(a, "raw")
    print("[remote] session ready:", line or a.session)
    return 0


def cmd_run(a) -> int:
    cmd = list(a.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: remote.py run [options] -- <script.py> [args...]")
        return 2
    if cmd[0] in ("python", "python3"):
        cmd = cmd[1:]
    script = ROOT / cmd[0]
    if not script.exists():
        print(f"[remote] {cmd[0]} not found in the project")
        return 2
    source = data_source_of(cmd)
    line, synced = prepare_with_heal(a, source)
    hw = hardware_label(line)

    commit, dirty = git_state()
    if dirty == "True" and source == "raw":
        print("[remote] NOTE: working tree has uncommitted changes; meta.json will record git_dirty=True")
    author = a.author or ""
    job_id = f"J{datetime.now():%Y%m%d-%H%M%S}-{ACCOUNT}-{a.session}"
    compute = {"host": "colab", "account": ACCOUNT, "session": a.session, "hardware": hw, "job_id": job_id,
               "code_bundle": synced.get("code", "unsynced (--no-sync)"), "env_hash": synced.get("env", ""),
               "research_data": synced.get("rdata", "")}
    if a.no_sync:  # code_bundle / env_hash / research_data are then the VM's markers from an earlier sync
        compute["no_sync"] = True
    env = {"SSC_ARCHIVE_ROOT": f"{REMOTE_JOBS}/{job_id}/archive", "SSC_GIT_COMMIT": commit,
           "SSC_GIT_DIRTY": dirty, "SSC_COMPUTE": json.dumps(compute), "PYTHONUNBUFFERED": "1",
           "MPLBACKEND": "Agg", "TZ": project_timezone()}  # run_ids use local time: keep VM == laptop
    if author:
        env["SSC_AUTHOR"] = author
    remote_cmd = [REMOTE_PY] + [c.replace("\\", "/") for c in cmd]
    started = time.time()
    created = now_iso()
    print(f"[remote] job {job_id} on '{a.session}' [{hw}]: {' '.join(cmd)}")
    launch(a.session, f"{REMOTE_JOBS}/{job_id}", remote_cmd, env, {"cmd": cmd, "created_at": created})
    try:
        ec = follow(a.session, f"{REMOTE_JOBS}/{job_id}")
    except KeyboardInterrupt:
        print(f"\n[remote] detached; the job keeps running. Collect it later with:\n"
              f"  .venv\\Scripts\\python.exe colab\\remote.py fetch {job_id} -a {ACCOUNT} -s {a.session} --data {source}")
        return 130
    runs, _ = fetch(a.session, job_id, source)
    record_job(job_id, created, a.session, hw, source, cmd, ec, time.time() - started, runs, commit, dirty, author)
    print(f"\n[remote] job {job_id} exit={ec}; merged runs: {', '.join(runs) or '(none)'}")
    if a.stop:
        colab("stop", "-s", a.session, timeout=120)
        print(f"[remote] session '{a.session}' stopped")
    return ec


def record_job(job_id, created, session, hw, source, cmd, ec, secs, runs, commit, dirty, author) -> None:
    path = paths.registry_dir(source) / "compute_jobs.csv"
    old = read_registry(path)
    if not old.empty and list(old.columns) != JOB_COLUMNS:  # header changed: rewrite with the new columns
        old.reindex(columns=JOB_COLUMNS).fillna("").to_csv(path, index=False, encoding="utf-8-sig")
    append_csv_row(path, JOB_COLUMNS, {
        "job_id": job_id, "created_at": created, "finished_at": now_iso(), "account": ACCOUNT, "session": session,
        "hardware": hw, "data_source": source, "command": " ".join(cmd), "exit_code": ec,
        "wall_seconds": f"{secs:.0f}", "runs": ";".join(runs), "git_commit": commit,
        "git_dirty": dirty, "author": author, "status": "done" if ec == 0 else "failed"})


def cmd_fetch(a) -> int:
    runs, ec = fetch(a.session, a.job_id, a.data)
    record_job(a.job_id, "", a.session, "", a.data, ["(fetched later)"], ec, 0, runs, *git_state(), a.author or "")
    print(f"[remote] fetched {a.job_id} exit={ec}; merged runs: {', '.join(runs) or '(none)'}")
    return 0


def cmd_down(a) -> int:
    print(colab("stop", "-s", a.session, timeout=120).strip())
    return 0


def cmd_jobs(a) -> int:
    reg = read_registry(paths.registry_dir(a.data) / "compute_jobs.csv")
    if reg.empty:
        print("(no jobs recorded)")
        return 0
    cols = ["job_id", "hardware", "exit_code", "wall_seconds", "runs", "command"]
    for row in reg[cols].tail(a.n).to_dict("records"):
        print(" | ".join(str(row[c])[:70] for c in cols))
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)  # live progress even when output is redirected
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-s", "--session", default=DEFAULT_SESSION, help=f"Colab session name (default {DEFAULT_SESSION})")
    common.add_argument("-a", "--account", default=DEFAULT_ACCOUNT,
                        help=f"Google account profile (default {DEFAULT_ACCOUNT}; others live in WSL ~/.colab-accounts/<name>)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", parents=[common], help="check WSL, Colab CLI, login, balance, sessions")
    for name in ("up", "run"):
        sp = sub.add_parser(name, parents=[common],
                            help="create+prepare the session" if name == "up" else "run a project script on Colab")
        sp.add_argument("--gpu", default=None, help="T4 | L4 | A100 ... (not needed for this project)")
        sp.add_argument("--high-mem", action="store_true", help="high-RAM shape (more CPU cores; Pro/Pro+)")
        if name == "run":
            sp.add_argument("--author", default=None)
            sp.add_argument("--no-sync", action="store_true", help="skip code/data upload")
            sp.add_argument("--no-heal", action="store_true", help="never recreate an unresponsive session")
            sp.add_argument("--stop", action="store_true", help="stop the session after the job")
            sp.add_argument("command", nargs=argparse.REMAINDER, help="-- <script.py> [args]")
    sf = sub.add_parser("fetch", parents=[common], help="collect a job that ran while detached")
    sf.add_argument("job_id")
    sf.add_argument("--data", default="raw", choices=paths.SOURCES)
    sf.add_argument("--author", default=None)
    sub.add_parser("down", parents=[common], help="stop the session (releases the VM)")
    sj = sub.add_parser("jobs", parents=[common], help="list recorded Colab jobs")
    sj.add_argument("--data", default="raw", choices=paths.SOURCES)
    sj.add_argument("-n", type=int, default=20)
    a = p.parse_args(argv)
    global ACCOUNT
    if not a.account.replace("-", "").replace("_", "").isalnum():
        p.error("--account must be alphanumeric (plus - and _)")
    ACCOUNT = a.account
    return {"doctor": cmd_doctor, "up": cmd_up, "run": cmd_run, "fetch": cmd_fetch,
            "down": cmd_down, "jobs": cmd_jobs}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
