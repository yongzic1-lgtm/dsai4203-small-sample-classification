"""R424 (research round 2, agent "r2_noise") - pack the archived r420 replica arrays (and the R922 / R421 likelihood
tables) into ONE input file for the pooled analysis research/r425 on Colab.

Pure file I/O, no computation and no Run (same pattern as research/r935_pack_r933.py): remote.py syncs only code
(src, scripts, extensions, research, ...) and data/raw, so r425 reads this packed copy, which travels in the research/
code bundle (archived under its hash in logs/code_bundles/). The manifest stores the sha256 of every source file so
r425 can show that it analysed exactly the archived numbers. Only slices are packed (to keep the file small):

  chk_pub8 / chk_priv8   float64 (n, 4, 16, 10): csv-score public / private AUCs at the 4 r933 levels, all procs,
                         for the bit-exact comparison with research/r935_r933_pooled.npz
  priv8 / pub8           float32 (n, L, 16, 4): csv-score private / public AUCs of S003, r201, r209, Bayes, all levels
  abc                    float32 rows with |S003 public - 0.99923| <= 1.5e-4 (columns in manifest)
  lik_r922 / lik_r421    likelihood tables (flip_y grid) from reports/tables/raw

    .venv\\Scripts\\python.exe research\\r424_pack_r420.py            # writes research/r424_r420_pooled.npz
Delete research/r424_r420_pooled.npz once r425 is frozen (it travels in every code bundle).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
TAB = ROOT / "reports" / "tables"
OUT = ROOT / "research" / "r424_r420_pooled.npz"
ORIG = [0.0, 0.0025, 0.005, 0.01]
SEL = ["s003_gmm2_hv", "r201_qda_block", "r209_selftrain_qda_lw", "bayes"]
STORE = 1.5e-4
OBS = 0.99923


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs = []
for d in sorted(EXP.glob("*_r420_replica_flip_grid_s*")):
    if d.name.endswith("_quick"):
        continue
    s = d / "summary.json"
    if not (s.exists() and json.loads(s.read_text(encoding="utf-8")).get("status") == "done"):
        continue
    runs.append(d)
if not runs:
    raise SystemExit("no finished r420 runs")

import yaml  # noqa: E402

parts = {k: [] for k in ("seeds", "chk_pub8", "chk_priv8", "priv8", "pub8", "abc")}
manifest = {"runs": [], "orig_levels": ORIG, "sel_procs": SEL, "abc_store": STORE, "obs": OBS}
flips0 = procs0 = abc_cols0 = None
for d in runs:
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    flips, procs = [float(x) for x in cfg["flips"]], list(cfg["procedures"])
    abc_cols = json.loads((d / "abc_columns.json").read_text(encoding="utf-8"))
    if flips0 is None:
        flips0, procs0, abc_cols0 = flips, procs, abc_cols
    elif (flips, procs, abc_cols) != (flips0, procs0, abc_cols0):
        raise SystemExit(f"inconsistent r420 configs: {d.name}")
    li = [flips.index(f) for f in ORIG]
    js = [procs.index(k) for k in SEL]
    files = {nm: d / f"{nm}.npy" for nm in ("seeds", "pub_auc_r8", "priv_auc_r8", "abc_rows")}
    A = {nm: np.load(p) for nm, p in files.items()}
    parts["seeds"].append(A["seeds"])
    parts["chk_pub8"].append(A["pub_auc_r8"][:, li])
    parts["chk_priv8"].append(A["priv_auc_r8"][:, li])
    parts["priv8"].append(A["priv_auc_r8"][..., js].astype(np.float32))
    parts["pub8"].append(A["pub_auc_r8"][..., js].astype(np.float32))
    ab = A["abc_rows"]
    i_s3 = abc_cols.index("pub8_s003_gmm2_hv")
    parts["abc"].append(ab[np.abs(ab[:, i_s3] - OBS) <= STORE + 1e-12].astype(np.float32))
    manifest["runs"].append({"run_id": d.name, "n_seeds": int(len(A["seeds"])), "n_abc": int(cfg["n_abc"]),
                             "n_draws": int(cfg["n_draws"]), "sha256": {nm: sha(p) for nm, p in files.items()}})
manifest.update({"flips": flips0, "procs": procs0, "abc_columns": abc_cols0})
arrs = {k: np.concatenate(v) for k, v in parts.items()}
if len(np.unique(arrs["seeds"])) != len(arrs["seeds"]):
    raise SystemExit("duplicate seeds across r420 runs")
for nm, fn, cols in (("lik_r922", "r922_likelihood_fine.csv", ["flip_y", "Ltrain_rel", "Lcomb_main"]),
                     ("lik_r421", "r421_likelihood_fine.csv", ["flip_y", "L_train_rel", "Lcomb_random", "Lcomb_strat197",
                                                              "Lcomb_strat198", "Lcomb_stratmix"])):
    p = TAB / fn
    if not p.exists():
        p = TAB / "raw" / fn
    if p.exists():
        arrs[nm] = pd.read_csv(p)[cols].to_numpy(float)
        manifest[nm] = {"file": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p), "columns": cols}
buf = io.BytesIO()
np.savez_compressed(buf, manifest_json=np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8), **arrs)
OUT.write_bytes(buf.getvalue())
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB): {len(runs)} runs, {len(arrs['seeds'])} replicas, "
      f"{len(arrs['abc'])} ABC rows; likelihoods: {[k for k in ('lik_r922', 'lik_r421') if k in arrs]}")
