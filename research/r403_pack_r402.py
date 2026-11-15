"""R403 - pack the archived r402 replica arrays into ONE input file for the pooled analysis on Colab (research/r404).

Pure file I/O, no computation and no Run (same pattern as research/r935_pack_r933.py): the r402 jobs ran on two
Colab sessions (pro2/r2ind, pro2/r2ind2) and were merged into the LOCAL archive, but remote.py syncs only code and
data/raw, so r404 reads this packed copy, which travels in the research/ code bundle (archived with the bundle hash
in logs/code_bundles/). The manifest stores the sha256 of every source file so r404 can prove it analysed exactly the
archived numbers. Also packed: the flip_y likelihood of R922 (reports/tables/raw/r922_likelihood_fine.csv, for the
flat-prior weights), the archived test_pred.npy of the reference runs (for the real-data reproduction check), and
the EB / reg choice counts from each run's replica_meta.json.

    .venv\\Scripts\\python.exe research\\r403_pack_r402.py            # writes research/r403_r402_pooled.npz
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--exp-dir", default=str(ROOT / "experiments"), help="smoke tests only: a scratch archive's experiments/")
ap.add_argument("--out", default=str(ROOT / "research" / "r403_r402_pooled.npz"))
ap.add_argument("--include-smoke", action="store_true", help="smoke tests only")
args = ap.parse_args()
EXP = Path(args.exp_dir)
OUT = Path(args.out)
ARRAYS = ["priv_auc_r8", "pub_auc_r8", "exp_auc", "bnd_stats", "seeds", "real_test_pred"]
R922 = ROOT / "reports" / "tables" / "r922_likelihood_fine.csv"
sys.path.insert(0, str(ROOT / "research"))
REF_RUNS = {   # = r400_inductive_lib.RUN_IDS (archived runs only)
    "s003_gmm2_hv": "20261022-050708_r003_gmm2_hv",
    "r255_bqda_eb_hv": "20261022-070200_r255_bqda_eb_hv",
    "r209_selftrain_qda_lw": "20261022-080632_r209_selftrain_qda_lw",
    "r201_qda_block": "20261022-071234_r201_qda_block",
    "r210_qda_block_nll": "20261022-082047_r210_qda_block_nll",
    "x_qda_block_nll_ext": "20261022-095943_r930_qda_block_nll_ext",
    "r255_bqda_eb_trainonly": "20261022-070202_r255_bqda_eb_trainonly",
    "r208_qda_block_lw": "20261022-075228_r208_qda_block_lw",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs = []
for d in sorted(EXP.glob("*_r402_inductive_replicas_s*")):
    s = d / "summary.json"
    if (d.name.endswith("_smoke") and not args.include_smoke) or not (s.exists() and (d / "priv_auc_r8.npy").exists()):
        continue
    if json.loads(s.read_text(encoding="utf-8")).get("status") != "done":
        continue
    runs.append(d)
if not runs:
    raise SystemExit("no finished r402 runs in experiments/")

payload, manifest = {}, {"runs": [], "references": {}, "r922": None}
for i, d in enumerate(runs):
    import yaml
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    metas = json.loads((d / "replica_meta.json").read_text(encoding="utf-8"))
    e = {"index": i, "run_id": d.name, "flips": cfg["flips"], "procedures": cfg["procedures"], "n_draws": cfg["n_draws"],
         "seeds": cfg["seeds"], "real_check": cfg.get("real_check"), "sha256": {},
         "real_check_rows": summ.get("real_check"), "real_eb_choice": summ.get("real_eb_choice")}
    for nm in ARRAYS:
        f = d / f"{nm}.npy"
        if f.exists():
            payload[f"{i}__{nm}"] = np.load(f)
            e["sha256"][f"{nm}.npy"] = sha(f)
    e["sha256"]["replica_meta.json"] = sha(d / "replica_meta.json")
    payload[f"{i}__meta_json"] = np.frombuffer(json.dumps(
        [{k: m[k] for k in ("seed", "chosen_reg", "eb_choice", "trainonly_block_size", "n_bayes_boundary",
                            "n_s003_boundary", "n_flip_test", "seconds")} for m in metas]).encode("utf-8"), dtype=np.uint8)
    manifest["runs"].append(e)
    print(f"packed {d.name}: {cfg['seeds']}")
for k, rid in REF_RUNS.items():
    f = ROOT / "experiments" / rid / "test_pred.npy"
    payload[f"ref__{k}"] = np.load(f)
    manifest["references"][k] = {"run_id": rid, "sha256": sha(f)}
lk = pd.read_csv(R922)
payload["r922__flip_y"] = lk["flip_y"].to_numpy(float)
payload["r922__Lcomb_main"] = lk["Lcomb_main"].to_numpy(float)
manifest["r922"] = {"file": "reports/tables/r922_likelihood_fine.csv", "sha256": sha(R922),
                    "run_id": "20261022-092146_r922_label_noise_private"}
payload["manifest_json"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
buf = io.BytesIO()
np.savez_compressed(buf, **payload)
OUT.write_bytes(buf.getvalue())
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), sha256 {hashlib.sha256(buf.getvalue()).hexdigest()}")
