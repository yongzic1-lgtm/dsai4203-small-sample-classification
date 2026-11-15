"""R935 - pack the archived r933 replica arrays into ONE input file for the pooled analysis on Colab (verifier "cand").

Pure file I/O, no computation and no Run: the r933 jobs ran on two Colab sessions (pro2/vcand, pro2/vcand2) and were
merged into the LOCAL archive; remote.py syncs only code (src, scripts, extensions, research, ...) and data/raw, so the
pooled analysis (research/r936_final_selection.py) reads this packed copy, which travels in the research/ code bundle
(and is therefore archived with the bundle hash in logs/code_bundles/). The manifest stores the sha256 of every source
array so r936 can prove it analysed exactly the archived numbers.

    .venv\\Scripts\\python.exe research\\r935_pack_r933.py            # writes research/r935_r933_pooled.npz
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "research" / "r935_r933_pooled.npz"
ARRAYS = ["priv_auc", "pub_auc", "extra", "seeds", "priv_auc_r8", "pub_auc_r8", "cv_auc", "real_cv_auc"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


runs = []
for d in sorted(EXP.glob("*_r933_replicas_*")):
    s = d / "summary.json"
    if not (s.exists() and (d / "priv_auc.npy").exists()):
        continue
    if json.loads(s.read_text(encoding="utf-8")).get("status") != "done":
        continue
    runs.append(d)
# a run whose seeds all occur in another finished run with the same flags is a re-run (e.g. the cross-VM determinism
# check 20261022-172258_r933_replicas_param_s136-143_tn0_r8_cv, bit-identical to s136-246): pool it only once
_flags = {d: (yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")).get("train_noise", True),
              set(np.load(d / "seeds.npy").tolist())) for d in runs}
runs = [d for d in runs if not any(o != d and _flags[o][0] == _flags[d][0] and _flags[d][1] < _flags[o][1] for o in runs)]
if not runs:
    sys.exit("no finished r933 runs in experiments/")

pack, manifest = {}, {"runs": []}
for i, d in enumerate(runs):
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
    meta = json.loads((d / "replica_meta.json").read_text(encoding="utf-8"))
    entry = {"index": i, "run_id": d.name, "variant": cfg["variant"], "flips": cfg["flips"],
             "procedures": cfg["procedures"], "n_draws": cfg["n_draws"],
             "train_noise": bool(cfg.get("train_noise", True)), "round_decimals": cfg.get("round_decimals"),
             "sha256": {}}
    for nm in ARRAYS:
        f = d / f"{nm}.npy"
        if f.exists():
            pack[f"{i}__{nm}"] = np.load(f)
            entry["sha256"][nm] = sha(f)
    pack[f"{i}__n_flip_train"] = np.array([m["n_flip_train"] for m in meta], dtype=int)
    pack[f"{i}__n_flip_test"] = np.array([m["n_flip_test"] for m in meta], dtype=int)
    pack[f"{i}__meta_seed"] = np.array([m["seed"] for m in meta], dtype=int)
    entry["n_replicas"] = int(len(meta))
    # nested-QDA reg choices (clean-train label vector = first label vector of every replica)
    regs = {}
    for m in meta:
        for k, v in (m.get("chosen_reg") or [{}])[0].items():
            regs.setdefault(k, []).extend(v)
    entry["chosen_reg_first_label_vector"] = {k: {str(r): int(c) for r, c in zip(*np.unique(v, return_counts=True))}
                                              for k, v in regs.items()}
    if "n_distinct_scores_rounded" in meta[0]:
        entry["n_distinct_scores_mean"] = {k: float(np.mean([m["n_distinct_scores"][k] for m in meta]))
                                           for k in meta[0]["n_distinct_scores"]}
        entry["n_distinct_scores_rounded_mean"] = {k: float(np.mean([m["n_distinct_scores_rounded"][k] for m in meta]))
                                                   for k in meta[0]["n_distinct_scores_rounded"]}
    manifest["runs"].append(entry)
    assert np.array_equal(pack[f"{i}__meta_seed"], pack[f"{i}__seeds"]), d.name

pack["manifest_json"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
buf = io.BytesIO()
np.savez_compressed(buf, **pack)
OUT.write_bytes(buf.getvalue())
print(f"packed {len(runs)} runs, {sum(e['n_replicas'] for e in manifest['runs'])} replicas -> {OUT} "
      f"({OUT.stat().st_size / 1e6:.2f} MB), sha256 {sha(OUT)[:16]}")
for e in manifest["runs"]:
    print(f"  {e['run_id']}: n={e['n_replicas']} train_noise={e['train_noise']} round={e['round_decimals']}")
