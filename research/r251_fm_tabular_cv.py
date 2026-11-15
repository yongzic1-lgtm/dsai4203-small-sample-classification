"""R251 - pretrained tabular foundation models (TabPFN v2, TabICL) on the 250 labelled rows.

Supervised in-context learning only: each fold model sees the fold's 200 training rows (features +
labels) as its context and scores the validation rows and the test rows. The unlabeled test
rows are never used as context. Official CV (project.yaml official_cv), test predictions =
fold_mean; each candidate is an experiment run.

  tabpfn2_all   TabPFN v2 weights (ModelVersion.V2, Prior Labs License = Apache-2.0 + attribution),
                raw 150 features
  tabpfn2_hv    TabPFN v2 on the label-free high-variance block (R003 basis, train+test scaling)
  tabicl_all    TabICL (checkpoint tabicl-classifier-v2-20260212, BSD-3), raw 150 features
  tabicl_hv     TabICL on the block

Licence / access were checked in R250 (20261022-063927_r250_fm_licence_check): both download
anonymously; TabPFN-2.5+ (default of tabpfn 9.x) needs a PriorLabs licence acceptance + token and
is therefore NOT used. All hub tokens are removed from the environment before the models load.

Runs in its own venv (/content/venv_tabpfn or /content/venv_tabicl; pinned project packages +
torch + the model package), built by lib_fm.ensure_venv on the GPU VM.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --gpu L4 -- research/r251_fm_tabular_cv.py --model tabpfn
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import lib_fm

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=["raw", "synthetic"])
p.add_argument("--author", default=None)
p.add_argument("--model", required=True, choices=["tabpfn", "tabicl"])
p.add_argument("--only", nargs="*", default=None, help="subset: all hv")
p.add_argument("--n-estimators", type=int, default=8)
p.add_argument("--test-pred", default="fold_mean", choices=["fold_mean", "refit"])
p.add_argument("--max-folds", type=int, default=None, help="synthetic smoke tests only")
a = p.parse_args()
if a.max_folds and a.data == "raw":
    raise SystemExit("--max-folds is for synthetic smoke tests only")

PKGS = {"tabpfn": ["tabpfn", "huggingface_hub"], "tabicl": ["tabicl", "huggingface_hub"]}
if lib_fm.ON_VM:
    py, venv_info = lib_fm.ensure_venv(a.model, PKGS[a.model])
    lib_fm.reexec_in(py)
else:
    venv_info = {}
lib_fm.apply_token_free_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ssc import paths  # noqa: E402
from ssc.tracking import Run  # noqa: E402
from ssc.transductive import SubspaceModel  # noqa: E402
from ssc.validation import cross_validate  # noqa: E402

src = a.data
ds, scheme, splits = lib_fm.official_setup(src)
if a.max_folds:
    splits = splits[: a.max_folds]
dev = "cuda" if torch.cuda.is_available() else "cpu"
versions = lib_fm.freeze_versions()
print(f"python={sys.executable} device={dev} gpu={torch.cuda.get_device_name(0) if dev == 'cuda' else None} "
      f"versions={versions}", flush=True)


def make_base():
    if a.model == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev, random_state=0,
                                                           n_estimators=a.n_estimators)
    from tabicl import TabICLClassifier
    return TabICLClassifier(device=dev, random_state=0, n_estimators=a.n_estimators)


base = make_base()
params = {k: str(v) for k, v in base.get_params().items()}
print("model params:", params, flush=True)
if a.model == "tabpfn" and "v2" not in params.get("model_path", "").lower():
    raise SystemExit(f"refusing: model_path {params.get('model_path')} is not a TabPFN v2 checkpoint")

CANDS = {
    "all": (lambda: make_base(), "raw 150 features", False),
    "hv": (lambda: SubspaceModel(make_base(), "highvar", source=src), "label-free high-variance block", True),
}
tag = {"tabpfn": "tabpfn2", "tabicl": "tabicl"}[a.model]
names = a.only or list(CANDS)

# reference = S003 configuration recomputed (identical code/splits; sklearn version pinned in this venv)
t0 = time.time()
ref_cv = lib_fm.r003_reference("r003_gmm2_hv", ds, splits, scheme, src)
s003 = ref_cv.test_pred
refs = {"r003_gmm2_hv": ref_cv}
for rn in ("r003_sup_qda_hv", "r003_sup_l1_all", "r003_sup_logreg_hv"):
    refs[rn] = lib_fm.r003_reference(rn, ds, splits, scheme, src, n_jobs=min(8, os.cpu_count() or 2))
print(f"references done in {time.time() - t0:.0f}s: " +
      ", ".join(f"{k}={v.fold_scores['auc'].mean():.4f}" for k, v in refs.items()), flush=True)

# timing probe on one fold (not archived)
t0 = time.time()
sp = splits[0]
m = CANDS[names[0]][0]().fit(ds.X.to_numpy()[sp.train_idx], ds.y[sp.train_idx])
_ = m.predict_proba(ds.X.to_numpy()[sp.val_idx])
print(f"timing probe ({names[0]}): fit + predict 50 validation rows = {time.time() - t0:.1f}s", flush=True)
if a.test_pred == "fold_mean":
    t0 = time.time()
    _ = m.predict_proba(ds.X_test.to_numpy())
    print(f"timing probe: predict 19,750 test rows = {time.time() - t0:.1f}s", flush=True)

rows = []
for name in names:
    make, desc, uses_unl = CANDS[name]
    run_name = f"r251_{tag}_{name}"
    cfg = {"name": run_name, "description": f"{a.model} ({'v2 weights' if a.model == 'tabpfn' else 'v2 ckpt'}), {desc}",
           "model_code": "research/r251_fm_tabular_cv.py + research/lib_fm.py", "cv": scheme.to_dict(),
           "test_pred": a.test_pred, "uses_unlabeled_test_features": uses_unl, "model_params": params,
           "n_estimators": a.n_estimators, "device": dev, "versions": versions,
           "venv": {k: v for k, v in venv_info.items() if k != "freeze"}, "venv_freeze": venv_info.get("freeze"),
           "pretrained_on": "synthetic prior data (no real datasets) per the model papers",
           "code_sha": lib_fm.code_hash(__file__)}
    hyp = (f"A tabular foundation model ({a.model}) used as a 200-row in-context learner should match or beat the "
           "best supervised baselines on the same inputs (L1 all-features 0.847; QDA block 0.996) because its "
           "synthetic prior covers covariance-type class differences; predicted CV AUC "
           + ("0.85-0.95 on all 150 features" if name == "all" else "0.97-0.999 on the block")
           + ", still below the transductive GMM (1.000). Refuted if below L1 (all) / below L2 logistic 0.898 (block).")
    with Run(run_name, cfg, kind="experiment", source=src, author=a.author, hypothesis=hyp) as run:
        t0 = time.time()
        cv = cross_validate(make(), ds.X, ds.y, splits, scheme, X_test=ds.X_test, test_pred=a.test_pred, n_jobs=1)
        secs = time.time() - t0
        s = run.save_cv(cv)
        cal = lib_fm.oof_calibration(cv.oof, ds.y)
        agr = lib_fm.reference_agreement(cv.test_pred, s003)
        pcs = lib_fm.paired_table({name: cv}, refs)
        run.save_table("paired_vs_references", pd.DataFrame(pcs))
        pc = {r["reference"]: r for r in pcs}
        row = {"candidate": run_name, "run_id": run.run_id, "auc_mean": s["auc_mean"], "auc_std_repeat": s["auc_std_repeat"],
               "oof_auc_mean": s.get("oof_auc_mean"), **cal, "cv_seconds": secs,
               **{f"diff_vs_{k}": v["mean_diff"] for k, v in pc.items()},
               **{f"se_vs_{k}": v["se_split"] for k, v in pc.items()},
               **{f"p_vs_{k}": v["corrected_p"] for k, v in pc.items()},
               **{f"s003_{k}": v for k, v in agr.items()}, "uses_unlabeled_test_features": uses_unl}
        rows.append(row)
        run.finish(extra_summary={"official_cv": True, **cal, "cv_seconds": secs, "agreement_with_s003": agr,
                                  "paired_vs_references": pcs, "uses_unlabeled_test_features": uses_unl})
        print(f"{run_name:18s} auc={s['auc_mean']:.5f} +/- {s['auc_std_repeat']:.5f} logloss={cal['cv_logloss']:.4f} "
              f"pseudoAUC={agr['pseudo_auc_confident']:.5f} side_agree_boundary={agr.get('side_agree_boundary', float('nan')):.3f} "
              f"({secs:.0f}s) run={run.run_id}", flush=True)

tables = paths.tables_dir(src)
tables.mkdir(parents=True, exist_ok=True)
out = tables / f"r251_{tag}_candidates.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows)[["candidate", "auc_mean", "auc_std_repeat", "cv_logloss", "s003_pseudo_auc_confident",
                          "s003_side_agree_boundary", "cv_seconds"]].to_string(index=False))
