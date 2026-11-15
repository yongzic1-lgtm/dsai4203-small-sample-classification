"""R250 - can pretrained tabular foundation models be used at all? (licence + token-free download)

Analysis run (kind = analysis). For TabPFN and TabICL it
  1. builds a separate venv on the Colab VM (pinned project requirements + the package; the
     shared /content/venv is not touched) and records package versions and licence metadata;
  2. asks the HuggingFace Hub, anonymously (token=False), whether the weight repositories are
     gated and which licence they declare, and downloads the weight licence text;
  3. tries one fit on RANDOM numbers (not competition data) with every token removed and the
     TabPFN browser login disabled: a model that needs a login / token / licence click-through
     fails here and is then NOT used by the later scripts.
No credential is ever entered. Results: run folder (fm_check.json, licence texts) and
reports/tables/r250_fm_licence_check.csv.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rfm --gpu L4 -- research/r250_fm_licence_check.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap

import pandas as pd

from ssc import paths
from ssc.tracking import Run

import lib_fm

p = argparse.ArgumentParser()
p.add_argument("--data", default="raw", choices=paths.SOURCES)
p.add_argument("--author", default=None)
a = p.parse_args()

COMMON = r'''
import json, os, sys, time, inspect, traceback
import numpy as np
from importlib import metadata
res = {"env_tokens_present": [k for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TABPFN_TOKEN") if os.environ.get(k)],
       "TABPFN_NO_BROWSER": os.environ.get("TABPFN_NO_BROWSER"), "HF_TOKEN_PATH": os.environ.get("HF_TOKEN_PATH")}
def ver(n):
    try: return metadata.version(n)
    except Exception: return None
def lic(n):
    try:
        md = metadata.metadata(n)
        cls = [c for c in (md.get_all("Classifier") or []) if "License" in c]
        return {"License": md.get("License"), "License-Expression": md.get("License-Expression"), "classifiers": cls}
    except Exception as e: return repr(e)
rng = np.random.default_rng(0)
Xr = rng.normal(size=(120, 12)); yr = (Xr[:, 0] + 0.5 * rng.normal(size=120) > 0).astype(int)
try:
    import torch
    res["torch"] = ver("torch"); res["cuda"] = bool(torch.cuda.is_available())
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e:
    res["torch_error"] = repr(e)
dev = "cuda" if res.get("cuda") else "cpu"
def hub(repo, files=("LICENSE", "README.md")):
    out = {}
    try:
        from huggingface_hub import model_info, hf_hub_download
        mi = model_info(repo, token=False)
        out["gated"] = mi.gated
        out["private"] = mi.private
        cd = mi.card_data.to_dict() if mi.card_data else {}
        out["card_license"] = cd.get("license"); out["card_license_name"] = cd.get("license_name")
        out["files"] = [s.rfilename for s in (mi.siblings or [])][:40]
        for f in files:
            if f in out["files"]:
                pth = hf_hub_download(repo, f, token=False)
                out[f + "_text"] = open(pth, encoding="utf-8", errors="replace").read()[:6000]
    except Exception as e:
        out["error"] = repr(e)[:1500]
    return out
'''

TABPFN = COMMON + r'''
res["tabpfn"] = ver("tabpfn"); res["tabpfn_license_meta"] = lic("tabpfn"); res["huggingface_hub"] = ver("huggingface_hub")
res["sklearn"] = ver("scikit-learn"); res["numpy"] = ver("numpy")
res["hub_v2_clf"] = hub("Prior-Labs/TabPFN-v2-clf")
from tabpfn import TabPFNClassifier
try:
    from tabpfn.constants import ModelVersion
    res["model_versions"] = [str(v) for v in ModelVersion]
except Exception as e:
    ModelVersion = None; res["model_versions_error"] = repr(e)
res["init_signature"] = str(inspect.signature(TabPFNClassifier.__init__))[:3000]
if hasattr(TabPFNClassifier, "create_default_for_version"):
    res["create_default_signature"] = str(inspect.signature(TabPFNClassifier.create_default_for_version))
def attempt(label, make):
    t = time.time()
    try:
        clf = make(); clf.fit(Xr, yr); pr = clf.predict_proba(Xr[:5])[:, 1]
        res[label] = {"status": "ok", "seconds": round(time.time() - t, 1), "pred": [round(float(v), 4) for v in pr],
                      "params": {k: str(v)[:200] for k, v in clf.get_params().items()}}
    except BaseException as e:
        res[label] = {"status": "error", "seconds": round(time.time() - t, 1), "error": (type(e).__name__ + ": " + str(e))[:2500]}
if ModelVersion is not None and hasattr(TabPFNClassifier, "create_default_for_version"):
    attempt("fit_v2", lambda: TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=dev, random_state=0))
attempt("fit_default_latest", lambda: TabPFNClassifier(device=dev, random_state=0))
print("<<RES>>" + json.dumps(res, default=str) + "<</RES>>")
'''

TABICL = COMMON + r'''
res["tabicl"] = ver("tabicl"); res["tabicl_license_meta"] = lic("tabicl"); res["huggingface_hub"] = ver("huggingface_hub")
res["sklearn"] = ver("scikit-learn"); res["numpy"] = ver("numpy")
res["hub_tabicl"] = hub("jingang/TabICL")
from tabicl import TabICLClassifier
res["init_signature"] = str(inspect.signature(TabICLClassifier.__init__))[:3000]
t = time.time()
try:
    clf = TabICLClassifier(device=dev, random_state=0); clf.fit(Xr, yr); pr = clf.predict_proba(Xr[:5])[:, 1]
    res["fit_default"] = {"status": "ok", "seconds": round(time.time() - t, 1), "pred": [round(float(v), 4) for v in pr],
                          "params": {k: str(v)[:200] for k, v in clf.get_params().items()}}
except BaseException as e:
    res["fit_default"] = {"status": "error", "seconds": round(time.time() - t, 1), "error": (type(e).__name__ + ": " + str(e))[:2500]}
print("<<RES>>" + json.dumps(res, default=str) + "<</RES>>")
'''

STACKS = {"tabpfn": (["tabpfn", "huggingface_hub"], TABPFN), "tabicl": (["tabicl", "huggingface_hub"], TABICL)}

cfg = {"name": "r250_fm_licence_check", "kind": "analysis", "stacks": {k: v[0] for k, v in STACKS.items()},
       "inputs": "random numbers only (no competition data)", "token_policy": "all hub tokens removed, browser login off"}
with Run("r250_fm_licence_check", cfg, kind="analysis", source=a.data, author=a.author,
         hypothesis="TabPFN v2 weights (Prior Labs License = Apache-2.0 + attribution) and TabICL weights "
                    "(BSD-3) download anonymously; TabPFN-2.5+ (non-commercial, PriorLabs login + licence "
                    "acceptance) refuses without a token and must not be used.") as run:
    out, rows = {}, []
    for name, (pkgs, code) in STACKS.items():
        info = {}
        try:
            py, info = lib_fm.ensure_venv(name, pkgs)
            r = subprocess.run([str(py), "-c", textwrap.dedent(code)], env=lib_fm.token_free_env(),
                               capture_output=True, text=True, timeout=1500, stdin=subprocess.DEVNULL)
            txt = r.stdout
            res = json.loads(txt.split("<<RES>>", 1)[1].split("<</RES>>", 1)[0]) if "<<RES>>" in txt else \
                {"error": "no result", "stderr": r.stderr[-4000:], "stdout": txt[-2000:]}
        except Exception as e:  # noqa: BLE001
            res = {"error": repr(e)[:2000]}
        res["venv_info"] = {k: v for k, v in info.items() if k != "freeze"}
        res["venv_freeze"] = info.get("freeze")
        out[name] = res
        for key in ("hub_v2_clf", "hub_tabicl"):
            h = res.get(key) or {}
            for f in ("LICENSE_text", "README.md_text"):
                if h.get(f):
                    (run.dir / f"{name}_{f.split('_')[0].replace('.md', '')}.txt").write_text(h[f], encoding="utf-8")
        fits = {k: v for k, v in res.items() if k.startswith("fit_")}
        hubinfo = res.get("hub_v2_clf") or res.get("hub_tabicl") or {}
        for k, v in fits.items():
            rows.append({"stack": name, "attempt": k, "package_version": res.get(name), "status": v.get("status"),
                         "seconds": v.get("seconds"), "error": (v.get("error") or "")[:300],
                         "hub_repo_gated": hubinfo.get("gated"), "hub_card_license": hubinfo.get("card_license"),
                         "package_license": json.dumps(res.get(f"{name}_license_meta"))[:300],
                         "cuda": res.get("cuda"), "gpu": res.get("gpu"), "torch": res.get("torch"),
                         "sklearn": res.get("sklearn"), "numpy": res.get("numpy"),
                         "pinned_resolution": res["venv_info"].get("pinned_resolution")})
        print(f"== {name}: " + json.dumps({k: v for k, v in res.items() if k not in ("venv_freeze",)}, default=str)[:6000])
    (run.dir / "fm_check.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    df = pd.DataFrame(rows)
    run.save_table("fm_check", df)
    t = paths.tables_dir(a.data)
    t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "r250_fm_licence_check.csv", index=False)
    print(df.to_string(index=False))
    run.finish(extra_summary={"official_cv": False, "checks": rows})
sys.exit(0)
