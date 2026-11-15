"""Smoke test for colab/remote.py: one tiny archived CV run.

    .venv\\Scripts\\python.exe colab\\remote.py run -- colab/selftest.py --data synthetic

With --data synthetic the merged run lands in sandbox/, never in the real archive.
"""

import argparse
import os
import platform

from ssc import data, models, validation
from ssc.tracking import Run

p = argparse.ArgumentParser()
p.add_argument("--data", default="synthetic", choices=["raw", "synthetic"])
a = p.parse_args()

ds = data.load(a.data)
cfg = {"name": "colab-selftest",
       "pipeline": {"model": {"name": "logreg", "params": {"C": 0.1, "l1_ratio": 1.0}}, "random_state": 0},
       "cv": {"kind": "kfold", "n_splits": 5, "n_repeats": 2, "seed": 2026}}
with Run("colab-selftest", cfg, kind="selftest", source=a.data, hypothesis="remote pipeline check") as run:
    cv = validation.run_cv(models.build_pipeline(cfg["pipeline"]), ds.X, ds.y,
                           validation.Scheme.from_dict(cfg["cv"]), X_test=ds.X_test)
    s = run.save_cv(cv)
    print(f"auc_mean={s['auc_mean']:.4f} split_sig={s['split_signature']}")
print(f"host={platform.node()} cpus={os.cpu_count()} python={platform.python_version()}")
