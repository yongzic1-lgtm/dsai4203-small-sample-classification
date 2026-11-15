"""R301 - run extension E1 or E2 (their unmodified run.py) with the extended model builder.

extensions/e1_validation_protocol/run.py and extensions/e2_learning_curve/run.py build
every model with ssc.models.build_pipeline, which knows neither QDA on the label-free
block nor the TransductiveGMM. This wrapper loads the extension's run.py as a module,
swaps its `build_pipeline` for research/lib_ext12.build_pipeline (same specs plus
subspace_qda / subspace_lda / subspace_logreg / transductive_gmm) and, with --prefix,
replaces the report-file prefix (e.g. e2_qda_) so several runs of one extension do not
overwrite each other's reports/tables|figures files. Everything else (protocol, seeds,
archiving as an kind="extension" run, manifest) is the extension's own code.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext12 --high-mem -- research/r301_run_extension.py \\
        --ext e2 --prefix e2_qda_ -- --config research/r302_e2_qda_block.yaml --n-jobs -1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import lib_ext12

ROOT = Path(__file__).resolve().parents[1]
EXT = {"e1": "e1_validation_protocol", "e2": "e2_learning_curve"}

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--ext", choices=sorted(EXT), required=True)
p.add_argument("--prefix", default=None, help="report-file prefix, must start with the extension's own e#_")
p.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the arguments of the extension's run.py")
a = p.parse_args()
rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
if a.prefix and not a.prefix.startswith(f"{a.ext}_"):
    raise SystemExit(f"--prefix must start with {a.ext}_")

# the extension's --data also decides which unlabeled feature matrix the extra models use
# (os.environ is inherited by the joblib worker processes started later)
if "--data" in rest:
    os.environ["SSC_DATA"] = rest[rest.index("--data") + 1]

path = ROOT / "extensions" / EXT[a.ext] / "run.py"
spec = importlib.util.spec_from_file_location(f"{a.ext}_run_ext12", path)
mod = importlib.util.module_from_spec(spec)  # deliberately NOT in sys.modules: joblib pickles its functions by value
spec.loader.exec_module(mod)
mod.build_pipeline = lib_ext12.build_pipeline
if a.prefix:
    mod.PREFIX = a.prefix
print(f"[r301] {path.relative_to(ROOT)} with lib_ext12.build_pipeline, prefix={mod.PREFIX!r}, args={rest}", flush=True)
sys.argv = [str(path)] + rest
raise SystemExit(mod.main(rest))
