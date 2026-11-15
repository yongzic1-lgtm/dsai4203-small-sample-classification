"""R207 - grid-edge check for R200's linear models on the in-fold block.

R200 chose the LARGEST C of its grid in 100/100 folds (L2 block, C = 30) and 78/100 folds
(L1 block, C = 10): the optimum may lie outside the grid. This reruns the two block models
with the grid extended to C = 1e4 (still nested, inner RepeatedStratifiedKFold(5, 2)), and
adds L2 logistic on R003's train+test block (13 columns) with the same wide grid, to separate
"weaker penalty" from "the extra borderline columns the in-fold rule adds (16-23 columns)".

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r207_sup_linear_widegrid.py
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lib_sup
from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates
from ssc.transductive import SubspaceModel

C_WIDE = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
H = ("R200 L2/L1 on the in-fold block reached 0.932 with C at the grid edge. Prediction: the wider grid moves the "
     "AUC by < 0.005 (C beyond 30 is almost unpenalised for 16-23 columns and 200 rows). L2 on the exact 13-column "
     "train+test block stays near R003's 0.898 (C=1) even with tuned C, i.e. the gain of R200's block models comes "
     "from the borderline columns, not from the penalty. Refuted if l2_block_tt_wide > 0.92.")


def lr(l1_ratio):
    return LogisticRegression(l1_ratio=l1_ratio, solver="liblinear", max_iter=20000, random_state=0)


def candidates():
    src = lib_sup.CURRENT["source"]
    blk = lambda m: Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", m)])  # noqa: E731
    return [
        Cand("l2_block_wide", lambda d: Tuned(blk(lr(0.0)), {"model__C": C_WIDE}, log_dir=d), "supervised-linear",
             "block(in-fold highvar)", False, len(C_WIDE), "L2 logistic, in-fold block, nested C up to 1e4", H,
             tuning=f"C in {C_WIDE}"),
        Cand("l1_block_wide", lambda d: Tuned(blk(lr(1.0)), {"model__C": C_WIDE}, log_dir=d), "supervised-linear",
             "block(in-fold highvar)", False, len(C_WIDE), "L1 logistic, in-fold block, nested C up to 1e4", H,
             tuning=f"C in {C_WIDE}"),
        Cand("l2_block_tt_wide", lambda d: Tuned(SubspaceModel(lr(0.0), "highvar", source=src), {"estimator__C": C_WIDE}, log_dir=d),
             "supervised-linear", "block(train+test highvar, train+test standardised)", True, len(C_WIDE),
             "L2 logistic on R003's 13-column train+test block, nested C up to 1e4", H, tuning=f"C in {C_WIDE}"),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r207", a.data, a.n_jobs, a.author, "research/r207_sup_linear_widegrid.py", a.only, a.max_folds)
