"""R209 - follow-ups to R204 (sklearn semi-supervised baselines), written after reading R204.

R204's grids were fixed before R201 showed that small reg_param (0.001-0.01) is best for QDA:
SelfTraining-QDA chose reg 0.03 (grid bottom) and LabelSpreading chose alpha 0.8 / k 30 (grid
corner). This script gives the semi-supervised models the same chance as the supervised ones
and adds clean paired ablations "self-training vs its own base classifier" at identical fixed
hyperparameters on the identical (train+test) block:

  sup_qda_tt_r003          supervised QDA(reg 0.003) on the train+test block       (pair A, base)
  selftrain_qda_r003       SelfTraining(QDA reg 0.003, threshold 0.99), 20,000 rows (pair A, + unlabeled)
  sup_qda_tt_r01           supervised QDA(reg 0.1) on the train+test block = R003 sup_qda_hv recomputed (pair B, base)
                           [pair B's self-training partner is R204 selftrain_qda_default (threshold 0.75)]
  selftrain_qda_fullgrid   SelfTraining(QDA), nested reg in {0.03, 0.01, 0.003, 0.001} x threshold {0.99, 0.9}
  selftrain_qda_lw         SelfTraining(QDA with per-class Ledoit-Wolf shrinkage), threshold 0.99, no tuning
  labelspread_widegrid     LabelSpreading(knn), nested k in {30, 60, 120} x alpha {0.8, 0.95, 0.99}

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r209_semisup_followup.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

import lib_sup
from lib_sup import Cand, PoolGraphSSL, PoolSelfTraining, Tuned, common_args, run_candidates
from ssc.transductive import SubspaceModel

ST_GRID = {"estimator__reg_param": [0.03, 0.01, 0.003, 0.001], "threshold": [0.99, 0.9]}
LS_GRID = {"n_neighbors": [30, 60, 120], "alpha": [0.8, 0.95, 0.99]}
H = ("R201/R204: strict supervised QDA (nested reg) 0.9999 > SelfTraining-QDA (R204 grid, reg >= 0.03) 0.9995 > "
     "LabelSpreading <= 0.944. Predictions: at identical base settings, self-training changes AUC by < 0.001 "
     "(the supervised QDA already ranks almost perfectly; pseudo-labels mainly sharpen probabilities, so log-loss "
     "should improve more than AUC); with the full reg grid SelfTraining-QDA reaches 0.9997-1.0000; the wider "
     "LabelSpreading grid stays < 0.97 (graph diffusion does not use the Gaussian shape).")


def candidates():
    src = lib_sup.CURRENT["source"]
    feat = "block(train+test highvar) + 20,000 unlabeled rows"
    n_st = len(ST_GRID["estimator__reg_param"]) * len(ST_GRID["threshold"])
    n_ls = len(LS_GRID["n_neighbors"]) * len(LS_GRID["alpha"])
    return [
        Cand("sup_qda_tt_r003", lambda d: Tuned(SubspaceModel(QDA(reg_param=0.003), "highvar", source=src), None, log_dir=d),
             "supervised-generative", "block(train+test highvar, train+test standardised)", True, 1,
             "supervised QDA reg 0.003 on the train+test block (base of pair A)", H,
             extra={"note": "reg 0.003 = modal nested choice of R201 qda_block (informed by R201)"}),
        Cand("selftrain_qda_r003", lambda d: Tuned(PoolSelfTraining(QDA(reg_param=0.003), threshold=0.99, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "SelfTraining(QDA reg 0.003, threshold 0.99) on 20,000 rows (pair A)", H,
             extra={"note": "reg 0.003 = modal nested choice of R201 qda_block (informed by R201)"}),
        Cand("sup_qda_tt_r01", lambda d: Tuned(SubspaceModel(QDA(reg_param=0.1), "highvar", source=src), None, log_dir=d),
             "supervised-generative", "block(train+test highvar, train+test standardised)", True, 1,
             "supervised QDA reg 0.1 on the train+test block (R003 sup_qda_hv recomputed; base of pair B)", H),
        Cand("selftrain_qda_fullgrid", lambda d: Tuned(PoolSelfTraining(QDA(), threshold=0.99, source=src), ST_GRID, inner_repeats=1, log_dir=d),
             "semi-supervised", feat, True, n_st, "SelfTraining(QDA) on 20,000 rows, nested reg (down to 0.001) x threshold", H,
             tuning=str(ST_GRID)),
        Cand("selftrain_qda_lw", lambda d: Tuned(PoolSelfTraining(QDA(solver="eigen", shrinkage="auto"), threshold=0.99, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "SelfTraining(QDA Ledoit-Wolf), threshold 0.99, 20,000 rows, no tuning", H),
        Cand("labelspread_widegrid", lambda d: Tuned(PoolGraphSSL("spreading", 30, 0.8, source=src), LS_GRID, inner_repeats=1, log_dir=d),
             "semi-supervised", feat, True, n_ls, "LabelSpreading(knn) on 20,000 rows, nested k up to 120 x alpha up to 0.99", H,
             tuning=str(LS_GRID)),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r209", a.data, a.n_jobs, a.author, "research/r209_semisup_followup.py", a.only, a.max_folds)
