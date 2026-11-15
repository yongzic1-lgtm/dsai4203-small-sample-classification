"""R210 - two follow-ups to R209, written after reading R209.

1. Clean pair for "self-training vs its own base" with the tuning-free Ledoit-Wolf QDA:
   R209 selftrain_qda_lw (SelfTraining(QDA-LW, threshold 0.99), 20,000 rows) scored 1.0000 / log-loss
   0.0071, but its supervised base on the SAME train+test block was not archived:
     sup_qda_tt_lw       supervised QDA-LW on the train+test block (base of pair C)
2. Tie-breaking hypothesis: nested tuning by inner-CV AUC chose LARGER reg than the best fixed value
   (R209 fullgrid self-training chose 0.01-0.03 -> 0.9996, fixed 0.003 -> 1.0000; R205 menu chose its
   grid bottom). When inner-fold AUCs saturate at 1.0 many grid entries tie, and GridSearchCV keeps the
   first (= most regularised) one. Tuning by inner log-loss (a proper scoring rule that keeps
   resolving after AUC saturates) should pick smaller reg:
     qda_block_nll       strict supervised in-fold block, nested reg chosen by inner neg_log_loss
     selftrain_qda_nll   SelfTraining(QDA) on 20,000 rows, nested reg x threshold by inner neg_log_loss
   (in fold_fit_info.csv, best_inner_auc then holds the inner negative log-loss)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r210_sup_tiebreak_lw.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lib_sup
from lib_sup import Cand, HighVarColumns, PoolSelfTraining, Tuned, common_args, run_candidates
from ssc.transductive import SubspaceModel

REG = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]
ST_GRID = {"estimator__reg_param": [0.03, 0.01, 0.003, 0.001], "threshold": [0.99, 0.9]}
H = ("Predictions: (1) supervised QDA-LW on the train+test block scores ~0.995 (R206 at n = 200: 0.9956), so "
     "self-training adds ~+0.004 AUC and lowers log-loss by > 0.05 for this tuning-free base; (2) with inner "
     "log-loss the nested reg moves to 0.001-0.003 and the outer AUC is >= the AUC-tuned versions "
     "(qda_block 0.9999, selftrain fullgrid 0.9996), with lower outer log-loss.")


def candidates():
    src = lib_sup.CURRENT["source"]
    n_st = len(ST_GRID["estimator__reg_param"]) * len(ST_GRID["threshold"])
    return [
        Cand("sup_qda_tt_lw", lambda d: Tuned(SubspaceModel(QDA(solver="eigen", shrinkage="auto"), "highvar", source=src), None, log_dir=d),
             "supervised-generative", "block(train+test highvar, train+test standardised)", True, 1,
             "supervised QDA-LW on the train+test block (base of pair C)", H, tuning="none (analytic Ledoit-Wolf shrinkage)"),
        Cand("qda_block_nll", lambda d: Tuned(Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", QDA())]),
                                              {"model__reg_param": REG}, scoring="neg_log_loss", log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, len(REG),
             "QDA on in-fold block, nested reg_param chosen by inner log-loss", H, tuning=f"reg_param in {REG}, inner neg_log_loss"),
        Cand("selftrain_qda_nll", lambda d: Tuned(PoolSelfTraining(QDA(), threshold=0.99, source=src), ST_GRID, inner_repeats=1,
                                                  scoring="neg_log_loss", log_dir=d),
             "semi-supervised", "block(train+test highvar) + 20,000 unlabeled rows", True, n_st,
             "SelfTraining(QDA) on 20,000 rows, nested reg x threshold chosen by inner log-loss", H,
             tuning=f"{ST_GRID}, inner neg_log_loss"),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r210", a.data, a.n_jobs, a.author, "research/r210_sup_tiebreak_lw.py", a.only, a.max_folds)
