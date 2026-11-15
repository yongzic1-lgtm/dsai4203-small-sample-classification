"""R208 - follow-ups to R201 (supervised-only QDA), written after reading R201's first results.

R201 showed that nested reg_param on the in-fold block picks 0.001-0.01 (AUC 0.9999), but two
R201 variants had reg grids fixed beforehand at 0.03-0.5 (qda_blockextra, qda_corrk), so their
comparison with qda_block is confounded by the grid; and qda_all failed because sklearn 1.9's
svd QDA needs more rows per class than features (n_k > 150). This script:
  qda_blockextra_full   in-fold block + k extra in-fold |corr| columns, k x FULL reg grid (nested)
  qda_corrk_full        in-fold top-k |corr| over all 150, k x FULL reg grid (nested)
  qda_all_lw            QDA with per-class Ledoit-Wolf shrinkage (solver='eigen'), all 150 features, no tuning
  qda_block_lw          QDA-LW on the in-fold block, no tuning (a tuning-free strict supervised model)
  qda_block13_infold    QDA on the in-fold block with ratio 1.65 (recovers the exact 13 columns in 94/100
                        folds; ratio chosen by looking at the label-free train+test block, i.e. informed by
                        test FEATURES, never by labels), nested reg

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r208_sup_qda_followup.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates
from ssc.features import CorrelationTopK

REG = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]
EXTRA_K = [0, 1, 2, 3, 5, 8]
K_CORR = [5, 10, 13, 20, 30]

H = ("R201: nested QDA on the in-fold block = 0.9999 with reg 0.001-0.01; the ratio-tuned variant (fewer columns) "
     "was slightly lower (0.9996), suggesting that the 3-10 borderline columns carry signal. Predictions: with the "
     "full reg grid, block+extra is within split noise of qda_block (the in-fold block already contains the useful "
     "borderline columns); corr-top-k stays clearly worse (< 0.99: univariate |corr| misses variance-only signal); "
     "QDA-LW on all 150 features < 0.95; tuning-free QDA-LW on the block 0.99-0.999 (LW over-shrinks a little); "
     "the exact 13-column in-fold block (ratio 1.65) is slightly below qda_block (0.9990-0.9998).")


def blk(sel, *steps):
    return Pipeline([("sel", sel), ("scale", StandardScaler()), *steps])


def qlw():
    return QDA(solver="eigen", shrinkage="auto")


def candidates():
    return [
        Cand("qda_blockextra_full", lambda d: Tuned(blk(HighVarColumns(extra_k=0), ("model", QDA())),
                                                    {"sel__extra_k": EXTRA_K, "model__reg_param": REG}, log_dir=d),
             "supervised-generative", "block(in-fold)+top-k corr extras(in-fold)", False, len(EXTRA_K) * len(REG),
             "QDA on in-fold block + k extra in-fold |corr| columns, nested k x full reg grid", H,
             tuning=f"extra_k in {EXTRA_K} x reg_param in {REG}"),
        Cand("qda_corrk_full", lambda d: Tuned(Pipeline([("scale", StandardScaler()), ("sel", CorrelationTopK(k=13)), ("model", QDA())]),
                                               {"sel__k": K_CORR, "model__reg_param": REG}, log_dir=d),
             "supervised-generative", "corr-topk(in-fold)", False, len(K_CORR) * len(REG),
             "QDA on in-fold top-k |corr| columns of all 150, nested k x full reg grid", H,
             tuning=f"k in {K_CORR} x reg_param in {REG}"),
        Cand("qda_all_lw", lambda d: Tuned(Pipeline([("scale", StandardScaler()), ("model", qlw())]), None, log_dir=d),
             "supervised-generative", "all 150", False, 1,
             "QDA with per-class Ledoit-Wolf shrinkage (eigen solver), all 150 features, no tuning", H,
             tuning="none (analytic Ledoit-Wolf shrinkage)"),
        Cand("qda_block_lw", lambda d: Tuned(blk(HighVarColumns(), ("model", qlw())), None, log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, 1,
             "QDA with per-class Ledoit-Wolf shrinkage on the in-fold block, no tuning", H,
             tuning="none (analytic Ledoit-Wolf shrinkage)"),
        Cand("qda_block13_infold", lambda d: Tuned(blk(HighVarColumns(ratio=1.65), ("model", QDA())), {"model__reg_param": REG}, log_dir=d),
             "supervised-generative", "block(in-fold highvar, ratio 1.65 ~ exact 13 columns)", False, len(REG),
             "QDA on the in-fold block with ratio 1.65 (~ the 13 train+test block columns), nested reg", H,
             tuning=f"reg_param in {REG}",
             extra={"note": "ratio 1.65 was picked by comparing in-fold blocks with the label-free train+test block"}),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    run_candidates(candidates(), "r208", a.data, a.n_jobs, a.author, "research/r208_sup_qda_followup.py", a.only, a.max_folds)
