"""R204 - semi-supervised baselines from scikit-learn (between supervised and TransductiveGMM).

All models see the FEATURES of all 20,000 rows (train + test): rows of the current training
fold carry their labels, every other row (validation fold, other train rows, all test rows)
is passed as -1. Labels of validation rows are never used, so the official CV stays honest.
The block (13 high-variance features) and its standardisation come from train+test features
(label-free, as in ssc.transductive).

  selftrain_qda_nested   SelfTrainingClassifier(QDA), nested reg_param x threshold
  selftrain_qda_default  SelfTrainingClassifier(QDA reg 0.1 = R003's untuned value), threshold 0.75 (sklearn default)
  selftrain_lr           SelfTrainingClassifier(L2 logistic C=1), threshold 0.75
  labelspread_block      LabelSpreading(knn, k=7, alpha=0.2) = sklearn defaults
  labelprop_block        LabelPropagation(knn, k=7)
  labelspread_nested     LabelSpreading(knn), nested n_neighbors x alpha
  labelspread_all        LabelSpreading(knn, k=7) on all 150 standardised features (graph built in noise)

The knn graph of the fixed 20,000-row pool is cached per process (exact; lib_sup.CachedLabelSpreading).
The dense rbf kernel (20,000 x 20,000) is not used.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r204_semisup_sklearn.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression

import lib_sup
from lib_sup import Cand, PoolGraphSSL, PoolSelfTraining, Tuned, common_args, run_candidates

ST_GRID = {"estimator__reg_param": [0.3, 0.1, 0.03], "threshold": [0.99, 0.9]}
LS_GRID = {"n_neighbors": [30, 15, 7], "alpha": [0.8, 0.5, 0.2]}

H_ST = ("Self-training adds confidently pseudo-labelled test rows; with a QDA base the pseudo-labels in the "
        "confident regions are almost always right, so the refitted QDA moves towards the mixture solution. "
        "Prediction: SelfTraining-QDA >= supervised QDA (0.996) and close to TransductiveGMM (1.000), 0.997-1.000. "
        "With a linear base (LR) self-training cannot fix the wrong model family: 0.85-0.92.")
H_LS = ("Graph label spreading on a kNN graph of the 20,000 block rows exploits cluster structure but not the "
        "global Gaussian shape. Prediction: 0.95-0.995 on the block, nested >= default; on all 150 features the "
        "graph is dominated by noise: < 0.80. All below TransductiveGMM.")


def candidates():
    src = lib_sup.CURRENT["source"]
    n_st = len(ST_GRID["estimator__reg_param"]) * len(ST_GRID["threshold"])
    n_ls = len(LS_GRID["n_neighbors"]) * len(LS_GRID["alpha"])
    feat = "block(train+test highvar) + 20,000 unlabeled rows"
    return [
        Cand("selftrain_qda_nested",
             lambda d: Tuned(PoolSelfTraining(QDA(), threshold=0.9, source=src), ST_GRID, inner_repeats=1, log_dir=d),
             "semi-supervised", feat, True, n_st, "SelfTrainingClassifier(QDA) on 20,000 rows, nested reg_param x threshold",
             H_ST, tuning=str(ST_GRID)),
        Cand("selftrain_qda_default",
             lambda d: Tuned(PoolSelfTraining(QDA(reg_param=0.1), threshold=0.75, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "SelfTrainingClassifier(QDA reg 0.1), threshold 0.75, 20,000 rows", H_ST),
        Cand("selftrain_lr",
             lambda d: Tuned(PoolSelfTraining(LogisticRegression(C=1.0, max_iter=5000), threshold=0.75, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "SelfTrainingClassifier(L2 logistic C=1), threshold 0.75, 20,000 rows", H_ST),
        Cand("labelspread_block", lambda d: Tuned(PoolGraphSSL("spreading", 7, 0.2, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "LabelSpreading(knn k=7, alpha=0.2) on 20,000 block rows", H_LS),
        Cand("labelprop_block", lambda d: Tuned(PoolGraphSSL("propagation", 7, source=src), None, log_dir=d),
             "semi-supervised", feat, True, 1, "LabelPropagation(knn k=7) on 20,000 block rows", H_LS),
        Cand("labelspread_nested", lambda d: Tuned(PoolGraphSSL("spreading", 7, 0.2, source=src), LS_GRID, inner_repeats=1, log_dir=d),
             "semi-supervised", feat, True, n_ls, "LabelSpreading(knn) on 20,000 block rows, nested n_neighbors x alpha",
             H_LS, tuning=str(LS_GRID)),
        Cand("labelspread_all", lambda d: Tuned(PoolGraphSSL("spreading", 7, 0.2, columns="all", source=src), None, log_dir=d),
             "semi-supervised", "all 150 (train+test standardised) + 20,000 unlabeled rows", True, 1,
             "LabelSpreading(knn k=7, alpha=0.2) on all 150 features, 20,000 rows", H_LS),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r204", a.data, a.n_jobs, a.author, "research/r204_semisup_sklearn.py", a.only, a.max_folds)
