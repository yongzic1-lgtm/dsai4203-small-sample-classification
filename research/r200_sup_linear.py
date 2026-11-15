"""R200 - supervised-only track, linear models (250 labels, no test features).

L1 / L2 / elastic-net logistic regression with NESTED tuning of C (and l1_ratio, k), on three
feature sets, plus shrinkage LDA:
  all     all 150 features (StandardScaler in-fold)
  block   the label-free high-variance block re-derived in each 200-row training fold
          (lib_sup.HighVarColumns: std > 1.5 x median std), no test features
  corrk   in-fold top-k by |corr(feature, y)|, k tuned jointly with C in the inner CV

Every candidate = one archived experiment run under the official CV (5-fold x 20, seed 2026),
with test predictions (fold_mean). Inner CV = RepeatedStratifiedKFold(5, 2) on the training
fold, scoring ROC AUC; n_configs = grid size is recorded in each run.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r200_sup_linear.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates
from ssc.features import CorrelationTopK

C_L1 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]      # small C first = most regularised
C_L2 = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
C_EN = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
R_EN = [0.2, 0.5, 0.8]
K_SEL = [5, 10, 20, 40]
C_K = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def lr(l1_ratio, solver="liblinear"):
    return LogisticRegression(l1_ratio=l1_ratio, solver=solver, max_iter=5000, random_state=0)


def pipe(feat, model):
    if feat == "all":
        return Pipeline([("scale", StandardScaler()), ("model", model)])
    if feat == "block":
        return Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", model)])
    if feat == "corrk":
        return Pipeline([("scale", StandardScaler()), ("sel", CorrelationTopK(k=10)), ("model", model)])
    raise ValueError(feat)


H_LIN = ("Linear decision rules cannot use the class difference in covariance (R003: QDA 0.996 vs LDA 0.878). "
         "Prediction: every linear model stays in 0.83-0.91 CV AUC whatever the penalty and feature set; "
         "block >= all features; nested tuning changes R003's untuned values by < 0.015. "
         "Refuted if any linear model reaches > 0.93.")
FEAT = {"all": "all 150", "block": "block(in-fold highvar)", "corrk": "corr-topk(in-fold)"}


def candidates():
    out = []
    for feat in ("all", "block"):
        out += [
            Cand(f"l1_{feat}", lambda d, f=feat: Tuned(pipe(f, lr(1.0)), {"model__C": C_L1}, log_dir=d),
                 "supervised-linear", FEAT[feat], False, len(C_L1), f"L1 logistic, {FEAT[feat]}, nested C", H_LIN,
                 tuning=f"C in {C_L1}"),
            Cand(f"l2_{feat}", lambda d, f=feat: Tuned(pipe(f, lr(0.0)), {"model__C": C_L2}, log_dir=d),
                 "supervised-linear", FEAT[feat], False, len(C_L2), f"L2 logistic, {FEAT[feat]}, nested C", H_LIN,
                 tuning=f"C in {C_L2}"),
            Cand(f"en_{feat}", lambda d, f=feat: Tuned(pipe(f, lr(0.5, "saga")),
                                                       {"model__C": C_EN, "model__l1_ratio": R_EN}, log_dir=d),
                 "supervised-linear", FEAT[feat], False, len(C_EN) * len(R_EN),
                 f"elastic-net logistic (saga), {FEAT[feat]}, nested C x l1_ratio", H_LIN,
                 tuning=f"C in {C_EN} x l1_ratio in {R_EN}"),
            Cand(f"lda_{feat}", lambda d, f=feat: Tuned(pipe(f, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                                                        None, log_dir=d),
                 "supervised-linear", FEAT[feat], False, 1,
                 f"shrinkage LDA (Ledoit-Wolf, analytic), {FEAT[feat]}", H_LIN, tuning="none (analytic shrinkage)"),
        ]
    out += [
        Cand("l1_corrk", lambda d: Tuned(pipe("corrk", lr(1.0)), {"sel__k": K_SEL, "model__C": C_K}, log_dir=d),
             "supervised-linear", FEAT["corrk"], False, len(K_SEL) * len(C_K),
             "L1 logistic on in-fold top-k |corr| features, nested k x C", H_LIN, tuning=f"k in {K_SEL} x C in {C_K}"),
        Cand("l2_corrk", lambda d: Tuned(pipe("corrk", lr(0.0)), {"sel__k": K_SEL, "model__C": C_K}, log_dir=d),
             "supervised-linear", FEAT["corrk"], False, len(K_SEL) * len(C_K),
             "L2 logistic on in-fold top-k |corr| features, nested k x C", H_LIN, tuning=f"k in {K_SEL} x C in {C_K}"),
        Cand("lda_corrk", lambda d: Tuned(pipe("corrk", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                                          {"sel__k": K_SEL}, log_dir=d),
             "supervised-linear", FEAT["corrk"], False, len(K_SEL),
             "shrinkage LDA on in-fold top-k |corr| features, nested k", H_LIN, tuning=f"k in {K_SEL}"),
    ]
    return out


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    run_candidates(candidates(), "r200", a.data, a.n_jobs, a.author, "research/r200_sup_linear.py", a.only, a.max_folds)
