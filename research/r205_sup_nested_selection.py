"""R205 - selection-aware (fully nested) estimate of the best SUPERVISED-ONLY procedure.

The best mean CV AUC over many evaluated candidates is biased upwards (winner's curse,
Varma & Simon 2006; Cawley & Talbot 2010). Here the whole "try a menu of model families and
hyperparameters, keep the best" procedure is itself one estimator: in every official outer
fold, GridSearchCV over a menu of 46 strictly supervised configurations (no test features;
the high-variance block is derived in-fold) runs an inner RepeatedStratifiedKFold(5, 2) on
the 200 training rows, picks the best configuration by inner AUC and refits it. The outer
AUC of this procedure is an honest estimate for "what can 250 labels alone achieve" after
model selection. The chosen family per fold is recorded (fold_fit_info.csv).

Menu: L2 / L1 logistic (all features), shrinkage LDA (all, block), QDA (block, variance
ratio 1.5 / 1.6 / 1.7 x reg_param), Gaussian NB (block), RBF-SVM with Platt probabilities
(block), kNN (block), degree-2 polynomial L2 logistic (block), LightGBM (block, fixed small).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r205_sup_nested_selection.py
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates


def lr(l1_ratio):
    return LogisticRegression(l1_ratio=l1_ratio, solver="liblinear", max_iter=5000, random_state=0)


HV = HighVarColumns(ratio=1.5)
MENU = [
    {"sel": ["passthrough"], "model": [lr(0.0)], "model__C": [0.003, 0.01, 0.03, 0.1, 0.3]},
    {"sel": ["passthrough"], "model": [lr(1.0)], "model__C": [0.05, 0.1, 0.2, 0.5, 1.0]},
    {"sel": ["passthrough", HV], "model": [LDA(solver="lsqr", shrinkage="auto")]},
    {"sel": [HighVarColumns(ratio=1.5), HighVarColumns(ratio=1.6), HighVarColumns(ratio=1.7)], "model": [QDA()],
     "model__reg_param": [0.5, 0.3, 0.1, 0.03, 0.01]},
    {"sel": [HV], "model": [GaussianNB()]},
    {"sel": [HV], "model": [SVC(kernel="rbf", probability=True, random_state=0)], "model__C": [1.0, 10.0, 100.0],
     "model__gamma": [0.01, 0.03, 0.1]},
    {"sel": [HV], "model": [KNeighborsClassifier()], "model__n_neighbors": [41, 25, 15, 9]},
    {"sel": [HV], "poly": [PolynomialFeatures(2, include_bias=False)], "scale2": [StandardScaler()], "model": [lr(0.0)],
     "model__C": [0.003, 0.01, 0.03, 0.1]},
    {"sel": [HV], "model": [LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                                           verbose=-1, n_jobs=1, random_state=0)]},
]


def n_menu():
    n = 0
    for g in MENU:
        k = 1
        for v in g.values():
            k *= len(v)
        n += k
    return n


def base():
    return Pipeline([("sel", "passthrough"), ("scale", StandardScaler()), ("poly", "passthrough"),
                     ("scale2", "passthrough"), ("model", lr(0.0))])


H = ("Winner's curse check. Prediction: the inner CV picks a QDA configuration in >= 90% of outer folds; the "
     "selection-aware outer AUC is within 0.003 of the best single nested supervised family (QDA, ~0.995), i.e. "
     "the optimism from choosing among ~70 supervised configurations is small because one family dominates. "
     "Refuted if the outer AUC is > 0.01 below the best single candidate.")


def candidates():
    return [Cand("select_supervised", lambda d: Tuned(base(), MENU, inner_repeats=2, log_dir=d), "selection",
                 "menu (all 150 / in-fold block)", False, n_menu(),
                 "nested model selection over a 46-config strictly supervised menu (families x hyperparameters)", H,
                 tuning="GridSearchCV over MENU (see script)")]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    run_candidates(candidates(), "r205", a.data, a.n_jobs, a.author, "research/r205_sup_nested_selection.py", a.only, a.max_folds)
