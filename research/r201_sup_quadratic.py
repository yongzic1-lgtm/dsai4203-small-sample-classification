"""R201 - supervised-only track, quadratic / generative / nonparametric models.

All tuning is NESTED (inner RepeatedStratifiedKFold(5, 2) on the 200-row training fold).
Feature sets: block = high-variance block re-derived in-fold from the 200 training rows
(no test features; with R001's ratio 1.5 the in-fold rule returns the 13 block columns plus
3-10 borderline noise columns, so one variant tunes the ratio in the nested CV); pcablock = in-fold PCA of that block keeping every non-null component
(the block has one exact linear redundancy, rank 12); block+extra = block plus the top-k
non-block features by in-fold |corr| with y; corrk = in-fold top-k |corr| over all 150;
block_tt = block chosen and standardised with train+test features (R003's SubspaceModel,
uses unlabeled test features, label-free) to show that this choice does not matter.

Candidates: QDA (reg_param nested) on block / pcablock / block+extra / corrk / all / block_tt,
Gaussian naive Bayes (block, all), logistic regression on degree-2 polynomial features of the
block (nested C), SVM-RBF (nested C x gamma; block, all), kNN (nested k x weights; block, all).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r201_sup_quadratic.py
"""

from __future__ import annotations

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

import lib_sup
from lib_sup import Cand, HighVarColumns, RankPCA, Tuned, common_args, run_candidates
from ssc.features import CorrelationTopK
from ssc.transductive import SubspaceModel

REG = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]   # strongest first (ties -> simpler)
REG_PCA = REG + [0.0]
RATIO = [1.5, 1.6, 1.7]   # in-fold variance rule; 1.5 = R001's a-priori rule (gives 16-23 columns on 200 rows)
REG_ALL = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3]
EXTRA_K = [0, 1, 2, 3, 5, 8]
REG_EXTRA = [0.3, 0.1, 0.03]
K_CORR = [5, 10, 13, 20, 30]
REG_CORR = [0.5, 0.3, 0.1]
C_POLY = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SVM_C = [0.1, 1.0, 10.0, 100.0]
SVM_G_BLOCK = [0.01, 0.03, 0.1, 0.3]
SVM_G_ALL = [0.0003, 0.001, 0.003, 0.01]
KNN_K = [41, 25, 15, 9, 5]
KNN_W = ["uniform", "distance"]


def block(*steps):
    return Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), *steps])


def allf(*steps):
    return Pipeline([("scale", StandardScaler()), *steps])


H_QDA = ("R003: QDA(reg 0.1) on the train+test block = 0.9958. Prediction: nested reg on the in-fold block gives "
         "0.994-0.997 (within 0.002 of R003). A smoke check showed that R001's ratio-1.5 rule on 200 rows returns a "
         "superset (16-23 columns, always containing the 13); prediction: the extra noise columns cost 0.001-0.005, "
         "so qda_blockratio (ratio tuned in the inner CV) and qda_block_tt are slightly better than qda_block; "
         "PCA basis within split noise; extra in-fold |corr| features change AUC by < 0.002; "
         "QDA on all 150 features or on corr-top-k is clearly worse (< 0.97) because covariance estimation in "
         "high dimension with 200 rows is poor. Refuted if nested QDA on the block < 0.99.")
H_GNB = ("Gaussian naive Bayes keeps class-specific variances but drops correlations. Prediction: between LDA "
         "(0.878, means only) and QDA (0.996, full covariance): 0.92-0.98 on the block; worse on all features.")
H_POLY = ("A linear logistic model on degree-2 features of the block can represent the QDA boundary; with 104 "
          "features and 200 rows it needs strong L2. Prediction: 0.97-0.995, below QDA.")
H_SVM = ("RBF-SVM can approximate the quadratic boundary. Prediction: block 0.97-0.995 (<= QDA), all features "
         "< 0.90 (RBF distances dominated by 137 noise features).")
H_KNN = ("kNN in 13 dims with 200 rows is data-starved. Prediction: block 0.90-0.97, all features < 0.85.")


def candidates():
    src = lib_sup.CURRENT["source"]
    return [
        Cand("qda_block", lambda d: Tuned(block(("model", QDA())), {"model__reg_param": REG}, log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, len(REG),
             "QDA on in-fold high-variance block, nested reg_param", H_QDA, tuning=f"reg_param in {REG}"),
        Cand("qda_blockratio", lambda d: Tuned(block(("model", QDA())), {"sel__ratio": RATIO, "model__reg_param": REG}, log_dir=d),
             "supervised-generative", "block(in-fold highvar, ratio tuned)", False, len(RATIO) * len(REG),
             "QDA on in-fold high-variance block, nested variance ratio x reg_param", H_QDA,
             tuning=f"ratio in {RATIO} x reg_param in {REG}"),
        Cand("qda_pcablock", lambda d: Tuned(block(("pca", RankPCA()), ("model", QDA())), {"model__reg_param": REG_PCA}, log_dir=d),
             "supervised-generative", "PCA(rank) of block(in-fold)", False, len(REG_PCA),
             "QDA on in-fold PCA basis (non-null components) of in-fold block, nested reg_param", H_QDA,
             tuning=f"reg_param in {REG_PCA}"),
        Cand("qda_blockextra", lambda d: Tuned(Pipeline([("sel", HighVarColumns(extra_k=0)), ("scale", StandardScaler()), ("model", QDA())]),
                                               {"sel__extra_k": EXTRA_K, "model__reg_param": REG_EXTRA}, log_dir=d),
             "supervised-generative", "block(in-fold)+top-k corr extras(in-fold)", False, len(EXTRA_K) * len(REG_EXTRA),
             "QDA on in-fold block plus k extra in-fold |corr| features, nested k x reg_param", H_QDA,
             tuning=f"extra_k in {EXTRA_K} x reg_param in {REG_EXTRA}"),
        Cand("qda_corrk", lambda d: Tuned(allf(("sel", CorrelationTopK(k=13)), ("model", QDA())),
                                          {"sel__k": K_CORR, "model__reg_param": REG_CORR}, log_dir=d),
             "supervised-generative", "corr-topk(in-fold)", False, len(K_CORR) * len(REG_CORR),
             "QDA on in-fold top-k |corr| features of all 150, nested k x reg_param", H_QDA,
             tuning=f"k in {K_CORR} x reg_param in {REG_CORR}"),
        Cand("qda_all", lambda d: Tuned(allf(("model", QDA())), {"model__reg_param": REG_ALL}, log_dir=d),
             "supervised-generative", "all 150", False, len(REG_ALL),
             "QDA on all 150 features, nested reg_param", H_QDA, tuning=f"reg_param in {REG_ALL}"),
        Cand("qda_block_tt", lambda d: Tuned(SubspaceModel(QDA(), "highvar", source=src), {"estimator__reg_param": REG}, log_dir=d),
             "supervised-generative", "block(train+test highvar, train+test standardised)", True, len(REG),
             "QDA on R003's train+test block (SubspaceModel), nested reg_param", H_QDA, tuning=f"reg_param in {REG}"),
        Cand("gnb_block", lambda d: Tuned(block(("model", GaussianNB())), None, log_dir=d),
             "supervised-generative", "block(in-fold highvar)", False, 1, "Gaussian naive Bayes on in-fold block", H_GNB),
        Cand("gnb_block_tt", lambda d: Tuned(SubspaceModel(GaussianNB(), "highvar", source=src), None, log_dir=d),
             "supervised-generative", "block(train+test highvar, train+test standardised)", True, 1,
             "Gaussian naive Bayes on R003's train+test block (SubspaceModel)", H_GNB),
        Cand("gnb_all", lambda d: Tuned(allf(("model", GaussianNB())), None, log_dir=d),
             "supervised-generative", "all 150", False, 1, "Gaussian naive Bayes on all 150 features", H_GNB),
        Cand("poly2lr_block", lambda d: Tuned(block(("poly", PolynomialFeatures(2, include_bias=False)), ("scale2", StandardScaler()),
                                                    ("model", LogisticRegression(l1_ratio=0.0, solver="liblinear", max_iter=5000, random_state=0))),
                                              {"model__C": C_POLY}, log_dir=d),
             "supervised-linear", "poly2(block in-fold)", False, len(C_POLY),
             "L2 logistic on degree-2 polynomial features of the in-fold block, nested C", H_POLY, tuning=f"C in {C_POLY}"),
        Cand("svm_block", lambda d: Tuned(block(("model", SVC(kernel="rbf", random_state=0))),
                                          {"model__C": SVM_C, "model__gamma": SVM_G_BLOCK}, log_dir=d),
             "supervised-kernel", "block(in-fold highvar)", False, len(SVM_C) * len(SVM_G_BLOCK),
             "RBF-SVM on in-fold block, nested C x gamma (decision_function scores)", H_SVM,
             tuning=f"C in {SVM_C} x gamma in {SVM_G_BLOCK}"),
        Cand("svm_all", lambda d: Tuned(allf(("model", SVC(kernel="rbf", random_state=0))),
                                        {"model__C": SVM_C, "model__gamma": SVM_G_ALL}, log_dir=d),
             "supervised-kernel", "all 150", False, len(SVM_C) * len(SVM_G_ALL),
             "RBF-SVM on all 150 features, nested C x gamma", H_SVM, tuning=f"C in {SVM_C} x gamma in {SVM_G_ALL}"),
        Cand("knn_block", lambda d: Tuned(block(("model", KNeighborsClassifier())),
                                          {"model__n_neighbors": KNN_K, "model__weights": KNN_W}, log_dir=d),
             "supervised-kernel", "block(in-fold highvar)", False, len(KNN_K) * len(KNN_W),
             "kNN on in-fold block, nested k x weights", H_KNN, tuning=f"k in {KNN_K} x weights in {KNN_W}"),
        Cand("knn_all", lambda d: Tuned(allf(("model", KNeighborsClassifier())),
                                        {"model__n_neighbors": KNN_K, "model__weights": KNN_W}, log_dir=d),
             "supervised-kernel", "all 150", False, len(KNN_K) * len(KNN_W),
             "kNN on all 150 features, nested k x weights", H_KNN, tuning=f"k in {KNN_K} x weights in {KNN_W}"),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r201", a.data, a.n_jobs, a.author, "research/r201_sup_quadratic.py", a.only, a.max_folds)
