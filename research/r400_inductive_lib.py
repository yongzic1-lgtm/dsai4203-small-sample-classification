"""r400_inductive_lib - strictly inductive candidate procedures for research round 2 (label "r2_inductive").

No archive writes. Imported by research/r401_*.py, r402_*.py with importlib (the module name starts with a digit):
    L4 = importlib.import_module("r400_inductive_lib")

Question (D010 switch 3): if the instructor forbids any use of the unlabeled test features, which two strictly
inductive submissions should be the final selections? Strictly inductive = the model of each outer training fold
sees only the 200 training rows of that fold (features and labels); no test row, not even its features, enters
any fit, block rule, scaling or tuning step. The test rows are only scored.

Procedures (array level, official CV 5x20 seed 2026, test score = mean of the 100 fold models' test scores, as
ssc.validation.cross_validate does it: running sum in split order, then / 100):

  r201_qda_block            HighVarColumns(1.5) in-fold -> StandardScaler -> QDA, nested reg (inner roc_auc)
  r210_qda_block_nll        same, inner neg_log_loss                                       } r931_cand_lib
  x_qda_block_nll_ext       same, inner neg_log_loss, grid extended to 1e-5 (= r930)       } .nested_qda_scores
  r255_bqda_eb_trainonly    lib_fm.TrainHighVar(1.5) in-fold -> lib_fm.BayesQDA() (EB NIW prior, Student-t)
  x_bqda_trainonly_wk       same, EB kappa0 grid widened beyond R255's edge 1.0 (WIDE_KAPPA, up to 300)
  r208_qda_block_lw         HighVarColumns(1.5) in-fold -> StandardScaler -> QDA(solver="eigen", shrinkage="auto")
  x_ens_r201_bqda_to        0.5 * (r201_qda_block + r255_bqda_eb_trainonly)   (probability average)

The ensemble is computed from the two fold-mean vectors; the official-CV run of the same ensemble
(research/r401, ProbAverage below) averages per fold, which differs only by floating-point rounding (~1e-17).

Transductive references (for comparison only; they use the test features): S003 (TransductiveGMM), r255_bqda_eb_hv,
r209_selftrain_qda_lw, all via research/r931_cand_lib.all_scores (checked bit-exact against the archive by r932).

ARCHIVED_SHA16: sha256[:16] of the float64 bytes of each archived test_pred.npy, so a real-data pass on the VM
(where experiments/ is not synced) can prove that every re-implementation reproduces the archived predictions.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

C = importlib.import_module("r931_cand_lib")
from lib_fm import BayesQDA, TrainHighVar  # noqa: E402
from lib_sup import HighVarColumns  # noqa: E402

WIDE_KAPPA = (0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)   # superset of R255's grid and R256's wide grid

TRANSDUCTIVE = ["s003_gmm2_hv", "r255_bqda_eb_hv", "r209_selftrain_qda_lw"]
INDUCTIVE = ["r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext", "r255_bqda_eb_trainonly",
             "x_bqda_trainonly_wk", "r208_qda_block_lw", "x_ens_r201_bqda_to"]
EVAL = TRANSDUCTIVE + INDUCTIVE + ["bayes"]

RUN_IDS = {
    "s003_gmm2_hv": "20261022-050708_r003_gmm2_hv",
    "r255_bqda_eb_hv": "20261022-070200_r255_bqda_eb_hv",
    "r209_selftrain_qda_lw": "20261022-080632_r209_selftrain_qda_lw",
    "r201_qda_block": "20261022-071234_r201_qda_block",
    "r210_qda_block_nll": "20261022-082047_r210_qda_block_nll",
    "x_qda_block_nll_ext": "20261022-095943_r930_qda_block_nll_ext",
    "r255_bqda_eb_trainonly": "20261022-070202_r255_bqda_eb_trainonly",
    "r208_qda_block_lw": "20261022-075228_r208_qda_block_lw",
    "x_bqda_trainonly_wk": None,       # new: research/r401 official run
    "x_ens_r201_bqda_to": None,        # new: research/r401 official run
}
ARCHIVED_SHA16 = {
    "s003_gmm2_hv": "71fb85e7168cde88",
    "r255_bqda_eb_hv": "e25a12aa3c506a3b",
    "r209_selftrain_qda_lw": "7ca0ee20d72004f1",
    "r201_qda_block": "0d7e9b64d238586a",
    "r210_qda_block_nll": "da02b1c4e7e7222b",
    "x_qda_block_nll_ext": "ec6dde0117266d4d",
    "r255_bqda_eb_trainonly": "412c1f9f1e3df3f5",
    "r208_qda_block_lw": "801a4eb5649c0165",
}


def sha16(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- new inductive procedures
def bqda_trainonly_scores(Xtr_raw, Xte_raw, y, splits, kappa_grid=None, cv=None):
    """Pipeline([("block", TrainHighVar(1.5)), ("model", BayesQDA(...))]) per outer fold (= r255_bqda_eb_trainonly
    with the default grid). Returns the fold-mean test score and the EB choice (kappa0, nu0_extra, prior) per fold."""
    Xtr_raw = np.asarray(Xtr_raw, dtype=float)
    Xte_raw = np.asarray(Xte_raw, dtype=float)
    kw = {} if kappa_grid is None else {"kappa_grid": tuple(kappa_grid)}
    acc, choices = None, []
    for s in splits:
        tf = TrainHighVar(1.5).fit(Xtr_raw[s.train_idx])
        bq = BayesQDA(**kw).fit(tf.transform(Xtr_raw[s.train_idx]), y[s.train_idx])
        sc = bq.predict_proba(tf.transform(Xte_raw))[:, 1]
        choices.append((float(bq.kappa0_), float(bq.nu0_extra_), str(bq.prior_), int(len(tf.columns_))))
        if cv is not None:
            cv.append(C.roc_auc_exact(y[s.val_idx], bq.predict_proba(tf.transform(Xtr_raw[s.val_idx]))[:, 1]))
        acc = sc if acc is None else acc + sc
    return acc / len(splits), choices


def qda_lw_block_scores(Xtr_raw, Xte_raw, y, splits, cv=None):
    """r208_qda_block_lw: HighVarColumns(1.5) in-fold -> StandardScaler -> QDA(solver='eigen', shrinkage='auto')."""
    Xtr_raw = np.asarray(Xtr_raw, dtype=float)
    Xte_raw = np.asarray(Xte_raw, dtype=float)
    acc = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in splits:
            Xo = Xtr_raw[s.train_idx]
            sel = HighVarColumns().fit(Xo)
            Xs = sel.transform(Xo)
            scl = StandardScaler().fit(Xs)
            q = QDA(solver="eigen", shrinkage="auto").fit(scl.transform(Xs), y[s.train_idx])
            sc = q.predict_proba(scl.transform(sel.transform(Xte_raw)))[:, 1]
            if cv is not None:
                cv.append(C.roc_auc_exact(y[s.val_idx], q.predict_proba(scl.transform(sel.transform(Xtr_raw[s.val_idx])))[:, 1]))
            acc = sc if acc is None else acc + sc
    return acc / len(splits)


def all_scores(lf, Xtr_raw, Xte_raw, y_train, scheme, procs=None):
    """Fold-mean test scores of every procedure in EVAL except 'bayes' (lf = r931_cand_lib.LabelFree with S003's
    mixture). Returns (scores, info) with the nested-QDA reg choices and the EB choices of both BQDA variants."""
    from ssc.validation import make_splits
    procs = list(procs or [k for k in EVAL if k != "bayes"])
    y = np.asarray(y_train).astype(int)
    splits = make_splits(y, scheme)
    ref = [k for k in procs if k in C.PROCS]
    if "x_ens_r201_bqda_to" in procs and "r201_qda_block" not in ref:
        ref.append("r201_qda_block")
    out, info = C.all_scores(lf, Xtr_raw, Xte_raw, y, scheme, procs=ref)
    if {"r255_bqda_eb_trainonly", "x_ens_r201_bqda_to"} & set(procs):
        out["r255_bqda_eb_trainonly"], info["eb_trainonly"] = bqda_trainonly_scores(Xtr_raw, Xte_raw, y, splits)
    if "x_bqda_trainonly_wk" in procs:
        out["x_bqda_trainonly_wk"], info["eb_trainonly_wk"] = bqda_trainonly_scores(Xtr_raw, Xte_raw, y, splits,
                                                                                   kappa_grid=WIDE_KAPPA)
    if "r208_qda_block_lw" in procs:
        out["r208_qda_block_lw"] = qda_lw_block_scores(Xtr_raw, Xte_raw, y, splits)
    if "x_ens_r201_bqda_to" in procs:
        out["x_ens_r201_bqda_to"] = 0.5 * (out["r201_qda_block"] + out["r255_bqda_eb_trainonly"])
    return {k: out[k] for k in procs}, info


# ----------------------------------------------------------------------------- estimators for the official CV (r401)
class ProbAverage(ClassifierMixin, BaseEstimator):
    """Average of the members' P(class 1). Every member is cloned and fitted on the same rows (the outer training
    fold), so the ensemble is exactly as inductive as its members."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.estimators_ = [clone(e).fit(X, y) for e in self.estimators]
        w = np.ones(len(self.estimators_)) if self.weights is None else np.asarray(self.weights, dtype=float)
        self.w_ = w / w.sum()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X):
        p = None
        for wi, e in zip(self.w_, self.estimators_):
            q = wi * np.asarray(e.predict_proba(X))[:, 1]
            p = q if p is None else p + q
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
