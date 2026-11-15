"""r931_cand_lib - array-level re-implementations of the final-candidate procedures (verifier label "cand").

No archive writes. Imported by research/r932_*.py and r933_*.py with importlib (the module name starts with a
digit):  C = importlib.import_module("r931_cand_lib").

Every procedure reproduces the code path of the archived run that produced the candidate's test_pred.npy
(official CV 5-fold x 20, seed 2026, test score = mean of the 100 fold models' test scores), but on an arbitrary
train+test feature matrix (train rows first), so the SAME procedure can be re-run on replica datasets with known
labels (research/lib_gen_replica.py):

  s003_gmm2_hv          ssc.transductive.TransductiveGMM("highvar", 2)                    run 20261022-050708_r003_gmm2_hv (S003)
  r105_gmm2_block_mle   lib_mix.MixGMM(block, 2, reg=1e-9, polish=True)                    run 20261022-074857_r105_gmm2_block_mle
  r155_gmm2_inf_tight   lib_gen_replica.TransductiveGMMTight(columns="informative")        run 20261022-080453_r155_gmm2_inf_tight
  r102_semisup_eps01    lib_mix.SemiSupEM(block, 1, eps=0.01)                              run 20261022-064500_r102_semisup_eps01
  r255_bqda_eb_hv       SubspaceModel(lib_fm.BayesQDA(), "highvar")                        run 20261022-070200_r255_bqda_eb_hv
  r209_selftrain_qda_lw lib_sup.PoolSelfTraining(QDA(solver="eigen", shrinkage="auto"), threshold=0.99)
                                                                                           run 20261022-080632_r209_selftrain_qda_lw
  r210_qda_block_nll    lib_sup.Tuned(HighVarColumns + StandardScaler + QDA, reg grid REG, scoring neg_log_loss)
                                                                                           run 20261022-082047_r210_qda_block_nll
  r201_qda_block        same, scoring roc_auc                                              run 20261022-071234_r201_qda_block
  x_qda_block_nll_ext   NOT archived before: r210 with the grid extended below its edge (REG + 3e-4 .. 1e-5)

The nested QDA is computed without GridSearchCV but with the identical inner splits
(RepeatedStratifiedKFold(5, 2, random_state=0) on the 200-row training fold), identical pipeline steps and
scorers (roc_auc on decision_function, log_loss on predict_proba[:, 1]) and GridSearchCV's selection rule
(np.average over splits, rankdata(-mean, "min"), first minimum), so one pass yields r201, r210 and the
extended-grid choice. research/r932 checks every re-implementation against the archived test predictions.

all_scores(..., cv_out={}) also returns the official-CV validation-fold AUCs of every procedure (same fold models,
validation rows scored as the archived runs scored them: ssc.models.predict_score = predict_proba[:, 1]); added for
research/r933 --cv-stats, checked against the archived fold_scores.csv by research/r933's real-data pass.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lib_gen_replica as G  # noqa: E402
import lib_mix as L  # noqa: E402
from lib_fm import BayesQDA  # noqa: E402
from lib_sup import HighVarColumns  # noqa: E402
from ssc.validation import make_splits  # noqa: E402

REG = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001]       # r201 / r210 grid (strongest first)
REG_EXT = REG + [3e-4, 1e-4, 3e-5, 1e-5]                                     # below the r210 grid edge
N_BASE = len(REG)

PROCS = ["s003_gmm2_hv", "r105_gmm2_block_mle", "r155_gmm2_inf_tight", "r102_semisup_eps01", "r255_bqda_eb_hv",
         "r209_selftrain_qda_lw", "r210_qda_block_nll", "r201_qda_block", "x_qda_block_nll_ext"]
RUN_IDS = {
    "s003_gmm2_hv": "20261022-050708_r003_gmm2_hv",
    "r105_gmm2_block_mle": "20261022-074857_r105_gmm2_block_mle",
    "r155_gmm2_inf_tight": "20261022-080453_r155_gmm2_inf_tight",
    "r102_semisup_eps01": "20261022-064500_r102_semisup_eps01",
    "r255_bqda_eb_hv": "20261022-070200_r255_bqda_eb_hv",
    "r209_selftrain_qda_lw": "20261022-080632_r209_selftrain_qda_lw",
    "r210_qda_block_nll": "20261022-082047_r210_qda_block_nll",
    "r201_qda_block": "20261022-071234_r201_qda_block",
    "x_qda_block_nll_ext": None,
}
USES_TEST_FEATURES = {"s003_gmm2_hv": True, "r105_gmm2_block_mle": True, "r155_gmm2_inf_tight": True,
                      "r102_semisup_eps01": True, "r255_bqda_eb_hv": True, "r209_selftrain_qda_lw": True,
                      "r210_qda_block_nll": False, "r201_qda_block": False, "x_qda_block_nll_ext": False}


class LabelFree:
    """All label-free fits of one train+test feature matrix (train rows first), shared by every label vector."""

    def __init__(self, X_all, n_train: int, need=PROCS):
        self.X = np.asarray(X_all, dtype=float)
        self.n_tr = int(n_train)
        self.need = set(need)
        self.seconds = {}
        t = time.perf_counter()
        self.hv = G.highvar_columns(self.X)                                     # std > 1.5 x median std
        if "s003_gmm2_hv" in self.need:
            self.R_s003 = G.UnsupervisedGMM(self.X, "highvar", 2).fit().R          # = ssc TransductiveGMM mixture
        self.seconds["s003"] = time.perf_counter() - t
        if self.need & {"r105_gmm2_block_mle", "r102_semisup_eps01"}:
            t = time.perf_counter()
            self.U = L.basis_arr(self.X, tuple(int(c) for c in self.hv))         # block, non-degenerate PCA basis
            if "r105_gmm2_block_mle" in self.need:
                self.R_mle = L.gmm_fit_arr(self.U, 2, "full", 5, 0, 1e-9)["post"]
            self.seconds["r105"] = time.perf_counter() - t
            if "r102_semisup_eps01" in self.need:
                t = time.perf_counter()
                self.post0 = L.gmm_fit_arr(self.U, 2, "full", 5, 0, 1e-6)["post"]
                self.seconds["r102_post0"] = time.perf_counter() - t
        if "r155_gmm2_inf_tight" in self.need:
            t = time.perf_counter()
            g = G.UnsupervisedGMM(self.X, "informative", 2, tol=1e-8, max_iter=5000).fit()
            self.R_inf, self.inf_cols = g.R, np.asarray(g.cols_)
            self.seconds["r155"] = time.perf_counter() - t
        B = self.X[:, self.hv]
        self.Zhv = (B - B.mean(0)) / B.std(0)                                     # SubspaceModel / pool_space scaling


# ----------------------------------------------------------------------------- label-dependent parts
def gmm_scores(R, n_tr, y, splits, alpha: float = 1.0, cv=None):
    """TransductiveGMM / MixGMM / TransductiveGMMTight fold-mean test score: mean_f clip(R_test @ rate_f)."""
    Rte = R[n_tr:]
    acc = None
    for s in splits:
        yt = y[s.train_idx]
        Rt = R[s.train_idx]
        rate = (Rt.T @ yt + alpha * yt.mean()) / (Rt.sum(0) + alpha)
        sc = np.clip(Rte @ rate, 0.0, 1.0)
        if cv is not None:
            cv.append(roc_auc_exact(y[s.val_idx], np.clip(R[s.val_idx] @ rate, 0.0, 1.0)))
        acc = sc if acc is None else acc + sc
    return acc / len(splits)


def semisup_scores(lf: LabelFree, y, splits, eps: float = 0.01, reg: float = 1e-6, cv=None):
    """lib_mix.SemiSupEM(block, k_per_class=1, eps): EM with the fold's training labels as soft constraints,
    started from the polished unsupervised fit; test score = unconstrained P(class 1 | x)."""
    n_tr = lf.n_tr
    acc, iters = None, []
    for s in splits:
        idx = s.train_idx
        yt = y[idx]
        rate = L.component_rates(lf.post0[idx], yt)
        cc = np.zeros(2, int)
        cc[np.argsort(rate)[::-1][:1]] = 1
        fit = L.em(lf.U, lf.post0, "full", reg, tol=1e-10, label_idx=idx, label_cls=yt, comp_class=cc, eps=eps)
        iters.append(fit["n_iter"])
        lp = L.component_logpdf(lf.U[n_tr:], fit["means"], fit["covs"]) + np.log(fit["weights"])[None]
        R = np.exp(lp - logsumexp(lp, axis=1)[:, None])
        sc = np.clip(R[:, cc == 1].sum(1), 0.0, 1.0)
        if cv is not None:
            lv = L.component_logpdf(lf.U[s.val_idx], fit["means"], fit["covs"]) + np.log(fit["weights"])[None]
            Rv = np.exp(lv - logsumexp(lv, axis=1)[:, None])
            cv.append(roc_auc_exact(y[s.val_idx], np.clip(Rv[:, cc == 1].sum(1), 0.0, 1.0)))
        acc = sc if acc is None else acc + sc
    return acc / len(splits), {"semisup_em_iter_mean": float(np.mean(iters)), "semisup_em_iter_max": int(np.max(iters))}


def bqda_scores(lf: LabelFree, y, splits, cv=None):
    """SubspaceModel(BayesQDA(), 'highvar'): EB Normal-Inverse-Wishart QDA on the train+test-standardised block."""
    Ztr, Zte = lf.Zhv[: lf.n_tr], lf.Zhv[lf.n_tr:]
    acc = None
    for s in splits:
        bq = BayesQDA().fit(Ztr[s.train_idx], y[s.train_idx])
        sc = bq.predict_proba(Zte)[:, 1]
        if cv is not None:
            cv.append(roc_auc_exact(y[s.val_idx], bq.predict_proba(Ztr[s.val_idx])[:, 1]))
        acc = sc if acc is None else acc + sc
    return acc / len(splits)


def selftrain_scores(lf: LabelFree, y, splits, cv=None):
    """lib_sup.PoolSelfTraining(QDA(solver='eigen', shrinkage='auto'), threshold=0.99) on all rows of the pool."""
    n = len(lf.Zhv)
    acc, npl = None, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in splits:
            yp = np.full(n, -1, dtype=int)
            yp[s.train_idx] = y[s.train_idx]
            st = SelfTrainingClassifier(QDA(solver="eigen", shrinkage="auto"), threshold=0.99, criterion="threshold",
                                        k_best=10, max_iter=10).fit(lf.Zhv, yp)
            npl.append(int((st.labeled_iter_ > 0).sum()))
            sc = st.predict_proba(lf.Zhv[lf.n_tr:])[:, 1]
            if cv is not None:
                cv.append(roc_auc_exact(y[s.val_idx], st.predict_proba(lf.Zhv[s.val_idx])[:, 1]))
            acc = sc if acc is None else acc + sc
    return acc / len(splits), {"selftrain_n_pseudo_mean": float(np.mean(npl))}


def _gs_best(M):
    """GridSearchCV's choice: mean over splits, rank 'min' of -mean, first minimum (ties -> first grid entry)."""
    means = np.average(M, axis=1)
    if np.isnan(means).all():
        return 0
    means = np.nan_to_num(means, nan=np.nanmin(means) - 1)
    return int(rankdata(-means, method="min").argmin())


class FastQDA:
    """sklearn 1.9 QuadraticDiscriminantAnalysis(solver='svd', reg_param=r, tol=1e-4) for MANY reg values at once:
    the per-class SVD does not depend on reg (scaling = (1 - reg) * S^2 / n_k + reg), so it is computed once.
    _scores() repeats sklearn's _decision_function expressions; decision_function / predict_proba as in sklearn's
    DiscriminantAnalysisPredictionMixin. A reg whose scalings are not all > tol raises LinAlgError like sklearn."""

    def __init__(self, Z, y, tol: float = 1e-4):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        _, cnts = np.unique(y, return_counts=True)
        self.priors_ = cnts / float(len(y))
        self.tol = tol
        self.d = Z.shape[1]
        self.means_, self.s2_, self.rot_ = [], [], []
        for c in self.classes_:
            Xc = Z[y == c, :]
            if Xc.shape[0] == 1:
                raise ValueError("class with one sample")
            self.means_.append(Xc.mean(0))
            mean = Xc.mean(0)
            _, S, Vt = np.linalg.svd(Xc - mean, full_matrices=False)
            self.s2_.append((S ** 2) / Xc.shape[0])
            self.rot_.append(Vt.T)

    def _scores(self, X, reg):
        norm2, u = [], []
        for i in range(len(self.classes_)):
            S = ((1 - reg) * self.s2_[i]) + reg
            if np.sum(S > self.tol) < self.d:
                raise np.linalg.LinAlgError("covariance not full rank: increase reg_param")
            R = self.rot_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2 ** 2, axis=1))
            u.append(np.sum(np.log(S)))
        norm2 = np.array(norm2).T
        return -0.5 * (norm2 + np.asarray(u)) + np.log(self.priors_)

    def decision_function(self, X, reg):
        s = self._scores(X, reg)
        return s[:, 1] - s[:, 0]

    def predict_proba(self, X, reg):
        s = self._scores(X, reg)
        ll = s - s.max(axis=1)[:, np.newaxis]
        return np.exp(ll - np.log(np.exp(ll).sum(axis=1)[:, np.newaxis]))


def roc_auc_exact(y_true, y_score):
    """sklearn 1.9 roc_auc_score for binary 0/1 labels without its input validation (same float operations:
    stable descending sort, tie-grouped thresholds, drop_intermediate, (0, 0) start, trapezoid). Checked
    bit-for-bit against sklearn in research/r932."""
    y = (np.asarray(y_true) == 1).astype(np.float64)
    s = np.asarray(y_score, dtype=float)
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    thr = np.r_[np.nonzero(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y * 1.0, dtype=np.float64)[thr]
    fps = 1 + thr.astype(np.float64) - tps
    if fps.shape[0] > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps, tps = fps[keep], tps[keep]
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]
    fpr, tpr = fps / fps[-1], tps / tps[-1]
    d = fpr[1:] - fpr[:-1]
    return float(np.sum(d * (tpr[1:] + tpr[:-1]) / 2.0, dtype=np.float64))


def log_loss_exact(y_true, p1):
    """sklearn 1.9 log_loss(y_true, p1) for binary 0/1 labels and a 1-D P(class 1) (columns [1 - p, p], clipped to
    [eps, 1 - eps], mean of -xlogy). Checked bit-for-bit against sklearn in research/r932."""
    p1 = np.asarray(p1, dtype=float)
    P = np.concatenate([(1 - p1)[:, None], p1[:, None]], axis=1)
    eps = np.finfo(P.dtype).eps
    P = np.clip(P, eps, 1 - eps)
    y = np.asarray(y_true)
    T = np.concatenate([(y == 0)[:, None], (y == 1)[:, None]], axis=1).astype(P.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        temp = T * np.log(P)
    loss = -np.sum(np.where(T == 0.0, 0.0, temp), axis=1)       # sklearn.utils._array_api._xlogy
    return float(np.average(loss))


def nested_qda_scores(Xtr_raw, Xte_raw, y, splits, grid=REG_EXT, n_base: int = N_BASE, inner_seed: int = 0,
                      exact_metrics: bool = False, val_out=None):
    """r201_qda_block (inner roc_auc), r210_qda_block_nll (inner neg_log_loss), both over grid[:n_base], and the
    extended-grid log-loss choice over the full grid. Pipeline per fit: HighVarColumns(1.5) -> StandardScaler -> QDA.
    Inner scorers are sklearn's (roc_auc_score on decision_function, log_loss on predict_proba[:, 1]); a reg that
    fails sklearn's rank check gives NaN (GridSearchCV error_score=nan)."""
    Xtr_raw = np.asarray(Xtr_raw, dtype=float)
    Xte_raw = np.asarray(Xte_raw, dtype=float)
    out = {"r201_qda_block": None, "r210_qda_block_nll": None, "x_qda_block_nll_ext": None}
    chosen = {k: [] for k in out}
    if exact_metrics:   # sklearn's own metric functions (slow input validation); default = bit-identical copies
        auc_f = roc_auc_score
        ll_f = log_loss
    else:
        auc_f, ll_f = roc_auc_exact, log_loss_exact
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in splits:
            Xo, yo = Xtr_raw[s.train_idx], y[s.train_idx]
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=inner_seed)
            inner = list(cv.split(Xo, yo))
            A = np.full((len(grid), len(inner)), np.nan)
            N = np.full((len(grid), len(inner)), np.nan)
            for j, (itr, iva) in enumerate(inner):
                sel = HighVarColumns().fit(Xo[itr])
                Xs = sel.transform(Xo[itr])
                sc = StandardScaler().fit(Xs)
                Zs = sc.transform(Xs)
                Zv = sc.transform(sel.transform(Xo[iva]))
                yv = yo[iva]
                try:
                    q = FastQDA(Zs, yo[itr])
                except np.linalg.LinAlgError:
                    continue
                for i, reg in enumerate(grid):
                    try:
                        A[i, j] = auc_f(yv, q.decision_function(Zv, reg))
                        N[i, j] = -ll_f(yv, q.predict_proba(Zv, reg)[:, 1])
                    except np.linalg.LinAlgError:
                        pass
            pick = {"r201_qda_block": _gs_best(A[:n_base]), "r210_qda_block_nll": _gs_best(N[:n_base]),
                    "x_qda_block_nll_ext": _gs_best(N)}
            sel = HighVarColumns().fit(Xo)
            Xs = sel.transform(Xo)
            scl = StandardScaler().fit(Xs)
            qf = FastQDA(scl.transform(Xs), yo)
            Zt = scl.transform(sel.transform(Xte_raw))
            Zval = scl.transform(sel.transform(Xtr_raw[s.val_idx])) if val_out is not None else None
            fitted = {}
            for k, i in pick.items():
                if i not in fitted:
                    try:
                        fitted[i] = qf.predict_proba(Zt, grid[i])[:, 1]
                    except np.linalg.LinAlgError:   # refit rank check fails (only possible below the r210 grid)
                        i = n_base - 1
                        fitted.setdefault(i, qf.predict_proba(Zt, grid[i])[:, 1])
                chosen[k].append(grid[i])
                if val_out is not None:
                    val_out.setdefault(k, []).append(roc_auc_exact(y[s.val_idx], qf.predict_proba(Zval, grid[i])[:, 1]))
                out[k] = fitted[i] if out[k] is None else out[k] + fitted[i]
    for k in out:
        out[k] = out[k] / len(splits)
    return out, chosen


def all_scores(lf: LabelFree, Xtr_raw, Xte_raw, y_train, scheme, procs=PROCS, cv_out=None):
    """Test scores (fold-mean over the official splits of y_train) of every requested procedure. With cv_out={} the
    validation-fold AUCs (split order) of every procedure are stored in cv_out[proc]."""
    y = np.asarray(y_train).astype(int)
    splits = make_splits(y, scheme)
    n_tr = lf.n_tr
    out, info, secs = {}, {}, {}
    t = time.perf_counter()
    want_cv = cv_out is not None

    def cvl(k):
        if not want_cv:
            return None
        cv_out[k] = []
        return cv_out[k]
    if "s003_gmm2_hv" in procs:
        out["s003_gmm2_hv"] = gmm_scores(lf.R_s003, n_tr, y, splits, cv=cvl("s003_gmm2_hv"))
    if "r105_gmm2_block_mle" in procs:
        out["r105_gmm2_block_mle"] = gmm_scores(lf.R_mle, n_tr, y, splits, cv=cvl("r105_gmm2_block_mle"))
    if "r155_gmm2_inf_tight" in procs:
        out["r155_gmm2_inf_tight"] = gmm_scores(lf.R_inf, n_tr, y, splits, cv=cvl("r155_gmm2_inf_tight"))
    secs["gmm_rates"] = time.perf_counter() - t
    if "r102_semisup_eps01" in procs:
        t = time.perf_counter()
        out["r102_semisup_eps01"], inf = semisup_scores(lf, y, splits, cv=cvl("r102_semisup_eps01"))
        info.update(inf)
        secs["r102"] = time.perf_counter() - t
    if "r255_bqda_eb_hv" in procs:
        t = time.perf_counter()
        out["r255_bqda_eb_hv"] = bqda_scores(lf, y, splits, cv=cvl("r255_bqda_eb_hv"))
        secs["r255"] = time.perf_counter() - t
    if "r209_selftrain_qda_lw" in procs:
        t = time.perf_counter()
        out["r209_selftrain_qda_lw"], inf = selftrain_scores(lf, y, splits, cv=cvl("r209_selftrain_qda_lw"))
        info.update(inf)
        secs["r209"] = time.perf_counter() - t
    if {"r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext"} & set(procs):
        t = time.perf_counter()
        qcv = {} if want_cv else None
        nq, chosen = nested_qda_scores(Xtr_raw, Xte_raw, y, splits, val_out=qcv)
        if want_cv:
            for k in ("r201_qda_block", "r210_qda_block_nll", "x_qda_block_nll_ext"):
                if k in procs:
                    cv_out[k] = qcv[k]
        for k, v in nq.items():
            if k in procs:
                out[k] = v
        info["chosen_reg"] = chosen
        secs["nested_qda"] = time.perf_counter() - t
    info["seconds"] = secs
    return out, info


def auc_fast(y, s):
    return G.auc_fast(y, s)


class SubsetAUC:
    """ROC AUC of one fixed score vector on many (labels, row-subset) combinations: the rows are sorted once
    (ties grouped, counted 1/2), each evaluation is O(n) with bincount. Equal to sklearn's roc_auc_score
    (and lib_gen_replica.auc_fast) up to floating-point rounding."""

    def __init__(self, s):
        s = np.asarray(s, dtype=float)
        self.o = np.argsort(s, kind="mergesort")
        ss = s[self.o]
        self.g = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
        self.G = int(self.g[-1]) + 1

    def __call__(self, y, mask=None):
        yo = np.asarray(y)[self.o]
        if mask is None:
            pos = (yo == 1).astype(float)
            neg = 1.0 - pos
        else:
            wo = np.asarray(mask)[self.o]
            pos = ((yo == 1) & wo).astype(float)
            neg = ((yo == 0) & wo).astype(float)
        npg = np.bincount(self.g, weights=pos, minlength=self.G)
        nng = np.bincount(self.g, weights=neg, minlength=self.G)
        below = np.cumsum(nng) - nng
        n1, n0 = npg.sum(), nng.sum()
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((npg * (below + 0.5 * nng)).sum() / (n1 * n0))
