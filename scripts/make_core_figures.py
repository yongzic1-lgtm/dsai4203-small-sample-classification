"""Figures and table views for the core report (reports/report_core.md), built from archived files only.

    .venv\\Scripts\\python.exe scripts\\make_core_figures.py              # write reports/figures/core_*.png, reports/tables/core_*
    .venv\\Scripts\\python.exe scripts\\make_core_figures.py --check      # compute, cross-check and print; write nothing
    .venv\\Scripts\\python.exe scripts\\make_core_figures.py --only F3    # one figure (F1..F6) plus every table view

Nothing is fitted and no Run is created: the script reads archived files and computes descriptive statistics
(means, quantiles, correlations, kernel density outlines) and closed-form linear algebra. Style: src/ssc/plotstyle.py
(Times New Roman, drawn at the printed width 6.27 in, 300 dpi). Every figure has a table view in reports/tables/ and
every input is listed with its sha256 in reports/tables/core_manifest.json.

Figures (Section of the core report):
  F1 core_data.png        (1) anatomy of the data: (a) correlations of the 20 structured features over the 20,000
                          train+test rows (the 130 noise features: a note); (b) sorted sd-ratio profile, 13-column
                          high-variance block;
                          (c) the two classes as two differently shaped clouds (S003's scores, closed-form projection)
  F2 core_protocol.png    (2) why repeated 5-fold x 20: (a) E1 paired differences, 50 executions per scheme, with MDD;
                          (b, c) R900 label-permutation nulls A and B, one dot per permutation
  F3 core_journey.png     (3) the six uploads on one nines axis: CV, pre-registered expectation, public score +- 1 SE
  F4 core_landscape.png   (5) the 100 official fold AUCs of 14 representative models on the nines axis
  F5 core_unlabeled.png   (5) (a) label-budget curves (R441); (b) AUC on boundary rows of 222 replicas (R404)
  F6 core_selection.png   (4) (a) per-replica private-AUC difference to S003 (R936); (b) S003's predictive
                          distribution of the private AUC (R922, flat prior)
Table views (reports/tables/): core_journey.{csv,md} (= Table 1 of the report), core_label_budget.{csv,md},
core_boundary.{csv,md} (as before), core_data_features.csv, core_data.md, core_protocol_schemes.{csv,md},
core_protocol_nulls.{csv,md}, core_landscape.{csv,md}, core_budget_curves.csv, core_selection.{csv,md},
core_predictive.{csv,md}; core_manifest.json (inputs with sha256, outputs, conventions).

Conventions (also in the manifest):
  Nines axis: an AUC x is drawn at -log10(1 - x); exact 1.0 is drawn at 1 - 1e-5 and labelled "= 1", above an axis
  break (F3, F5a); F4 ends its axis at 0.9993 and counts the folds at exactly 1 in a separate column.
  Names: one display name per entity (plotstyle.LABEL); reference values are dashed lines (plotstyle.refline).
  Public-LB SE: Hanley & McNeil (1982) via ssc.validation.auc_standard_error at the pre-registered expected AUC for
  395 public rows split 198 positive / 197 negative (as the audited journey table R440). z(exp) = (public - expected)
  / SE, z(CV) = (public - CV mean) / SE. (scripts/record_lb.py prints a slightly different SE: at the observed score,
  179/216 rows; registry/beliefs.md quotes that one.)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common as C
from ssc import paths
from ssc import plotstyle as ps
from ssc.tracking import now_iso

SOURCE = "raw"
N_POS, N_NEG = 198, 197          # 395 public rows (2% of 19,750); test set is ~50% positive (as R440)
CAP = ps.CAP                      # where an AUC of exactly 1.0 is drawn on the nines axis

S003_RUN = "20261022-050708_r003_gmm2_hv"
E1_RUN = "20261022-064054_e1_validation_protocol"
E3_RUN = "20261022-063908_e3_leakage"
R901_RUN = "20261022-094812_r901_id_order_dupes"
R922_RUN = "20261022-092146_r922_label_noise_private"
R936_RUN = "20261022-172146_r936_final_selection"
R441_RUN = "20261022-182022_r441_label_budget_rerun"
R404_RUN = "20261022-194238_r404_inductive_pooled"
R900_RUNS = "20261022-091506_r900_perm_null_a + 20261022-092943_r900_perm_null_b"
R936_NPZ = "research/r935_r933_pooled.npz"     # R936's archived input (sha256 checked against its config.yaml)

# label-free feature taxonomy (research/reports/gen.md, R150; R901 summary.json informative12)
INFORMATIVE = [6, 15, 30, 46, 52, 61, 64, 80, 100, 112, 142, 144]
REDUNDANT = [8, 17, 26, 49, 57, 67, 77, 98]

# ---------------------------------------------------------------------------------------- presentation only
SUB_LABEL = {  # (model name in Table 1, unlabelled test features used, short name under the x axis of F3)
    "S001": ("L1 logistic, all features", "no", "L1 logistic, all 150"),
    "S002": ("QDA, label-free block", "block", "QDA, label-free block"),
    "S003": ("Transductive GMM (final)", "yes", "GMM (final)"),
    "S004": ("E3: corr top-20 + L2", "no", "E3 top-20 + L2"),
    "S005": ("Strict QDA, nested reg (final)", "no", "strict QDA (final)"),
    "S006": ("Self-training QDA-LW", "yes", "self-training QDA"),
}
QUESTION = {"S001": "Does linear\nCV hold?", "S002": "Is the signal in\nthe covariance?",
            "S003": "Does unlabelled\nstructure transfer?", "S004": "Does honest CV\npredict the LB?",
            "S005": "Strict QDA as a\nrule-safe backup?", "S006": "Self-training as\nthe alternative?"}
F1_H, F1C_W = 1.32, 1.52     # F1: panel height and width of panel (c), inches (also sets its x squeeze)

# F2 (a): E1 schemes shown, (scheme id in paired_estimates.csv, label)
E1_SCHEMES = [("holdout20", "Hold-out 20%"), ("5fold", "5-fold"), ("10fold", "10-fold"),
              ("5foldx10", "5-fold × 10"), ("5foldx20", "5-fold × 20")]
# F2 (b, c): R900 pipelines, (pipeline id, label, colour key)
R900_ROWS = [("s003_gmm2_hv", ps.name("s003"), "s003"), ("r155_gmm2_inf_tight", ps.name("r155"), "other"),
             ("r201_qda_block", ps.name("s005"), "s005"), ("r210_qda_block_nll", ps.name("r210"), "other"),
             ("r209_selftrain_qda_lw", ps.name("s006"), "s006"), ("r255_bqda_eb_hv", ps.name("bqda"), "bqda"),
             ("e3_honest", "E3 honest filter", "other"), ("ctrl_e3_leaky", "Leaky filter (control)", "leaky"),
             ("ctrl_disk_reader", "Reads labels (control)", "leaky"), ("ctrl_random", "Random scores (control)", "muted")]

# F4: representative models, (run id, label, group, colour key); groups in this order
LANDSCAPE = [
    ("20261022-050647_r003_sup_l1_all", ps.name("s001"), "labels", "other"),
    ("20261022-070352_r200_lda_block", "LDA, block", "labels", "other"),
    ("20261022-071710_r201_knn_block", "kNN, block", "labels", "other"),
    ("20261022-074613_r203_lgbm_block_nested", "LightGBM, block (best trees)", "labels", "other"),
    ("20261022-070236_r200_l1_block", "L1 logistic (best linear), block", "labels", "other"),
    ("20261022-071605_r201_svm_block", "RBF-SVM, block", "labels", "other"),
    ("20261022-072331_r202_gpc_ard_block", "GP classifier (ARD), block", "labels", "other"),
    ("20261022-071234_r201_qda_block", ps.name("s005"), "labels", "s005"),
    ("20261022-050659_r003_sup_qda_hv", ps.name("s002"), "test", "other"),
    ("20261022-070200_r255_bqda_eb_hv", ps.name("bqda"), "test", "bqda"),
    ("20261022-080632_r209_selftrain_qda_lw", ps.name("s006"), "test", "s006"),
    ("20261022-050708_r003_gmm2_hv", ps.name("s003"), "test", "s003"),
    ("20261022-073935_r251_tabpfn2_hv", ps.name("tabpfn") + ", block", "pretrained", "foundation"),
    ("20261022-074643_r251_tabicl_hv", ps.name("tabicl") + ", block", "pretrained", "foundation"),
]
GROUP_NAME = {"labels": "Labels only", "test": "Test-row features used",
              "pretrained": "Pretrained elsewhere (analysis only)"}

# F5 / Table views: the six procedures that both R441 (label budget) and R404 (boundary rows) evaluated
# (label, colour key, `model` in r441_label_budget.csv, `proc` in r404_boundary.csv)
ROWS = [
    ("S003 GMM (test features)", "s003", "s003_gmm2_hv", "s003_gmm2_hv"),
    ("S006 self-training QDA-LW (test features)", "s006", "r209_selftrain_qda_lw", "r209_selftrain_qda_lw"),
    ("Bayesian QDA, train+test block (r255)", "bqda", "r255_bqda_eb_hv", "r255_bqda_eb_hv"),
    ("QDA, log-loss reg (r210, labels only)", "other", "r210_qda_block_nll", "r210_qda_block_nll"),
    ("S005 strict QDA (r201, labels only)", "s005", "r201_qda_block", "r201_qda_block"),
    ("Bayesian QDA, in-fold block (r255, labels only)", "bqda", "r255_bqda_eb_trainonly", "r255_bqda_eb_trainonly"),
]
BUDGET_CURVES = [("s003_gmm2_hv", ps.name("s003"), "s003"), ("r209_selftrain_qda_lw", ps.name("s006"), "s006"),
                 ("r255_bqda_eb_hv", ps.name("bqda"), "bqda"), ("r201_qda_block", ps.name("s005"), "s005")]
# table views also list these (not in the figure)
# reg 0.003 was taken from R201's nested choices on all 250 labels (research/reports/r2_audit.md 2.3), so these two
# rows borrow information from the full label set; the report quotes them only with that caveat
BUDGET_EXTRA = [("eq206_qda_tt_r003", "QDA reg 0.003 (from R201, all 250 labels), train+test block (R206 model)"),
                ("eq206_qda_infold_r003", "strict QDA reg 0.003 (from R201, all 250 labels), in-fold block (R206 model)")]
BOUNDARY_EXTRA = [("bayes", "True class probability (Bayes posterior, ceiling)"),
                  ("x_qda_block_nll_ext", "QDA, ext. reg grid (r930, labels only)")]

# F6 (a): R936 candidates, (proc, label, colour key); primary convention of the report: group tn0 (test-label noise
# only), csv8 scores (8 decimals, like the submission files), noise levels weighted by R156
SELECTION = [("r209_selftrain_qda_lw", ps.name("s006"), "s006"),
             ("r255_bqda_eb_hv", ps.name("bqda"), "bqda"),
             ("r210_qda_block_nll", ps.name("r210"), "other"),
             ("r201_qda_block", ps.name("s005"), "s005"),
             ("x_qda_block_nll_ext", ps.name("ext_qda"), "other")]      # post hoc (R936 lists it as a candidate)
W_R156 = {0.0: 0.0773, 0.0025: 0.4069, 0.005: 0.3608, 0.01: 0.1550}

ROLE_COLOR = {"s003": ps.BLUE, "s005": ps.ORANGE, "s006": ps.AQUA, "bqda": ps.YELLOW, "foundation": ps.VIOLET,
              "leaky": ps.RED, "other": ps.INK2, "muted": ps.MUTED, "bayes": ps.INK}

INPUTS: dict[str, Path] = {}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(p: Path) -> Path:
    """An input file: must exist; recorded for the manifest."""
    if not p.exists():
        C.die(f"input {p} not found (this script reads the real archive; is $SSC_ARCHIVE_ROOT set?)", 2)
    INPUTS[C.rel(p).replace("\\", "/")] = p
    return p


def check(ok: bool, what: str) -> None:
    if not ok:
        C.die(f"cross-check failed: {what}", 1)


def signed(v: float, nd: int) -> str:
    """Signed number with a typographic minus (U+2212), as printed in the report."""
    return f"{v:+.{nd}f}".replace("-", "−")


def hm_se(auc: float) -> float:
    from ssc.validation import auc_standard_error
    return auc_standard_error(float(auc), N_POS, N_NEG)


def write_md(df: pd.DataFrame, path: Path, sources: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<!-- generated by scripts/make_core_figures.py from {sources}; do not edit by hand -->\n")
        f.write(df.to_markdown(index=False, disable_numparse=True) + "\n")


def exp_dir() -> Path:
    return paths.experiments_dir(SOURCE)


# ============================================================================================ data: tables

def journey_table(subs_csv: Path, hist_csv: Path) -> pd.DataFrame:
    s = pd.read_csv(subs_csv, dtype=str).fillna("")
    s = s[s["public_lb"] != ""]
    rows = []
    for _, r in s.iterrows():
        sid = r["sub_id"]
        name, tf, _ = SUB_LABEL.get(sid, (r["description"], "", sid))
        cv, sd = float(r["cv_auc_mean"]), float(r["cv_auc_std_repeat"])
        exp, pub = float(r["expected_public"]), float(r["public_lb"])
        se = hm_se(exp)
        rows.append({"sub_id": sid, "day_utc": r["submitted_at_utc"][:10], "run_id": r["run_id"], "model": name,
                     "test_features": tf, "cv_auc_mean": cv, "cv_auc_sd": sd, "expected_public": exp,
                     "public_lb": pub, "hm_se_at_expected": se, "public_minus_expected": pub - exp,
                     "public_minus_cv": pub - cv, "z_expected": (pub - exp) / se, "z_cv": (pub - cv) / se,
                     "sha256_prefix": r["sha256"]})
    t = pd.DataFrame(rows)
    if hist_csv.exists():  # the same numbers must come out of scripts/submission_history.py
        need(hist_csv)
        h = pd.read_csv(hist_csv).set_index("sub_id")
        for _, r in t.iterrows():
            if r["sub_id"] in h.index:
                hr = h.loc[r["sub_id"]]
                for a in ("cv_auc_mean", "expected_public", "public_lb"):
                    check(abs(float(hr[a]) - r[a]) <= 1e-9, f"{r['sub_id']}: {a} differs from {hist_csv.name}")
    return t


def journey_md(t: pd.DataFrame) -> pd.DataFrame:
    """Report Table 1 (7 columns, fits the page); day, test-feature use and the SE are in core_journey.csv."""
    return pd.DataFrame({
        "Sub": t["sub_id"],
        "Model": t["model"],
        "CV AUC ± sd": [f"{m:.4f} ± {s:.4f}" for m, s in zip(t["cv_auc_mean"], t["cv_auc_sd"])],
        "Expected": [f"{v:.4f}" for v in t["expected_public"]],
        "Public": [f"{v:.5f}" for v in t["public_lb"]],
        "z(exp)": [signed(v, 2) for v in t["z_expected"]],
        "z(CV)": [signed(v, 2) for v in t["z_cv"]],
    })


def budget_models() -> list[tuple[str, str, bool]]:
    """(r441 model, label, in the figure) for the table view."""
    curves = {m for m, *_ in BUDGET_CURVES}
    return [(m, lab, m in curves) for lab, _, m, _ in ROWS] + [(m, lab, False) for m, lab in BUDGET_EXTRA]


def budget_table(r441_csv: Path) -> pd.DataFrame:
    b = pd.read_csv(r441_csv)
    keep = [m for m, *_ in budget_models()]
    missing = sorted(set(keep) - set(b["model"]))
    check(not missing, f"{r441_csv.name} has no rows for {missing}")
    b = b[b["model"].isin(keep)].copy()
    b["defined_all"] = (b["n_failed"] == 0) & b["auc_mean"].notna()
    return b[["model", "n_labels", "n_evals", "n_failed", "defined_all", "auc_mean", "auc_median", "auc_q10",
              "auc_se_cluster", "uses_test_features"]].reset_index(drop=True)


def first_hit(b: pd.DataFrame, model: str, level: float = 0.99):
    """Smallest budget whose mean AUC reaches `level`, among budgets where every evaluation is defined
    (the rule of R440/R441: a mean over the few fits that succeed would be biased)."""
    d = b[(b["model"] == model) & b["defined_all"] & (b["auc_mean"] >= level)]
    return int(d["n_labels"].min()) if len(d) else None


def budget_md(b: pd.DataFrame) -> pd.DataFrame:
    budgets = sorted(b["n_labels"].unique())
    out = {"Model": [], **{str(n): [] for n in budgets}, "First ≥ 0.99": []}
    for m, label, in_fig in budget_models():
        out["Model"].append(label + ("" if in_fig else " *"))
        for n in budgets:
            r = b[(b["model"] == m) & (b["n_labels"] == n)]
            out[str(n)].append(f"{r['auc_mean'].iloc[0]:.4f}" if len(r) and bool(r["defined_all"].iloc[0]) else "–")
        fh = first_hit(b, m)
        out["First ≥ 0.99"].append(str(fh) if fh else "never")
    return pd.DataFrame(out)


def budget_curves(evals_csv: Path, b: pd.DataFrame, fh_csv: Path) -> pd.DataFrame:
    """Per model and budget: mean, 10% and 90% quantiles of the validation AUC over the 400 evaluations
    (80 folds x 5 label draws; 80 at n = 200), only where every evaluation is defined."""
    ev = pd.read_csv(evals_csv)
    fh = pd.read_csv(fh_csv).set_index("model")
    rows = []
    for m, label, role in BUDGET_CURVES:
        e = ev[ev["model"] == m]
        for n, g in e.groupby("n_labels"):
            a = g["auc"].to_numpy(float)
            defined = bool(np.isfinite(a).all())
            r = {"model": m, "label": label, "n_labels": int(n), "n_evals": len(a), "defined_all": defined,
                 "auc_mean": np.nan, "auc_q10": np.nan, "auc_q50": np.nan, "auc_q90": np.nan, "frac_eq_1": np.nan}
            if defined:
                r.update(auc_mean=a.mean(), auc_q10=np.quantile(a, 0.10), auc_q50=np.quantile(a, 0.5),
                         auc_q90=np.quantile(a, 0.90), frac_eq_1=float((a == 1.0).mean()))
                s = b[(b["model"] == m) & (b["n_labels"] == n)]
                if len(s):   # the summary table of R441 must give the same mean / q10
                    check(abs(float(s["auc_mean"].iloc[0]) - r["auc_mean"]) < 1e-12, f"R441 mean {m} n={n}")
                    check(abs(float(s["auc_q10"].iloc[0]) - r["auc_q10"]) < 1e-12, f"R441 q10 {m} n={n}")
            rows.append(r)
        hit = first_hit(b, m)
        arch = fh.loc[m, "min_labels_mean_ge_0.99"]
        check(hit == (int(arch) if pd.notna(arch) else None), f"first hit {m}: {hit} vs archived {arch}")
    t = pd.DataFrame(rows)
    t["first_hit_0.99"] = t["model"].map({m: first_hit(b, m) for m, *_ in BUDGET_CURVES})
    return t


def boundary_table(r404_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(r404_csv).set_index("proc")
    procs = [(p, lab, role, True) for lab, role, _, p in ROWS] + [(p, lab, "other", False) for p, lab in BOUNDARY_EXTRA]
    missing = sorted({p for p, *_ in procs} - set(d.index))
    check(not missing, f"{r404_csv.name} has no rows for {missing}")
    rows = []
    for proc, label, role, in_fig in procs:
        r = d.loc[proc]
        n = int(r["n_replicas"])
        rows.append({"proc": proc, "label": label, "in_figure": in_fig, "inductive": bool(r["inductive"]),
                     "n_replicas": n, "boundary_rows_mean": float(r["n_bnd_rows_mean"]),
                     "auc": float(r["bnd_auc_clean_mean"]), "auc_mc_se": float(r["bnd_auc_clean_mc_se"]),
                     # R404 computes the MC SE as sd(ddof=1) / sqrt(n) over the replicas, so this is the replica sd
                     "auc_sd_replicas": float(r["bnd_auc_clean_mc_se"]) * np.sqrt(n),
                     "s003_minus_proc": float(r["bnd_auc_clean_s003_minus_proc"]),
                     "share_s003_better": float(r["bnd_auc_clean_share_s003_better"]), "role": role})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def boundary_md(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Procedure": [lab + ("" if f else " *") for lab, f in zip(t["label"], t["in_figure"])],
        "Boundary-row AUC (MC SE)": [f"{a:.3f} ({s:.3f})" for a, s in zip(t["auc"], t["auc_mc_se"])],
        "S003 minus procedure": [("–" if p == "s003_gmm2_hv" else signed(v, 3)) for p, v in zip(t["proc"], t["s003_minus_proc"])],
        "Replicas where S003 is better": [("–" if p == "s003_gmm2_hv" else f"{v * n:.0f}/{n}")
                                          for p, v, n in zip(t["proc"], t["share_s003_better"], t["n_replicas"])],
    })


# ---------------------------------------------------------------------------------------- F1 data anatomy

def data_anatomy(train_csv: Path, test_csv: Path, sub_csv: Path, s003_pred: Path, r901_json: Path) -> dict:
    """Correlations, sd ratios and the closed-form two-class projection of the standardised block."""
    import scipy.linalg as sl
    tr, te, ss = pd.read_csv(train_csv), pd.read_csv(test_csv), pd.read_csv(sub_csv)
    check(bool((te["id"].to_numpy() == ss["id"].to_numpy()).all()), "test.csv id order = sample_submission.csv")
    feats = [str(i) for i in range(150)]
    X = np.vstack([tr[feats].to_numpy(float), te[feats].to_numpy(float)])            # 20,000 x 150
    y = tr["target"].to_numpy(int)
    sd = X.std(axis=0, ddof=1)
    ratio = sd / np.median(sd)
    order_r = np.argsort(-ratio, kind="stable")
    block = sorted(int(j) for j in np.flatnonzero(ratio > 1.5))
    r901 = json.loads(r901_json.read_text(encoding="utf-8"))["block"]
    check(block == r901["block13"], f"block {block} vs R901 block13")
    top16 = [round(float(v), 3) for v in ratio[order_r[:16]]]
    check(top16 == r901["ratio_profile"]["all20k"]["top16_ratios"], f"top-16 sd ratios {top16} vs R901")
    check(sorted(r901["informative12"]) == INFORMATIVE, "informative columns vs R901")
    noise = [j for j in range(150) if j not in INFORMATIVE + REDUNDANT]
    order = INFORMATIVE + REDUNDANT + noise
    Cr = np.corrcoef(X, rowvar=False)
    off = Cr - np.eye(150)
    maxabs = np.abs(off).max(axis=1)
    S = INFORMATIVE + REDUNDANT

    # two classes as two clouds: weight each row by S003's score p (training rows: their label)
    p_test = np.load(s003_pred).astype(float)
    check(p_test.shape == (len(te),), "S003 test_pred length")
    p = np.r_[y.astype(float), p_test]
    Z = (X[:, block] - X[:, block].mean(0)) / X[:, block].std(0, ddof=1)               # standardised block
    ev, U = np.linalg.eigh(np.cov(Z, rowvar=False))
    keep = ev > 1e-9 * ev.max()                                                         # block rank (12)
    Y = (Z - Z.mean(0)) @ U[:, keep]

    def wcov(w):
        m = (w[:, None] * Y).sum(0) / w.sum()
        D = Y - m
        return m, (w[:, None] * D).T @ D / w.sum()

    m1, S1 = wcov(p)
    m0, S0 = wcov(1.0 - p)
    lam, V = sl.eigh(S1, S0)                       # S1 v = lam S0 v, V' S0 V = I, lam ascending
    j_hi, j_lo = int(np.argmax(lam)), int(np.argmin(lam))
    Wd = V[:, [j_hi, j_lo]]
    P2 = (Y - m0) @ Wd                             # class-0 weighted mean at the origin; class-0 sd = 1 on both axes
    mu = {1: (m1 - m0) @ Wd, 0: np.zeros(2)}
    cov = {1: Wd.T @ S1 @ Wd, 0: Wd.T @ S0 @ Wd}
    # axis limits of F1c: the 99.9% quantile of |coordinate| on each axis (x 1.02); the panel is square-ish, so the
    # x axis is squeezed relative to y by (x range / panel width) / (y range / panel height)
    lim = (float(np.quantile(np.abs(P2[:, 0]), 0.999)) * 1.02, float(np.quantile(np.abs(P2[:, 1]), 0.999)) * 1.02)
    outside = int(((np.abs(P2[:, 0]) > lim[0]) | (np.abs(P2[:, 1]) > lim[1])).sum())
    check(outside <= 0.005 * len(P2), "F1c: at least 99.5% of the rows inside the axes")
    squeeze = (2 * lim[0] / F1C_W) / (2 * lim[1] / F1_H)
    feat_tab = pd.DataFrame({
        "feature": np.arange(150), "group": ["informative" if j in INFORMATIVE else "redundant" if j in REDUNDANT
                                             else "noise" for j in range(150)],
        "sd_20000": sd, "sd_ratio": ratio, "rank_sd_ratio": np.argsort(order_r) + 1,
        "in_block": [j in block for j in range(150)], "max_abs_corr_other": maxabs,
        "order_in_heatmap": [order.index(j) + 1 for j in range(150)]})
    return {"X_rows": len(X), "corr": Cr[np.ix_(order, order)], "order": order, "ratio": ratio,
            "order_r": order_r, "block": block, "feat": feat_tab, "P2": P2, "p": p, "n_train": len(tr), "y": y,
            "mu": mu, "cov": cov, "lam": lam, "lim": lim, "n_outside": outside, "squeeze": squeeze, "lam_hi": float(lam[j_hi]), "lam_lo": float(lam[j_lo]),
            "rank": int(keep.sum()), "max_noise": float(np.abs(off[np.ix_(noise, noise)]).max()),
            "max_struct_noise": float(np.abs(off[np.ix_(S, noise)]).max()),
            "max_struct": float(np.abs(off[np.ix_(S, S)]).max()), "gap": (float(ratio[order_r[12]]),
                                                                          float(ratio[order_r[13]]))}


def data_md(D: dict) -> pd.DataFrame:
    rows = [("Rows used (train + test, features only)", f"{D['X_rows']:,}"),
            ("High-variance block: sd / median sd > 1.5", ", ".join(map(str, D["block"])) + " (13; = R901 block13)"),
            ("13th / 14th largest sd ratio (the gap)", f"{D['gap'][0]:.3f} / {D['gap'][1]:.3f}"),
            ("Max |correlation|: among the 20 structured features", f"{D['max_struct']:.3f}"),
            ("Max |correlation|: structured vs noise", f"{D['max_struct_noise']:.3f}"),
            ("Max |correlation|: among the 130 noise features", f"{D['max_noise']:.3f}"),
            ("Rank of the standardised block", str(D["rank"])),
            ("Generalised eigenvalues (p-weighted / (1−p)-weighted variance), ascending",
             ", ".join(f"{v:.4g}" for v in D["lam"])),
            ("Projection axes (F1c)", f"largest ratio {D['lam_hi']:.3g}; smallest ratio {D['lam_lo']:.3g} "
                                      f"(= 1/{1 / D['lam_lo']:.0f})"),
            ("F1c: sd of the two weighted classes on (v1, v2)",
             "(1−p)-weighted: 1 × 1; p-weighted: " + " × ".join(f"{v:.3g}" for v in np.sqrt(np.diag(D["cov"][1])))),
            ("F1c: axis limits (±), rows outside", f"v1 ±{D['lim'][0]:.2f}, v2 ±{D['lim'][1]:.2f}; "
                                                   f"{D['n_outside']} of {D['X_rows']:,} rows outside"),
            ("F1c: x axis squeezed relative to y", f"{D['squeeze']:.2f}×")]
    return pd.DataFrame(rows, columns=["Quantity", "Value"])


# ---------------------------------------------------------------------------------------- F2 protocol

def protocol_tables(e1_est: Path, e1_pd: Path, e1_rep: Path, nullA: Path, nullB: Path):
    est = pd.read_csv(e1_est)
    pdiff = pd.read_csv(e1_pd).set_index("scheme")
    rep = pd.read_csv(e1_rep).set_index("scheme")
    rows = []
    for sch, label in E1_SCHEMES:
        d = est.loc[est["scheme"] == sch, "diff"].to_numpy(float)
        r = pdiff.loc[sch]
        check(len(d) == int(r["n_estimates"]) == 50, f"E1 {sch}: 50 executions")
        check(abs(d.std(ddof=1) - float(r["std_diff"])) < 1e-12, f"E1 {sch}: sd of the paired differences")
        check(abs(d.mean() - float(r["mean_diff"])) < 1e-12, f"E1 {sch}: mean of the paired differences")
        check(abs(float(r["mdd_2sd"]) - float(rep.loc[sch, "mdd_2sd"])) < 1e-12, f"E1 {sch}: report table")
        rows.append({"scheme": sch, "label": label, "n_executions": len(d), "fits_per_decision": int(r["fits_per_decision"]),
                     "mean_diff": d.mean(), "sd_diff": d.std(ddof=1), "min_diff": d.min(), "max_diff": d.max(),
                     "mdd_2sd": float(r["mdd_2sd"]), "mdd_lo": float(r["mdd_lo"]), "mdd_hi": float(r["mdd_hi"]),
                     "p_correct_at_0.01": float(r["p_correct_at_target"])})
    schemes = pd.DataFrame(rows)
    A, B = pd.read_csv(nullA), pd.read_csv(nullB)
    A = A[A["null"] == "A"]
    nr = []
    for pid, label, role in R900_ROWS:
        for nm, T in (("A", A), ("B", B)):
            v = T.loc[T["pipeline"] == pid, "auc_mean"].to_numpy(float)
            check(len(v) == 20, f"R900 null {nm} {pid}: 20 permutations")
            nr.append({"pipeline": pid, "label": label, "null": nm, "n_perm": len(v), "mean": v.mean(),
                       "min": v.min(), "max": v.max(), "n_above_0.5": int((v > 0.5).sum())})
    nulls = pd.DataFrame(nr)
    return est, schemes, A, B, nulls


def schemes_md(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Scheme": s["label"], "Fits per decision": s["fits_per_decision"].astype(str),
        "sd of paired difference": [f"{v:.4f}" for v in s["sd_diff"]],
        "MDD = 2 sd [95% CI]": [f"{m:.4f} [{lo:.4f}, {hi:.4f}]" for m, lo, hi in zip(s["mdd_2sd"], s["mdd_lo"], s["mdd_hi"])],
        "P(correct order at 0.01)": [f"{v:.2f}" for v in s["p_correct_at_0.01"]]})


def nulls_md(n: pd.DataFrame) -> pd.DataFrame:
    w = n.pivot(index="label", columns="null", values=["mean", "min", "max"])
    lab = [lab for _, lab, _ in R900_ROWS]
    return pd.DataFrame({"Pipeline": lab,
                         "Null A mean [min, max]": [f"{w.loc[l, ('mean', 'A')]:.3f} [{w.loc[l, ('min', 'A')]:.3f}, "
                                                    f"{w.loc[l, ('max', 'A')]:.3f}]" for l in lab],
                         "Null B mean [min, max]": [f"{w.loc[l, ('mean', 'B')]:.3f} [{w.loc[l, ('min', 'B')]:.3f}, "
                                                    f"{w.loc[l, ('max', 'B')]:.3f}]" for l in lab]})


# ---------------------------------------------------------------------------------------- F4 landscape

def landscape_table() -> tuple[pd.DataFrame, dict]:
    exps = pd.read_csv(need(paths.experiments_csv(SOURCE)), dtype=str).set_index("run_id")
    rows, folds = [], {}
    for run, label, group, role in LANDSCAPE:
        fs = pd.read_csv(need(exp_dir() / run / "fold_scores.csv"))
        a = fs["auc"].to_numpy(float)
        check(len(a) == 100, f"{run}: 100 official folds")
        check(exps.loc[run, "split_sig"] == "5708ce5c3828", f"{run}: official split signature")
        rep_means = fs.groupby("repeat")["auc"].mean()
        check(abs(rep_means.mean() - float(exps.loc[run, "auc_mean"])) < 6e-6, f"{run}: CV mean vs experiments.csv")
        folds[run] = a
        rows.append({"run_id": run, "label": label, "group": group, "cv_auc_mean": a.mean(),
                     "cv_auc_sd_repeat": rep_means.std(ddof=1), "fold_auc_min": a.min(),
                     "fold_auc_median": np.median(a), "folds_eq_1": int((a == 1.0).sum()), "n_folds": len(a)})
    return pd.DataFrame(rows), folds


def landscape_md(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Model": t["label"], "Group": t["group"].map({"labels": "labels only", "test": "test features",
                                                                      "pretrained": "pretrained"}),
                         "CV AUC ± sd": [f"{m:.4f} ± {s:.4f}" for m, s in zip(t["cv_auc_mean"], t["cv_auc_sd_repeat"])],
                         "Lowest fold": [f"{v:.3f}" for v in t["fold_auc_min"]],
                         "Folds = 1.0": [f"{v}/100" for v in t["folds_eq_1"]], "Run": t["run_id"]})


# ---------------------------------------------------------------------------------------- F6 selection

def selection_data(npz: Path, r936_cfg: Path, r936_private: Path):
    """Per-replica private-AUC difference to S003 (tn0, csv8 scores, R156-weighted noise levels), recomputed from
    R936's archived input exactly as R936 does, and checked against R936's private.csv."""
    blob = npz.read_bytes()
    cfg_sha = None
    for line in r936_cfg.read_text(encoding="utf-8").splitlines():
        if line.startswith("input_sha256:"):
            cfg_sha = line.split(":", 1)[1].strip()
    check(hashlib.sha256(blob).hexdigest() == cfg_sha, "r935_r933_pooled.npz sha256 = R936 config input_sha256")
    Z = np.load(io.BytesIO(blob))
    man = json.loads(bytes(Z["manifest_json"]).decode("utf-8"))
    parts, procs, flips = [], None, None
    for e in man["runs"]:
        if e["train_noise"]:
            continue                                     # tn0: test-label noise only (clean training labels)
        procs, flips = e["procedures"], e["flips"]
        parts.append(Z[f"{e['index']}__priv_auc_r8"])    # replicas x levels x draws x procedures, 8-decimal scores
    PRIV = np.concatenate(parts)
    w = np.array([W_R156[f] for f in flips])
    w = w / w.sum()
    j0 = procs.index("s003_gmm2_hv")
    arch = pd.read_csv(r936_private)
    arch = arch[(arch["group"] == "tn0") & (arch["scores"] == "csv8") & (arch["flip_y"] == "weighted_r156")].set_index("proc")
    per_rep, draws, rows = {}, {}, []
    for k, label, role in SELECTION:
        j = procs.index(k)
        D = PRIV[..., j] - PRIV[..., j0]                              # n x L x Dr
        dd = np.tensordot(D, w, axes=([1], [0])).mean(1)               # per replica: weighted levels, mean over draws
        beat = np.tensordot((D > 0).astype(float), w, axes=([1], [0])).mean()
        check(abs(dd.mean() - float(arch.loc[k, "diff_vs_s003_mean"])) < 1e-12, f"R936 mean difference {k}")
        check(abs(beat - float(arch.loc[k, "p_beats_s003"])) < 1e-9, f"R936 P(beats S003) {k}")
        per_rep[k] = dd
        draws[k] = (D.transpose(0, 2, 1).reshape(-1, len(w)), w)          # (replica x draw) x level, level weights
        rows.append({"proc": k, "label": label, "n_replicas": len(dd), "diff_mean": dd.mean(),
                     "diff_mc_se": float(arch.loc[k, "diff_mc_se"]), "replica_q05": np.quantile(dd, 0.05),
                     "replica_q95": np.quantile(dd, 0.95), "share_replicas_mean_gt_0": float((dd > 0).mean()),
                     "p_beats_s003_replica_splits": beat})
    return pd.DataFrame(rows), per_rep, draws


def selection_md(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Candidate": t["label"],
                         "Mean private AUC − S003": [f"{v * 1e4:+.2f}·10⁻⁴".replace("+-", "−").replace("-", "−")
                                                     for v in t["diff_mean"]],
                         "Beats S003 (replica splits)": [f"{v:.0%}" for v in t["p_beats_s003_replica_splits"]],
                         "Replica means > 0": [f"{v:.0%}" for v in t["share_replicas_mean_gt_0"]]})


def predictive(lik_csv: Path, pbf_csv: Path, r922_json: Path):
    """S003's predictive distribution of the private AUC, flat prior on flip_y in [0, 0.1] (R922 'uniform_0_0.1'):
    posterior weights = prior x L_train x L_public (Lcomb_main) on the fine grid, mapped to the nearest simulated
    noise level as R922 does; within a level the private AUC is taken as normal(mean, sd) of R922's archived draws
    (the draws themselves are not archived). Checked against R922's archived summary."""
    from scipy import stats
    L = pd.read_csv(lik_csv)
    P = pd.read_csv(pbf_csv)
    f = L["flip_y"].to_numpy(float)
    w = (L["Ltrain_rel"] * L["Lpub_conv_exact_round5"]).to_numpy(float)     # flat prior: weights = likelihood
    w = w / w.sum()
    fp = P["flip_y"].to_numpy(float)
    near = np.abs(f[:, None] - fp[None, :]).argmin(1)
    wf = np.bincount(near, weights=w, minlength=len(fp))
    mu, sd = P["mean"].to_numpy(float), np.maximum(P["sd"].to_numpy(float), 1e-9)
    x = np.linspace(0.95, 1.0, 50001)
    comp = np.array([wf[j] * stats.norm.pdf(x, mu[j], sd[j]) for j in range(len(fp))])   # level x grid
    dens = comp.sum(0)
    cdf = np.cumsum(dens) * (x[1] - x[0])
    cdf /= cdf[-1]
    q = lambda t: float(x[np.searchsorted(cdf, t)])                                         # noqa: E731
    p999 = float(sum(wf[j] * stats.norm.sf(0.999 - 5e-6, mu[j], sd[j]) for j in range(len(fp))))
    ours = {"mean": float((wf * mu).sum()), "q05": q(0.05), "q50": q(0.50), "q95": q(0.95), "P_ge_0.999": p999,
            "post_mean_flip": float((w * f).sum())}
    arch = json.loads(r922_json.read_text(encoding="utf-8"))["posterior_by_prior"]["uniform_0_0.1"]
    ref = {"mean": arch["pred_private_mean"], "q05": arch["pred_private_q05"], "q50": arch["pred_private_q50"],
           "q95": arch["pred_private_q95"], "P_ge_0.999": arch["P_B004_private_ge_0.999"],
           "post_mean_flip": arch["post_mean_flip"]}
    tol = {"mean": 2e-6, "q05": 1e-4, "q50": 1e-4, "q95": 1e-4, "P_ge_0.999": 0.005, "post_mean_flip": 2e-6}
    for k in ref:
        check(abs(ours[k] - ref[k]) <= tol[k], f"R922 predictive {k}: {ours[k]:.6f} vs archived {ref[k]:.6f}")
    table = pd.DataFrame([{"quantity": k, "this_script": ours[k], "r922_archived": ref[k], "abs_diff": abs(ours[k] - ref[k])}
                          for k in ref])
    # probability mass per 0.0005-wide bin of the private AUC, per simulated noise level (figure F6b)
    edges = np.round(np.arange(0.986, 1.0000001, 0.0005), 6)
    edges[-1] = 1.0 + 1e-9                                  # last bin includes AUC = 1
    mass = np.array([wf[j] * np.diff(stats.norm.cdf(edges, mu[j], sd[j])) for j in range(len(fp))])
    return {"x": x, "dens": dens, "comp": comp, "fp": fp, "wf": wf, "ours": ours, "ref": ref, "edges": edges,
            "mass": mass}, table


# ============================================================================================ figures

def place(fig, x, y, w, h):
    """Axes at (x, y) with size (w, h), all in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, y / H, w / W, h / H))


def ellipse(ax, mean, cov, k, ls, z=5):
    """k-sd ellipse of a 2-d covariance: 0.8 pt ink with a white halo, so it shows on any cloud."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(xy=mean, width=2 * k * np.sqrt(vals[1]), height=2 * k * np.sqrt(vals[0]), angle=ang,
                         fill=False, ec=ps.INK, lw=0.8, ls=ls, zorder=z,
                         path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]))


def check_break(values, what: str) -> None:
    """Nothing may be plotted between the axis break of a nines axis and 1 (exact 1.0 is drawn at the cap)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo = float(ps.nines_inv(ps.break_at()))
    check(not bool(((v > lo) & (v < 1.0)).any()), f"{what}: a value lies between the axis break and 1")


def fig_data(D: dict, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(ps.FULL_W, 1.96))
    Y0, HA = 0.36, F1_H
    cmap = ps.diverging_cmap()
    cmap.set_bad(ps.MID)
    M = D["corr"].copy()
    np.fill_diagonal(M, np.nan)

    # (a) correlation heatmap of the 20 structured features (roles as inferred under make_classification);
    # the 130 noise features (all |r| <= max_noise, a flat grey square) are a note, not a picture ------------
    zx = place(fig, 0.03, Y0, HA, HA)
    im = zx.imshow(M[:20, :20], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    for s in zx.spines.values():
        s.set_visible(True)
        s.set_color(ps.INK)
        s.set_linewidth(0.6)
    zx.plot([11.5, 11.5], [-0.5, 19.5], color=ps.INK, lw=0.6)
    zx.plot([-0.5, 19.5], [11.5, 11.5], color=ps.INK, lw=0.6)
    zx.set_xticks([5.5, 15.5])
    zx.set_xticklabels(["informative\n(1–12)", "redundant\n(13–20)"], fontsize=7.5, color=ps.INK2, linespacing=1.0)
    zx.set_yticks([])
    zx.tick_params(axis="x", length=0, pad=2)
    zx.text(1.21, 0.0, "+ 130 noise\nfeatures,\nall |r| ≤ " + f"{D['max_noise']:.2f}" + "\n(not shown)",
            transform=zx.transAxes, ha="left", va="bottom", fontsize=7.5, color=ps.INK2, linespacing=1.1)
    cax = place(fig, 0.03 + HA + 0.06, Y0 + 0.62, 0.05, HA - 0.70)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["−1", "0", "1"])
    cax.tick_params(length=1.5, width=0.4, color=ps.AXIS, labelsize=7.5, labelcolor=ps.INK2, pad=1)
    cax.set_title("r", fontsize=7.5, color=ps.INK2, pad=2, loc="left", style="italic")
    ps.panel_title(zx, "(a) The 20 structured features")

    # (b) sd-ratio profile -------------------------------------------------------------------------------
    bx = place(fig, 2.74, Y0, 1.10, HA)
    ratio, order_r, block = D["ratio"], D["order_r"], set(D["block"])
    struct = set(INFORMATIVE + REDUNDANT)
    rk = np.arange(1, 151)
    r_sorted = ratio[order_r]
    isb = np.array([j in block for j in order_r])
    iss = np.array([(j in struct) and (j not in block) for j in order_r])
    isn = ~(isb | iss)
    ps.refline(bx, 1.5, axis="y")
    bx.scatter(rk[isn], r_sorted[isn], s=5, color=ps.MUTED, lw=0, alpha=0.8, zorder=2, label="noise (130)")
    bx.scatter(rk[iss], r_sorted[iss], s=12, facecolor="white", edgecolor=ps.INK, lw=0.6, zorder=3,
               label="other structured (7)")
    bx.scatter(rk[isb], r_sorted[isb], s=13, color=ps.INK, edgecolor="white", lw=0.4, zorder=4)
    bx.set_yscale("log")
    bx.set_ylim(0.25, 13)
    bx.set_yticks([0.5, 1, 1.5, 2, 4, 8])
    bx.set_yticklabels(["0.5", "1", "1.5", "2", "4", "8"])
    bx.minorticks_off()
    bx.set_xlim(-2, 152)
    bx.set_xticks([1, 50, 100, 150])
    ps.clean(bx, grid="y")
    bx.set_xlabel("features ranked by sd ratio")
    bx.set_ylabel("sd / median sd", labelpad=1)
    # bracket over the 13 block features
    yb = 9.6
    bx.plot([0.6, 0.6, 13.4, 13.4], [yb * 0.9, yb, yb, yb * 0.9], color=ps.INK, lw=0.6, zorder=5)
    ps.label(bx, 16, yb, "high-variance block (13)", ha="left", size=8, color=ps.INK)
    g0, g1 = D["gap"]
    ps.label(bx, 16.5, g0, f"{g0:.2f}", ha="left", size=8, color=ps.INK)
    ps.callout(bx, f"{g1:.2f}", xy=(14.3, g1 * 0.96), xytext=(24, 0.88), ha="left", va="center", size=8,
               color=ps.INK)
    bx.legend(loc="upper right", fontsize=8, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25,
              markerscale=1.0, bbox_to_anchor=(1.03, 0.86))
    ps.panel_title(bx, "(b) Spread of each feature")

    # (c) two classes, two shapes ----------------------------------------------------------------------
    cx = place(fig, 4.32, Y0, F1C_W, HA)
    P2, p = D["P2"], D["p"]
    unc = np.abs(p - 0.5)
    idx = np.argsort(-unc, kind="stable")             # most certain first, uncertain rows drawn last (on top)
    sure = unc[idx] >= 0.4
    cx.scatter(P2[idx[sure], 0], P2[idx[sure], 1], c=p[idx[sure]], cmap=cmap, vmin=0, vmax=1, s=0.5, lw=0,
               alpha=0.55, rasterized=True, zorder=1)
    cx.scatter(P2[idx[~sure], 0], P2[idx[~sure], 1], c=p[idx[~sure]], cmap=cmap, vmin=0, vmax=1, s=2.2, lw=0,
               alpha=0.95, rasterized=True, zorder=2)
    for cls in (0, 1):
        ellipse(cx, D["mu"][cls], D["cov"][cls], 1, (0, (2.5, 1.5)), z=4)
        ellipse(cx, D["mu"][cls], D["cov"][cls], 2, "-", z=4)
    nt, yy = D["n_train"], D["y"]
    T = P2[:nt]
    cx.scatter(T[yy == 0, 0], T[yy == 0, 1], marker="o", s=4, facecolor="white", edgecolor=ps.INK, lw=0.4, zorder=6)
    cx.scatter(T[yy == 1, 0], T[yy == 1, 1], marker="^", s=5, color=ps.INK, edgecolor="white", lw=0.25, zorder=6)
    lx, ly = D["lim"]
    cx.set_xlim(-lx, lx)
    cx.set_ylim(-ly, ly)
    tx = [v for v in (-10, -5, 0, 5, 10) if abs(v) < lx]
    ty = [v for v in (-2, 0, 2) if abs(v) < ly]
    cx.set_xticks(tx)
    cx.set_xticklabels([f"{v:g}".replace("-", "−") for v in tx])
    cx.set_yticks(ty)
    cx.set_yticklabels([f"{v:g}".replace("-", "−") for v in ty])
    cx.tick_params(labelsize=7.5)
    ps.clean(cx, grid="")
    cx.set_xlabel(rf"$v_1$ (variance ratio {D['lam_hi']:.1f})")
    cx.set_ylabel(rf"$v_2$ (variance ratio 1/{1 / D['lam_lo']:.0f})", labelpad=1)
    s1 = np.sqrt(np.diag(D["cov"][1]))
    m1 = D["mu"][1]
    ps.callout(cx, "sd 1 × 1", xy=(1.0, 1.73), xytext=(3.6, 2.05), ha="left", va="center", size=8, color=ps.INK)
    xe = 0.75 * 2 * s1[0]                                   # a point on the lower edge of the 2-sd p-ellipse
    ye = m1[1] - 2 * s1[1] * np.sqrt(1 - 0.75 ** 2)
    ps.callout(cx, f"sd {s1[0]:.1f} × {s1[1]:.2f}", xy=(m1[0] + xe, ye), xytext=(3.4, -0.75), ha="left",
               va="center", size=8, color=ps.INK)
    ps.label(cx, lx * 0.97, -ly * 0.97, f"x squeezed\n{D['squeeze']:.1f}×", ha="right", va="bottom", size=7.5,
             color=ps.INK2, linespacing=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    ps.colorbar(cx, sm, "", ticks=[0, 0.5, 1], ticklabels=["0", "0.5", "1"], loc=(1.05, 0.0, 0.05, 0.45))
    cx.text(1.05, 0.49, "S003\nscore", transform=cx.transAxes, fontsize=7.5, color=ps.INK2, ha="left", va="bottom",
            linespacing=1.0)
    h = [Line2D([], [], marker="^", ls="none", ms=3.6, color=ps.INK, mec="white", mew=0.3, label="train, y = 1"),
         Line2D([], [], marker="o", ls="none", ms=3.4, mfc="white", mec=ps.INK, mew=0.5, label="train, y = 0")]
    cx.legend(handles=h, loc="upper left", fontsize=7.5, handletextpad=0.1, borderaxespad=0.05, labelspacing=0.15,
              bbox_to_anchor=(-0.02, 1.02))
    ps.panel_title(cx, "(c) S003's two groups, two shapes")
    ps.save(fig, out)
    plt.close(fig)


FITS_GREY = {2: "#e1e0d9", 10: "#c3c2b7", 20: "#a3a29b", 100: "#6e6d68", 200: "#52514e"}


def fig_protocol(est: pd.DataFrame, s: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame, out: Path) -> None:
    plt = ps.setup()
    from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(ps.FULL_W, 2.17))
    H_AX, Y0 = 1.43, 0.38
    # (a) ridgeline ----------------------------------------------------------------------------------------
    ax = place(fig, 0.72, Y0, 1.33, H_AX)
    si = s.set_index("scheme")
    c0 = float(si.loc["5foldx20", "mean_diff"])            # band: pooled 5 x 20 mean +- 0.01, as in E1's figure
    ax.axvspan(c0 - 0.01, c0 + 0.01, color=ps.MID, lw=0, zorder=0)
    ps.refline(ax, 0.0, axis="x", zorder=1)
    XL = 0.095
    xs = np.linspace(-XL, XL, 1501)
    n = len(E1_SCHEMES)
    for i, (sch, label) in enumerate(E1_SCHEMES):
        y0 = n - 1 - i
        d = est.loc[est["scheme"] == sch, "diff"].to_numpy(float)
        dens = gaussian_kde(d)(xs)
        dens = dens / dens.max() * 0.80
        m = dens > 0.01 * 0.80
        r = si.loc[sch]
        fits = int(r["fits_per_decision"])
        ax.fill_between(xs[m], y0, y0 + dens[m], color=FITS_GREY[fits], lw=0, zorder=2 + i)
        ax.plot(xs[m], y0 + dens[m], color=ps.INK, lw=0.6, zorder=2 + i)
        ax.plot([xs[m][0], xs[m][-1]], [y0, y0], color=ps.AXIS, lw=0.4, zorder=2 + i)
        mdd = float(r["mdd_2sd"])
        ax.plot([-mdd, mdd], [y0 - 0.07, y0 - 0.07], color=ps.INK, lw=1.2, solid_capstyle="butt", zorder=9)
        for v in (-mdd, mdd):
            ax.plot([v, v], [y0 - 0.15, y0 + 0.01], color=ps.INK, lw=0.7, zorder=9)
        ax.scatter(d, np.full(len(d), y0 - 0.25), marker="|", s=7, color=ps.INK2, lw=0.45, alpha=0.8, zorder=8)
        ax.text(1.03, y0 + 0.05, f"{mdd:.3f}" if mdd > 0.01 else f"{mdd:.4f}", transform=ax.get_yaxis_transform(),
                fontsize=8, color=ps.INK, ha="left", va="center")
        ax.text(-0.06, y0 + 0.40, label, transform=ax.get_yaxis_transform(), fontsize=8, color=ps.INK, ha="right",
                va="center")
        ax.text(-0.06, y0 + 0.00, f"{fits} fits", transform=ax.get_yaxis_transform(), fontsize=7.5, color=ps.INK2,
                ha="right", va="center")
    ax.text(1.03, n - 0.25, "MDD", transform=ax.get_yaxis_transform(), fontsize=8, color=ps.INK2, ha="left",
            va="bottom")
    ax.set_ylim(-0.42, n - 0.15)
    ax.set_yticks([])
    ax.set_xlim(-XL, XL)
    ax.set_xticks([-0.05, 0, 0.05])
    ax.set_xticklabels(["−0.05", "0", "0.05"])
    ps.clean(ax, grid="", left=False)
    ax.set_xlabel("paired AUC difference (band: 5 × 20 mean ± 0.01)")
    ps.panel_title(ax, "(a) One model pair, 50 split seeds per scheme", x=-0.53,
                   sub="E1: corr top-30 vs top-50 + L2; bar = ±MDD")

    # (b, c) permutation nulls, one shared AUC scale -------------------------------------------------------
    rows = R900_ROWS
    nr = len(rows)
    ypos = {pid: (nr - 1 - i) + (0 if i < 7 else -0.45) for i, (pid, _, _) in enumerate(rows)}
    rng = np.random.default_rng(900)
    for k, (T, title, sub, x0, fires) in enumerate((
            (A, "(b) Null A", "permuted throughout", 4.02, "ctrl_e3_leaky"),
            (B, "(c) Null B", "scored on true labels", 5.20, "ctrl_disk_reader"))):
        bx = place(fig, x0, Y0, 0.98, H_AX)
        ps.refline(bx, 0.5, axis="x", zorder=1)
        for pid, label, role in rows:
            v = T.loc[T["pipeline"] == pid, "auc_mean"].to_numpy(float)
            yj = ypos[pid] + rng.uniform(-0.24, 0.24, len(v))
            if role in ("leaky", "muted"):                       # controls: x markers, red only where it fires
                col = ps.RED if pid == fires else ps.INK2
                bx.scatter(v, yj, s=9, marker="x", color=col, lw=0.6, alpha=0.9, zorder=3)
                if pid == fires:                                 # value label on the empty side of the cluster
                    right = v.mean() < 0.8
                    bx.text(v.max() + 0.05 if right else v.min() - 0.06, ypos[pid], f"{v.mean():.2f}",
                            ha="left" if right else "right", va="center", fontsize=8, color=ps.INK)
            else:
                bx.scatter(v, yj, s=7, color=ROLE_COLOR[role], edgecolor="white", lw=0.3, alpha=0.85, zorder=3)
            bx.plot([v.mean()] * 2, [ypos[pid] - 0.36, ypos[pid] + 0.36], color=ps.INK, lw=1.0, zorder=4)
        bx.axhline(ypos["e3_honest"] - 0.72, color=ps.GRID, lw=0.6)
        bx.set_xlim(-0.03, 1.05)
        bx.set_xticks([0, 0.5, 1])
        bx.set_xticklabels(["0", "0.5", "1"])
        bx.set_ylim(min(ypos.values()) - 0.6, nr - 0.4)
        ps.clean(bx, grid="", left=False)
        bx.set_yticks([ypos[pid] for pid, _, _ in rows])
        bx.set_yticklabels([lab for _, lab, _ in rows] if k == 0 else [], fontsize=8, color=ps.INK)
        bx.tick_params(axis="y", length=0, pad=3)
        bx.set_xlabel("AUC per permutation")
        ps.panel_title(bx, title, sub=sub)
    ps.save(fig, out)
    plt.close(fig)


def fig_journey(t: pd.DataFrame, leaky_cv: float, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(ps.FULL_W, 2.06))
    AX_X, AX_W = 0.50, 5.72
    ax = place(fig, AX_X, 0.56, AX_W, 1.20)
    ti = t.set_index("sub_id")
    subs = list(ti.index)
    dx = 0.2
    ps.nines(ax, "y", lo=0.5, cap=True, cap_label="= 1", brk=True, ticks=(0.9, 0.99, 0.999, 0.9999))
    ps.clean(ax, grid="y", bottom=True)
    ax.grid(axis="y", which="minor", visible=False)
    t_top = float(ps.nines_inv(ps.break_at() - 0.22))      # clipped bars stop here, with a caret
    shown = []

    def bar(x, lo, hi, col):
        clipped = hi >= 1.0
        top = t_top if clipped else hi
        ax.plot([x, x], [lo, top], color=col, lw=1.0, alpha=0.7, zorder=3, solid_capstyle="butt")
        if clipped:
            ax.plot(x, top, marker="^", ms=3.0, color=col, mew=0, alpha=0.9, zorder=3)
        shown.extend([lo, top])

    xs = [i + dx for i in range(len(subs))]
    ax.plot(xs, ps.clip_auc(ti["public_lb"].to_numpy()), color=ps.AXIS, lw=0.9, zorder=2)
    # value labels of the public scores: placed clear of the path (offset in points, alignment)
    vpos = {"S001": (0.10, 1, "left", "top"), "S002": (0.08, -4, "left", "top"),
            "S003": (0.08, 3, "left", "bottom"), "S004": (0.12, 0, "left", "center"),
            "S005": (0.08, 3, "left", "bottom"), "S006": (-0.08, 3, "right", "bottom")}
    for i, sid in enumerate(subs):
        r = ti.loc[sid]
        cv, sd = r["cv_auc_mean"], r["cv_auc_sd"]
        if sd > 0:
            bar(i - dx, cv - sd, cv + sd, ps.INK2)
        ax.plot(i - dx, min(cv, CAP), "o", ms=4.6, color=ps.INK2, mec="white", mew=0.5, zorder=4)
        ax.plot(i, r["expected_public"], marker="D", ms=4.2, mfc="white", mec=ps.INK, mew=0.9, ls="none", zorder=4)
        se, pub = r["hm_se_at_expected"], r["public_lb"]
        col = ps.color(sid)
        bar(i + dx, pub - se, pub + se, col)
        ax.annotate("", xy=(i + dx, pub), xytext=(i, r["expected_public"]), zorder=3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", color=ps.INK2, lw=0.5,
                                    shrinkA=3.2, shrinkB=3.6))
        ax.plot(i + dx, pub, "s", ms=4.8, color=col, mec="white", mew=0.5, zorder=5)
        shown.extend([cv, r["expected_public"], pub])
        ox, oy, ha, va = vpos[sid]
        ax.annotate(f"{pub:.3f}" if pub < 0.99 else f"{pub:.5f}", (i + dx + ox, pub), xytext=(0, oy),
                    textcoords="offset points", ha=ha, va=va, fontsize=8, color=ps.INK, zorder=6)
        if sid == "S004":
            ax.plot(i - dx, leaky_cv, "o", ms=4.6, mfc="white", mec=ps.RED, mew=1.0, zorder=4)
            ax.annotate("", xy=(i + dx, pub), xytext=(i - dx, leaky_cv), zorder=3,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=ps.RED, lw=0.7,
                                        shrinkA=3.0, shrinkB=3.4, connectionstyle="arc3,rad=-0.45"))
            ax.annotate(f"leaky CV {leaky_cv:.3f}", (i - dx - 0.09, leaky_cv), xytext=(0, -1),
                        textcoords="offset points", ha="right", va="bottom", fontsize=8, color=ps.INK)
            ax.annotate(f"honest CV {cv:.3f}", (i - dx - 0.09, cv), xytext=(0, -1.5), textcoords="offset points",
                        ha="right", va="center", fontsize=8, color=ps.INK2)
            shown.append(leaky_cv)
    check_break(shown, "F3")
    ax.set_xlim(-0.55, len(subs) - 0.45)
    ax.set_xticks(range(len(subs)))
    final = {"S003", "S005"}
    ax.set_xticklabels([f"{sid}" + (" (final)" if sid in final else "") + f"\n{ps.name(sid)[5:]}" for sid in subs],
                       fontsize=8, linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2.5)
    for lab, sid in zip(ax.get_xticklabels(), subs):
        lab.set_color(ps.INK if sid in final else ps.INK2)
        if sid in final:
            lab.set_fontweight("bold")
    ax.set_ylabel("ROC AUC (nines scale)", labelpad=2)
    # questions: one row above the frame, centred on each step
    for i, sid in enumerate(subs):
        ax.annotate(QUESTION[sid], (i, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=ps.INK, style="italic",
                    linespacing=1.0, annotation_clip=False)
    # upload days: brackets under the tick labels
    days = sorted(set(ti["day_utc"]))
    for d in days:
        idx = [i for i, sid in enumerate(subs) if ti.loc[sid, "day_utc"] == d]
        a, b = min(idx) - 0.42, max(idx) + 0.42
        tr = ax.get_xaxis_transform()
        yb = -0.27
        ax.plot([a, a, b, b], [yb + 0.03, yb, yb, yb + 0.03], color=ps.MUTED, lw=0.5, transform=tr, clip_on=False)
        lab = f"{int(d[8:])} {pd.Timestamp(d).strftime('%b')} (UTC)"
        ax.text((a + b) / 2, yb - 0.02, lab, transform=tr, ha="center", va="top", fontsize=8, color=ps.INK2,
                clip_on=False)
    handles = [Line2D([], [], marker="o", ls="-", color=ps.INK2, mec="white", ms=4.6, lw=1.0,
                      label="CV mean ± sd (5 × 20)"),
               Line2D([], [], marker="D", ls="none", mfc="white", mec=ps.INK, mew=0.9, ms=4.2,
                      label="pre-registered expectation"),
               Line2D([], [], marker="s", ls="-", color=ps.INK2, mec="white", ms=4.8, lw=1.0,
                      label="public score ± 1 SE"),
               Line2D([], [], marker="o", ls="none", mfc="white", mec=ps.RED, mew=1.0, ms=4.6,
                      label="leaky CV (E3)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 1.02), fontsize=8, labelspacing=0.2,
              handletextpad=0.4, borderaxespad=0.2)
    ps.save(fig, out)
    plt.close(fig)


def fig_landscape(lt: pd.DataFrame, folds: dict, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(ps.FULL_W, 2.24))
    MX, MW, Y0, HY = 1.56, 3.34, 0.34, 1.86
    LO, HI = 0.68, 0.9993
    ax = place(fig, MX, Y0, MW, HY)
    groups = ["labels", "test", "pretrained"]
    ypos, y, headers = {}, 0.0, {}
    for g in groups:                      # top to bottom
        headers[g] = y
        y -= 0.85
        for run, label, grp, role in LANDSCAPE:
            if grp == g:
                ypos[run] = y
                y -= 1.0
        y -= 0.12
    ylo, yhi = y + 0.5, 0.45
    rng = np.random.default_rng(4203)
    ps.nines(ax, "x", lo=LO, hi=HI, ticks=(0.8, 0.9, 0.99, 0.995, 0.999))
    ps.clean(ax, grid="x", left=False)
    ax.grid(axis="x", which="minor", visible=False)
    ps.refline(ax, 0.947, axis="x", zorder=1, ymax=(headers["labels"] - 0.42 - ylo) / (yhi - ylo))
    for run, label, grp, role in LANDSCAPE:
        a = folds[run]
        below = a[a < 1.0]
        yj = ypos[run] + rng.uniform(-0.32, 0.32, len(below))
        ax.scatter(below, yj, s=4.8, color=ROLE_COLOR[role], lw=0, alpha=0.45, zorder=3)
        m = float(a.mean())
        if m < HI:
            ax.plot([m, m], [ypos[run] - 0.42, ypos[run] + 0.42], color=ps.INK, lw=1.2, zorder=5)
        elif m < 1.0:                     # S005: mean 0.9999 lies beyond the break
            ax.plot(HI, ypos[run], marker=">", ms=4.0, color=ps.INK, mew=0, clip_on=False, zorder=6)
            ax.annotate(f"mean {m:.4f}", (HI, ypos[run]), xytext=(-3, -3.2), textcoords="offset points", ha="right",
                        va="top", fontsize=8, color=ps.INK, zorder=7,     # white box: masks the group divider
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    ps.nines_break(ax, "x", float(ps.nines_fwd(HI)) - 0.012)
    ax.set_ylim(ylo, yhi)
    ax.set_yticks([ypos[r] for r, *_ in LANDSCAPE])
    ax.set_yticklabels([lab for _, lab, _, _ in LANDSCAPE], fontsize=8, color=ps.INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel("fold AUC (nines scale)", labelpad=2)
    for g in groups:
        ax.text(-MX / MW + 0.005, headers[g] - 0.06, GROUP_NAME[g], transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=8, color=ps.INK, style="italic")
        if g != "labels":
            ax.axhline(headers[g] + 0.46, color=ps.GRID, lw=0.6, xmin=-MX / MW + 0.005, xmax=(6.2 - MX) / MW,
                       clip_on=False)
    # key (top right of the main axis)
    kh = [Line2D([], [], marker="o", ls="none", ms=2.4, color=ps.INK2, alpha=0.6, mew=0, label="one fold"),
          Line2D([], [], marker="|", ls="none", ms=6, color=ps.INK, mew=1.2, label="CV mean"),
          Line2D([], [], ls=(0, (4.0, 2.5)), lw=0.6, color=ps.INK2, label="0.947: best linear rule (population)")]
    ax.legend(handles=kh, loc="upper right", bbox_to_anchor=(1.0, 1.035), ncol=3, fontsize=8, handlelength=1.4,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0)
    # AUC = 1 column: bar length = share of the 100 folds at exactly 1 ------------------------------------
    CX, CB = 5.04, 0.62
    cx = place(fig, CX, Y0, 1.1, HY)
    cx.set_xlim(0, 1.1 / CB)
    cx.set_ylim(ylo, yhi)
    cx.axis("off")
    for run, label, grp, role in LANDSCAPE:
        n1 = int((folds[run] == 1.0).sum())
        if n1:
            cx.barh(ypos[run], n1 / 100, height=0.56, left=0, color=ROLE_COLOR[role], alpha=0.85 if role != "other"
                    else 0.55, lw=0, zorder=2)
        cx.text(n1 / 100 + 0.04, ypos[run], f"{n1}", ha="left", va="center", fontsize=8,
                color=ps.INK if n1 == 100 else ps.INK2, fontweight="bold" if n1 == 100 else "normal")
    cx.plot([0, 0], [ylo, yhi], color=ps.AXIS, lw=0.5, zorder=1)
    cx.text(0, headers["labels"] - 0.06, "folds at AUC = 1", ha="left", va="center", fontsize=8, color=ps.INK2)
    cx.text(0.0, -0.012, "= 1", transform=cx.get_xaxis_transform(), ha="center", va="top", fontsize=8,
            color=ps.INK2)
    # CV saturated: bracket over the three rows whose 100 folds are all 1
    sat = [ypos[r] for r, _, g, _ in LANDSCAPE if (folds[r] == 1.0).all()]
    xb = -0.06
    cx.plot([xb + 0.03, xb, xb, xb + 0.03], [max(sat) + 0.35, max(sat) + 0.35, min(sat) - 0.35, min(sat) - 0.35],
            color=ps.INK, lw=0.6, clip_on=False)
    cx.text(xb - 0.04, float(np.mean(sat)), "CV saturated: all 100 folds = 1", ha="right", va="center", fontsize=8,
            color=ps.INK, clip_on=False)
    ps.save(fig, out)
    plt.close(fig)


def fig_unlabeled(cur: pd.DataFrame, bt: pd.DataFrame, out: Path) -> None:
    plt = ps.setup()
    fig = plt.figure(figsize=(ps.FULL_W, 2.2))
    Y0, HY = 0.36, 1.52
    # (a) label budget curves -------------------------------------------------------------------------------
    ax = place(fig, 0.52, Y0, 2.42, HY)
    ticks = (0.6, 0.9, 0.99, 0.999, 0.9999)
    ps.nines(ax, "y", lo=0.6, cap=True, cap_label="= 1", brk=True, ticks=ticks)
    ps.clean(ax, grid="y")
    ax.grid(axis="y", which="minor", visible=False)
    ax.set_xscale("log")
    ps.refline(ax, 0.99, axis="y")
    order = ["bqda", "s005", "s006", "s003"]               # drawing order: S005 after Bayesian QDA
    shown, hits = [], []
    for m, label, role in sorted(BUDGET_CURVES, key=lambda r: order.index(r[2])):
        c = cur[(cur["model"] == m) & cur["defined_all"]].sort_values("n_labels")
        col = ROLE_COLOR[role]
        n = c["n_labels"].to_numpy(float)
        lo, hi, mu = ps.clip_auc(c["auc_q10"]), ps.clip_auc(c["auc_q90"]), ps.clip_auc(c["auc_mean"])
        shown.extend(list(c["auc_q10"]) + list(c["auc_q90"]) + list(c["auc_mean"]))
        if role == "s005":                                  # hatched, so the overlap with Bayesian QDA reads
            hc = ps.shade(col, -0.25)
            with plt.rc_context({"hatch.linewidth": 0.4, "hatch.color": hc}):
                ax.fill_between(n, lo, hi, facecolor="none", edgecolor=hc, hatch="//////", lw=0, zorder=3)
        else:
            ax.fill_between(n, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(n, mu, color=col, lw=1.3, zorder=4)
        ax.plot(n, mu, "o", ms=2.6, color=col, mec="white", mew=0.3, zorder=4)
        fh = int(c["first_hit_0.99"].iloc[0])
        yh = float(mu[n == fh][0])
        ax.plot(fh, yh, "o", ms=6.0, mfc="none", mec=ps.INK, mew=1.0, zorder=7)
        hits.append(fh)
    check_break(shown, "F5a")
    ax.annotate(ps.name("s003"), (2.0, CAP), xytext=(1, -4), textcoords="offset points", ha="left", va="top",
                fontsize=8, color=ps.INK, zorder=8)
    ps.callout(ax, "S006 self-\ntraining", xy=(12.5, 0.99), xytext=(7.5, 0.9985), ha="right", va="center", size=8,
               color=ps.INK, linespacing=1.0)
    ax.text(1.85, 0.935, "Bayesian QDA,\nlabel-free block", ha="left", va="center", fontsize=8, color=ps.INK,
            linespacing=1.0, zorder=8)
    ps.callout(ax, "S005\nstrict QDA", xy=(80, 0.982), xytext=(150, 0.93), ha="center", va="center", size=8,
               color=ps.INK, linespacing=1.0)
    ax.set_xlim(1.7, 260)
    xt = [2, 4, 10, 20, 40, 80, 120, 200]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in xt])
    for tl, v in zip(ax.get_xticklabels(), xt):
        if v in hits:
            tl.set_fontweight("bold")
            tl.set_color(ps.INK)
    ax.minorticks_off()
    ax.set_xlabel("labelled training rows (log scale)")
    ax.set_ylabel("validation AUC (nines scale)", labelpad=2)
    ps.panel_title(ax, "(a) Labels needed: mean and 10–90% band",
                   sub="ring and bold tick: first budget with mean ≥ 0.99")

    # (b) boundary rows ----------------------------------------------------------------------------------
    BX, BW = 4.42, 1.42
    bx = place(fig, BX, Y0, BW, HY)
    b = bt[bt["in_figure"]].set_index("proc")
    bayes = float(bt.set_index("proc").loc["bayes", "auc"])
    others = b.drop(index="s003_gmm2_hv")
    check(bool((others["share_s003_better"] == 1.0).all()) and bool((b["n_replicas"] == 222).all()),
          "F5b: S003 higher than every other procedure in 222/222 replicas")
    n = len(ROWS)
    ps.refline(bx, bayes, axis="x", zorder=1)
    bx.text(bayes - 0.004, n - 0.38, f"ceiling {bayes:.3f}", ha="right", va="center", fontsize=8, color=ps.INK2)
    for i, (label, role, _, p) in enumerate(ROWS):
        y = n - 1 - i
        r = b.loc[p]
        col = ROLE_COLOR[role]
        sdv, se2 = r["auc_sd_replicas"], 2 * r["auc_mc_se"]
        bx.plot([r["auc"] - sdv, r["auc"] + sdv], [y, y], color=col, lw=5.0, alpha=0.25, solid_capstyle="butt",
                zorder=2)
        bx.plot([r["auc"] - se2, r["auc"] + se2], [y, y], color=ps.shade(col, 0.35), lw=2.6, solid_capstyle="butt",
                zorder=4)
        hollow = bool(r["inductive"]) and role == "bqda"
        bx.plot(r["auc"], y, "o", ms=3.0, mfc="white" if hollow else ps.shade(col, 0.35), mec="white" if not hollow
                else ps.shade(col, 0.35), mew=0.4 if not hollow else 0.8, zorder=5)
        bx.text(1.0 + 0.32 / BW, y, f"{r['auc']:.3f}", transform=bx.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, color=ps.INK)
    bx.text(1.0 + 0.32 / BW, n - 0.38, "AUC", transform=bx.get_yaxis_transform(), ha="right", va="center",
            fontsize=8, color=ps.INK2)
    bx.set_ylim(-0.55, n - 0.05)
    bx.set_yticks([n - 1 - i for i in range(n)])
    names = {"s003_gmm2_hv": ps.name("s003"), "r209_selftrain_qda_lw": ps.name("s006"),
             "r255_bqda_eb_hv": ps.name("bqda").replace(", ", ",\n"), "r210_qda_block_nll": ps.name("r210"),
             "r201_qda_block": ps.name("s005"), "r255_bqda_eb_trainonly": ps.name("bqda_infold").replace(", ", ",\n")}
    bx.set_yticklabels([names[p] for _, _, _, p in ROWS], fontsize=8, color=ps.INK, linespacing=1.0)
    bx.tick_params(axis="y", length=0, pad=3)
    bx.set_xlim(0.575, 0.855)
    bx.set_xticks([0.6, 0.7, 0.8])
    ps.clean(bx, grid="x", left=False)
    bx.set_xlabel("AUC, rows with true probability 0.1–0.9", labelpad=2)
    ps.panel_title(bx, "(b) Boundary rows, 222 replicas", x=-0.98 / BW,
                   sub="S003 higher than every other in 222/222 (paired)")
    ps.save(fig, out)
    plt.close(fig)


SEL_RAMP = ("#cde2fb", "#8dbcf2", "#2a78d6", "#174f9e", "#0b2a55")


def fig_selection(st: pd.DataFrame, per_rep: dict, draws: dict, pred: dict, out: Path) -> None:
    plt = ps.setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde
    fig = plt.figure(figsize=(ps.FULL_W, 2.24))
    Y0, HY = 0.36, 1.50
    # (a) paired private-AUC difference -------------------------------------------------------------------
    AW = 1.86
    ax = place(fig, 1.16, Y0, AW, HY)
    XL = (-10.0, 3.6)
    xs = np.linspace(XL[0], XL[1], 1361)
    rng = np.random.default_rng(936)
    ax.axvspan(0, XL[1], color=ps.MID, lw=0, zorder=0)
    ps.refline(ax, 0.0, axis="x", zorder=1)
    sti = st.set_index("proc")
    n = len(SELECTION)
    ypos = {k: (n - 1 - i) + (0 if i < n - 1 else -0.25) for i, (k, _, _) in enumerate(SELECTION)}
    for k, label, role in SELECTION:
        y = ypos[k]
        col = ROLE_COLOR[role]
        D, w = draws[k]
        vals = (D * 1e4).ravel()
        wts = np.tile(w, D.shape[0])
        dens = gaussian_kde(vals, weights=wts, bw_method=0.15)(xs)
        dens = dens / dens.max() * 0.52
        m = (dens > 0.012 * 0.52) & (xs >= vals.min()) & (xs <= vals.max())
        post_hoc = k == "x_qda_block_nll_ext"
        ax.fill_between(xs[m], y, y + dens[m], color=col, alpha=0.12 if post_hoc else 0.28, lw=0, zorder=2)
        ax.plot(xs[m], y + dens[m], color=col, lw=0.8, ls=(0, (3, 1.6)) if post_hoc else "-", zorder=3)
        dd = per_rep[k] * 1e4
        ax.scatter(dd, y - 0.06 - rng.uniform(0, 0.3, len(dd)), s=3.0, color=col, lw=0, alpha=0.6, zorder=3)
        mu = dd.mean()
        ax.plot([mu, mu], [y - 0.4, y + 0.56], color=ps.INK, lw=1.2, zorder=5)
        ax.text(1.0 + 0.40 / AW, y + 0.06, signed(mu, 2), transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8, color=ps.INK)
        ax.text(1.0 + 0.78 / AW, y + 0.06, f"{sti.loc[k, 'p_beats_s003_replica_splits']:.0%}",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=ps.INK)
    ax.axhline(ypos["x_qda_block_nll_ext"] + 0.72, color=ps.GRID, lw=0.6, xmin=-1.15 / AW, xmax=1.0 + 0.8 / AW,
               clip_on=False)
    top = n - 0.25
    ax.text(1.0 + 0.40 / AW, top, "mean\n(×10⁻⁴)", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=ps.INK2, linespacing=1.0)
    ax.text(1.0 + 0.78 / AW, top, "P(beats\nS003)", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=ps.INK2, linespacing=1.0)
    ax.text(0.3, top + 0.02, "beats\nS003 →", ha="left", va="bottom", fontsize=7.5, color=ps.INK2, linespacing=1.0)
    kh = [Patch(facecolor=ps.LIGHT, edgecolor=ps.INK2, lw=0.6, label="replica × noise draw"),
          Line2D([], [], marker="o", ls="none", ms=2.2, color=ps.INK2, mew=0, label="replica mean (16 draws)"),
          Line2D([], [], marker="|", ls="none", ms=6, color=ps.INK, mew=1.2, label="mean")]
    ax.legend(handles=kh, loc="lower left", bbox_to_anchor=(-1.17 / AW, 1.0), ncol=3, fontsize=8, handlelength=1.0,
              handleheight=0.6, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)
    ax.set_ylim(ypos["x_qda_block_nll_ext"] - 0.55, top + 0.75)
    ax.set_yticks([ypos[k] + 0.06 for k, _, _ in SELECTION])
    labs = [label.replace(" (r930", "\n(r930") if " (r930" in label else
            label.replace(", ", ",\n") if len(label) > 20 else label for k, label, role in SELECTION]
    ax.set_yticklabels(labs, fontsize=8, color=ps.INK, linespacing=1.0)
    for tl, (k, _, _) in zip(ax.get_yticklabels(), SELECTION):
        if k == "x_qda_block_nll_ext":
            tl.set_style("italic")
            tl.set_color(ps.INK2)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(*XL)
    ax.set_xticks([-8, -6, -4, -2, 0, 2])
    ax.set_xticklabels(["−8", "−6", "−4", "−2", "0", "2"])
    ps.clean(ax, grid="", left=False)
    ax.set_xlabel("private AUC − S003 (×10⁻⁴)", labelpad=2)
    ps.panel_title(ax, "(a) Candidates vs S003, 222 replicas", x=-1.15 / AW, pad=16)

    # (b) predictive distribution of S003's private AUC ------------------------------------------------------
    BXW = 1.80
    bx = place(fig, 4.33, Y0, BXW, HY - 0.36)
    sx = place(fig, 4.33, Y0 + HY - 0.30, BXW, 0.30)        # strip above the bars: median and 90% interval
    edges, mass = pred["edges"], pred["mass"]               # probability per 0.0005-wide bin, per noise level
    # (lo, hi]: labelled by the simulated levels inside (R922 levels step 0.0005)
    bands = [(-1, 0.0, "0"), (0.0, 0.0025, "0.0005–0.0025"), (0.0025, 0.005, "0.003–0.005"),
             (0.005, 0.01, "0.0055–0.01"), (0.01, 1.0, "> 0.01")]
    fp = pred["fp"]
    base = np.zeros(len(edges) - 1)
    wbin = edges[1] - edges[0]
    lv = np.round(fp, 6)
    for (lo, hi, lab), col in zip(bands, SEL_RAMP):
        sel = (fp > lo) & (fp <= hi)
        if lab[0].isdigit():                                 # the label names the lowest and highest level inside
            ends = [float(e) for e in lab.split("–")]
            check([lv[sel].min(), lv[sel].max()] == [ends[0], ends[-1]], f"F6b band label {lab}")
        part = mass[sel].sum(0) * 100
        bx.bar(edges[:-1], part, bottom=base, width=wbin, align="edge", color=col, ec="white", lw=0.3, zorder=2,
               label=lab)
        base = base + part
    o = pred["ref"]
    top = base.max()
    YT = top * 1.40
    bx.set_ylim(0, YT)
    # outline of the bars at or above 0.999 (their total is P(>= 0.999))
    k0 = int(np.argmin(np.abs(edges - 0.999)))
    check(abs(edges[k0] - 0.999) < 1e-9, "0.999 is a bin edge")
    xe = edges[k0:]
    ye = base[k0:]
    stx = np.repeat(xe, 2)[1:-1]
    sty = np.repeat(ye, 2)
    bx.plot(np.r_[xe[0], stx, xe[-1] - 1e-9], np.r_[0, sty, 0], color=ps.INK, lw=0.8, zorder=4)
    ps.refline(bx, 0.999, axis="x", zorder=3, ymax=1.22 / 1.40)        # stops below the "public" label
    bx.plot([0.99923, 0.99923], [0, YT], color=ps.INK, lw=0.8, zorder=4)
    bx.text(0.99915, top * 1.30, "public 0.99923", ha="right", va="center", fontsize=8, color=ps.INK)
    bx.set_xlim(0.986, 1.0003)
    bx.set_xticks([0.986, 0.99, 0.995, 1.0])
    bx.set_xticklabels(["0.986", "0.990", "0.995", "1.000"])
    ps.clean(bx, grid="y")
    bx.set_yticks([t for t in (0, 5, 10) if t < YT])
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    bx.set_ylabel("probability per 0.0005 bin", labelpad=2)
    bx.set_xlabel("S003 private AUC", labelpad=2)
    lh, ll = bx.get_legend_handles_labels()
    lh.append(Patch(facecolor="white", edgecolor=ps.INK, lw=0.8))
    ll.append(f"P(≥ 0.999) = {o['P_ge_0.999']:.2f}")
    bx.legend(lh, ll, title="label-noise rate $f$", loc="upper left", fontsize=7.5, title_fontsize=8,
              handlelength=0.9, handleheight=0.7, labelspacing=0.1, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0),
              alignment="left")
    sx.set_xlim(*bx.get_xlim())
    sx.set_ylim(0, 1)
    sx.axis("off")
    yl = 0.18
    sx.plot([o["q05"], o["q95"]], [yl, yl], color=ps.INK, lw=1.0)
    for v in (o["q05"], o["q95"]):
        sx.plot([v, v], [yl - 0.14, yl + 0.14], color=ps.INK, lw=1.0)
    sx.plot(o["q50"], yl, "o", ms=4.0, color=ps.INK, mec="white", mew=0.5, zorder=5)
    sx.text(1.0, 0.52, f"median {o['q50']:.4f}, 90%: {o['q05']:.4f}–{o['q95']:.4f}", transform=sx.transAxes,
            ha="right", va="bottom", fontsize=8, color=ps.INK)
    ps.panel_title(sx, "(b) What to expect from S003", x=-0.30, pad=16)
    ps.save(fig, out)
    plt.close(fig)


# ============================================================================================ main

def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compute and print only; write nothing")
    ap.add_argument("--only", default="", help="comma-separated subset of F1..F6 to draw (tables are always built)")
    args = ap.parse_args()
    C.print_archive(SOURCE)
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()}

    tdir, fdir, xdir = paths.tables_dir(SOURCE), paths.figures_dir(SOURCE), exp_dir()
    raw = paths.data_dir(SOURCE)
    # --- inputs
    subs_csv = need(paths.submissions_csv(SOURCE))
    hist_csv = tdir / "core_submission_history.csv"
    e3_json = need(xdir / E3_RUN / "summary.json")
    r441_csv = need(tdir / "r441_label_budget.csv")
    r441_evals = need(xdir / R441_RUN / "label_budget_evals.csv")
    r441_fh = need(xdir / R441_RUN / "label_budget_first_hit.csv")
    r404_csv = need(tdir / "r404_boundary.csv")
    train_csv, test_csv = need(raw / "train.csv"), need(raw / "test.csv")
    sub_csv = need(raw / "sample_submission.csv")
    s003_pred = need(xdir / S003_RUN / "test_pred.npy")
    r901_json = need(xdir / R901_RUN / "summary.json")
    e1_est = need(xdir / E1_RUN / "paired_estimates.csv")
    e1_pd = need(xdir / E1_RUN / "paired_difference.csv")
    e1_rep = need(tdir / "e1_paired_difference.csv")
    nullA = need(tdir / "r900_perm_null_A_per_perm.csv")
    nullB = need(tdir / "r900_perm_null_B_per_perm.csv")
    npz = need(paths.ROOT / R936_NPZ)
    r936_cfg = need(xdir / R936_RUN / "config.yaml")
    r936_priv = need(xdir / R936_RUN / "private.csv")
    r922_lik = need(tdir / "r922_likelihood_fine.csv")
    r922_pbf = need(tdir / "r922_private_by_flip.csv")
    r922_json = need(xdir / R922_RUN / "summary.json")

    # --- tables (also the cross-checks against the archive)
    jt = journey_table(subs_csv, hist_csv)
    leaky_cv = float(json.loads(e3_json.read_text(encoding="utf-8"))["leaky_auc_mean"])
    bt_budget = budget_table(r441_csv)
    bt_bound = boundary_table(r404_csv)
    cur = budget_curves(r441_evals, pd.read_csv(r441_csv).assign(
        defined_all=lambda d: (d["n_failed"] == 0) & d["auc_mean"].notna()), r441_fh)
    D = data_anatomy(train_csv, test_csv, sub_csv, s003_pred, r901_json)
    est, schemes, A, B, nulls = protocol_tables(e1_est, e1_pd, e1_rep, nullA, nullB)
    lt, folds = landscape_table()
    st, per_rep, draws = selection_data(npz, r936_cfg, r936_priv)
    pred, pred_tab = predictive(r922_lik, r922_pbf, r922_json)
    jmd, bmd, dmd = journey_md(jt), budget_md(bt_budget), boundary_md(bt_bound)

    from tabulate import tabulate

    def show(title, df):
        print(f"\n{title}")
        print(tabulate(df.to_numpy().tolist(), headers=list(df.columns), tablefmt="simple", disable_numparse=True))

    show(f"Table 1 (core_journey; SE = Hanley-McNeil at the expected AUC, {N_POS}/{N_NEG} public rows):", jmd)
    print(f"E3 leaky CV (S004 model, {E3_RUN}): {leaky_cv:.4f}")
    show("F1 data anatomy (core_data.md):", data_md(D))
    show(f"F2 (a) E1 schemes ({E1_RUN}):", schemes_md(schemes))
    show(f"F2 (b, c) R900 nulls ({R900_RUNS}):", nulls_md(nulls))
    show("F4 landscape (official fold scores):", landscape_md(lt))
    print(f"\nF5 (a) label budget (R441, {R441_RUN}): first budget with mean AUC >= 0.99 (all evaluations defined):")
    for m, label, in_fig in budget_models():
        print(f"  {label:<50} {first_hit(bt_budget, m)}{'' if in_fig else '   (table only)'}")
    show(f"F5 (b) boundary rows (R404, {R404_RUN}; mean {bt_bound['boundary_rows_mean'].iloc[0]:.0f} rows per replica):",
         dmd)
    show(f"F6 (a) private-AUC difference to S003 (R936, {R936_RUN}; tn0, csv8, R156 weights):", selection_md(st))
    show("F6 (b) S003 private-AUC predictive, flat prior (R922): this script vs archived summary",
         pred_tab.assign(**{c: pred_tab[c].map(lambda v: f"{v:.6f}") for c in ("this_script", "r922_archived", "abs_diff")}))
    if args.check:
        print("\n--check: nothing written")
        return 0

    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    figs = {"F1": ("core_data.png", lambda p: fig_data(D, p)),
            "F2": ("core_protocol.png", lambda p: fig_protocol(est, schemes, A, B, p)),
            "F3": ("core_journey.png", lambda p: fig_journey(jt, leaky_cv, p)),
            "F4": ("core_landscape.png", lambda p: fig_landscape(lt, folds, p)),
            "F5": ("core_unlabeled.png", lambda p: fig_unlabeled(cur, bt_bound, p)),
            "F6": ("core_selection.png", lambda p: fig_selection(st, per_rep, draws, pred, p))}
    out = {}
    # table views
    jt.to_csv(tdir / "core_journey.csv", index=False)
    write_md(jmd, tdir / "core_journey.md", "registry/submissions.csv (SE: Hanley-McNeil at the expected AUC, 198/197)")
    bt_budget.to_csv(tdir / "core_label_budget.csv", index=False)
    write_md(bmd, tdir / "core_label_budget.md", f"reports/tables/r441_label_budget.csv (run {R441_RUN})")
    bt_bound.to_csv(tdir / "core_boundary.csv", index=False)
    write_md(dmd, tdir / "core_boundary.md", f"reports/tables/r404_boundary.csv (run {R404_RUN})")
    D["feat"].to_csv(tdir / "core_data_features.csv", index=False)
    write_md(data_md(D), tdir / "core_data.md", f"data/raw/train.csv + test.csv, S003 test_pred.npy ({S003_RUN}), "
                                                f"R901 summary.json ({R901_RUN})")
    schemes.to_csv(tdir / "core_protocol_schemes.csv", index=False)
    write_md(schemes_md(schemes), tdir / "core_protocol_schemes.md", f"run {E1_RUN} paired_estimates.csv / paired_difference.csv")
    nulls.to_csv(tdir / "core_protocol_nulls.csv", index=False)
    write_md(nulls_md(nulls), tdir / "core_protocol_nulls.md", "reports/tables/r900_perm_null_{A,B}_per_perm.csv")
    lt.to_csv(tdir / "core_landscape.csv", index=False)
    write_md(landscape_md(lt), tdir / "core_landscape.md", "experiments/<run>/fold_scores.csv of the runs listed")
    cur.to_csv(tdir / "core_budget_curves.csv", index=False)
    st.to_csv(tdir / "core_selection.csv", index=False)
    write_md(selection_md(st), tdir / "core_selection.md", f"{R936_NPZ} (input of run {R936_RUN}), checked against its private.csv")
    pred_tab.to_csv(tdir / "core_predictive.csv", index=False)
    write_md(pred_tab.assign(**{c: pred_tab[c].map(lambda v: f"{v:.5f}") for c in ("this_script", "r922_archived", "abs_diff")}),
             tdir / "core_predictive.md", f"reports/tables/r922_likelihood_fine.csv + r922_private_by_flip.csv (run {R922_RUN})")
    for name in ("core_journey.csv", "core_journey.md", "core_label_budget.csv", "core_label_budget.md",
                 "core_boundary.csv", "core_boundary.md", "core_data_features.csv", "core_data.md",
                 "core_protocol_schemes.csv", "core_protocol_schemes.md", "core_protocol_nulls.csv",
                 "core_protocol_nulls.md", "core_landscape.csv", "core_landscape.md", "core_budget_curves.csv",
                 "core_selection.csv", "core_selection.md", "core_predictive.csv", "core_predictive.md"):
        out[name] = tdir / name
    from PIL import Image
    for key, (name, draw) in figs.items():
        out[name] = fdir / name
        if not only or key in only:
            draw(fdir / name)
            w, h = (v / ps.DPI for v in Image.open(fdir / name).size)
            print(f"  {key} {name}: {w:.2f} x {h:.2f} in" + ("   WARNING: wider than the text width 6.27 in "
                                                                "(scaled down when included)" if w > 6.30 else ""))

    manifest = {
        "generated_by": "scripts/make_core_figures.py", "generated_at": now_iso(),
        "style": "src/ssc/plotstyle.py (Times New Roman, 300 dpi, drawn at 6.27 in)",
        "inputs": {k: sha256(p) for k, p in sorted(INPUTS.items())},
        "source_runs": {
            "core_data (F1)": f"data/raw (train+test features); {S003_RUN} test_pred.npy; {R901_RUN}",
            "core_protocol (F2)": f"{E1_RUN}; {R900_RUNS}",
            "core_journey (F3)": f"registry/submissions.csv (S001-S006); {E3_RUN}",
            "core_landscape (F4)": "; ".join(r for r, *_ in LANDSCAPE),
            "core_unlabeled (F5)": f"(a) {R441_RUN}; (b) {R404_RUN}",
            "core_selection (F6)": f"(a) {R936_RUN} via {R936_NPZ}; (b) {R922_RUN}"},
        "conventions": {
            "nines_axis": "AUC x drawn at -log10(1 - x); exact 1.0 drawn at 1 - 1e-5 and labelled '= 1', with an axis "
                          "break between 0.9999 and the cap (F3, F5a; checked: no plotted value lies between the break "
                          "and 1); in F4 the main axis ends at 0.9993 and folds at exactly 1 are counted in a separate "
                          "column (bar = share of the 100 folds)",
            "reference_lines": "every reference value (0, 0.5, 1.5, 0.947, 0.99, 0.999, 0.806) is a 0.6 pt dashed line",
            "F1a": "Pearson correlation of the 150 features over the 20,000 train+test rows; drawn: the 20 structured "
                   f"features, informative {INFORMATIVE} then redundant {REDUNDANT} (roles as inferred under "
                   "make_classification), diagonal left blank; the 130 noise features are only a note (max |r| among "
                   "them: core_data.md)",
            "F1b": "sd over the 20,000 rows divided by the median sd of the 150 features; block = ratio > 1.5",
            "F1c": "block standardised with train+test mean and sd, reduced to its rank-12 span; every row weighted by "
                   "p (test rows: S003's archived score; training rows: their label) and by 1 - p; weighted means and "
                   "covariances; generalised eigenvectors of (p-weighted, (1-p)-weighted) covariance; axes = the "
                   "directions with the largest and the smallest variance ratio, scaled to unit (1-p)-weighted sd, "
                   "origin at the (1-p)-weighted mean; ellipses = 1 sd (dashed) and 2 sd (solid) of each weighted "
                   "class; limits = 99.9% quantile of |coordinate| x 1.02 per axis (rows outside: core_data.md); the "
                   "panel is not equal-aspect: x is squeezed relative to y by the factor printed in the panel; rows "
                   "with 0.1 < p < 0.9 are drawn last and larger",
            "F2a": "E1 paired difference (model A corr top-30 + L2 minus model B corr top-50 + L2) per execution, "
                   "50 executions (split seeds) per scheme; outline = Gaussian kernel density (Scott), scaled to equal peak "
                   "height, drawn where it exceeds 1% of its peak; fill grey by fits per decision; ink bar = +-MDD, "
                   "MDD = 2 sd of the 50 differences (E1 paired_difference.csv); ticks = the 50 estimates; grey band "
                   "= pooled 5 x 20 mean +- 0.01 (as in E1's figure)",
            "F2bc": "R900: mean AUC over 100 folds (5 x 20, seed 2026) for each of 20 label permutations; Null A "
                    "trained and scored on permuted labels, its folds stratified on the permuted labels (so not the "
                    "official folds); Null B trained on permuted labels on the official folds and scored on the true "
                    "ones; one x scale for both; black tick = mean over the 20 permutations; controls drawn as x (the "
                    "leaky filter and the label reader should fire, random scores should not), red only where the "
                    "control fires (leaky filter in A, label reader in B)",
            "F3_public_se": f"Hanley-McNeil at the pre-registered expected AUC, {N_POS} pos / {N_NEG} neg (as R440); "
                            "bars that reach 1 stop below the axis break with a caret; arrows: expectation -> public "
                            "score, and (red) E3 leaky CV -> S004 public score",
            "F4": "fold_scores.csv of each run (100 official folds, split signature 5708ce5c3828); dots = folds with "
                  "AUC < 1 (jitter +-0.32 row); tick = CV mean (S005's 0.9999 is beyond the main axis: marker at its "
                  "end); right column: number of folds with AUC exactly 1 (bar length = share of 100)",
            "F5a": "R441 label_budget_evals.csv: mean and 10%/90% quantiles over 400 evaluations (80 folds x 5 label "
                   "draws; 80 at n = 200), only budgets where every evaluation is defined; first hit = smallest such "
                   "budget with mean >= 0.99 (equals R441 label_budget_first_hit.csv), ringed on the curve, its x tick "
                   "in bold; S005's band hatched (no fill) so that it reads where it overlaps Bayesian QDA's",
            "F5b": "R404 boundary rows (true class probability in (0.1, 0.9)), clean labels, 222 replicas; dot = mean; "
                   "dark bar = +-2 MC SE of the mean; light bar = +-1 sd across replicas (= MC SE x sqrt(222), R404 "
                   "computes the MC SE as sd/sqrt(n)); per-replica values are not archived; hollow = in-fold "
                   "(labels-only) Bayesian QDA; 222/222 = R404 bnd_auc_clean_share_s003_better = 1 for every other "
                   "procedure",
            "F6a": "R936 primary convention: group tn0 (clean training labels), csv8 scores, levels 0/0.0025/0.005/0.01 "
                   "weighted 0.0773/0.4069/0.3608/0.1550 (R156); dots = per-replica mean difference over the 16 draws; "
                   "outline = weighted kernel density (bandwidth factor 0.15) of all replica x draw differences, drawn "
                   "within the data range where it exceeds 1.2% of its peak; P(beats S003) over replica x draw "
                   "splits; recomputed from the archived npz and equal to R936 private.csv; ext.-grid QDA (r930) is a "
                   "post-hoc variant, drawn last, dashed, below a divider",
            "F6b": "R922 flat prior on flip_y in [0, 0.1]: weights Ltrain_rel x Lpub_conv_exact_round5 mapped to the "
                   "nearest simulated level; within a level normal(mean, sd) of R922's private_by_flip.csv (draws not "
                   "archived); reproduces R922's mean, quantiles and P(>= 0.999) (core_predictive.csv); "
                   "printed numbers are R922's archived ones; bars stacked by noise level (5 groups, each labelled by the "
                   "lowest and highest simulated level inside it), the bars at or "
                   "above 0.999 outlined (their total = P(>= 0.999)); strip above: median and 90% interval"},
        "outputs": sorted(C.rel(p).replace("\\", "/") for p in out.values()),
    }
    (tdir / "core_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwritten:")
    for p in list(out.values()) + [tdir / "core_manifest.json"]:
        print(f"  {C.rel(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
