"""R440 - audit of the numbers the core report will use (research round 2, label "r2_audit").

Every number that research/reports/{sup,fm,gen}.md, research/reports/verify.md and registry/submissions.csv offer to
the core report is re-derived from the archived run files (summary.json, fold_scores.csv, run tables, report tables):
CV means and sds are recomputed from fold_scores.csv, paired differences with ssc.validation.paired_compare, the rest
is read from the run that produced it. The script also compares the independent label-budget re-run R441 with R206
and R304 (D009 evidence 1) and builds a figure-ready "journey" table (S001 -> S006 + candidates).

Two steps, because the Colab VM has no copy of experiments/ and reports/:
  1. --pack   (laptop; pure file copy, no Run, nothing written to the archive): reads every file named below and
              writes research/r440_archive_pack.json.gz = {relative path: {"sha256": <of the bytes>, "text": <utf-8>}}
              plus {"sha256", "n_lines"} of each submission CSV. The pack travels to Colab in the code bundle.
  2. default  (Colab; kind = analysis Run): checks every packed sha256, runs the audit, writes
              reports/tables/r440_core_numbers.{csv,md}  (claim, value, source file, verified yes/no, note, ...)
              reports/tables/r440_r249_recheck.csv       (every row of r249_comparison.csv recomputed)
              reports/tables/r440_journey.{csv,md}       (figure-ready journey table)
              reports/tables/r440_label_budget_compare.{csv,md}, r440_label_budget_first_hit.{csv,md}
              reports/figures/r440_label_budget_compare.png

    .venv\\Scripts\\python.exe research\\r440_core_numbers.py --pack
    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s r2aud -- research/r440_core_numbers.py
"""

from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PACK = os.path.join(HERE, "r440_archive_pack.json.gz")

# ----------------------------------------------------------------------------------------------- runs referenced
R = {
    "l1_r003": "20261022-050647_r003_sup_l1_all", "lda_r003": "20261022-050653_r003_sup_lda_hv",
    "qda_r003": "20261022-050659_r003_sup_qda_hv", "lr_r003": "20261022-050704_r003_sup_logreg_hv",
    "gmm": "20261022-050708_r003_gmm2_hv", "e3": "20261022-063908_e3_leakage",
    "r201": "20261022-071234_r201_qda_block", "r209": "20261022-080632_r209_selftrain_qda_lw",
    "r210": "20261022-082047_r210_qda_block_nll", "r205": "20261022-072747_r205_select_supervised",
    "tabpfn_all": "20261022-073743_r251_tabpfn2_all", "tabpfn_hv": "20261022-073935_r251_tabpfn2_hv",
    "tabicl_all": "20261022-074251_r251_tabicl_all", "tabicl_hv": "20261022-074643_r251_tabicl_hv",
    "hs_all": "20261022-073018_r253_hs_all", "hs_quad": "20261022-073222_r253_hs_quadhv",
    "hs_indep": "20261022-075227_r257_hs_indep",
    "bqda": "20261022-070200_r255_bqda_eb_hv", "bqda_plugin": "20261022-070536_r256_bqda_eb_plugin_hv",
    "bqda_train": "20261022-070202_r255_bqda_eb_trainonly", "qda001": "20261022-070543_r256_qda_reg0.01_hv",
    "qda003": "20261022-070545_r256_qda_reg0.03_hv", "qda03": "20261022-070546_r256_qda_reg0.3_hv",
    "gpc_fm": "20261022-070207_r255_gpc_ard_hv", "dpgmm": "20261022-071054_r254_dpgmm_hv",
    "dp_evid": "20261022-070634_r254_dpgmm_evidence",
    "r150": "20261022-064606_r150_taxonomy", "r151": "20261022-072608_r151_replica_validation",
    "r154": "20261022-065605_r154_class_structure", "r156": "20261022-081314_r156_flip_posterior",
    "r920": "20261022-100551_r920_generator_alternatives", "r921": "20261022-094951_r921_bayes_regret_replicas",
    "r922": "20261022-092146_r922_label_noise_private", "r206": "20261022-074712_r206_label_budget",
    "r304": "20261022-070600_r304_label_requirement", "r936": "20261022-172146_r936_final_selection",
    "r930_ext": "20261022-095943_r930_qda_block_nll_ext",
}
TABLES = ["r249_comparison.csv", "r206_label_budget.csv", "r304_label_requirement.csv", "r922_posterior_by_prior.csv",
          "r936_private.csv", "r936_choice.csv", "r936_public_abc.csv", "r936_cv_power.csv", "r154_population_aucs.csv",
          "r254_replica_auc_summary.csv", "r921_summary.csv", "r255_eb_choice_frequency.csv",
          "r153_param_summary_models.csv", "r932_pairwise_agreement.csv", "r930_paired_vs_gmm.csv",
          "e1_paired_difference.csv", "r900_perm_null_A_summary.csv", "r900_perm_null_B_summary.csv"]
RUN_EXTRA = {"r206": ["label_budget_evals.csv"], "r304": ["label_requirement_all.csv"], "e3": ["cells.csv"]}
SUB_IDS = ["S001", "S002", "S003", "S004", "S005", "S006"]


def rel(*p):
    return "/".join(p)


# ================================================================================================ step 1: pack
def make_pack() -> None:
    files = {}

    def add(path_rel):
        full = os.path.join(ROOT, *path_rel.split("/"))
        if not os.path.exists(full):
            print(f"[pack] missing: {path_rel}")
            return
        b = open(full, "rb").read()
        files[path_rel] = {"sha256": hashlib.sha256(b).hexdigest(), "text": b.decode("utf-8")}

    r249 = pd.read_csv(os.path.join(ROOT, "reports", "tables", "r249_comparison.csv"))
    runs = set(R.values()) | set(r249["run_id"])
    r441 = sorted(glob.glob(os.path.join(ROOT, "experiments", "*_r441_label_budget_rerun")))
    if r441:
        runs.add(os.path.basename(r441[-1]))
    for run in sorted(runs):
        for f in ("summary.json", "fold_scores.csv"):
            if os.path.exists(os.path.join(ROOT, "experiments", run, f)):
                add(rel("experiments", run, f))
    for key, fs in RUN_EXTRA.items():
        for f in fs:
            add(rel("experiments", R[key], f))
    if r441:
        for f in ("label_budget_evals.csv", "label_budget_summary.csv", "label_budget_first_hit.csv"):
            add(rel("experiments", os.path.basename(r441[-1]), f))
    for t in TABLES:
        add(rel("reports", "tables", t))
    add("registry/submissions.csv")
    subs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "submissions", "S00*.csv"))):
        b = open(f, "rb").read()
        subs[os.path.basename(f)] = {"sha256": hashlib.sha256(b).hexdigest(), "n_lines": b.count(b"\n")}
    pack = {"files": files, "submission_files": subs, "r441_run": os.path.basename(r441[-1]) if r441 else None}
    with gzip.open(PACK, "wt", encoding="utf-8") as fh:
        json.dump(pack, fh)
    print(f"[pack] {len(files)} files, {len(subs)} submission hashes -> {PACK} ({os.path.getsize(PACK) / 1e3:.0f} kB); "
          f"r441 run = {pack['r441_run']}")


# ================================================================================================ step 2: audit
class Archive:
    def __init__(self, pack: dict):
        self.files = pack["files"]
        self.subs = pack["submission_files"]
        self.r441 = pack.get("r441_run")
        bad = [p for p, v in self.files.items() if hashlib.sha256(v["text"].encode("utf-8")).hexdigest() != v["sha256"]]
        if bad:
            raise ValueError(f"pack corrupted: {bad[:5]}")
        self._cache = {}

    def text(self, path):
        return self.files[path]["text"].lstrip("﻿")

    def csv(self, path):
        if path not in self._cache:
            self._cache[path] = pd.read_csv(io.StringIO(self.text(path)))
        return self._cache[path]

    def summary(self, run):
        return json.loads(self.text(rel("experiments", run, "summary.json")))

    def s(self, key, *path):
        """summary value; string dicts (agreement_with_s003, ...) are parsed with ast.literal_eval."""
        v = self.summary(R.get(key, key))
        for p in path:
            if isinstance(v, str):
                v = ast.literal_eval(v)
            v = v[p]
        return v

    def folds(self, key):
        return self.csv(rel("experiments", R.get(key, key), "fold_scores.csv"))

    def fold_mean(self, key):
        return float(self.folds(key)["auc"].mean())

    def fold_sd(self, key):
        return float(self.folds(key).groupby("repeat")["auc"].mean().std(ddof=1))

    def paired(self, a, b):
        from ssc.validation import paired_compare
        return paired_compare(self.folds(a), self.folds(b))

    def table(self, name):
        return self.csv(rel("reports", "tables", name))

    def sub(self, sid, col):
        t = self.csv("registry/submissions.csv")
        return t.loc[t["sub_id"] == sid, col].iloc[0]


# ------------------------------------------------------------------------------------------------ claim checking
def _num(s: str) -> float:
    return float(s.replace("−", "-").replace(",", "").replace("+", "").replace("%", "").strip())


def _tol(s: str) -> float:
    s = s.replace("−", "-").replace(",", "").replace("%", "").strip().lstrip("+-")
    if "e" in s.lower():
        mant, exp = s.lower().split("e")
        d = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10 ** (int(exp) - d) * (1 + 1e-9)
    d = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-d) * (1 + 1e-9) + 1e-15


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, (int, float, np.floating)) for x in v):
        return f"{_fmt(v[0])} to {_fmt(v[1])}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0 or math.isnan(v):
            return str(v)
        return f"{v:.6g}"
    return str(v)


def check(claimed: str, archived, mode: str) -> bool:
    if mode == "info":  # not checkable from the archive
        return False
    if mode == "eq":
        try:
            c = ast.literal_eval(claimed)
        except (ValueError, SyntaxError):
            c = claimed
        return list(c) == list(archived) if isinstance(c, (list, tuple)) else c == archived
    if mode == "true":
        return bool(archived)
    if mode == "pct":
        return abs(float(archived) * 100 - _num(claimed.rstrip("%"))) <= _tol(claimed.rstrip("%"))
    if mode == "round":
        return abs(float(archived) - _num(claimed)) <= _tol(claimed)
    if mode == "ge":
        return float(archived) >= _num(claimed) - _tol(claimed)
    if mode == "le":
        return float(archived) <= _num(claimed) + _tol(claimed)
    lo_s, hi_s = re.split(r"\s*(?:–|--|to)\s*|(?<=\d)-(?=\d)", claimed)[:2]
    if mode == "range":  # archived (lo, hi) must round to the claimed ends
        return abs(archived[0] - _num(lo_s)) <= _tol(lo_s) and abs(archived[1] - _num(hi_s)) <= _tol(hi_s)
    if mode == "in":  # archived scalar inside the claimed interval
        return _num(lo_s) - _tol(lo_s) <= float(archived) <= _num(hi_s) + _tol(hi_s)
    raise ValueError(mode)


CLAIMS: list[dict] = []


def C(cid, section, claim, value, claimed_in, source, get, mode="round", note="", status=""):
    CLAIMS.append(dict(id=cid, section=section, claim=claim, value=value, claimed_in=claimed_in, source=source,
                       get=get, mode=mode, note=note, status=status))


def ex(run, f="summary.json"):
    return rel("experiments", R.get(run, run), f)


def tb(name):
    return rel("reports", "tables", name)


def define_claims(A: Archive, data_facts: dict, hm, hm_any) -> None:
    SUPM, FMM, GENM, VER, SUBS = ("research/reports/sup.md", "research/reports/fm.md", "research/reports/gen.md",
                                  "research/reports/verify.md", "registry/submissions.csv")
    # ---------------------------------------------------------------- 1 problem formulation / data
    C("D1", "data", "labelled training rows", "250", "task brief; report_core.md template", "data/raw/train.csv",
      lambda A: data_facts["n_train"], "eq")
    C("D2", "data", "unlabelled test rows", "19750", "task brief", "data/raw/test.csv", lambda A: data_facts["n_test"], "eq")
    C("D3", "data", "features", "150", "task brief", "data/raw/train.csv", lambda A: data_facts["n_features"], "eq")
    C("D4", "data", "training positive rate", "0.452", "R150 summary (train_pos_rate)", "data/raw/train.csv",
      lambda A: data_facts["pos_rate"], note=f"{data_facts['n_pos']} positives of {data_facts['n_train']}")
    C("D5", "data", "test rows in the public LB", "395", "verify.md §5 (about 2% of 19,750)", "Kaggle competition page",
      lambda A: 0.02 * data_facts["n_test"], "info", status="external (not archived)",
      note="0.02 x 19,750 = 395; the exact public size is stated on the Kaggle page only; archive the rules page")
    C("D6", "data", "Hanley-McNeil SE of a public AUC of 0.79 (395 rows, 198/197)", "0.023", "verify.md §1.2",
      "computed (ssc.validation.auc_standard_error)", lambda A: hm(0.79))
    C("D7", "data", "Hanley-McNeil SE of a public AUC of 0.82", "0.021", "verify.md §1.2", "computed", lambda A: hm(0.82))
    C("D8", "data", "Hanley-McNeil SE of a public AUC of 0.99913", "0.0015", "verify.md §1.2", "computed", lambda A: hm(0.99913))
    C("D9", "data", "Hanley-McNeil SE of a public AUC of 0.99884", "0.0017", "verify.md §1.2", "computed", lambda A: hm(0.99884))
    C("D10", "data", "Hanley-McNeil SE of one 50-row validation fold at AUC 0.80 (23 / 27 rows)", "0.065",
      "reports/report_core.md template ('one 50-row fold SE ~0.065')", "computed",
      lambda A: hm_any(0.80, round(50 * data_facts["pos_rate"]), 50 - round(50 * data_facts["pos_rate"])))
    C("D11", "data", "features / training rows (p/n)", "0.6", "reports/report_core.md template", "data/raw/train.csv",
      lambda A: data_facts["n_features"] / data_facts["n_train"])

    # ---------------------------------------------------------------- 2 generator (gen.md, corrected by verify)
    C("G1", "generator", "informative columns", "[6, 15, 30, 46, 52, 61, 64, 80, 100, 112, 142, 144]", GENM, ex("r150"),
      lambda A: A.s("r150", "informative"), "eq", note="R920 basis search picks the same set (basis_best_equals_claim)")
    C("G2", "generator", "R920 label-free basis search returns the same 12 columns", "True", VER + " A2", ex("r920"),
      lambda A: A.s("r920", "basis_best_equals_claim"), "true")
    C("G3", "generator", "R920 basis margin (nats)", "13.0", VER + " A2", ex("r920"), lambda A: A.s("r920", "basis_margin_nats"))
    C("G4", "generator", "R150 basis log-det margin (nats)", "1.09", GENM, ex("r150"), lambda A: A.s("r150", "basis_logdet_margin"),
      note="different statistic from R920's 13.0 nats; quote R920 in the report")
    C("G5", "generator", "redundant columns", "[8, 17, 26, 49, 57, 67, 77, 98]", GENM, ex("r150"),
      lambda A: A.s("r150", "redundant"), "eq")
    C("G6", "generator", "noise columns", "130", GENM, ex("r150"), lambda A: A.s("r150", "role_counts", "noise"), "eq")
    C("G7", "generator", "structured columns / rank", "(20, 12)", GENM, ex("r150"),
      lambda A: (A.s("r150", "n_structured"), A.s("r150", "rank_structured")), "eq")
    C("G8", "generator", "1 - R^2 of the 8 exact dependencies (max)", "4e-14", GENM + " ('R^2 >= 1-4e-14')", ex("r150"),
      lambda A: 1 - A.s("r150", "dependency_r2_min"),
      note="archived 1 - R^2 = 4.07e-14, i.e. R^2 = 1 - 4.1e-14, marginally below the stated bound; write 'R^2 > 1 - 1e-13'")
    C("G9", "generator", "max |residual| of the dependencies", "4.8e-6", GENM, ex("r150"),
      lambda A: A.s("r150", "dependency_max_abs_residual"))
    C("G10", "generator", "noise std ~ U(0.5, 3): KS p", "0.26", GENM, ex("r150"),
      lambda A: A.s("r150", "noise_tests", "noise_std_ks_U(0.5,3)"))
    C("G11", "generator", "noise mean ~ U(-5, 5): KS p", "0.87", GENM, ex("r150"),
      lambda A: A.s("r150", "noise_tests", "noise_mean_ks_U(-5,5)"))
    C("G12", "generator", "implied B range", "-1.08 to 1.11", GENM, ex("r150"),
      lambda A: (A.s("r150", "B_unscaled_min"), A.s("r150", "B_unscaled_max")), "range")
    C("G13", "generator", "B ~ U: KS p", "0.73", GENM, ex("r150"), lambda A: A.s("r150", "B_unscaled_ks_uniform_p"))
    C("G14", "generator", "class balance of the 20,000 rows (hard GMM labels, class 1)", "10016", GENM + " (10,016 : 9,984)",
      ex("r150"), lambda A: A.s("r150", "n_class1_hard"), "eq")
    C("G15", "generator", "max |z| of the 8 zero-difference informative columns", "2.08", GENM, ex("r150"),
      lambda A: A.s("r150", "zero_group_abs_z_max"))
    C("G16", "generator", "min |z| of the 4 active informative columns", "47", GENM + " ('|z| >= 47')", ex("r150"),
      lambda A: A.s("r150", "active_abs_z_min"), "ge")
    C("G17", "generator", "class_sep MLE (R920)", "0.703", VER + " A4; gen.md errata 4", ex("r920"), lambda A: A.s("r920", "class_sep_mle"))
    C("G18", "generator", "class_sep 95% LR interval (R920)", "0.639 to 0.760", VER + " A4", ex("r920"),
      lambda A: tuple(A.s("r920", "class_sep_95ci_LR")), "range")
    C("G19", "generator", "class_sep (R150, superseded)", "0.71", "gen.md before the errata", ex("r150"),
      lambda A: A.s("r150", "class_sep_hat"), status="superseded by R920 (0.70, 0.64-0.76)")
    C("G20", "generator", "13-column high-variance block; 7 structured columns outside it", "(13, 7)", GENM, ex("r150"),
      lambda A: (len(A.s("r150", "highvar_block")), len(A.s("r150", "structured_outside_block"))), "eq")
    C("G21", "generator", "population Bayes AUC (quadratic rule)", "0.99995", GENM, ex("r154"), lambda A: A.s("r154", "auc_bayes_quadratic"))
    C("G22", "generator", "best linear population AUC", "0.947", GENM, ex("r154"), lambda A: A.s("r154", "auc_best_linear_formula"),
      note="Monte Carlo 0.9466, closed form 0.9468")
    C("G23", "generator", "Bayes error rate", "0.33%", VER + " 7.1 (R154)", ex("r154"), lambda A: A.s("r154", "bayes_error_rate"), "pct")
    C("G24", "generator", "expected test rows with posterior in (0.1, 0.9)", "186", VER + " 7.1 (R154)", ex("r154"),
      lambda A: A.s("r154", "expected_test_rows_0.1_0.9"))
    C("G25", "generator", "replica ranking of the R003 models identical (parametric replicas)", "97%", GENM, ex("r151"),
      lambda A: A.s("r151", "ranking_order_match_rate", "parametric"), "pct", note="29 of 30 replicas")
    C("G26", "generator", "S003 regret vs Bayes on parametric replicas (expected AUC)", "2.6e-7", VER + " B1", ex("r921"),
      lambda A: A.s("r921", "summary", "param", "regret_eauc_mean"))
    C("G27", "generator", "mis-ordered private pairs per replica (S003 vs Bayes)", "24", VER + " B1 ('24 of 9.4e7')", ex("r921"),
      lambda A: A.s("r921", "summary", "param", "regret_clean_pairs_mean"))
    C("G28", "generator", "generic make_classification draws where the block misses a direction", "5.5%", VER + " B1", ex("r921"),
      lambda A: A.s("r921", "summary", "mc_not_spanning", "n") / A.s("r921", "summary", "mc_all", "n"), "pct")
    C("G29", "generator", "regret of S003 on those draws", "5.4e-4", VER + " B1", ex("r921"),
      lambda A: A.s("r921", "summary", "mc_not_spanning", "regret_eauc_mean"))
    C("G30", "generator", "flip_y MLE (train labels + S003 public)", "0.003", VER + " B6; gen.md errata 1", ex("r922"),
      lambda A: A.s("r922", "flip_mle_combined"))
    C("G31", "generator", "flip_y likelihood support (>= 1/8 of max)", "0 to 0.0147", VER + " B6", ex("r922"),
      lambda A: tuple(A.s("r922", "flip_support_1_8")), "range", note="archived upper end 0.01475")
    C("G32", "generator", "likelihood ratio flip_y = 0 vs MLE", "0.20", VER + " B5", ex("r922"), lambda A: A.s("r922", "LR_flip0_vs_mle"))
    C("G33", "generator", "likelihood ratio flip_y = 0.01 vs MLE", "0.37", VER + " B6", ex("r922"), lambda A: A.s("r922", "LR_flip0.01_vs_mle"))
    pri = lambda A, k: A.s("r922", "posterior_by_prior", "uniform_0_0.1", k)  # noqa: E731
    C("G34", "generator", "S003 private AUC, posterior mean (flat prior)", "0.9968", VER + " B7", ex("r922"), lambda A: pri(A, "pred_private_mean"))
    C("G35", "generator", "S003 private AUC, posterior median (flat prior)", "0.9973", VER + " B7", ex("r922"), lambda A: pri(A, "pred_private_q50"))
    C("G36", "generator", "S003 private AUC, 90% interval (flat prior)", "0.9923 to 0.9996", VER + " B7", ex("r922"),
      lambda A: (pri(A, "pred_private_q05"), pri(A, "pred_private_q95")), "range")
    C("G37", "generator", "P(S003 private >= 0.999) (B004)", "0.16", VER + " B8", ex("r922"), lambda A: pri(A, "P_B004_private_ge_0.999"))
    C("G38", "generator", "R156 posterior mean flip_y", "0.0044", "gen.md before the errata", ex("r156"),
      lambda A: A.s("r156", "posterior_mean_flip_y"), status="superseded by R922 (4-point prior truncated at 0.01)")
    C("G39", "generator", "R156 expected private AUC of S003", "0.9977", "gen.md before the errata; B004", ex("r156"),
      lambda A: A.s("r156", "expected_private_auc_s003"), status="superseded by R922 (0.997, 90% 0.992-0.9996)")
    g2 = lambda A, f, m, c: A.table("r153_param_summary_models.csv").query("flip_y == @f and model == @m")[c].iloc[0]  # noqa: E731
    C("G40", "generator", "Table G2: Bayes private AUC, flip_y 0", "0.99995", GENM + " Table G2", tb("r153_param_summary_models.csv"),
      lambda A: g2(A, 0.0, "bayes", "private_mean"))
    C("G41", "generator", "Table G2: Bayes private AUC, flip_y 0.005", "0.99742", GENM + " Table G2", tb("r153_param_summary_models.csv"),
      lambda A: g2(A, 0.005, "bayes", "private_mean"))
    C("G42", "generator", "Table G2: S003 (gmm2_hv) regret, flip_y 0 (100 replicas)", "3e-7", GENM + " Table G2", tb("r153_param_summary_models.csv"),
      lambda A: g2(A, 0.0, "gmm2_hv", "regret_mean"), note="R921 (1000 replicas) gives 2.6e-7; quote R921")
    C("G43", "generator", "Table G2: QDA reg 0.1 private AUC, flip_y 0", "0.99314", GENM + " Table G2", tb("r153_param_summary_models.csv"),
      lambda A: g2(A, 0.0, "qda_hv", "private_mean"))
    C("G44", "generator", "Table G2: L1 logistic private AUC, flip_y 0", "0.8602", GENM + " Table G2", tb("r153_param_summary_models.csv"),
      lambda A: g2(A, 0.0, "l1_all", "private_mean"))

    # ---------------------------------------------------------------- 3 submissions (registry vs runs)
    for sid in SUB_IDS:
        run = A.sub(sid, "run_id")
        C(f"S{sid[1:]}a", "submissions", f"{sid} CV AUC mean (registry)", f"{float(A.sub(sid, 'cv_auc_mean')):.5f}", SUBS,
          ex(run, "fold_scores.csv") if sid != "S004" else ex(run),
          (lambda A, r=run: A.fold_mean(r)) if sid != "S004" else (lambda A, r=run: A.summary(r)["auc_mean"]),
          note="recomputed from fold_scores.csv" if sid != "S004" else "E3 writes no fold_scores.csv; summary.json")
        C(f"S{sid[1:]}b", "submissions", f"{sid} CV AUC sd over repeats (registry)", f"{float(A.sub(sid, 'cv_auc_std_repeat')):.5f}",
          SUBS, ex(run, "fold_scores.csv") if sid != "S004" else ex(run),
          (lambda A, r=run: A.fold_sd(r)) if sid != "S004" else (lambda A, r=run: A.summary(r)["auc_std_repeat"]))
        fn = os.path.basename(A.sub(sid, "file"))
        C(f"S{sid[1:]}c", "submissions", f"{sid} submission file sha256 prefix = registry", str(A.sub(sid, "sha256")), SUBS,
          "submissions/" + fn, lambda A, f=fn: A.subs[f]["sha256"][:16] if f in A.subs else "file missing", "eq",
          note="sha256 of the local file, computed at pack time")
        pub = A.sub(sid, "public_lb")
        C(f"S{sid[1:]}d", "submissions", f"{sid} public LB", "pending" if pd.isna(pub) else f"{float(pub):.5f}", SUBS,
          SUBS, lambda A, p=pub: "pending" if pd.isna(p) else float(p), "eq" if pd.isna(pub) else "round",
          status="registry only (Kaggle page not re-read)" if not pd.isna(pub) else "not uploaded yet",
          note="recorded by record_lb.py; the only archived copy" if not pd.isna(pub) else "S005/S006 created and pre-registered (B006/B007)")
    C("E3a", "submissions", "E3 leaky CV (k_feat 20)", "0.867", "S004 description; ext34.md", ex("e3"), lambda A: A.s("e3", "leaky_auc_mean"))
    cell = lambda A, c: A.csv(ex("e3", "cells.csv")).query("k_feat == 20 and k_fold == 5")[c].iloc[0]  # noqa: E731
    C("E3b", "submissions", "E3 leakage gap (k_feat 20)", "0.073", VER + " D4", ex("e3", "cells.csv"), lambda A: cell(A, "gap_mean"))
    C("E3c", "submissions", "E3 gap SD over 20 split seeds", "0.014", VER + " D4", ex("e3", "cells.csv"), lambda A: cell(A, "gap_std"))
    C("E3d", "submissions", "E3 gap Nadeau-Bengio corrected SE", "0.020", VER + " D4", ex("e3", "cells.csv"),
      lambda A: cell(A, "gap_mean") / cell(A, "corrected_t"))
    C("E3e", "submissions", "E3 gap corrected p", "0.0003", VER + " D4", ex("e3", "cells.csv"), lambda A: cell(A, "corrected_p"))
    C("E3f", "submissions", "E3 leaky CV - S004 public", "0.075", VER + " F8", ex("e3") + " + " + SUBS,
      lambda A: A.s("e3", "leaky_auc_mean") - float(A.sub("S004", "public_lb")))
    C("E3g", "submissions", "E3 honest CV - S004 public", "0.002", VER + " F8", ex("e3") + " + " + SUBS,
      lambda A: A.s("e3", "auc_mean") - float(A.sub("S004", "public_lb")))

    # ---------------------------------------------------------------- 4 supervised ablations (sup.md, r249)
    T = A.table("r249_comparison.csv")
    grp = lambda g: T[T["group"] == g]  # noqa: E731
    C("P1", "supervised", "LDA (means only)", "0.86 to 0.88", SUPM, ex("r200_lda_block", "fold_scores.csv") if False else tb("r249_comparison.csv") + " + " + ex("lda_r003"),
      lambda A: (A.fold_mean("20261022-070352_r200_lda_block"), A.fold_mean("lda_r003")), "range",
      note="r200_lda_block 0.859 (in-fold block) and R003 lda_hv 0.878 (train+test block)")
    C("P2", "supervised", "Gaussian naive Bayes", "0.87", SUPM, ex("20261022-071541_r201_gnb_block", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-071541_r201_gnb_block"), note="train+test block version 0.872")
    C("P3", "supervised", "best linear model (penalised logistic, block)", "0.938", SUPM, tb("r249_comparison.csv"),
      lambda A: grp("supervised-linear")["auc_mean"].max(), note="r207_l2_block_tt_wide (block chosen on train+test features)")
    C("P4", "supervised", "degree-2 polynomial logistic", "0.930", SUPM, ex("20261022-071544_r201_poly2lr_block", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-071544_r201_poly2lr_block"))
    C("P5", "supervised", "RBF-SVM (block)", "0.951", SUPM, ex("20261022-071605_r201_svm_block", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-071605_r201_svm_block"))
    C("P6", "supervised", "kNN (block)", "0.902", SUPM, ex("20261022-071710_r201_knn_block", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-071710_r201_knn_block"))
    C("P7", "supervised", "GP classifier ARD (block, sup track)", "0.976", SUPM + " (English text and table)",
      ex("20261022-072331_r202_gpc_ard_block", "fold_scores.csv"), lambda A: A.fold_mean("20261022-072331_r202_gpc_ard_block"),
      note="sup.md Chinese summary says 0.978 = fm's r255_gpc_ard_hv (0.9778, train+test block); use one run per sentence")
    C("P8", "supervised", "tree ensembles at most", "0.914", SUPM, tb("r249_comparison.csv"),
      lambda A: grp("supervised-trees")["auc_mean"].max(), "le", note="best = r203_lgbm_block_nested 0.9136")
    C("P9", "supervised", "LightGBM block nested: mean", "0.9136", SUPM, ex("20261022-074613_r203_lgbm_block_nested", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-074613_r203_lgbm_block_nested"))
    C("P10", "supervised", "LightGBM block nested: sd", "0.0113", SUPM, ex("20261022-074613_r203_lgbm_block_nested", "fold_scores.csv"),
      lambda A: A.fold_sd("20261022-074613_r203_lgbm_block_nested"))
    C("P11", "supervised", "XGBoost block", "0.9033", tb("r249_comparison.csv"), ex("20261022-074610_r203_xgb_block", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-074610_r203_xgb_block"))
    C("P12", "supervised", "XGBoost all 150", "0.8658", tb("r249_comparison.csv"), ex("20261022-074518_r203_xgb_all", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-074518_r203_xgb_all"))
    C("P13", "supervised", "random forest / extra trees (block)", "0.8965 to 0.9038", tb("r249_comparison.csv"),
      ex("20261022-074525_r203_rf_block", "fold_scores.csv"),
      lambda A: (A.fold_mean("20261022-074525_r203_rf_block"), A.fold_mean("20261022-074546_r203_et_block")), "range")
    C("P14", "supervised", "gain from tuning LightGBM (nested - default)", "0.006", SUPM, tb("r249_comparison.csv"),
      lambda A: A.fold_mean("20261022-074613_r203_lgbm_block_nested") - A.fold_mean("20261022-074604_r203_lgbm_block"))
    C("P15", "supervised", "label spreading, best (kNN graph, wide grid)", "0.957", SUPM, ex("20261022-080634_r209_labelspread_widegrid", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-080634_r209_labelspread_widegrid"))
    C("P16", "supervised", "label spreading on all 150 features", "0.62", SUPM, ex("20261022-072134_r204_labelspread_all", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-072134_r204_labelspread_all"))
    C("P17", "supervised", "strict supervised QDA r201 (S005): mean", "0.9999", SUPM, ex("r201", "fold_scores.csv"), lambda A: A.fold_mean("r201"))
    C("P18", "supervised", "strict supervised QDA r201: sd", "0.0002", SUPM, ex("r201", "fold_scores.csv"), lambda A: A.fold_sd("r201"))
    C("P19", "supervised", "r201 OOF log-loss", "0.0146", SUPM, ex("r201"), lambda A: A.s("r201", "oof_logloss"))
    C("P20", "supervised", "r201 nested reg choices span 0.001-0.01", "[0.001, 0.003, 0.01]", SUPM, ex("r201"),
      lambda A: sorted(json.loads(k)["model__reg_param"] for k in A.s("r201", "chosen_params_top")), "eq",
      note="0.003 in 49 folds, 0.001 in 41, 0.01 in 10")
    C("P21", "supervised", "r210 (log-loss inner): mean", "1.0000", SUPM, ex("r210", "fold_scores.csv"), lambda A: A.fold_mean("r210"),
      note="0.99998; 1 of 100 folds below 1.0")
    C("P22", "supervised", "r210: sd", "0.0001", SUPM, ex("r210", "fold_scores.csv"), lambda A: A.fold_sd("r210"))
    C("P23", "supervised", "r210 OOF log-loss", "0.0087", SUPM, ex("r210"), lambda A: A.s("r210", "oof_logloss"))
    C("P24", "supervised", "r210 picks reg 0.001 (grid floor) in every fold", "100", SUPM, ex("r210"),
      lambda A: A.s("r210", "chosen_params_top").get('{"model__reg_param": 0.001}', 0), "eq")
    C("P25", "supervised", "selection-aware estimate r205: mean", "0.9995", SUPM, ex("r205", "fold_scores.csv"), lambda A: A.fold_mean("r205"))
    C("P26", "supervised", "r205: sd", "0.0006", SUPM, ex("r205", "fold_scores.csv"), lambda A: A.fold_sd("r205"))
    C("P27", "supervised", "configurations tried in the sup track", "587", SUPM, tb("r249_comparison.csv"),
      lambda A: int(T.loc[T["group"] != "transductive", "n_configs"].sum()), "eq")
    C("P28", "supervised", "candidates in the sup track", "63", SUPM, tb("r249_comparison.csv"),
      lambda A: int((T["group"] != "transductive").sum()), "eq")
    C("P29", "supervised", "QDA on in-fold top-k |corr| features (full reg grid)", "0.9926", SUPM,
      ex("20261022-075111_r208_qda_corrk_full", "fold_scores.csv"), lambda A: A.fold_mean("20261022-075111_r208_qda_corrk_full"))
    C("P30", "supervised", "linear: top-k |corr| vs block (L1)", "0.86 to 0.93", SUPM, tb("r249_comparison.csv"),
      lambda A: (A.fold_mean("20261022-070353_r200_l1_corrk"), A.fold_mean("20261022-070236_r200_l1_block")), "range")
    C("P31", "supervised", "no method exceeded 0.88 on all 150 features", "0.88", SUPM + " (Text for the report)",
      tb("r249_comparison.csv") + " + fm runs (r251, r253)",
      lambda A: max(T.loc[T["features"].str.startswith("all 150"), "auc_mean"].max(), A.fold_mean("tabicl_all"),
                    A.fold_mean("tabpfn_all"), A.fold_mean("hs_all")), "le",
      note="true inside the sup track (max 0.875, r208_qda_all_lw) but false for the project: on all 150 features "
           "TabICL v2 0.9929, TabPFN v2 0.9851, horseshoe logistic 0.9261 (fm). Write 'no sup-track model'")
    C("P32", "supervised", "in-fold block size (mean columns)", "19.1", SUPM, ex("r201"), lambda A: A.s("r201", "support_size_mean"))
    C("P33", "supervised", "self-training lifts untuned QDA-LW: supervised QDA-LW", "0.9962", SUPM,
      ex("20261022-082044_r210_sup_qda_tt_lw", "fold_scores.csv"), lambda A: A.fold_mean("20261022-082044_r210_sup_qda_tt_lw"))
    C("P34", "supervised", "self-training QDA-LW (S006) CV", "1.0000", SUPM, ex("r209", "fold_scores.csv"), lambda A: A.fold_mean("r209"))
    C("P35", "supervised", "log-loss QDA-LW supervised -> self-trained", "0.082 to 0.0071", SUPM, ex("r209"),
      lambda A: (A.s("20261022-082044_r210_sup_qda_tt_lw", "oof_logloss"), A.s("r209", "oof_logloss")), "range")
    C("P36", "supervised", "log-loss QDA reg 0.003 supervised -> self-trained", "0.0109 to 0.0120", SUPM, tb("r249_comparison.csv"),
      lambda A: (A.s("20261022-080546_r209_sup_qda_tt_r003", "oof_logloss"), A.s("20261022-080549_r209_selftrain_qda_r003", "oof_logloss")),
      "range", status="confirmed; not nested-honest", note="reg 0.003 = modal R201 choice over all outer folds (sup.md discloses)")
    C("P37", "supervised", "self-training with an LR base vs supervised LR", "0.878 to 0.898", SUPM, tb("r249_comparison.csv"),
      lambda A: (A.fold_mean("20261022-072003_r204_selftrain_lr"), A.fold_mean("lr_r003")), "range")
    C("P38", "supervised", "untuned QDA-LW on the in-fold block", "0.986", SUPM, ex("20261022-075228_r208_qda_block_lw", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-075228_r208_qda_block_lw"))
    C("P39", "supervised", "QDA-LW on all 150 features", "0.875", SUPM, ex("20261022-075225_r208_qda_all_lw", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-075225_r208_qda_all_lw"))
    C("P40", "supervised", "LDA / L2 on all 150 features", "0.796 to 0.787", SUPM, tb("r249_comparison.csv"),
      lambda A: (A.fold_mean("20261022-070235_r200_lda_all"), A.fold_mean("20261022-065651_r200_l2_all")), "range")
    C("P41", "supervised", "L1 logistic all 150 (sup, nested C)", "0.8693", SUPM, ex("20261022-065631_r200_l1_all", "fold_scores.csv"),
      lambda A: A.fold_mean("20261022-065631_r200_l1_all"))
    C("P42", "supervised", "Spearman with S003 on 204 boundary rows: self-training (r209)", "0.70", SUPM, tb("r932_pairwise_agreement.csv"),
      lambda A: A.table("r932_pairwise_agreement.csv").query("a == 's003_gmm2_hv' and b == 'r209_selftrain_qda_lw'")["spearman_s003_boundary"].iloc[0])
    C("P43", "supervised", "Spearman with S003 on boundary rows: supervised QDA (r201, r210)", "0.45 to 0.49", SUPM, tb("r932_pairwise_agreement.csv"),
      lambda A: tuple(A.table("r932_pairwise_agreement.csv").query("a == 's003_gmm2_hv' and b in ['r201_qda_block', 'r210_qda_block_nll']")
                      .sort_values("spearman_s003_boundary")["spearman_s003_boundary"]), "range")
    C("P44", "supervised", "GMM OOF log-loss", "0.0171", SUPM, tb("r249_comparison.csv"),
      lambda A: T.loc[T["group"] == "transductive", "oof_logloss"].iloc[0], status="confirmed (R249 table only)",
      note="the S003 run folder has no oof_logloss; R249 recomputed it")
    # paired differences in the sup table (recomputed from fold scores)
    for cid, run, val_d, val_p in [("P45", "r201", "-0.0001", "0.77"), ("P46", "20261022-080551_r209_sup_qda_tt_r01", "-0.0042", "0.12"),
                                   ("P47", "gmm", "0.0000", "0.84"), ("P48", "r205", "-0.0004", "0.56")]:
        C(cid + "a", "supervised", f"paired diff vs r210: {run}", val_d, SUPM, ex(run, "fold_scores.csv") + " + r210",
          lambda A, r=run: A.paired(r, "r210")["mean_diff"])
        C(cid + "b", "supervised", f"corrected p vs r210: {run}", val_p, SUPM, ex(run, "fold_scores.csv") + " + r210",
          lambda A, r=run: A.paired(r, "r210")["corrected_p"])
    C("P49", "supervised", "D009: r210 vs GMM paired 95% interval half-width", "1.6e-4", "D009 wording ('about 2-5e-4')",
      tb("r930_paired_vs_gmm.csv"), lambda A: _ci_half(A, "r210", "gmm"), note="r201: see P50; the two together give 'about 2-5e-4'")
    C("P50", "supervised", "D009: r201 vs GMM paired 95% interval half-width", "5.1e-4", "D009 wording", ex("r201", "fold_scores.csv"),
      lambda A: _ci_half(A, "r201", "gmm"))
    C("P51", "supervised", "D009: folds where r210 / r201 are worse than the GMM", "(1, 4)", "D009 wording ('1-4')", ex("r201", "fold_scores.csv"),
      lambda A: (int((A.folds("r210")["auc"] < 1).sum()), int((A.folds("r201")["auc"] < 1).sum())), "eq",
      note="the GMM is 1.0 in all 100 folds, so 'worse' = folds below 1.0; better in 0 folds")

    # ---------------------------------------------------------------- 4b validation protocol / leakage audit
    e1 = lambda A, c: A.table("e1_paired_difference.csv").query("scheme == '5foldx20'")[c].iloc[0]  # noqa: E731
    C("V1", "validation", "E1: sd of the paired difference under 5x20 CV (R940 reproduces 0.003056)", "0.003056",
      VER + " F1", tb("e1_paired_difference.csv"), lambda A: e1(A, "std_diff"))
    C("V2", "validation", "E1: 5x20 minimum detectable difference (2 sd, ~50% power)", "0.0061", VER + " F1", tb("e1_paired_difference.csv"),
      lambda A: e1(A, "mdd_2sd"))
    C("V3", "validation", "E1: 95% interval of that MDD", "0.0051 to 0.0076", VER + " F1", tb("e1_paired_difference.csv"),
      lambda A: (e1(A, "mdd_lo"), e1(A, "mdd_hi")), "range")
    na = lambda A: A.table("r900_perm_null_A_summary.csv")  # noqa: E731
    C("V4", "validation", "Null A: permuted-label CV AUC of the 7 honest pipelines", "0.477 to 0.496", VER + " D1",
      tb("r900_perm_null_A_summary.csv"),
      lambda A: (na(A).loc[~na(A)["pipeline"].str.startswith("ctrl"), "perm_auc_mean"].min(),
                 na(A).loc[~na(A)["pipeline"].str.startswith("ctrl"), "perm_auc_mean"].max()), "range")
    C("V5", "validation", "Null A: positive control (E3 leaky pipeline)", "0.697", VER + " D1", tb("r900_perm_null_A_summary.csv"),
      lambda A: na(A).query("pipeline == 'ctrl_e3_leaky'")["perm_auc_mean"].iloc[0])
    C("V6", "validation", "Null B: positive control (disk reader) AUC", "1.000", VER + " D1", tb("r900_perm_null_B_summary.csv"),
      lambda A: A.table("r900_perm_null_B_summary.csv").query("pipeline == 'ctrl_disk_reader'")["perm_auc_mean"].iloc[0])
    C("V7", "validation", "S003 boundary rows (score in (0.1, 0.9)) / share of test rows", "204", "S003 note ('~1% ambiguous'); fm.md",
      ex("bqda"), lambda A: A.s("bqda", "agreement_with_s003", "n_boundary"), "eq", note="204 / 19,750 = 1.03%")

    # ---------------------------------------------------------------- 5 foundation models / Bayesian (fm.md)
    for cid, key, m, sd, ll in [("F1", "tabpfn_all", "0.9851", "0.0041", "0.167"), ("F2", "tabpfn_hv", "0.9888", "0.0031", "0.146"),
                                ("F3", "tabicl_all", "0.9929", "0.0021", "0.120"), ("F4", "tabicl_hv", "0.9989", "0.0008", "0.047"),
                                ("F5", "hs_all", "0.9261", "0.0070", "0.347"), ("F6", "hs_quad", "0.9673", "0.0054", "0.248"),
                                ("F7", "bqda", "1.0000", "0.0000", "0.0094"), ("F8", "bqda_train", "0.9990", "0.0011", "0.041"),
                                ("F9", "gpc_fm", "0.9778", "0.0046", "0.248"), ("F10", "dpgmm", "1.0000", "0.0000", "0.0171")]:
        C(cid + "a", "fm", f"{R[key][16:]}: CV AUC mean", m, FMM, ex(key, "fold_scores.csv"), lambda A, k=key: A.fold_mean(k))
        C(cid + "b", "fm", f"{R[key][16:]}: sd over repeats", sd, FMM, ex(key, "fold_scores.csv"), lambda A, k=key: A.fold_sd(k))
        C(cid + "c", "fm", f"{R[key][16:]}: CV log-loss (fold mean)", ll, FMM, ex(key), lambda A, k=key: A.s(k, "cv_logloss"),
          note="fold-mean log-loss, not OOF; do not compare with sup's OOF log-loss (r209 OOF 0.0071)" if key == "bqda" else "")
    C("F11", "fm", "BQDA plug-in control: CV log-loss", "0.0095", FMM, ex("bqda_plugin"), lambda A: A.s("bqda_plugin", "cv_logloss"))
    C("F12", "fm", "classical QDA reg 0.01 (block)", "0.9998", FMM, ex("qda001", "fold_scores.csv"), lambda A: A.fold_mean("qda001"))
    C("F13", "fm", "classical QDA reg 0.03", "0.9990", FMM, ex("qda003", "fold_scores.csv"), lambda A: A.fold_mean("qda003"),
      note="archived 0.998950 rounds to 0.9989 (fm.md prints 0.9990); immaterial")
    C("F14", "fm", "classical QDA reg 0.3", "0.9832", FMM, ex("qda03", "fold_scores.csv"), lambda A: A.fold_mean("qda03"))
    C("F15", "fm", "horseshoe all vs L1 (R003): paired diff", "0.079", FMM, ex("hs_all", "fold_scores.csv") + " + l1_r003",
      lambda A: A.paired("hs_all", "l1_r003")["mean_diff"])
    C("F16", "fm", "horseshoe all vs L1: split SE", "0.0025", FMM, ex("hs_all", "fold_scores.csv"), lambda A: A.paired("hs_all", "l1_r003")["se_split"])
    C("F17", "fm", "horseshoe all vs L1: corrected p", "0.0005", FMM, ex("hs_all", "fold_scores.csv"), lambda A: A.paired("hs_all", "l1_r003")["corrected_p"])
    C("F18", "fm", "horseshoe all vs L1: fold win rate", "0.99", FMM, ex("hs_all", "fold_scores.csv"), lambda A: A.paired("hs_all", "l1_r003")["win_rate_fold"])
    C("F19", "fm", "horseshoe all vs block L2 (R003): paired diff", "0.028", FMM, ex("hs_all", "fold_scores.csv") + " + lr_r003",
      lambda A: A.paired("hs_all", "lr_r003")["mean_diff"])
    C("F20", "fm", "horseshoe all vs block L2: corrected p", "0.03", FMM, ex("hs_all", "fold_scores.csv"), lambda A: A.paired("hs_all", "lr_r003")["corrected_p"])
    C("F21", "fm", "'horseshoe 0.926 is the best linear model'", "0.926", FMM + " (Text for the report)", tb("r249_comparison.csv"),
      lambda A: T.loc[T["group"] == "supervised-linear", "auc_mean"].max(), "le",
      note="overstated: best linear on ALL 150 features only; on the block, L2 logistic reaches 0.938 (r207_l2_block_tt_wide)")
    C("F22", "fm", "BQDA vs QDA reg 0.1: paired diff", "0.0042", FMM, ex("bqda", "fold_scores.csv") + " + qda_r003",
      lambda A: A.paired("bqda", "qda_r003")["mean_diff"])
    C("F23", "fm", "BQDA vs QDA reg 0.1: split SE", "0.00025", FMM, ex("bqda", "fold_scores.csv"), lambda A: A.paired("bqda", "qda_r003")["se_split"])
    C("F24", "fm", "BQDA vs QDA reg 0.1: corrected p", "0.12", FMM, ex("bqda", "fold_scores.csv"), lambda A: A.paired("bqda", "qda_r003")["corrected_p"])
    agr = lambda A, k, f: A.s(k, "agreement_with_s003", f)  # noqa: E731
    C("F25", "fm", "BQDA: same side of 0.5 as S003 on the 204 boundary rows", "74%", FMM, ex("bqda"), lambda A: agr(A, "bqda", "side_agree_boundary"), "pct",
      note="73.5%")
    C("F26", "fm", "BQDA: Spearman with S003 on boundary rows", "0.55", FMM, ex("bqda"), lambda A: agr(A, "bqda", "spearman_boundary"))
    C("F27", "fm", "BQDA: same side as S003 on all test rows", "99.5%", FMM, ex("bqda"), lambda A: agr(A, "bqda", "side_agree_all"), "pct",
      note="archived 99.554% rounds to 99.6% (fm.md truncates); immaterial")
    C("F28", "fm", "BQDA: pseudo-AUC vs S003's confident labels", "0.99998", FMM, ex("bqda"), lambda A: agr(A, "bqda", "pseudo_auc_confident"))
    C("F29", "fm", "BQDA train-only: boundary side agreement", "63%", FMM, ex("bqda_train"), lambda A: agr(A, "bqda_train", "side_agree_boundary"), "pct")
    C("F30", "fm", "TabICL block / all: boundary side agreement", "57% to 51%", FMM, ex("tabicl_hv"),
      lambda A: (agr(A, "tabicl_hv", "side_agree_boundary") * 100, agr(A, "tabicl_all", "side_agree_boundary") * 100), "range")
    C("F31", "fm", "TabPFN block / all: boundary side agreement", "49% to 51%", FMM, ex("tabpfn_hv"),
      lambda A: (agr(A, "tabpfn_hv", "side_agree_boundary") * 100, agr(A, "tabpfn_all", "side_agree_boundary") * 100), "range")
    C("F32", "fm", "FM Spearman with S003 on boundary rows", "0.19 to 0.38", FMM, ex("tabicl_hv"),
      lambda A: (min(agr(A, k, "spearman_boundary") for k in ("tabpfn_all", "tabpfn_hv", "tabicl_all", "tabicl_hv")),
                 max(agr(A, k, "spearman_boundary") for k in ("tabpfn_all", "tabpfn_hv", "tabicl_all", "tabicl_hv"))), "range")
    C("F33", "fm", "FM test rows with score in (0.1, 0.9)", "6% to 23%", FMM, ex("tabpfn_all"),
      lambda A: (100 * min(agr(A, k, "frac_pred_in_0.1_0.9") for k in ("tabpfn_all", "tabpfn_hv", "tabicl_all", "tabicl_hv")),
                 100 * max(agr(A, k, "frac_pred_in_0.1_0.9") for k in ("tabpfn_all", "tabpfn_hv", "tabicl_all", "tabicl_hv"))), "range")
    C("F34", "fm", "DP-GMM: same side as S003 on boundary rows", "99%", FMM, ex("dpgmm"), lambda A: agr(A, "dpgmm", "side_agree_boundary"), "pct")
    C("F35", "fm", "DP-GMM: effective components", "2", FMM, ex("dpgmm"), lambda A: A.s("dpgmm", "mixture", "n_eff_1pct"), "eq")
    ev = lambda A: A.s("dp_evid", "evidence")  # noqa: E731
    C("F36", "fm", "plug-in BIC: DP-GMM (T=10, gamma=1) vs K=2", "343920 to 343911", FMM, ex("dp_evid"),
      lambda A: (next(e["bic_plugin"] for e in ev(A) if e["model"].startswith("BGMM dirichlet_process T=10 gamma=1.0")),
                 next(e["bic_plugin"] for e in ev(A) if e["model"] == "GMM K=2")), "range")
    C("F37", "fm", "EB choice kappa0 = 1 (grid edge) in 100/100 folds", "100", VER + " 7.1", tb("r255_eb_choice_frequency.csv"),
      lambda A: int(A.table("r255_eb_choice_frequency.csv").query("kappa0 == 1.0")["n_folds"].sum()), "eq",
      note="kappa grid (0.001, 0.01, 0.1, 1.0): the choice sits at the upper edge")
    rep = lambda A, m: A.table("r254_replica_auc_summary.csv").query("generator == 'G1_gmm2' and model == @m")  # noqa: E731
    C("F38", "fm", "replicas: GMM AUC on the GMM boundary rows", "0.825", FMM, tb("r254_replica_auc_summary.csv"),
      lambda A: rep(A, "GMM K=2")["auc_on_gmm_boundary_rows"].iloc[0], status="confirmed; 6 replicas only (verify C12)")
    C("F39", "fm", "replicas: BQDA AUC on the GMM boundary rows", "0.702", FMM, tb("r254_replica_auc_summary.csv"),
      lambda A: rep(A, "BayesQDA (250 labels only)")["auc_on_gmm_boundary_rows"].iloc[0], status="confirmed; 6 replicas only")
    C("F40", "fm", "replicas: overall test AUC GMM vs BQDA", "0.9949 to 0.9948", FMM, tb("r254_replica_auc_summary.csv"),
      lambda A: (rep(A, "GMM K=2")["test_auc"].iloc[0], rep(A, "BayesQDA (250 labels only)")["test_auc"].iloc[0]), "range",
      status="confirmed; 6 replicas only")
    C("F41", "fm", "L1 logistic all 150 (R003 = S001)", "0.8469", FMM, ex("l1_r003", "fold_scores.csv"), lambda A: A.fold_mean("l1_r003"))
    C("F42", "fm", "BQDA test rows exactly 1.0", "1493", FMM, ex("bqda", "test_pred.npy"), lambda A: "not packed", "eq",
      status="not checked (needs test_pred.npy)")

    # ---------------------------------------------------------------- 6 replicas / final selection (r936)
    pv = lambda A, proc: A.table("r936_private.csv").query(  # noqa: E731
        "group == 'tn0' and scores == 'csv8' and flip_y == 'weighted_r156' and proc == @proc")["diff_vs_s003_mean"].iloc[0]
    for cid, proc, val in [("X1", "r209_selftrain_qda_lw", "-6.9e-5"), ("X2", "r255_bqda_eb_hv", "-1.16e-4"),
                           ("X3", "r210_qda_block_nll", "-1.68e-4"), ("X4", "r201_qda_block", "-2.57e-4"),
                           ("X5", "x_qda_block_nll_ext", "-6.3e-5")]:
        C(cid, "replicas", f"private AUC difference vs S003 on 222 replicas: {proc}", val, VER + " C8", tb("r936_private.csv"),
          lambda A, p=proc: pv(A, p), note="tn0, CSV-formatted scores, R156 level weights")
    ch = lambda A, proc: A.table("r936_choice.csv").query("second_selection == @proc")["gain_vs_s003_alone"].iloc[0]  # noqa: E731
    for cid, proc, val in [("X6", "r209_selftrain_qda_lw", "1.45e-5"), ("X7", "r255_bqda_eb_hv", "1.37e-5"),
                           ("X8", "r210_qda_block_nll", "5.0e-6"), ("X9", "r201_qda_block", "4.1e-6")]:
        C(cid, "replicas", f"gain of the pair S003 + {proc} over S003 alone", val, VER + " §6", tb("r936_choice.csv"), lambda A, p=proc: ch(A, p))
    cvp = lambda A, proc, c: A.table("r936_cv_power.csv").query("proc == @proc")[c].iloc[0]  # noqa: E731
    C("X10", "replicas", "CV paired test non-significant in 222/222 replicas (r201, r210)", "1.0", VER + " C5", tb("r936_cv_power.csv"),
      lambda A: min(cvp(A, p, "share_cv_not_significant") for p in ("r201_qda_block", "r210_qda_block_nll")))
    C("X11", "replicas", "true gap negative in 100% of replicas (r201, r210)", "1.0", VER + " C5", tb("r936_cv_power.csv"),
      lambda A: min(cvp(A, p, "share_true_gap_negative") for p in ("r201_qda_block", "r210_qda_block_nll")))
    ab = lambda A, proc, c: A.table("r936_public_abc.csv").query("scores == 'csv8' and window == 0.00015 and proc == @proc")[c].iloc[0]  # noqa: E731
    C("X12", "replicas", "S006 (r209) predicted public given S003's 0.99923", "0.99913", VER + " §5 / B007", tb("r936_public_abc.csv"),
      lambda A: ab(A, "r209_selftrain_qda_lw", "public_mean"))
    C("X13", "replicas", "S006 predicted public sd", "0.00033", VER + " §5", tb("r936_public_abc.csv"), lambda A: ab(A, "r209_selftrain_qda_lw", "public_pred_sd"))
    C("X14", "replicas", "S005 (r201) predicted public", "0.99884", VER + " §5 / B006", tb("r936_public_abc.csv"), lambda A: ab(A, "r201_qda_block", "public_mean"))
    C("X15", "replicas", "S005 predicted public sd", "0.00066", VER + " §5", tb("r936_public_abc.csv"), lambda A: ab(A, "r201_qda_block", "public_pred_sd"))
    C("X16", "replicas", "P(S006 public > S003 public)", "0.28", VER + " §5", tb("r936_public_abc.csv"), lambda A: ab(A, "r209_selftrain_qda_lw", "p_public_higher"))
    C("X17", "replicas", "P(S005 public > S003 public)", "0.20", VER + " §5", tb("r936_public_abc.csv"), lambda A: ab(A, "r201_qda_block", "p_public_higher"))

    # ---------------------------------------------------------------- 7 label budget (R206 as cited in sup.md; R304 in E2)
    r206 = A.table("r206_label_budget.csv")
    g6 = lambda m, n, c="auc_mean": r206.query("model == @m and n_labels == @n")[c].iloc[0]  # noqa: E731
    C("L1", "label budget", "R206: GMM AUC at 4 labels", "1.000", SUPM, tb("r206_label_budget.csv"), lambda A: g6("trans_gmm2", 4))
    C("L2", "label budget", "R206: self-training QDA-LW (thr 0.75) at 20 labels", "0.995", SUPM, tb("r206_label_budget.csv"),
      lambda A: g6("ssl_selftrain_lw", 20), note="first budget >= 0.99; threshold 0.75, not the S006 setting 0.99")
    C("L3", "label budget", "R206: QDA reg 0.003, train+test block, at 40 labels", "0.9933", tb("r206_label_budget.md"), tb("r206_label_budget.csv"),
      lambda A: g6("sup_qda_tt_r003", 40), note="reg 0.003 is informed by R201's nested choice")
    C("L4", "label budget", "R206: strict QDA reg 0.003 (in-fold block) at 80 labels", "0.9937", tb("r206_label_budget.md"), tb("r206_label_budget.csv"),
      lambda A: g6("sup_qda_infold_r003", 80))
    C("L5", "label budget", "R206: strict QDA reg 0.003 at 40 labels (defined evaluations)", "6", tb("r206_label_budget.csv"), tb("r206_label_budget.csv"),
      lambda A: int(g6("sup_qda_infold_r003", 40, "n")), "eq",
      note="the 0.9476 in r206_label_budget.md at 40 labels is the mean of only 6 of 60 evaluations (54 undefined)")
    C("L6", "label budget", "R206: best supervised model at 20 labels", "0.851", SUPM, tb("r206_label_budget.csv"),
      lambda A: r206.query("n_labels == 20 and model.str.startswith('sup_')", engine="python")["auc_mean"].max())
    C("L7", "label budget", "R304 (E2): GMM AUC >= 0.99 with 2 labels in 50/50 subsets", "1.0", "verify.md C13; ext12.md", tb("r304_label_requirement.csv"),
      lambda A: A.table("r304_label_requirement.csv").query("model == 'gmm2_hv' and n_labels == 2")["frac_auc_ge_0.99"].iloc[0])
    C("L8", "label budget", "R304 (E2): QDA reg 0.1 median AUC >= 0.99 from", "75", "verify.md C13", tb("r304_label_requirement.csv"),
      lambda A: int(A.table("r304_label_requirement.csv").query("model == 'qda_hv' and median >= 0.99")["n_labels"].min()), "eq")


def _ci_half(A, a, b):
    """half-width of the Nadeau-Bengio 95% interval of the paired mean difference (fold scores)."""
    from scipy import stats
    pc = A.paired(a, b)
    d = (A.folds(a).merge(A.folds(b), on=["repeat", "fold"], suffixes=("_a", "_b")).eval("auc_a - auc_b")).to_numpy()
    se = math.sqrt((1 / len(d) + 50 / 200) * d.var(ddof=1))
    assert abs(pc["mean_diff"] - d.mean()) < 1e-12
    return stats.t.ppf(0.975, len(d) - 1) * se


def run_claims(A: Archive) -> pd.DataFrame:
    rows = []
    for c in CLAIMS:
        try:
            v = c["get"](A)
            ok = check(c["value"], v, c["mode"])
            arch, err = (str(tuple(v)) if c["mode"] == "eq" and isinstance(v, tuple) else _fmt(v)), ""
        except Exception as e:  # noqa: BLE001
            ok, arch, err = False, "", f"error: {type(e).__name__}: {e}"
        status = c["status"] or ("confirmed" if ok else "mismatch")
        if not ok and c["status"] and not c["status"].startswith(("external", "not ")):
            status = "mismatch; " + c["status"]
        if c["mode"] == "le" and not ok and "overstated" not in status:
            status = "refuted" if not c["status"] else status
        note = "; ".join(x for x in (c["note"], err) if x)
        rows.append({"id": c["id"], "section": c["section"], "claim": c["claim"], "value": c["value"],
                     "archived_value": arch, "source_file": c["source"], "verified": "yes" if ok else "no",
                     "status": status, "note": note, "claimed_in": c["claimed_in"], "check": c["mode"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ r249 recheck
def r249_recheck(A: Archive) -> pd.DataFrame:
    T = A.table("r249_comparison.csv")
    ref = "20261022-082047_r210_qda_block_nll"
    out = []
    for _, r in T.iterrows():
        run = r["run_id"]
        s = A.summary(run)
        m, sd = A.fold_mean(run), A.fold_sd(run)
        pc = A.paired(run, ref)
        row = {"candidate": r["candidate"], "run_id": run, "group": r["group"],
               "table_auc_mean": r["auc_mean"], "fold_auc_mean": m, "summary_auc_mean": s.get("auc_mean"),
               "table_auc_sd": r["auc_std_repeat"], "fold_auc_sd": sd,
               "table_oof_logloss": r["oof_logloss"], "summary_oof_logloss": s.get("oof_logloss"),
               "table_diff": r["diff_vs_best_sup"], "fold_diff": pc["mean_diff"],
               "table_se_split": r["se_split_vs_best_sup"], "fold_se_split": pc["se_split"],
               "table_p_corr": r["p_corr_vs_best_sup"], "fold_p_corr": pc["corrected_p"],
               "summary_uses_test_features": s.get("uses_unlabeled_test_features"),
               "table_uses_test_features": r["uses_unlabeled_test_features"]}
        row["ok_mean"] = abs(row["table_auc_mean"] - m) < 5e-5 and (row["summary_auc_mean"] is None or abs(row["summary_auc_mean"] - m) < 1e-9)
        row["ok_sd"] = abs(row["table_auc_sd"] - sd) < 5e-5
        row["ok_logloss"] = (pd.isna(row["table_oof_logloss"]) and row["summary_oof_logloss"] is None) or (
            row["summary_oof_logloss"] is not None and abs(row["table_oof_logloss"] - row["summary_oof_logloss"]) < 5e-5) or (
            row["summary_oof_logloss"] is None and r["group"] == "transductive")
        row["ok_paired"] = (abs(row["table_diff"] - pc["mean_diff"]) < 5e-5 and abs(row["table_se_split"] - pc["se_split"]) < 5e-5
                            and abs(row["table_p_corr"] - pc["corrected_p"]) <= max(5e-4, 0.005 * pc["corrected_p"]))
        row["ok_all"] = bool(row["ok_mean"] and row["ok_sd"] and row["ok_logloss"] and row["ok_paired"])
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ label budget
R206_OF = {"eq206_qda_infold_r003": "sup_qda_infold_r003", "eq206_qda_infold_lw": "sup_qda_infold_lw",
           "eq206_qda_tt_r003": "sup_qda_tt_r003", "eq206_qda_tt_r01": "sup_qda_tt_r01",
           "eq206_selftrain_lw_t075": "ssl_selftrain_lw", "s003_gmm2_hv": "trans_gmm2"}


def budget_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, n), g in df.groupby(["model", "n_labels"]):
        ok = g.dropna(subset=["auc"])
        fm = ok.groupby(["repeat", "fold"])["auc"].mean()
        rows.append({"model": m, "n_labels": int(n), "n_evals": len(g), "n_failed": int(g["auc"].isna().sum()),
                     "mean": ok["auc"].mean() if len(ok) else np.nan, "median": ok["auc"].median() if len(ok) else np.nan,
                     "se": fm.std(ddof=1) / math.sqrt(len(fm)) if len(fm) > 1 else np.nan, "n_folds": len(fm),
                     "frac_ge_099": float((ok["auc"] >= 0.99).mean()) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def first_hit(st: pd.DataFrame, model: str, col="mean", budgets=None):
    t = st[(st["model"] == model) & (st["n_failed"] == 0)].sort_values("n_labels")
    if budgets is not None:
        t = t[t["n_labels"].isin(budgets)]
    h = t.loc[t[col] >= 0.99, "n_labels"]
    return int(h.iloc[0]) if len(h) else None


def label_budget(A: Archive):
    e206 = A.csv(ex("r206", "label_budget_evals.csv"))
    e441 = A.csv(rel("experiments", A.r441, "label_budget_evals.csv"))
    s206, s441 = budget_stats(e206), budget_stats(e441)
    inv = {v: k for k, v in R206_OF.items()}
    s206 = s206[s206["model"].isin(inv)].assign(model=lambda d: d["model"].map(inv))
    cmp_ = s441.merge(s206, on=["model", "n_labels"], how="inner", suffixes=("_r441", "_r206"))
    cmp_["diff"] = cmp_["mean_r441"] - cmp_["mean_r206"]
    cmp_["se_diff"] = np.sqrt(cmp_["se_r441"].fillna(0) ** 2 + cmp_["se_r206"].fillna(0) ** 2)
    cmp_["z"] = np.where(cmp_["se_diff"] > 0, cmp_["diff"] / cmp_["se_diff"].where(cmp_["se_diff"] > 0), np.where(cmp_["diff"].abs() < 1e-12, 0.0, np.inf))
    both_full = (cmp_["n_failed_r441"] == 0) & (cmp_["n_failed_r206"] == 0)
    cmp_["comparable"] = both_full
    cmp_["flag_3se"] = both_full & (cmp_["z"].abs() > 3)
    cmp_.insert(1, "r206_model", cmp_["model"].map(R206_OF))
    budgets206 = sorted(e206["n_labels"].unique())
    hits = []
    for m in sorted(set(s441["model"])):
        r = {"model": m, "r206_model": R206_OF.get(m, ""),
             "r441_first_mean_ge_099": first_hit(s441, m), "r441_first_mean_ge_099_on_r206_grid": first_hit(s441, m, budgets=budgets206),
             "r441_first_median_ge_099": first_hit(s441, m, "median"),
             "r206_first_mean_ge_099": first_hit(s206, m) if m in R206_OF else None}
        t = s441[(s441["model"] == m)].sort_values("n_labels")
        dfd = t.loc[t["n_failed"] < t["n_evals"], "n_labels"]
        r["r441_first_defined"] = int(dfd.iloc[0]) if len(dfd) else None
        full = t.loc[t["n_failed"] == 0, "n_labels"]
        r["r441_first_fully_defined"] = int(full.iloc[0]) if len(full) else None
        for n in (2, 4, 20, 40, 80):
            q = t[t["n_labels"] == n]
            r[f"r441_mean_at_{n}"] = float(q["mean"].iloc[0]) if len(q) and q["n_failed"].iloc[0] == 0 else np.nan
        hits.append(r)
    hits = pd.DataFrame(hits)
    r304 = A.table("r304_label_requirement.csv")
    return cmp_, hits, s441, s206, r304


def label_budget_figure(path, s441, s206, r304):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    lab = {"s003_gmm2_hv": "Transductive GMM (S003)", "eq206_selftrain_lw_t075": "Self-training QDA-LW, thr 0.75",
           "eq206_qda_tt_r003": "QDA reg 0.003, train+test block", "eq206_qda_infold_r003": "QDA reg 0.003, in-fold block (strict)",
           "eq206_qda_infold_lw": "QDA-LW, in-fold block (strict)", "eq206_qda_tt_r01": "QDA reg 0.1, train+test block (S002)",
           "r209_selftrain_qda_lw": "Self-training QDA-LW, thr 0.99 (S006)", "r201_qda_block": "Strict QDA, nested reg (S005)",
           "r255_bqda_eb_trainonly": "Bayesian QDA, in-fold block (strict)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    ax = axes[0]
    for i, m in enumerate(["s003_gmm2_hv", "eq206_selftrain_lw_t075", "eq206_qda_tt_r003", "eq206_qda_infold_r003", "eq206_qda_infold_lw"]):
        c = "#0b0b0b" if m == "s003_gmm2_hv" else slots[i]
        for st, ls, mk, tag in ((s441, "-", "o", "R441"), (s206, "--", "s", "R206")):
            d = st[(st["model"] == m) & (st["n_failed"] == 0)].sort_values("n_labels")
            ax.plot(d["n_labels"], d["mean"], ls, color=c, lw=1.8 if tag == "R441" else 1.2, marker=mk, ms=3.5,
                    label=f"{lab[m]} ({tag})" if tag == "R441" else None, alpha=1 if tag == "R441" else 0.7)
    ax.plot([], [], "--", color="#8a8984", label="dashed = R206 (repeats 0-3)")
    ax.set_title("Same models, new folds and label draws: R441 (solid) vs R206 (dashed)", fontsize=9.5)
    ax = axes[1]
    for i, m in enumerate(["s003_gmm2_hv", "r209_selftrain_qda_lw", "r201_qda_block", "r255_bqda_eb_trainonly", "eq206_qda_tt_r01"]):
        c = "#0b0b0b" if m == "s003_gmm2_hv" else slots[i]
        d = s441[(s441["model"] == m) & (s441["n_failed"] == 0)].sort_values("n_labels")
        ax.plot(d["n_labels"], d["mean"], "-", color=c, lw=1.8, marker="o", ms=3.5, label=lab[m])
        ax.fill_between(d["n_labels"], d["mean"] - 2 * d["se"], np.minimum(1, d["mean"] + 2 * d["se"]), color=c, alpha=0.12, lw=0)
    for m, c, txt in (("gmm2_hv", "#0b0b0b", "R304 GMM (median, out-of-subset)"), ("qda_hv", slots[4], "R304 QDA reg 0.1 (median)")):
        d = r304[(r304["model"] == m) & (r304["n_errors"] == 0)].sort_values("n_labels")
        ax.plot(d["n_labels"], d["median"], ":", color=c, lw=1.2, marker="x", ms=4, label=txt)
    ax.set_title("Candidate procedures (R441) and R304's E2 curves", fontsize=9.5)
    for ax in axes:
        ax.axhline(0.99, color="#8a8984", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 6, 10, 20, 40, 80, 120, 200])
        ax.set_xticklabels(["2", "4", "6", "10", "20", "40", "80", "120", "200"])
        ax.set_ylim(0.5, 1.005)
        ax.set_xlabel("labelled training rows (log scale)")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("validation AUC (mean; only fully defined budgets)")
    fig.suptitle("Value of unlabeled data: labels needed per model (R206, R441, R304)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


# ------------------------------------------------------------------------------------------------ journey
def journey(A: Archive, hm) -> pd.DataFrame:
    T = A.csv("registry/submissions.csv")
    use = {"S001": "labels only", "S002": "block from train+test features", "S003": "yes (transductive)",
           "S004": "labels only", "S005": "labels only (strictly inductive)", "S006": "yes (self-training on test rows)"}
    short = {"S001": "L1 logistic, all 150 features", "S002": "QDA reg 0.1 on the 13-column block",
             "S003": "Transductive GMM K=2 on the block", "S004": "E3 final: in-fold corr top-20 + L2 logistic",
             "S005": "Strict supervised QDA, nested reg (r201)", "S006": "Self-training QDA-LW (r209)"}
    proc = {"S003": "s003_gmm2_hv", "S005": "r201_qda_block", "S006": "r209_selftrain_qda_lw"}
    pv = A.table("r936_private.csv").query("group == 'tn0' and scores == 'csv8' and flip_y == 'weighted_r156'").set_index("proc")
    abc = A.table("r936_public_abc.csv").query("scores == 'csv8' and window == 0.00015").set_index("proc")
    s003_priv = A.s("r922", "posterior_by_prior", "uniform_0_0.1", "pred_private_mean")
    rows = []
    for i, r in T.iterrows():
        sid = r["sub_id"]
        run = r["run_id"]
        cv_m = A.fold_mean(run) if sid != "S004" else A.summary(run)["auc_mean"]
        cv_s = A.fold_sd(run) if sid != "S004" else A.summary(run)["auc_std_repeat"]
        pub = r["public_lb"]
        exp = float(r["expected_public"])
        p = proc.get(sid)
        rows.append({"step": i + 1, "sub_id": sid, "run_id": run, "model": short.get(sid, r["description"]),
                     "uses_test_features": use.get(sid, ""), "cv_auc_mean": round(cv_m, 5), "cv_auc_sd": round(cv_s, 5),
                     "expected_public": exp, "public_lb": None if pd.isna(pub) else float(pub),
                     "public_minus_cv": None if pd.isna(pub) else round(float(pub) - cv_m, 5),
                     "hm_se_public": round(hm(exp), 4),
                     "surprise_se": None if pd.isna(pub) else round((float(pub) - exp) / hm(exp), 2),
                     "replica_public_pred": round(float(abc.loc[p, "public_mean"]), 5) if p in abc.index and sid != "S003" else None,
                     "replica_public_pred_sd": round(float(abc.loc[p, "public_pred_sd"]), 5) if p in abc.index and sid != "S003" else None,
                     "replica_private_diff_vs_s003": float(pv.loc[p, "diff_vs_s003_mean"]) if p in pv.index else None,
                     "status": ("uploaded " + str(r["submitted_at_utc"])[:10]) if not pd.isna(pub) else "created, not uploaded",
                     "note": r["notes"]})
    for run, name, p, use_ in [("20261022-082047_r210_qda_block_nll", "Strict QDA, nested reg, log-loss inner (r210)", "r210_qda_block_nll", "labels only"),
                               ("20261022-070200_r255_bqda_eb_hv", "Bayesian QDA EB, train+test block (r255)", "r255_bqda_eb_hv", "block from train+test features"),
                               ("20261022-070202_r255_bqda_eb_trainonly", "Bayesian QDA EB, in-fold block (r255)", None, "labels only (strictly inductive)")]:
        rows.append({"step": None, "sub_id": "-", "run_id": run, "model": name, "uses_test_features": use_,
                     "cv_auc_mean": round(A.fold_mean(run), 5), "cv_auc_sd": round(A.fold_sd(run), 5), "expected_public": None,
                     "public_lb": None, "public_minus_cv": None, "hm_se_public": None, "surprise_se": None,
                     "replica_public_pred": round(float(abc.loc[p, "public_mean"]), 5) if p in abc.index else None,
                     "replica_public_pred_sd": round(float(abc.loc[p, "public_pred_sd"]), 5) if p in abc.index else None,
                     "replica_private_diff_vs_s003": float(pv.loc[p, "diff_vs_s003_mean"]) if p in pv.index else None,
                     "status": "candidate, not submitted", "note": ""})
    J = pd.DataFrame(rows)
    J.attrs["s003_private_mean"] = s003_priv
    return J


def journey_md(J: pd.DataFrame, run_id: str) -> str:
    def f(v, d=5):
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    lines = ["| Step | Sub | Model | Test features used | CV AUC (sd) | Expected public | Public LB | Public - CV | Surprise (HM SE) | Replica private vs S003 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in J.iterrows():
        word = "pending" if str(r["sub_id"]).startswith("S") else "not submitted"
        pub = f(r["public_lb"]) if r["public_lb"] is not None and not (isinstance(r["public_lb"], float) and math.isnan(r["public_lb"])) else (
            f"{word} (pred. {f(r['replica_public_pred'])} ± {f(r['replica_public_pred_sd'])})" if r["replica_public_pred"] is not None
            and not (isinstance(r["replica_public_pred"], float) and math.isnan(r["replica_public_pred"])) else "-")
        dv = r["replica_private_diff_vs_s003"]
        dvs = "" if dv is None or (isinstance(dv, float) and math.isnan(dv)) else ("0 (reference)" if dv == 0 else f"{dv:+.1e}")
        lines.append(f"| {f(r['step'], 0) if r['step'] is not None and not (isinstance(r['step'], float) and math.isnan(r['step'])) else '-'} | {r['sub_id']} | {r['model']} | "
                     f"{r['uses_test_features']} | {f(r['cv_auc_mean'], 4)} ({f(r['cv_auc_sd'], 4)}) | {f(r['expected_public'], 4)} | {pub} | "
                     f"{f(r['public_minus_cv'], 4)} | {f(r['surprise_se'], 2)} | {dvs} |")
    return ("Journey from the naive baseline to the final candidates (official 5x20 CV; public LB from registry/submissions.csv; "
            "HM SE = Hanley-McNeil SE of the expected public AUC on 395 rows (198/197); surprise = (public - expected) / HM SE; "
            "pending rows show the replica-ABC prediction given S003's public 0.99923 (r936_public_abc, CSV scores); "
            "last column = expected private-AUC difference to S003 on 222 parametric replicas (r936_private, tn0, csv8, R156 weights); "
            f"S003's own private expectation is about 0.997 (R922, flat prior). Run {run_id}.)\n\n" + "\n".join(lines) + "\n")


# ================================================================================================ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", action="store_true")
    p.add_argument("--data", default="raw")
    p.add_argument("--author", default=None)
    a = p.parse_args()
    if a.pack:
        make_pack()
        return
    sys.path.insert(0, HERE)
    from ssc import data, paths
    from ssc.tracking import Run
    from ssc.validation import auc_standard_error
    with gzip.open(PACK, "rt", encoding="utf-8") as fh:
        A = Archive(json.load(fh))
    ds = data.load(a.data)
    y = np.asarray(ds.y).astype(int)
    facts = {"n_train": int(len(y)), "n_test": int(len(ds.X_test)), "n_features": int(ds.X.shape[1]),
             "n_pos": int(y.sum()), "pos_rate": float(y.mean())}
    hm = lambda auc: auc_standard_error(auc, 198, 197)  # noqa: E731
    define_claims(A, facts, hm, auc_standard_error)
    hyp = ("Audit hypothesis (written before running): every CV number in sup.md, fm.md and registry/submissions.csv "
           "re-computes from fold_scores.csv to the printed precision; the generator facts in gen.md match the R150/R154/"
           "R920-R922 summaries; the r249 table reproduces row by row. Expected exceptions are wording, not numbers: "
           "'no method exceeded 0.88 on all 150 features' (fm's TabICL/TabPFN/horseshoe on all 150 features are higher), "
           "'horseshoe = best linear model' (block L2 0.938 is higher), and R206's partial means at budgets where most fits are "
           "undefined. For the label budget, R441 (new folds and draws) gives the same first budget reaching AUC 0.99 as R206 "
           "for every matched model, within one grid step, and no budget with both runs fully defined differs by > 3 SE.")
    cfg = {"name": "r440_core_numbers", "pack": "research/r440_archive_pack.json.gz", "n_packed_files": len(A.files),
           "r441_run": A.r441, "n_claims": len(CLAIMS), "data_facts": facts,
           "hm_se_counts": [198, 197], "code": "research/r440_core_numbers.py"}
    with Run("r440_core_numbers", cfg, kind="analysis", source=a.data, author=a.author, hypothesis=hyp) as run:
        tables, figs = paths.tables_dir(a.data), paths.figures_dir(a.data)
        tables.mkdir(parents=True, exist_ok=True)
        figs.mkdir(parents=True, exist_ok=True)
        claims = run_claims(A)
        rc = r249_recheck(A)
        cmp_, hits, s441, s206, r304 = label_budget(A)
        # replication claims (R206 statements vs R441)
        hit = hits.set_index("model")
        rep = []
        for cid, m, claim, val in [("L9", "s003_gmm2_hv", "R441: GMM first budget with mean AUC >= 0.99", "4"),
                                   ("L10", "eq206_selftrain_lw_t075", "R441: self-training QDA-LW thr 0.75, first budget >= 0.99", "20"),
                                   ("L11", "eq206_qda_tt_r003", "R441: QDA reg 0.003 train+test block, first budget >= 0.99", "40"),
                                   ("L12", "eq206_qda_infold_r003", "R441: strict QDA reg 0.003 in-fold, first budget >= 0.99", "80"),
                                   ("L13", "r209_selftrain_qda_lw", "R441: S006 procedure (self-training thr 0.99), first budget >= 0.99", "20"),
                                   ("L14", "r201_qda_block", "R441: S005 procedure (strict nested QDA), first budget >= 0.99", "80")]:
            got = hit.loc[m, "r441_first_mean_ge_099_on_r206_grid"] if m in hit.index else None
            got_own = hit.loc[m, "r441_first_mean_ge_099"] if m in hit.index else None
            ok = got is not None and not pd.isna(got) and int(got) == int(val)
            rep.append({"id": cid, "section": "label budget", "claim": claim + " (R206 grid)", "value": val,
                        "archived_value": "" if got is None or pd.isna(got) else str(int(got)),
                        "source_file": rel("experiments", A.r441, "label_budget_evals.csv"), "verified": "yes" if ok else "no",
                        "status": ("replicated" if ok else "not replicated") if cid in ("L9", "L10", "L11", "L12")
                        else ("R206 statement holds for the candidate" if ok else "R206 statement does not hold for the candidate"),
                        "check": "eq",
                        "note": f"own grid (incl. 2 labels): {'' if got_own is None or pd.isna(got_own) else int(got_own)}; "
                                f"first fully defined: {hit.loc[m, 'r441_first_fully_defined'] if m in hit.index else ''}",
                        "claimed_in": "research/reports/sup.md (R206)" if cid in ("L9", "L10", "L11", "L12") else "R206 design applied to the candidate"})
        claims = pd.concat([claims, pd.DataFrame(rep)], ignore_index=True)
        n_ok = int(rc["ok_all"].sum())
        claims = pd.concat([claims, pd.DataFrame([{
            "id": "P0", "section": "supervised", "claim": "every r249_comparison.csv row reproduces from fold_scores.csv / summary.json",
            "value": str(len(rc)), "archived_value": str(n_ok), "source_file": "reports/tables/r440_r249_recheck.csv",
            "verified": "yes" if n_ok == len(rc) else "no", "status": "confirmed" if n_ok == len(rc) else "mismatch",
            "note": "; ".join(rc.loc[~rc["ok_all"], "candidate"].tolist()[:10]), "claimed_in": "reports/tables/r249_comparison.csv", "check": "eq"}])],
            ignore_index=True)
        cols = ["claim", "value", "source_file", "verified", "note", "id", "section", "archived_value", "status", "claimed_in", "check"]
        claims = claims[cols]
        claims.to_csv(tables / "r440_core_numbers.csv", index=False)
        (tables / "r440_core_numbers.md").write_text(
            f"Core-report number audit (run {run.run_id}); verified = archived value equals the claim at the printed precision.\n\n"
            + claims[["id", "section", "claim", "value", "archived_value", "verified", "status", "source_file", "note"]].to_markdown(index=False) + "\n",
            encoding="utf-8")
        run.save_table("core_numbers", claims)
        rc.to_csv(tables / "r440_r249_recheck.csv", index=False)
        run.save_table("r249_recheck", rc)
        cmp_.to_csv(tables / "r440_label_budget_compare.csv", index=False)
        hits.to_csv(tables / "r440_label_budget_first_hit.csv", index=False)
        run.save_table("label_budget_compare", cmp_)
        run.save_table("label_budget_first_hit", hits)
        show = cmp_[["model", "n_labels", "mean_r206", "n_failed_r206", "mean_r441", "n_failed_r441", "diff", "se_diff", "z", "comparable", "flag_3se"]]
        (tables / "r440_label_budget_compare.md").write_text(
            f"R441 (official repeats 4-19, new draws) vs R206 (repeats 0-3) for the matched models; se = fold-cluster SE; "
            f"comparable = all evaluations defined in both runs (run {run.run_id}).\n\n" + show.round(4).to_markdown(index=False) + "\n",
            encoding="utf-8")
        (tables / "r440_label_budget_first_hit.md").write_text(
            f"First label budget with mean validation AUC >= 0.99 (fully defined budgets only; run {run.run_id}).\n\n"
            + hits.to_markdown(index=False) + "\n", encoding="utf-8")
        fig = label_budget_figure(figs / "r440_label_budget_compare.png", s441, s206, r304)
        run.save_figure("label_budget_compare", fig)
        J = journey(A, hm)
        J.to_csv(tables / "r440_journey.csv", index=False)
        (tables / "r440_journey.md").write_text(journey_md(J, run.run_id), encoding="utf-8")
        run.save_table("journey", J)
        vc = claims["verified"].value_counts().to_dict()
        print(claims[claims["verified"] == "no"][["id", "claim", "value", "archived_value", "status", "note"]].to_string(), flush=True)
        print(show.round(4).to_string(), flush=True)
        print(hits.to_string(), flush=True)
        print(journey_md(J, run.run_id), flush=True)
        run.finish(extra_summary={"n_claims": int(len(claims)), "verified_counts": vc,
                                  "not_verified": claims.loc[claims["verified"] == "no", "id"].tolist(),
                                  "r249_rows_ok": n_ok, "r249_rows": int(len(rc)),
                                  "label_budget_flags_3se": int(cmp_["flag_3se"].sum()),
                                  "label_budget_comparable_cells": int(cmp_["comparable"].sum())})


if __name__ == "__main__":
    main()
