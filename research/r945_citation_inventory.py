"""R945 - read-only citation inventory (verify round, lens "rules").

Lists every third-party module imported anywhere in research/, extensions/, src/ssc/, scripts/, colab/
(AST walk, so imports inside functions count too) and a fixed list of methods detected by code patterns,
and checks each against the BibTeX keys and `% status:` comments in registry/references.md.

It reads files only. It writes nothing to the archive and creates no Run (no data, no model).
Re-run before the final export to see which entries are still missing or still marked `candidate`:

    .venv\\Scripts\\python.exe research\\r945_citation_inventory.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
CODE_DIRS = ["research", "extensions", "src/ssc", "scripts", "colab"]
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
LOCAL = {"ssc", "_common", "lib_sup", "lib_fm", "lib_mix", "lib_gen_replica", "lib_ext12", "lib_ext34"}
# research scripts import each other (e.g. `from r201_sup_quadratic import ...`): treat any module that
# exists as research/<name>.py as local code
LOCAL |= {p.stem for p in (ROOT / "research").glob("*.py")}

# module -> references.md key it should be cited under (None = software note, no BibTeX entry needed)
MODULE_KEY = {
    "sklearn": "sklearn", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "matplotlib": "matplotlib", "lightgbm": "lightgbm", "xgboost": "xgboost",
    "tabpfn": "hollmann2025", "tabicl": "qu2025", "torch": "pytorch", "numpyro": "numpyro",
    "jax": "jax", "pyro": "bingham2019", "pymc": "pymc",
    "huggingface_hub": None, "joblib": None, "yaml": None, "tabulate": None, "threadpoolctl": None,
}
# imports written inside strings (code run in a subprocess, e.g. research/r250_fm_licence_check.py) are
# invisible to the AST walk; catch them with a line regex
STRING_IMPORT = re.compile(r"^\s*(?:from|import)\s+(tabpfn|tabicl|torch|numpyro|jax|pyro|pymc)\b", re.M)
# patterns that show code PRESENCE only; whether the method was actually used must be checked by hand
PRESENCE_ONLY = {"Kaggle API upload": "S001-S004 were uploaded through the web page (journal/memory); "
                                       "keep kaggleapi as candidate unless an API upload happens"}

# method -> (regex over code, expected key)
METHODS = {
    "Nadeau-Bengio corrected t-test": (r"nadeau|corrected.*t.?test|paired_compare", "nadeau2003"),
    "Hanley-McNeil AUC SE": (r"hanley|auc_standard_error", "hanley1982"),
    "bootstrap": (r"bootstrap", "efron1979"),
    "parametric bootstrap / replicas": (r"parametric|sample_parametric", "efron1993"),
    "EM / Gaussian mixture": (r"GaussianMixture|SemiSupEM", "dempster1977"),
    "BIC": (r"\.bic\(|\bBIC\b", "schwarz1978"),
    "make_classification generator": (r"make_classification", "guyon2003"),
    "QDA regularisation (RDA)": (r"QuadraticDiscriminantAnalysis|reg_param", "friedman1989"),
    "Ledoit-Wolf shrinkage": (r"Ledoit|shrinkage=.auto", "ledoit2004"),
    "L1 / lasso": (r"l1_ratio\s*[:=]\s*1|Lasso\(", "tibshirani1996"),
    "self-training": (r"SelfTrainingClassifier|selftrain", "yarowsky1995"),
    "label spreading": (r"LabelSpreading", "zhou2004"),
    "label propagation": (r"LabelPropagation", "zhu2002"),
    "semi-supervised EM": (r"SemiSupEM", "nigam2000"),
    "DP variational mixture": (r"BayesianGaussianMixture", "blei2006"),
    "horseshoe prior": (r"horseshoe", "carvalho2010"),
    "regularised horseshoe": (r"regulari[sz]ed.horseshoe|slab_scale|tau0", "piironen2017"),
    "NUTS": (r"\bNUTS\b", "hoffman2014"),
    "R-hat (split Gelman-Rubin)": (r"split_gelman_rubin|r_hat|rhat", "vehtari2021"),
    "NIW conjugate / Student-t predictive": (r"multigammaln|NIW|inverse.?wishart", "murphy2007"),
    "GP classification": (r"GaussianProcessClassifier", "rasmussen2006"),
    "random forest": (r"RandomForestClassifier", "breiman2001"),
    "extra trees": (r"ExtraTreesClassifier", "geurts2006"),
    "SVM (LIBSVM)": (r"\bSVC\(", "chang2011"),
    "liblinear": (r"liblinear", "fan2008"),
    "saga": (r"saga", "defazio2014"),
    "k-NN": (r"KNeighborsClassifier", "cover1967"),
    "nested CV": (r"GridSearchCV|nested", "varma2006"),
    "KS / D'Agostino normality tests": (r"normaltest", "dagostino1973"),
    "Mardia multivariate skewness": (r"mardia", "mardia1970"),
    "rejection ABC": (r"\bABC\b|abc_", "beaumont2002"),
    "likelihood-ratio chi-square": (r"chi2\.sf|likelihood.ratio|\bLRT\b", "wilks1938"),
    "power-law learning curve": (r"curve_fit", "figueroa2012"),
    "pivoted QR": (r"qr\(.*pivot", "golub2013"),
    "TabPFN": (r"TabPFNClassifier", "hollmann2025"),
    "TabICL": (r"TabICLClassifier", "qu2025"),
    "Kaggle API upload": (r"kaggle competitions submit", "kaggleapi"),
    # classical methods: one textbook entry is enough (Hastie et al. 2009 covers LDA, naive Bayes,
    # logistic regression, PCA, polynomial basis expansion, kNN)
    "LDA / naive Bayes / PCA / polynomial logistic (textbook)":
        (r"LinearDiscriminantAnalysis|GaussianNB|\bPCA\(|PolynomialFeatures", "hastie2009"),
    "rank correlation / KS tests (SciPy)": (r"kendalltau|spearmanr|ks_2samp|kstest", "scipy"),
}


def code_files():
    for d in CODE_DIRS:
        yield from sorted((ROOT / d).rglob("*.py"))


def imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def reference_status() -> dict[str, str]:
    text = (ROOT / "registry" / "references.md").read_text(encoding="utf-8")
    status = {}
    for m in re.finditer(r"@\w+\{(\w+),([^\n]*)", text):
        s = re.search(r"%\s*status:\s*(\w+)", m.group(2))
        status[m.group(1)] = s.group(1) if s else "?"
    return status


def main() -> None:
    files = [f for f in code_files() if f.resolve() != SELF]   # the pattern table must not match itself
    texts = {f: f.read_text(encoding="utf-8", errors="replace") for f in files}
    refs = reference_status()
    mods: dict[str, set[str]] = {}
    for f in files:
        found = imports(f) | set(STRING_IMPORT.findall(texts[f]))
        for m in found - STDLIB - LOCAL:
            mods.setdefault(m, set()).add(str(f.relative_to(ROOT)).replace("\\", "/"))

    def verdict(key):
        if key is None:
            return "software note (no BibTeX needed)"
        st = refs.get(key)
        return "MISSING" if st is None else ("ok" if st == "used" else f"present, status={st} -> set used")

    print(f"{len(files)} code files; references.md keys: {len(refs)}\n")
    print("## Imported third-party modules")
    print("| module | files | expected key | verdict |\n|---|---:|---|---|")
    for m in sorted(mods):
        key = MODULE_KEY.get(m, "UNMAPPED")
        v = verdict(key) if key != "UNMAPPED" else "UNMAPPED (add to MODULE_KEY and references.md)"
        print(f"| {m} | {len(mods[m])} | {key} | {v} |")
    print("\n## Methods detected in code")
    print("| method | files | expected key | verdict |\n|---|---:|---|---|")
    for name, (pat, key) in METHODS.items():
        hit = [f for f, t in texts.items() if re.search(pat, t, flags=re.I)]
        if hit:
            v = verdict(key)
            if name in PRESENCE_ONLY:
                v = f"code present only: {PRESENCE_ONLY[name]}"
            where = ", ".join(sorted(f.stem for f in hit)[:6]) + (" ..." if len(hit) > 6 else "")
            print(f"| {name} | {len(hit)} ({where}) | {key} | {v} |")
    unused = sorted(k for k, s in refs.items() if s == "used" and k not in set(MODULE_KEY.values())
                    and k not in {v[1] for v in METHODS.values()})
    if unused:
        print("\n`used` keys not matched by any detector (check by hand):", ", ".join(unused))


if __name__ == "__main__":
    main()
