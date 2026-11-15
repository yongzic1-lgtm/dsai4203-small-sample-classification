"""R203 - supervised-only track, small regularised tree ensembles (expected negative results).

RandomForest, ExtraTrees, LightGBM, XGBoost with fixed, deliberately conservative settings
chosen a priori for n = 200 training rows (no tuning, n_configs = 1), on all 150 features and
on the in-fold high-variance block; plus one LightGBM with a small NESTED grid on the block
(inner RepeatedStratifiedKFold(5, 1)), to check that tuning would not rescue trees.

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r203_sup_trees.py
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lib_sup import Cand, HighVarColumns, Tuned, common_args, run_candidates


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=1, random_state=0)


def et():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=1, random_state=0)


def lgbm(**kw):
    p = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, n_jobs=1, random_state=0)
    p.update(kw)
    return LGBMClassifier(**p)


def xgb():
    return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.5,
                         min_child_weight=2, reg_lambda=1.0, n_jobs=1, eval_metric="logloss", random_state=0)


LGBM_GRID = {"model__num_leaves": [3, 7], "model__min_child_samples": [20, 5], "model__n_estimators": [100, 400]}

H_TREE = ("Axis-aligned splits approximate a quadratic (covariance) boundary in 13 correlated dimensions badly "
          "with 200 rows, and on all features the 137 noise columns dilute the splits. Prediction: block "
          "0.80-0.93, all features 0.70-0.88, all well below QDA (0.996); boosting not better than RF; the nested "
          "LightGBM grid gains < 0.02 over the fixed one. Expected negative result.")


def candidates():
    out = []
    for feat, label in (("all", "all 150"), ("block", "block(in-fold highvar)")):
        def mk(m, f=feat):
            steps = [("sel", HighVarColumns())] if f == "block" else []
            return Pipeline(steps + [("scale", StandardScaler()), ("model", m)])
        out += [
            Cand(f"rf_{feat}", lambda d, mk=mk: Tuned(mk(rf()), None, log_dir=d), "supervised-trees", label, False, 1,
                 f"RandomForest(500, min_leaf 3, sqrt), {label}", H_TREE),
            Cand(f"et_{feat}", lambda d, mk=mk: Tuned(mk(et()), None, log_dir=d), "supervised-trees", label, False, 1,
                 f"ExtraTrees(500, min_leaf 3, sqrt), {label}", H_TREE),
            Cand(f"lgbm_{feat}", lambda d, mk=mk: Tuned(mk(lgbm()), None, log_dir=d), "supervised-trees", label, False, 1,
                 f"LightGBM(300, lr 0.03, 7 leaves, min_child 10, subsample 0.8, colsample 0.5, L2 1), {label}", H_TREE),
            Cand(f"xgb_{feat}", lambda d, mk=mk: Tuned(mk(xgb()), None, log_dir=d), "supervised-trees", label, False, 1,
                 f"XGBoost(300, lr 0.03, depth 3, subsample 0.8, colsample 0.5, mcw 2, L2 1), {label}", H_TREE),
        ]
    n_grid = 1
    for v in LGBM_GRID.values():
        n_grid *= len(v)
    out.append(Cand("lgbm_block_nested",
                    lambda d: Tuned(Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", lgbm())]),
                                    LGBM_GRID, inner_repeats=1, log_dir=d),
                    "supervised-trees", "block(in-fold highvar)", False, n_grid,
                    "LightGBM on in-fold block, nested num_leaves x min_child_samples x n_estimators", H_TREE,
                    tuning=str(LGBM_GRID)))
    return out


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    run_candidates(candidates(), "r203", a.data, a.n_jobs, a.author, "research/r203_sup_trees.py", a.only, a.max_folds)
