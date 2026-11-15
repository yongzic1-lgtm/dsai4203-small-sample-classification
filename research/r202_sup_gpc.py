"""R202 - supervised-only track, Gaussian-process classifiers (Laplace approximation).

GaussianProcessClassifier with C * RBF kernel; kernel hyperparameters (amplitude, one length
scale or one per feature = ARD) are fitted by maximising the marginal likelihood on the
200-row training fold (type-II maximum likelihood, 2 optimizer restarts). No inner CV is
needed for these hyperparameters, so n_configs = 1 (the restart count and the kernel family
are fixed a priori). Feature sets: in-fold high-variance block (no test features), and all
150 features (isotropic only; ARD with 150 length scales on 200 rows is not identifiable).

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro2 -s rsup -- research/r202_sup_gpc.py
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lib_sup
from lib_sup import Cand, GPCAuto, HighVarColumns, Tuned, common_args, run_candidates
from ssc.transductive import SubspaceModel

H_GPC = ("A GP classifier with an RBF kernel is a flexible smooth boundary; on the 13-d block it can represent "
         "the quadratic class boundary but must learn it from 200 rows. Prediction: ARD block 0.985-0.996 "
         "(<= QDA 0.996, because QDA has the correct parametric form), isotropic block a bit lower, all 150 "
         "features < 0.90 (isotropic distance dominated by noise features). ARD length scales should be "
         "shortest on block features with the largest class variance difference.")


def candidates():
    blk = lambda m: Pipeline([("sel", HighVarColumns()), ("scale", StandardScaler()), ("model", m)])  # noqa: E731
    return [
        Cand("gpc_rbf_block", lambda d: Tuned(blk(GPCAuto(ard=False)), None, log_dir=d),
             "supervised-kernel", "block(in-fold highvar)", False, 1,
             "GPC, C*RBF isotropic, in-fold block, ML-II hyperparameters", H_GPC, tuning="ML-II (marginal likelihood)"),
        Cand("gpc_ard_block", lambda d: Tuned(blk(GPCAuto(ard=True)), None, log_dir=d),
             "supervised-kernel", "block(in-fold highvar)", False, 1,
             "GPC, C*RBF with ARD length scales, in-fold block, ML-II hyperparameters", H_GPC,
             tuning="ML-II (marginal likelihood)"),
        Cand("gpc_ard_block_tt", lambda d: Tuned(SubspaceModel(GPCAuto(ard=True), "highvar", source=lib_sup.CURRENT["source"]), None, log_dir=d),
             "supervised-kernel", "block(train+test highvar, train+test standardised)", True, 1,
             "GPC, C*RBF ARD, R003's train+test block (SubspaceModel), ML-II hyperparameters", H_GPC,
             tuning="ML-II (marginal likelihood)"),
        Cand("gpc_rbf_all", lambda d: Tuned(Pipeline([("scale", StandardScaler()), ("model", GPCAuto(ard=False, length_scale=10.0))]),
                                            None, log_dir=d),
             "supervised-kernel", "all 150", False, 1,
             "GPC, C*RBF isotropic, all 150 features, ML-II hyperparameters", H_GPC, tuning="ML-II (marginal likelihood)"),
    ]


if __name__ == "__main__":
    a = common_args(__doc__.splitlines()[0]).parse_args()
    lib_sup.CURRENT["source"] = a.data
    run_candidates(candidates(), "r202", a.data, a.n_jobs, a.author, "research/r202_sup_gpc.py", a.only, a.max_folds)
