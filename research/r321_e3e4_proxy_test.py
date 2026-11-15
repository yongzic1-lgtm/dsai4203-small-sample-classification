"""R321 (ext34) - proxy test AUC for the E3 final models (every k_feat) and E4 feature sets.

The E3 brief compares the leaky CV, the honest CV and the leaderboard. The public LB has only
~395 rows (one number, SE ~0.02) and costs a submission. This analysis adds a large-sample
proxy: the transductive GMM (K=2 on the label-free high-variance block, S003 public 0.99923,
R003 CV 1.0000) gives near-certain labels for all 19,750 test rows, so

    proxy AUC = AUC(final model's test scores, GMM labels)

estimates the test AUC of any other model. It is NOT a leaderboard score: it inherits the
GMM's own errors (~1% of test rows have a GMM posterior in (0.1, 0.9)); the soft version
weights every row by its posterior instead of rounding it. A simulation of random 395-row
public subsets gives the expected public-LB spread of the E3 model before it is submitted.

Models (all fitted on all 250 rows, as a final model would be):
  E3  scaler + CorrelationTopK(k_feat) + L2 logreg (C=0.1), the E3 final model, every k_feat of
      extensions/e3_leakage/config.yaml (k_feat=20 is the model whose test_pred.npy E3 archived)
  E4  L2 logreg and QDA(reg 0.1) on: all 150, the label-free block, the bootstrap-stable set
      (frequency >= 0.5 in the top-10, B and seed of the E4 config, all rows), block - stable,
      block | stable. (GMM-on-block is the label source, so it is not scored.)

    .venv\\Scripts\\python.exe colab\\remote.py run -a pro1 -s rext34 --high-mem -- research/r321_e3e4_proxy_test.py --data raw
"""

from __future__ import annotations

import argparse
import copy
import hashlib

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from lib_ext34 import FoldColumnsModel
from ssc import data, paths
from ssc.data import FEATURES
from ssc.features import bootstrap_topk_frequency
from ssc.models import build_pipeline, predict_score
from ssc.tracking import Run
from ssc.transductive import TransductiveGMM, resolve_columns, unlabeled_features

P = len(FEATURES)


def soft_auc(p: np.ndarray, s: np.ndarray) -> float:
    """AUC with fractional labels: sum_ij p_i (1 - p_j) [s_i > s_j] (+ 1/2 ties) / (sum p)(sum 1 - p)."""
    order = np.argsort(s, kind="mergesort")
    s, p = s[order], p[order]
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    neg_before = np.concatenate([[0.0], np.cumsum(1 - p)])[start]  # negative mass strictly below each tie group
    num = 0.0
    self_pairs = float((p * (1 - p)).sum())  # pairs (i, i) are excluded from numerator and denominator
    for st, c, nb in zip(start, counts, neg_before):
        g = p[st:st + c]
        num += g.sum() * (nb + 0.5 * (1 - g).sum()) - 0.5 * float((g * (1 - g)).sum())
    return float(num / (p.sum() * (1 - p).sum() - self_pairs))


def public_sim(labels: np.ndarray, s: np.ndarray, n_pub: int, n_sim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(n_sim):
        idx = rng.choice(len(s), size=n_pub, replace=False)
        if labels[idx].min() == labels[idx].max():
            continue
        a.append(roc_auc_score(labels[idx], s[idx]))
    a = np.array(a)
    return {"pub_sim_mean": float(a.mean()), "pub_sim_sd": float(a.std(ddof=1)),
            "pub_sim_q05": float(np.quantile(a, 0.05)), "pub_sim_q95": float(np.quantile(a, 0.95))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="raw", choices=paths.SOURCES)
    ap.add_argument("--author", default=None)
    ap.add_argument("--n-sim", type=int, default=5000, help="random public subsets")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    src = a.data

    e3 = yaml.safe_load((paths.ROOT / "extensions" / "e3_leakage" / "config.yaml").read_text(encoding="utf-8"))
    e4 = yaml.safe_load((paths.ROOT / "extensions" / "e4_feature_stability" / "config.yaml").read_text(encoding="utf-8"))
    proj = yaml.safe_load(paths.PROJECT_YAML.read_text(encoding="utf-8"))
    kfeats = sorted({int(k) for k in e3["grid"]["k_feat"]} | {int(e3["leaderboard"]["k_feat"])})
    B = 200 if a.quick else int(e4["bootstrap"]["n_boot"])
    n_sim = 200 if a.quick else a.n_sim

    ds = data.load(src)
    X, y, Xt = ds.X.to_numpy(float), ds.y, ds.X_test.to_numpy(float)
    n_pub = int(round(float(proj["competition"]["public_fraction"]) * len(Xt)))
    block = resolve_columns("highvar", unlabeled_features(src))

    cfg = {"name": "r321_e3e4_proxy_test", "label_source": "TransductiveGMM('highvar', 2) fitted on all rows",
           "e3_k_feat": kfeats, "e3_pipeline": e3["pipeline"], "n_public_rows": n_pub, "n_sim": n_sim,
           "e4_stable_rule": f"top-10 bootstrap frequency >= 0.5, B={B}, seed={e4['bootstrap']['seed']}",
           "block_features": [FEATURES[j] for j in block], "uses_unlabeled_test_features": True, "quick": a.quick}
    hyp = ("On ~19,750 GMM-proxy-labelled test rows the E3 final model scores close to the honest CV "
           "(k_feat=20: ~0.79-0.81) and far below the leaky CV (~0.87); the proxy AUC peaks near k_feat=7 like the "
           "honest CV, not at k_feat=50 like the leaky CV. For E4, the label-free block beats the stable set "
           "for QDA and for L2 logreg.")
    with Run("r321_e3e4_proxy_test", cfg, kind="analysis", source=src, author=a.author, hypothesis=hyp) as run:
        gmm = TransductiveGMM("highvar", 2, source=src).fit(X, y)
        post = gmm.predict_proba(Xt)[:, 1]
        hard = (post >= 0.5).astype(int)
        gmm_info = {"test_pos_rate_hard": float(hard.mean()), "n_boundary_0.1_0.9": int(((post > 0.1) & (post < 0.9)).sum()),
                    "train_auc_gmm": float(roc_auc_score(y, gmm.predict_proba(X)[:, 1]))}
        print("GMM proxy labels:", gmm_info)

        rows, e3_pred = [], None
        for k in kfeats:
            spec = copy.deepcopy(e3["pipeline"])
            spec["selector"] = {"name": "corr_topk", "k": k, "method": e3["selection"]["method"]}
            m = build_pipeline(spec).fit(X, y)
            s = predict_score(m, Xt)
            if k == int(e3["leaderboard"]["k_feat"]):
                e3_pred = s
            rows.append({"k_feat": k, "proxy_auc_hard": float(roc_auc_score(hard, s)), "proxy_auc_soft": soft_auc(post, s),
                         **public_sim(hard, s, n_pub, n_sim, seed=2026)})
            print(f"  E3 k_feat={k:3d} proxy AUC hard {rows[-1]['proxy_auc_hard']:.4f} soft {rows[-1]['proxy_auc_soft']:.4f}"
                  f"  public-sim {rows[-1]['pub_sim_mean']:.4f} +/- {rows[-1]['pub_sim_sd']:.4f}")
        e3_tab = pd.DataFrame(rows)
        run.save_array("e3_final_k20_test_pred", e3_pred)

        freq = bootstrap_topk_frequency(X, y, 10, n_boot=B, random_state=int(e4["bootstrap"]["seed"]),
                                        method=e4["bootstrap"]["method"])
        stable = set(int(i) for i in np.flatnonzero(freq >= 0.5))
        bset = set(int(c) for c in block)
        sets = {"all150": set(range(P)), "block": bset, "stable_k10": stable,
                "block_minus_stable": bset - stable, "block_or_stable": bset | stable}
        rows = []
        for fam in ("logreg", "qda"):
            for name, cols in sets.items():
                if fam == "qda" and name == "all150":
                    continue  # not estimable (fewer rows per class than features)
                mdl = FoldColumnsModel(fam, columns=tuple(sorted(cols)), source=src).fit(X, y)
                s = mdl.predict_proba(Xt)[:, 1]
                rows.append({"model": fam, "set": name, "n_features": len(cols),
                             "features": " ".join(FEATURES[i] for i in sorted(cols)) if len(cols) < P else "all",
                             "proxy_auc_hard": float(roc_auc_score(hard, s)), "proxy_auc_soft": soft_auc(post, s)})
                print(f"  E4 {fam:6s} {name:20s} ({len(cols):3d}) proxy AUC hard {rows[-1]['proxy_auc_hard']:.4f}")
        e4_tab = pd.DataFrame(rows)

        run.save_table("e3_proxy", e3_tab)
        run.save_table("e4_proxy", e4_tab)
        tdir = run.dir / "report_preview" if a.quick else paths.tables_dir(src)
        tdir.mkdir(parents=True, exist_ok=True)
        e3_tab.to_csv(tdir / "r321_e3_proxy.csv", index=False)
        e4_tab.to_csv(tdir / "r321_e4_proxy.csv", index=False)
        run.summary.update({"gmm_proxy": gmm_info, "n_public_rows": n_pub,
                            "e3_k20_test_pred_sha1": hashlib.sha1(np.round(e3_pred, 12).tobytes()).hexdigest()[:16],
                            "e3_proxy_auc_hard": {int(r["k_feat"]): round(r["proxy_auc_hard"], 5) for _, r in e3_tab.iterrows()},
                            "e4_stable_set_k10": sorted(FEATURES[i] for i in stable)})
        run.finish()
    print("\n" + e3_tab.round(4).to_string(index=False))
    print("\n" + e4_tab.drop(columns=["features"]).round(4).to_string(index=False))
    print(f"run folder: {run.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
