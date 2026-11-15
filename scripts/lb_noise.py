"""How noisy is the ~395-row public leaderboard?

(a) Analytic: Hanley & McNeil (1982) standard error of an AUC for AUC = 0.70 .. 0.95,
    with the train positive rate of --data, for the public LB (~2% of 19,750 rows),
    the private LB, one CV validation fold (50 rows) and all 250 train rows.
(b) Simulation on the synthetic data (hidden labels known): fit L2 logistic
    regression on synthetic train, score the 19,750 synthetic test rows, then draw
    --n-sim random public subsets of the same size and report the spread of the
    public AUC. To ask "can the public LB order two submissions whose true AUC
    differs by delta?", a second score B is built with full-test AUC exactly delta
    lower and the swap probability P(public AUC of B > public AUC of A) is counted:
      nested       B = A + noise (two variants of one model: strongly correlated)
      independent  A and B each get their own noise (different models: less correlated)
    Correlated pairs swap less often, because most of the subset noise is shared.

    .venv\\Scripts\\python.exe scripts\\lb_noise.py                      # positive rate from data/raw
    .venv\\Scripts\\python.exe scripts\\lb_noise.py --data synthetic --n-sim 500

Writes lb_noise_se.csv, lb_noise_spread.csv, lb_noise_swap.csv and lb_noise_draws.csv
to ssc.paths.tables_dir(--data) (reports/tables for raw, sandbox for synthetic).
The simulation only illustrates noise of this size; it says nothing about the real model.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from tabulate import tabulate

import _common as C
from ssc import data, paths, synthetic
from ssc.models import build_pipeline
from ssc.validation import auc_standard_error

AUC_GRID = np.round(np.arange(0.70, 0.951, 0.05), 2)
SIM_MODEL = {"scaler": "standard", "model": {"name": "logreg", "params": {"C": 0.05, "l1_ratio": 0.0}},
             "random_state": 0}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count 1/2); y boolean."""
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def with_auc(y, base, z, target: float, iters: int = 50) -> tuple[np.ndarray, float]:
    """base + sigma * z with full-data AUC == target (bisection on sigma)."""
    lo, hi = 0.0, float(np.std(base))
    while auc(y, base + hi * z) > target:
        hi *= 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        if auc(y, base + mid * z) > target:
            lo = mid
        else:
            hi = mid
    s = base + hi * z
    return s, auc(y, s)


def analytic_table(source: str) -> tuple[pd.DataFrame, dict]:
    y = data.load(source).y
    n_test = len(data.load_sample_submission(source))
    n_pub = int(round(C.public_fraction() * n_test))
    rate = float(y.mean())

    def se(a, n):
        pos = int(round(n * rate))
        return auc_standard_error(a, pos, n - pos)

    rows = [{"auc": a, "n_public": n_pub, "se_public": se(a, n_pub), "ci95_halfwidth_public": 1.96 * se(a, n_pub),
             "se_private": se(a, n_test - n_pub), "se_cv_fold_50": se(a, 50), "se_train_250": se(a, len(y))}
            for a in AUC_GRID]
    return pd.DataFrame(rows), {"positive_rate": rate, "n_public": n_pub, "n_test": n_test}


def load_synthetic():
    if (paths.DATA_SYNTHETIC / "test_labels.csv").exists():
        ds = data.load("synthetic")
        labels = pd.read_csv(paths.DATA_SYNTHETIC / "test_labels.csv").set_index("id").loc[ds.ids_test]
        return ds.X.to_numpy(), ds.y, ds.X_test.to_numpy(), labels["target"].to_numpy().astype(bool), \
            labels["public"].to_numpy().astype(bool), "data/synthetic"
    f = synthetic.make()  # same generator and default seed, built in memory
    from ssc.data import FEATURES
    return (f["train"][FEATURES].to_numpy(), f["train"]["target"].to_numpy(), f["test"][FEATURES].to_numpy(),
            f["test_labels"]["target"].to_numpy().astype(bool), f["test_labels"]["public"].to_numpy().astype(bool),
            "ssc.synthetic.make() in memory")


def simulate(n_public: int, n_sim: int, deltas: list[float], seed: int):
    X, y, Xt, yt, pub_mask, origin = load_synthetic()
    model = build_pipeline(SIM_MODEL).fit(X, y)
    s_a = np.asarray(model.decision_function(Xt), dtype=float)
    auc_a = auc(yt, s_a)
    rng = np.random.default_rng(seed)
    n_test = len(yt)

    # pairs: (label, score A, score B, full AUC A, full AUC B, delta)
    pairs = []
    for d in deltas:
        z = rng.standard_normal(n_test)
        s_b, auc_b = with_auc(yt, s_a, z, auc_a - d)
        pairs.append(("nested", d, s_a, s_b, auc_a, auc_b))
        a2, auc_a2 = with_auc(yt, s_a, rng.standard_normal(n_test), auc_a - 0.02)
        b2, auc_b2 = with_auc(yt, s_a, rng.standard_normal(n_test), auc_a2 - d)
        pairs.append(("independent", d, a2, b2, auc_a2, auc_b2))

    draws = np.empty(n_sim)
    diffs = np.empty((len(pairs), n_sim))
    for i in range(n_sim):
        idx = rng.choice(n_test, size=n_public, replace=False)
        yi = yt[idx]
        draws[i] = auc(yi, s_a[idx])
        for j, (_, _, sa, sb, _, _) in enumerate(pairs):
            diffs[j, i] = auc(yi, sa[idx]) - auc(yi, sb[idx])

    pos = int(round(yt.mean() * n_public))
    hm = auc_standard_error(auc_a, pos, n_public - pos)
    spread = pd.DataFrame([{
        "sim_data": origin, "model": "L2 logreg C=0.05", "n_public": n_public, "n_sim": n_sim,
        "auc_full_test": auc_a, "auc_actual_public_mask": auc(yt[pub_mask], s_a[pub_mask]),
        "public_auc_mean": draws.mean(), "public_auc_sd": draws.std(ddof=1),
        "public_auc_p2.5": np.percentile(draws, 2.5), "public_auc_p97.5": np.percentile(draws, 97.5),
        "public_auc_min": draws.min(), "public_auc_max": draws.max(), "hanley_mcneil_se": hm,
    }])
    swap = pd.DataFrame([{
        "pair": lab, "delta_target": d, "delta_full_test": fa - fb, "auc_a_full": fa, "auc_b_full": fb,
        "spearman_a_b": float(spearmanr(sa, sb)[0]),
        "public_diff_mean": diffs[j].mean(), "public_diff_sd": diffs[j].std(ddof=1),
        "p_swap": float((diffs[j] < 0).mean() + 0.5 * (diffs[j] == 0).mean()),
        "swapped_on_actual_public_mask": bool(auc(yt[pub_mask], sb[pub_mask]) > auc(yt[pub_mask], sa[pub_mask])),
    } for j, (lab, d, sa, sb, fa, fb) in enumerate(pairs)])
    return spread, swap, pd.DataFrame({"draw": np.arange(n_sim), "public_auc": draws})


def main() -> int:
    C.setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_data_arg(ap)
    ap.add_argument("--n-sim", type=int, default=2000, help="random public subsets to draw (default 2000)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03],
                    help="true AUC differences to test (default 0.005 0.01 0.02 0.03)")
    ap.add_argument("--seed", type=int, default=2026, help="seed of the simulation (default 2026)")
    args = ap.parse_args()
    source = paths.resolve_source(args.data)
    if args.n_sim < 10:
        C.die("--n-sim must be >= 10", 2)

    se_tab, info = analytic_table(source)
    print(f"(a) Hanley-McNeil SE of AUC  [data {source}: positive rate {info['positive_rate']:.3f}, "
          f"public = {info['n_public']} of {info['n_test']} test rows]")
    print(tabulate(se_tab[["auc", "se_public", "ci95_halfwidth_public", "se_private", "se_cv_fold_50",
                           "se_train_250"]].to_numpy().tolist(),
                   headers=["AUC", "SE public", "95% +/- public", "SE private", "SE 50-row fold", "SE 250 rows"],
                   floatfmt=".4f", tablefmt="simple"))

    spread, swap, draws = simulate(info["n_public"], args.n_sim, sorted(set(args.deltas)), args.seed)
    s = spread.iloc[0]
    print(f"\n(b) Simulation on synthetic data ({s['sim_data']}; {int(s['n_sim'])} random {int(s['n_public'])}-row "
          "public subsets)")
    print(f"  full-test AUC {s['auc_full_test']:.4f}; public AUC mean {s['public_auc_mean']:.4f}, "
          f"sd {s['public_auc_sd']:.4f} (Hanley-McNeil SE {s['hanley_mcneil_se']:.4f}), "
          f"95% range {s['public_auc_p2.5']:.4f} .. {s['public_auc_p97.5']:.4f}")
    print(tabulate(swap[["pair", "delta_full_test", "spearman_a_b", "public_diff_sd", "p_swap"]].to_numpy().tolist(),
                   headers=["pair", "true diff", "spearman A,B", "sd of public diff", "P(order swaps)"],
                   floatfmt=".4f", tablefmt="simple"))
    one = swap[np.isclose(swap["delta_target"], 0.01)]
    if len(one):
        p = ", ".join(f"{r.pair} {r.p_swap:.2f}" for r in one.itertuples())
        print(f"  => true difference 0.01: the public LB shows the wrong order with probability {p}.")

    out = paths.tables_dir(source)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("se", se_tab), ("spread", spread), ("swap", swap), ("draws", draws)):
        df.to_csv(out / f"lb_noise_{name}.csv", index=False)
    print(f"\nwrote lb_noise_se.csv, lb_noise_spread.csv, lb_noise_swap.csv, lb_noise_draws.csv to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
