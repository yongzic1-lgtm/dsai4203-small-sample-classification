# 20261022-072942_r321_e3e4_proxy_test

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

On ~19,750 GMM-proxy-labelled test rows the E3 final model scores close to the honest CV (k_feat=20: ~0.79-0.81) and far below the leaky CV (~0.87); the proxy AUC peaks near k_feat=7 like the honest CV, not at k_feat=50 like the leaky CV. For E4, the label-free block beats the stable set for QDA and for L2 logreg.

## Result

(see summary.json)

## Conclusion (write after reading the results)

(ext34, 2026-10-22) GMM proxy labels (203 test rows with posterior in 0.1-0.9). E3 final model (k_feat=20, test_pred identical to 20261022-063908_e3_leakage) has proxy test AUC 0.820 (soft 0.813) vs leaky CV 0.867 and honest CV 0.793 (10-fold 0.805): leaky is ~0.047 too high, honest slightly pessimistic; expected public 0.820 +/- 0.021 (90%: 0.785-0.854). Proxy AUC is highest at k_feat=10 (0.864) / 7 (0.852), not at k_feat=50 (0.825) where the leaky CV peaks. E4: L2 logreg block 0.887 > stable 0.852 > all150 0.802; QDA block 0.994 > stable 0.962 > block-stable 0.917 (proxy labels come from a Gaussian model on the block, so QDA-on-block is favoured; not a leaderboard score).
