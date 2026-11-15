# 20261022-100551_r920_generator_alternatives

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The structural parameters are pinned down far beyond 'equally good' alternatives: rank 12 / 8 exact dependencies fix n_informative = 12, n_redundant = 8 deterministically; K = 4 vs K = 2 LR on the real data lies inside the H0 replica range and orders of magnitude below H1 (2 clusters/class); hypercube=False is disfavoured by a large factor through the 8 exact zero mean differences; class_sep is ~0.7 with a 95 % interval roughly 0.6-0.85, so 0.6-0.9 are NOT all equally supported (0.9 is clearly worse); flip_y is not identifiable from features at all.

## Result

(see summary.json)

## Conclusion (write after reading the results)

The structural claims survive every alternative tried: the 8 dependencies are exact up to 6-decimal rounding (residual sd / rounding-predicted sd = 0.992-1.004, eigenvalues 13-20 at 0.87x the rounding floor vs lambda_12 = 0.19), so n_informative = 12 / n_redundant = 8 / n_repeated = 0 is deterministic; the label-free Wishart profile likelihood picks the claimed informative basis with a 13.0-nat margin (17.7 with the hypercube constraint), 20/20 bootstraps and 24/24 known-truth replicas; LR(K4 vs K2) = 480 lies inside H0 replicas [290, 1254] and 250x below 2-clusters-per-class replicas [1.2e5, 1.8e5]; hypercube=True is favoured ~360:1 (local density; P(joint) 0.36 vs 4.7e-5); noise columns carry no class signal (KS p 0.31/0.58); rows are shuffled and the clean classes are 10,000/10,000 (z = -0.54). class_sep MLE 0.703, LR 95 % [0.64, 0.76] (replica coverage only 21/24, MLE sd 0.038, so honest ~[0.63, 0.78]); 0.6 (LR 0.008), 0.8 (0.004) and 0.9 (~0) are NOT equally good. flip_y cannot be identified from X (it only changes labels): see R922.
