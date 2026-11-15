# 20261022-183558_r421_public_split_joint

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. (b) Stratification fixes the observed public n1 at 198 (or 197) instead of letting it vary with sd ~10; the AUC window for 0.99923 is the single value 2D = 60 for every n1 near 197.5, and the 2D distribution given k depends on n1 only through n1*n0 (+-0.3 %), so P(0.99923 | k) changes by < 10 % for each k, the flip_y support [0, ~0.015] and MLE ~0.003 stay, and S003's private 90 % interval moves by < 3e-4. (a2) Conditional on S003's exact public 0.99923 and with real-feature labels, r201's public AUC is about 0.9985-0.9990 (B006 point 0.9988) and r209's about 0.9990-0.9993 (B007 point 0.9991); both are below S003's in most draws (P(higher) 0.1-0.3), and the random-split and stratified answers differ by < 1e-4. The random-split window probabilities reproduce R922's (1.24e-4, 2.44e-3, 4.1e-4 for k = 0, 1, 2) within MC error.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. The three score vectors were recreated with value hashes equal to submissions/S003, S005 and S006. Random split, 12M draws per k: P(public rounds to 0.99923 | k) = 1.17e-4 / 2.42e-3 / 4.1e-4 for k = 0 / 1 / 2 (R922: 1.24e-4 / 2.44e-3 / 4.1e-4). A stratified split (observed public n1 fixed at 197 or 198) gives 1.26-1.38e-4 / 2.55-2.63e-3 / 4.2-4.6e-4 (k = 0 rests on 61 hits at n1 = 198, ~13 % MC error); the flat-prior support stays [0, 0.0148] (1/8) and [0, 0.0203] (1/32), MLE 0.003, LR(f = 0) 0.19 (random) vs 0.19-0.20 (stratified), and S003's private predictive stays mean 0.9968, 90 % [0.9923, 0.9996] (differences <= 1e-4). Conditional on S003's exact public 0.99923 (flat prior), r201's public AUC is 0.99852 (80 % 0.99767-0.99933, P(higher than S003) 0.14, P(in B006's 80 % interval) 0.79) and r209's is 0.99912 (80 % 0.99884-0.99941, P(higher) 0.29, P(equal) 0.045, P(in B007's interval) 0.94); stratified and random splits agree within 4e-5. Caveat: the clean labels are drawn from the converged GMM posterior, which ranks like S003, so r201's loss is the upper end (private diff -3.6e-4 here vs -2.6e-4 on replicas, r936). (r2_noise)
