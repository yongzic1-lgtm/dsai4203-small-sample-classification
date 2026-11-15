# 20261022-191747_r425_flip_grid_reweight

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running (after reading only the r420 quick summaries of the runs, not these pooled numbers). (1) Bit-exact agreement with the archived r933 tn0 arrays at the 4 shared levels. (2) Under R922's flat-prior weights S003's private predictive has mean ~0.9968 and 90 % interval ~[0.992, 0.9996] (R922 on real features: 0.9968, [0.9923, 0.9996]), i.e. ~1e-3 lower and twice as wide as r936's 0.9978 (R156 weights). (3) The paired private differences to S003 are flat in flip_y, so the second-selection gains (r209 +1.5e-5, r201 +4e-6) and D010's ranking do not change. (4) Conditional on S003's public 0.99923 the predicted public AUC is ~0.9988 for r201 and ~0.9991 for r209 under both weightings (within 1.5e-4 of B006 / B007), P(r201 > S003) ~0.2, P(r209 > S003) ~0.25-0.3. (5) The r936-design restriction reproduces r936_public_abc exactly.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported on all five points. (1) Bit-exact: 142,080/142,080 csv-score public and private AUCs equal the archived r933 tn0 arrays. (2) With R922's flat-prior likelihood binned onto the 24 levels (posterior mean f 0.0062; 0.09 % of the mass lies beyond 0.03), S003's private predictive is mean 0.99683, median 0.99738, 90 % [0.99218, 0.99958], P(>= 0.999) 0.16, P(|0.99923 - private| <= 0.001) 0.33 - the R922 real-feature answer (0.9968, [0.9923, 0.9996]) - against 0.99775, [0.99468, 0.99995], 0.14 under r936's R156 weights. The stratified-split likelihood of r421 gives the same within 1e-5. Conditioning on S003's public inside the ABC changes it by < 2e-5 (0.99683 +- 3e-5). (3) Second-selection gains E[max(S003, X)] - E[S003]: r209 +2.0e-5, r201 +7.0e-6 (r936 weights: +1.45e-5, +4.1e-6); both grow slightly with the noise level but stay < 1/100 of the private sd, so D010's ranking and conclusion are unchanged. (4) Conditional on S003's public 0.99923 (+-1.5e-4, R922 weights): r201 public 0.99868 (MC SE 2.4e-5; sd 6.7e-4; 80 % [0.99782, 0.99941]; P(> S003) 0.19, P(= S003) 0.013; P(in B006's 80 % interval) 0.85); r209 0.99910 (MC SE 7e-6; sd 2.8e-4; 80 % [0.99882, 0.99941]; P(>) 0.27, P(=) 0.034; P(in B007's interval) 0.92). Window +-5e-5 and pure ABC agree within 4e-5. (5) The r936 design restricted to 4 levels and 16 subsets reproduces r936_public_abc exactly (125 accepted, 68 replicas, r201 0.998836, r209 0.999129). (r2_noise)
