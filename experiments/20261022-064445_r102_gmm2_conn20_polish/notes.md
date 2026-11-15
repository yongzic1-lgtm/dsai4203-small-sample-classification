# 20261022-064445_r102_gmm2_conn20_polish

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Same 12-dim span as the block (R100), so identical ranking up to rounding noise.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed: the 20 connected features span the same 12-dim space; ranking differs from S003 only by rounding/regularisation effects (Kendall 0.996, log-loss 0.0171, expected AUC unchanged). Label-free 'discovery' gives no new information (R100).
