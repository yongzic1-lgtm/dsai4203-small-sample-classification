# 20261022-064148_e1_validation_protocol_qda_pair

- author: TODO_ID_1
- kind: extension

## Hypothesis (written before running)

Near the AUC ceiling (0.996 vs 0.983) the split noise of every scheme is several times smaller than for the linear pair at AUC 0.80, so already a plain 5-fold (or even a 30 % hold-out) resolves this 0.0126 gap more often than for the linear pair; the MDD of the same scheme therefore depends strongly on the pair, and repeated 5-fold is still needed for the linear pair.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Real data, ceiling pair QDA reg 0.1 vs 0.3 (reference A-B = +0.0122, SE 0.0003): split noise is 5-6x smaller than for the linear pair; every k-fold scheme (even 3-fold, MDD 0.0057 [0.0048, 0.0072]) resolves 0.01, hold-outs still do not (MDD 0.021 / 0.016). The scheme needed for a 0.01 decision depends on the AUC regime and on the pair. (ext12)
