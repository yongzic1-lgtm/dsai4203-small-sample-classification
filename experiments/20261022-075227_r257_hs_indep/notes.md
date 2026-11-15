# 20261022-075227_r257_hs_indep

- author: TODO_ID_1
- kind: experiment

## Hypothesis (written before running)

Removing the exactly dependent columns changes the parametrisation, not the linear predictor space: CV AUC within 2 split-SE of r253_hs_all (0.926).

## Result

(see summary.json)

## Conclusion (write after reading the results)

CV AUC 0.9232 +/- 0.0074, vs r253_hs_all -0.0029 (split-SE 0.0008, corrected p 0.64): same predictive performance with an identifiable parametrisation. Hypothesis 'within 2 split-SE' narrowly refuted in the split-noise view (3.5 SE lower) but not in the data-sampling view; either way the reparametrisation does not help prediction.
