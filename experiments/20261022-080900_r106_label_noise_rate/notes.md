# 20261022-080900_r106_label_noise_rate

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

The observed public 0.99923 and the absence of confidently mislabelled training rows are jointly most consistent with a wrong-label rate of roughly 0.1-0.5 % (flip 0.002-0.01); the implied private AUC of S003 is ~0.995-0.999, far below the clean-label Bayes AUC 0.99995, and identical for all near-Bayes candidates.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed. With a uniform prior over the flip grid, the joint likelihood of (public 0.99923 within +-0.0004, no confidently mislabelled training row) peaks at flip 0.002-0.004 (wrong-label rate 0.1-0.2 %); posterior mean wrong-label rate 0.21 %, clean labels keep 8.6 % posterior mass but P(public <= 0.99923 | clean) = 0.002. Implied expected private AUC of S003 = 0.9978 (0.99995 if clean, 0.9949 at sklearn's default flip_y 0.01): the private LB is set by label noise, which no ranking can fix, and every near-Bayes candidate (R103: within 1e-6 of S003) gets the same private score. Use this for the private-LB belief, not for model choice.
