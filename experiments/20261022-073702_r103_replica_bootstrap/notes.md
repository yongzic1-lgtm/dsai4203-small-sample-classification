# 20261022-073702_r103_replica_bootstrap

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Under the fitted generator the S003 procedure is within 2e-5 test AUC of the Bayes ceiling; refinements (polishing, restarts, bagging, semi-supervised EM, discovery) differ from S003 by < 1e-5 (not detectable on a 19,355-row private LB); self-training QDA, supervised selection, tied covariance, supervised QDA and the 142-dim mixture are measurably worse. Without label noise a public AUC <= 0.99923 is rare; with 0.5 % wrong labels it is typical, i.e. the residual public error is label noise, not boundary ranking.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed. 40 + 40 replicas of 20,000 rows from the fitted K=2 generator (highvar rule recovers the block and discovery adds 0 features in every replica). Clean labels (G0): Bayes test AUC 0.999951; the S003 procedure is only 1.7e-7 (expected AUC 2.5e-7) below it, although its boundary ranking agrees with the Bayes ranking only at Kendall 0.934: the ~186 boundary rows are nearly coin flips, so re-ordering them costs almost nothing. Paired vs S003 (expected AUC, mean +- SE over 40 replicas): reg->0 MLE +2.3e-9 +- 0.8e-9 (better in 78 %), semi-supervised EM +1.6e-10 (n.s.), polishing/discovery/seed ensemble ~0, bagging -2.4e-8, self-training QDA -2.1e-8, semi-supervised 2 comp/class -3.7e-6, GMM K=4 -3.4e-5, 142-dim GMM -3.4e-5, supervised top-13 selection -3.5e-4, quadratic logit on pseudo-labels -4.5e-4, supervised QDA -6.7e-3, tied -0.23. With 0.5 % wrong labels (G1, sklearn flip_y = 0.01) test AUC drops to 0.99478 for every near-Bayes procedure (differences +-2e-5 = noise). Public LB check: P(public AUC <= 0.99923) = 0.001 under G0 but 0.83 under G1, so S003's real public loss is label noise, not ranking; no candidate can move the private LB by a detectable amount (headroom 2.5e-7 vs private SE ~1e-4).
