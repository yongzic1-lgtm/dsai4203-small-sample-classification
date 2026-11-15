| parameter            | estimate                                         | evidence                                                                                      |
|:---------------------|:-------------------------------------------------|:----------------------------------------------------------------------------------------------|
| n_samples            | 20000                                            | 250 train + 19,750 test; GMM weights 0.4998/0.5002                                            |
| n_features           | 150                                              | columns 0..149                                                                                |
| n_informative        | 12                                               | rank of the 20 structured columns = 12 (8 eigenvalues < 1e-9 x max)                           |
| n_redundant          | 8                                                | 8 exact linear dependencies, min R^2 1.000000000000                                           |
| n_repeated           | 0                                                | no pair with |corr| > 1-1e-8                                                                  |
| n_useless (noise)    | 130                                              | independent Gaussian columns, max |pairwise corr| 0.0291 (null sd 0.0071)                     |
| n_clusters_per_class | 1                                                | BIC best K=2 overall; within each class best K={'class0_post<=0.5': 1, 'class1_post>0.5': 1}  |
| weights              | None (balanced)                                  | 10016 vs 9984 rows by posterior > 0.5                                                         |
| hypercube            | True                                             | 8 informative columns with zero class-mean difference (max |z| 2.08); active |z| >= 47.3      |
| class_sep            | 0.709 (raw estimate; see R152 calibration)       | hamming distance of the two centroids = 4; per active coordinate [0.688, 0.669, 0.812, 0.669] |
| flip_y               | MLE 0.0000, 95% upper 0.0152 (train labels only) | 0 confident contradictions in 250 train rows                                                  |
| shift / scale        | 0.0 / 1.0, then x*a+b with a~U(0.5,3), b~U(-5,5) | noise std KS p=0.256 vs U(0.5,3); mean KS p=0.865 vs U(-5,5); |mean| independent of std       |
| A_k entries U(-1,1)  | consistent                                       | within-class u-cov off-diagonal sd 1.047 (expected 1.155)                                     |
| B entries U(-1,1)    | consistent                                       | implied B in [-1.08, 1.11], KS p vs U(-1,1) 0.732                                             |
| shuffle              | True                                             | roles scattered over the column indices                                                       |