# References（规则 4：所有用到的库、方法、教程都要引用）

规则（`docs/ARCHIVE_RULES.md` §7）：用到新的库、方法、教程、博客或问答帖时，当天加一条。报告的 References 只从这里拷贝。**Status**：`used`（我们确实用了，必须出现在报告里）/ `candidate`（用到才引用；用了就改成 `used`）。教程和网页要写 URL 和访问日期。

## Software

```bibtex
@article{sklearn,          % status: used (all models, CV, metrics)
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and others},
  title   = {Scikit-learn: Machine Learning in {P}ython},
  journal = {Journal of Machine Learning Research}, volume = {12}, pages = {2825--2830}, year = {2011},
  note    = {version 1.9.1}
}
@article{numpy,            % status: used
  author  = {Harris, C. R. and Millman, K. J. and van der Walt, S. J. and others},
  title   = {Array programming with {NumPy}},
  journal = {Nature}, volume = {585}, pages = {357--362}, year = {2020}, note = {version 2.4.6}
}
@inproceedings{pandas,     % status: used
  author    = {McKinney, W.},
  title     = {Data Structures for Statistical Computing in {P}ython},
  booktitle = {Proceedings of the 9th Python in Science Conference}, pages = {56--61}, year = {2010},
  note      = {pandas version 3.0.6}
}
@article{scipy,            % status: used (t-test, rank transform)
  author  = {Virtanen, P. and Gommers, R. and Oliphant, T. E. and others},
  title   = {{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in {P}ython},
  journal = {Nature Methods}, volume = {17}, pages = {261--272}, year = {2020}, note = {version 1.17.1}
}
@article{matplotlib,       % status: used (figures)
  author  = {Hunter, J. D.},
  title   = {Matplotlib: A 2D Graphics Environment},
  journal = {Computing in Science \& Engineering}, volume = {9}, number = {3}, pages = {90--95}, year = {2007},
  note    = {version 3.11}
}
@inproceedings{lightgbm,   % status: used (r203 LightGBM runs, negative result in sup.md; installed 4.7.0)
  author    = {Ke, G. and Meng, Q. and Finley, T. and Wang, T. and Chen, W. and Ma, W. and Ye, Q. and Liu, T.-Y.},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems 30}, pages = {3146--3154}, year = {2017},
  note      = {version 4.7}
}
@inproceedings{xgboost,    % status: used (r203 XGBoost runs, negative result in sup.md; installed 3.2.0)
  author    = {Chen, T. and Guestrin, C.},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794}, year = {2016}, note = {version 3.2}
}
@article{seaborn,          % status: candidate (not imported anywhere, R945 on 2026-10-22: do not cite)
  author  = {Waskom, M. L.},
  title   = {seaborn: statistical data visualization},
  journal = {Journal of Open Source Software}, volume = {6}, number = {60}, pages = {3021}, year = {2021}
}
@inproceedings{statsmodels, % status: candidate (not imported anywhere, R945 on 2026-10-22: do not cite)
  author    = {Seabold, S. and Perktold, J.},
  title     = {statsmodels: Econometric and Statistical Modeling with {P}ython},
  booktitle = {Proceedings of the 9th Python in Science Conference}, pages = {92--96}, year = {2010}
}
@misc{kaggleapi,           % status: candidate (cite if scripts/kaggle_submit.py is used; S001-S004 were uploaded through the web page)
  author = {{Kaggle}}, title = {Kaggle API (kaggle Python package, version 2.2.4)},
  howpublished = {\url{https://github.com/Kaggle/kaggle-api}}
}
```

Other packages in `requirements-lock.txt` that we use directly: PyYAML (configs), joblib (parallel CV), tabulate (tables), JupyterLab (exploration). List them in the report's software note with versions.

### Software added 2026-10-22（Verify 第 1 轮，`research/reports/verify_rules.md` §2；已用 `research/r945_citation_inventory.py` 核对代码里的 import）

```bibtex
@article{hollmann2025, % status: used (analysed comparison only, R250/R251)
  author={Hollmann, N. and M{\"u}ller, S. and Purucker, L. and Krishnakumar, A. and K{\"o}rfer, M. and Hoo, S. B. and Schirrmeister, R. T. and Hutter, F.},
  title={Accurate Predictions on Small Data with a Tabular Foundation Model}, journal={Nature}, volume={637}, pages={319--326}, year={2025}, doi={10.1038/s41586-024-08328-6},
  note={tabpfn 9.2.0; weights tabpfn-v2-classifier-finetuned-zk73skhh.ckpt (= v2\_default), huggingface.co/Prior-Labs/TabPFN-v2-clf, accessed 2026-10-22; Prior Labs License (Apache-2.0 with attribution)}}
@inproceedings{qu2025, % status: used (R250/R251)
  author={Qu, J. and Holzm{\"u}ller, D. and Varoquaux, G. and Le Morvan, M.}, title={{TabICL}: A Tabular Foundation Model for In-Context Learning on Large Data}, booktitle={International Conference on Machine Learning}, year={2025}}
@misc{qu2026, % status: used (R250/R251)
  author={Qu, J. and Holzm{\"u}ller, D. and Varoquaux, G. and Le Morvan, M.}, title={{TabICLv2}: A Better, Faster, Scalable, and Open Tabular Foundation Model}, howpublished={arXiv preprint arXiv:2602.11139}, year={2026},
  note={tabicl 2.2.0, checkpoint tabicl-classifier-v2-20260212, huggingface.co/jingang/TabICL, BSD-3-Clause}}
@inproceedings{pytorch, % status: used (backend of TabPFN/TabICL)
  author={Paszke, A. and Gross, S. and Massa, F. and others}, title={{PyTorch}: An Imperative Style, High-Performance Deep Learning Library}, booktitle={Advances in Neural Information Processing Systems 32}, pages={8024--8035}, year={2019}, note={torch 2.14.1}}
@misc{numpyro, % status: used (horseshoe NUTS, R253/R257)
  author={Phan, D. and Pradhan, N. and Jankowiak, M.}, title={Composable Effects for Flexible and Accelerated Probabilistic Programming in {NumPyro}}, howpublished={arXiv:1912.11554}, year={2019}, note={numpyro 0.22.0}}
@article{bingham2019, % status: used (cited with NumPyro)
  author={Bingham, E. and Chen, J. P. and Jankowiak, M. and Obermeyer, F. and Pradhan, N. and Karaletsos, T. and Singh, R. and Szerlip, P. and Horsfall, P. and Goodman, N. D.}, title={Pyro: Deep Universal Probabilistic Programming}, journal={Journal of Machine Learning Research}, volume={20}, number={28}, pages={1--6}, year={2019}}
@misc{jax, % status: used (NumPyro backend)
  author={Bradbury, J. and Frostig, R. and Hawkins, P. and Johnson, M. J. and Leary, C. and Maclaurin, D. and Necula, G. and Paszke, A. and VanderPlas, J. and Wanderman-Milne, S. and Zhang, Q.}, title={{JAX}: Composable Transformations of {P}ython+{N}um{P}y Programs}, howpublished={\url{https://github.com/jax-ml/jax}}, year={2018}, note={jax/jaxlib 0.10.2}}
```

Exact versions for the software note (`requirements-lock.txt`; Colab-only packages from the run configs, `verify_rules.md` §2 Table 2): matplotlib 3.11.2, lightgbm 4.7.0, xgboost 3.2.0, joblib 1.6.0, PyYAML 6.0.3, tabulate 0.10.0, threadpoolctl 3.7.0, huggingface_hub 2.2.0 (R250/R251). Compute platform: Google Colab, driven by the Colab CLI (`colab/remote.py`). PyMC and Pyro are not imported anywhere (R945); Pyro is cited only as NumPyro's companion reference (`bingham2019`).

## Methods

```bibtex
@article{nadeau2003,       % status: used (corrected resampled t-test, ssc.validation.paired_compare)
  author  = {Nadeau, C. and Bengio, Y.},
  title   = {Inference for the Generalization Error},
  journal = {Machine Learning}, volume = {52}, number = {3}, pages = {239--281}, year = {2003}
}
@article{hanley1982,       % status: used (AUC standard error, ssc.validation.auc_standard_error)
  author  = {Hanley, J. A. and McNeil, B. J.},
  title   = {The meaning and use of the area under a receiver operating characteristic ({ROC}) curve},
  journal = {Radiology}, volume = {143}, number = {1}, pages = {29--36}, year = {1982}
}
@article{efron1979,        % status: used (bootstrap, E4 and ssc.features.bootstrap_topk_frequency)
  author  = {Efron, B.},
  title   = {Bootstrap Methods: Another Look at the Jackknife},
  journal = {The Annals of Statistics}, volume = {7}, number = {1}, pages = {1--26}, year = {1979}
}
@book{efron1993,           % status: used (parametric bootstrap, R103, R153)
  author = {Efron, B. and Tibshirani, R. J.},
  title  = {An Introduction to the Bootstrap}, publisher = {Chapman \& Hall}, year = {1993}
}
@book{hastie2009,          % status: used (one-standard-error rule, VALIDATION_PROTOCOL.md section 5; E3 text Sec. 7.10.2; textbook LDA, Gaussian NB, polynomial logistic regression, PCA in r003, r200, r201, r205)
  author = {Hastie, T. and Tibshirani, R. and Friedman, J.},
  title  = {The Elements of Statistical Learning}, edition = {2nd}, publisher = {Springer}, year = {2009},
  note   = {Sec. 7.10, cross-validation}
}
@article{varma2006,        % status: used (optimism of the selected model; nested selection r205, protocol)
  author  = {Varma, S. and Simon, R.},
  title   = {Bias in error estimation when using cross-validation for model selection},
  journal = {BMC Bioinformatics}, volume = {7}, pages = {91}, year = {2006}
}
@article{cawley2010,       % status: used (over-fitting in model selection; r205, protocol)
  author  = {Cawley, G. C. and Talbot, N. L. C.},
  title   = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  journal = {Journal of Machine Learning Research}, volume = {11}, pages = {2079--2107}, year = {2010}
}
@article{ambroise2002,     % status: candidate (E3: feature-selection bias outside CV; set to used when E3 or E4 is submitted)
  author  = {Ambroise, C. and McLachlan, G. J.},
  title   = {Selection bias in gene extraction on the basis of microarray gene-expression data},
  journal = {Proceedings of the National Academy of Sciences}, volume = {99}, number = {10}, pages = {6562--6566}, year = {2002}
}
@article{meinshausen2010,  % status: candidate (E4: stability selection; set to used when E4 is submitted)
  author  = {Meinshausen, N. and B{\"u}hlmann, P.},
  title   = {Stability selection},
  journal = {Journal of the Royal Statistical Society: Series B}, volume = {72}, number = {4}, pages = {417--473}, year = {2010}
}
@article{ledoit2004,       % status: used (shrinkage LDA, sklearn shrinkage='auto': r003, r200; QDA-LW r206-r210)
  author  = {Ledoit, O. and Wolf, M.},
  title   = {A well-conditioned estimator for large-dimensional covariance matrices},
  journal = {Journal of Multivariate Analysis}, volume = {88}, number = {2}, pages = {365--411}, year = {2004}
}
@article{tibshirani1996,   % status: used (L1 logistic regression: S001, r200, r300)
  author  = {Tibshirani, R.},
  title   = {Regression Shrinkage and Selection via the Lasso},
  journal = {Journal of the Royal Statistical Society: Series B}, volume = {58}, number = {1}, pages = {267--288}, year = {1996}
}
@inproceedings{kohavi1995, % status: candidate (E1: CV vs hold-out; set to used when E1 is submitted)
  author    = {Kohavi, R.},
  title     = {A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection},
  booktitle = {Proceedings of the 14th International Joint Conference on Artificial Intelligence}, pages = {1137--1143}, year = {1995}
}
@article{bengio2004,       % status: candidate (E1: no unbiased estimator of the k-fold CV variance; set to used when E1 is submitted)
  author  = {Bengio, Y. and Grandvalet, Y.},
  title   = {No Unbiased Estimator of the Variance of K-Fold Cross-Validation},
  journal = {Journal of Machine Learning Research}, volume = {5}, pages = {1089--1105}, year = {2004}
}
@article{figueroa2012,     % status: used (E2: inverse power law learning curve, cited in the E2 text)
  author  = {Figueroa, R. L. and Zeng-Treitler, Q. and Kandula, S. and Ngo, L. H.},
  title   = {Predicting sample size required for classification performance},
  journal = {BMC Medical Informatics and Decision Making}, volume = {12}, pages = {8}, year = {2012},
  doi     = {10.1186/1472-6947-12-8}
}
```

### Methods added 2026-10-22（Verify 第 1 轮，`research/reports/verify_rules.md` §2；已用 `research/r945_citation_inventory.py` 和手工 grep 核对代码）

```bibtex
@article{dempster1977, % status: used (EM; TransductiveGMM = S003)
  author={Dempster, A. P. and Laird, N. M. and Rubin, D. B.}, title={Maximum Likelihood from Incomplete Data via the {EM} Algorithm},
  journal={Journal of the Royal Statistical Society: Series B}, volume={39}, number={1}, pages={1--38}, year={1977}}
@article{schwarz1978, % status: used (BIC, R101/R150/R254)
  author={Schwarz, G.}, title={Estimating the Dimension of a Model}, journal={The Annals of Statistics}, volume={6}, number={2}, pages={461--464}, year={1978}}
@misc{guyon2003, % status: used (generator identified as sklearn make_classification, R150-R153)
  author={Guyon, I.}, title={Design of Experiments for the {NIPS} 2003 Variable Selection Benchmark}, year={2003}}
@article{friedman1989, % status: used (QDA reg_param: S002, r201, r210)
  author={Friedman, J. H.}, title={Regularized Discriminant Analysis}, journal={Journal of the American Statistical Association}, volume={84}, number={405}, pages={165--175}, year={1989}}
@article{fan2008, % status: used (liblinear: S001, S004)
  author={Fan, R.-E. and Chang, K.-W. and Hsieh, C.-J. and Wang, X.-R. and Lin, C.-J.}, title={{LIBLINEAR}: A Library for Large Linear Classification}, journal={Journal of Machine Learning Research}, volume={9}, pages={1871--1874}, year={2008}}
@inproceedings{defazio2014, % status: used (saga, r200)
  author={Defazio, A. and Bach, F. and Lacoste-Julien, S.}, title={{SAGA}: A Fast Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives}, booktitle={Advances in Neural Information Processing Systems 27}, pages={1646--1654}, year={2014}}
@article{chang2011, % status: used (SVC, r201/r205)
  author={Chang, C.-C. and Lin, C.-J.}, title={{LIBSVM}: A Library for Support Vector Machines}, journal={ACM Transactions on Intelligent Systems and Technology}, volume={2}, number={3}, pages={27:1--27:27}, year={2011}}
@incollection{platt1999, % status: used (SVC probability=True, r205)
  author={Platt, J. C.}, title={Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods}, booktitle={Advances in Large Margin Classifiers}, publisher={MIT Press}, pages={61--74}, year={1999}}
@article{cover1967, % status: used (kNN, r201)
  author={Cover, T. M. and Hart, P. E.}, title={Nearest Neighbor Pattern Classification}, journal={IEEE Transactions on Information Theory}, volume={13}, number={1}, pages={21--27}, year={1967}}
@article{breiman2001, % status: used (random forest, r203)
  author={Breiman, L.}, title={Random Forests}, journal={Machine Learning}, volume={45}, number={1}, pages={5--32}, year={2001}}
@article{geurts2006, % status: used (extra trees, r203)
  author={Geurts, P. and Ernst, D. and Wehenkel, L.}, title={Extremely Randomized Trees}, journal={Machine Learning}, volume={63}, number={1}, pages={3--42}, year={2006}}
@book{rasmussen2006, % status: used (GP classification, r202/r255)
  author={Rasmussen, C. E. and Williams, C. K. I.}, title={Gaussian Processes for Machine Learning}, publisher={MIT Press}, year={2006}}
@inproceedings{yarowsky1995, % status: used (self-training, r204/r209/r210)
  author={Yarowsky, D.}, title={Unsupervised Word Sense Disambiguation Rivaling Supervised Methods}, booktitle={Proceedings of the 33rd Annual Meeting of the Association for Computational Linguistics}, pages={189--196}, year={1995}}
@inproceedings{lee2013, % status: used (pseudo-labels, r102_quadlogit_pseudo)
  author={Lee, D.-H.}, title={Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks}, booktitle={ICML 2013 Workshop on Challenges in Representation Learning}, year={2013}}
@inproceedings{zhou2004, % status: used (LabelSpreading, r204/r209)
  author={Zhou, D. and Bousquet, O. and Lal, T. N. and Weston, J. and Sch{\"o}lkopf, B.}, title={Learning with Local and Global Consistency}, booktitle={Advances in Neural Information Processing Systems 16}, pages={321--328}, year={2004}}
@techreport{zhu2002, % status: used (LabelPropagation, r204)
  author={Zhu, X. and Ghahramani, Z.}, title={Learning from Labeled and Unlabeled Data with Label Propagation}, institution={Carnegie Mellon University}, number={CMU-CALD-02-107}, year={2002}}
@article{nigam2000, % status: used (semi-supervised EM, r102_semisup_eps01)
  author={Nigam, K. and McCallum, A. K. and Thrun, S. and Mitchell, T.}, title={Text Classification from Labeled and Unlabeled Documents using {EM}}, journal={Machine Learning}, volume={39}, number={2--3}, pages={103--134}, year={2000}}
@article{blei2006, % status: used (DP-GMM, R254)
  author={Blei, D. M. and Jordan, M. I.}, title={Variational Inference for {D}irichlet Process Mixtures}, journal={Bayesian Analysis}, volume={1}, number={1}, pages={121--143}, year={2006}}
@book{bishop2006, % status: used (evidence approximation, R254/R255)
  author={Bishop, C. M.}, title={Pattern Recognition and Machine Learning}, publisher={Springer}, year={2006}}
@techreport{murphy2007, % status: used (NIW conjugate Bayesian QDA, R255/R256)
  author={Murphy, K. P.}, title={Conjugate {B}ayesian Analysis of the {G}aussian Distribution}, institution={University of British Columbia}, year={2007}}
@book{gelman2013, % status: used (Student-t posterior predictive, R255)
  author={Gelman, A. and Carlin, J. B. and Stern, H. S. and Dunson, D. B. and Vehtari, A. and Rubin, D. B.}, title={Bayesian Data Analysis}, edition={3rd}, publisher={CRC Press}, year={2013}}
@article{carvalho2010, % status: used (horseshoe, R253/R257)
  author={Carvalho, C. M. and Polson, N. G. and Scott, J. G.}, title={The Horseshoe Estimator for Sparse Signals}, journal={Biometrika}, volume={97}, number={2}, pages={465--480}, year={2010}}
@article{piironen2017, % status: used (regularised horseshoe, R253/R257)
  author={Piironen, J. and Vehtari, A.}, title={Sparsity Information and Regularization in the Horseshoe and Other Shrinkage Priors}, journal={Electronic Journal of Statistics}, volume={11}, number={2}, pages={5018--5051}, year={2017}}
@article{hoffman2014, % status: used (NUTS, R253/R257)
  author={Hoffman, M. D. and Gelman, A.}, title={The {No-U-Turn} Sampler: Adaptively Setting Path Lengths in {H}amiltonian {M}onte {C}arlo}, journal={Journal of Machine Learning Research}, volume={15}, pages={1593--1623}, year={2014}}
@article{vehtari2021, % status: used (R-hat, R257)
  author={Vehtari, A. and Gelman, A. and Simpson, D. and Carpenter, B. and B{\"u}rkner, P.-C.}, title={Rank-Normalization, Folding, and Localization: An Improved $\widehat{R}$ for Assessing Convergence of {MCMC}}, journal={Bayesian Analysis}, volume={16}, number={2}, pages={667--718}, year={2021}}
@article{gelman1992, % status: used (split Gelman-Rubin, r257:69)
  author={Gelman, A. and Rubin, D. B.}, title={Inference from Iterative Simulation Using Multiple Sequences}, journal={Statistical Science}, volume={7}, number={4}, pages={457--472}, year={1992}}
@book{golub2013, % status: used (QR with column pivoting, lib_fm.py:568)
  author={Golub, G. H. and Van Loan, C. F.}, title={Matrix Computations}, edition={4th}, publisher={Johns Hopkins University Press}, year={2013}}
@article{mardia1970, % status: used (multivariate normality, R150/R151)
  author={Mardia, K. V.}, title={Measures of Multivariate Skewness and Kurtosis with Applications}, journal={Biometrika}, volume={57}, number={3}, pages={519--530}, year={1970}}
@article{dagostino1973, % status: used (scipy.stats.normaltest, R002/R150)
  author={D'Agostino, R. and Pearson, E. S.}, title={Tests for Departure from Normality. Empirical Results for the Distributions of $b_2$ and $\sqrt{b_1}$}, journal={Biometrika}, volume={60}, number={3}, pages={613--622}, year={1973}}
@article{pritchard1999, % status: used (rejection ABC, R153/R156)
  author={Pritchard, J. K. and Seielstad, M. T. and Perez-Lezaun, A. and Feldman, M. W.}, title={Population Growth of Human {Y} Chromosomes: A Study of {Y} Chromosome Microsatellites}, journal={Molecular Biology and Evolution}, volume={16}, number={12}, pages={1791--1798}, year={1999}}
@article{beaumont2002, % status: used (ABC, R156)
  author={Beaumont, M. A. and Zhang, W. and Balding, D. J.}, title={Approximate {B}ayesian Computation in Population Genetics}, journal={Genetics}, volume={162}, number={4}, pages={2025--2035}, year={2002}}
@article{wilks1938, % status: used (likelihood-ratio chi-square, R100/R150)
  author={Wilks, S. S.}, title={The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses}, journal={The Annals of Mathematical Statistics}, volume={9}, number={1}, pages={60--62}, year={1938}}
```

## Tutorials, posts, course material

| Date accessed | Title | URL | Used for | Status |
|---|---|---|---|---|
| | | | | |
| 2026-10-22 | TabPFN v2 classifier model card (Prior Labs, Hugging Face) | https://huggingface.co/Prior-Labs/TabPFN-v2-clf | R250 licence check (card saved as `experiments/20261022-063927_r250_fm_licence_check/tabpfn_README.txt`); R251 weights `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (= v2_default) | used (analysis only, never in a submission) |
| 2026-10-22 | Prior Labs License (Apache 2.0 with additional attribution requirement), `priorlabs-1-1` | https://priorlabs.ai/tabpfn-license/ | licence of the TabPFN v2 weights; link taken from the archived model card (R250). The licence text itself is not archived yet (TODO, `verify_rules.md` §4) | used |
| 2026-10-22 | TabICL model card (Hugging Face) | https://huggingface.co/jingang/TabICL | R250 licence check (card saved as `experiments/20261022-063927_r250_fm_licence_check/tabicl_README.txt`); R251 checkpoint `tabicl-classifier-v2-20260212`, BSD-3-Clause | used (analysis only, never in a submission) |

## AI tools

Generative-AI use is disclosed separately (`registry/ai_usage.md`), not as a reference entry, unless the course asks otherwise.
