"""ssc: shared code for the DSAI4203 Small-Sample Classification Challenge.

Modules
-------
paths       canonical project paths (no hard-coded strings elsewhere)
data        load / verify the Kaggle data (raw or synthetic)
synthetic   mock data with known test labels, for pipeline tests
validation  split generation, repeated cross-validation, paired comparison
features    leakage-safe feature selectors (fit inside each fold)
models      named model factory -> sklearn Pipelines
tracking    experiment archive: run folders + registry/experiments.csv
submission  Kaggle submission files + registry/submissions.csv
"""

__version__ = "0.1.0"
