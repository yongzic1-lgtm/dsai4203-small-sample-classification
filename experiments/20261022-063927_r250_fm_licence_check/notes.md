# 20261022-063927_r250_fm_licence_check

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

TabPFN v2 weights (Prior Labs License = Apache-2.0 + attribution) and TabICL weights (BSD-3) download anonymously; TabPFN-2.5+ (non-commercial, PriorLabs login + licence acceptance) refuses without a token and must not be used.

## Result

(see summary.json)

## Conclusion (write after reading the results)

Confirmed. tabpfn 9.2.0 (code Apache-2.0): ModelVersion.V2 weights (HF Prior-Labs/TabPFN-v2-clf, not gated, licence priorlabs-1-1 = Prior Labs License, Apache-2.0 + attribution 'Built with PriorLabs-TabPFN' when distributing) downloaded anonymously and fitted on random numbers in 4.6 s on an L4. The package default (TabPFN-2.5+ family, non-commercial weights) raised TabPFNLicenseError (licence acceptance at ux.priorlabs.ai + TABPFN_TOKEN required) -> NOT used. tabicl 2.2.0 (BSD-3; HF jingang/TabICL not gated, bsd-3-clause; checkpoint tabicl-classifier-v2-20260212) downloaded anonymously and fitted. Both extra venvs kept the pinned numpy 2.4.6 / sklearn 1.9.1. No credential was entered anywhere.
