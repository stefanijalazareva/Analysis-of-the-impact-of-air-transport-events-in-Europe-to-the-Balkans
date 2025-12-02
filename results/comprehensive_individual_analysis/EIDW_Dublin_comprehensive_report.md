# Comprehensive Distribution Analysis: Dublin (EIDW)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 134,851 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -692244 | -692205 | 0.0276 | 0.0000 | No | 39.5 |
| 2 | Burr XII | 963229 | 963269 | 0.0159 | 0.0000 | No | 39.5 |
| 3 | F-Distribution | 963701 | 963740 | 0.0203 | 0.0000 | No | 40.0 |
| 4 | Generalized Gamma | 964981 | 965021 | 0.0218 | 0.0000 | No | 40.4 |
| 5 | Pareto | 965222 | 965252 | 0.0381 | 0.0000 | No | 41.1 |
| 6 | Lomax | 965222 | 965252 | 0.0381 | 0.0000 | No | 41.1 |
| 7 | Weibull | 966560 | 966590 | 0.0277 | 0.0000 | No | 39.9 |
| 8 | Exponential | 966634 | 966653 | 0.0279 | 0.0000 | No | 39.7 |
| 9 | Log-Normal | 968618 | 968648 | 0.0295 | 0.0000 | No | 42.0 |
| 10 | Inverse Gaussian | 969283 | 969313 | 0.0298 | 0.0000 | No | 41.5 |
| 11 | Log-Logistic | 971699 | 971729 | 0.0293 | 0.0000 | No | 49.1 |
| 12 | T-Distribution | 1029655 | 1029684 | 0.1492 | 0.0000 | No | 27.7 |
| 13 | Laplace | 1047055 | 1047075 | 0.1781 | 0.0000 | No | 29.8 |
| 14 | Logistic | 1057551 | 1057571 | 0.1556 | 0.0000 | No | 30.0 |
| 15 | Normal | 1119092 | 1119112 | 0.1938 | 0.0000 | No | 38.5 |
| 16 | Chi-Square | 1409314 | 1409343 | 0.6135 | 0.0000 | No | 8.4 |
| 17 | Uniform | 1658556 | 1658575 | 0.8707 | 0.0000 | No | 445.1 |
| 18 | Gamma | 2257239 | 2257268 | 0.8659 | 0.0000 | No | 1.6 |

## Best Distribution Details
**Beta Distribution**
- AIC: -692243.86
- BIC: -692204.62
- KS Statistic: 0.0276
- p-value: 0.000000
- Parameters: (np.float64(1.033894809014933), np.float64(1813.8911084250808), np.float64(0.0004987570640614466), np.float64(49.623441912326825))

## Predictions vs Actual Data
- **90th Percentile:** Model: 30.5 min, Data: 28.1 min
- **95th Percentile:** Model: 39.5 min, Data: 38.1 min  
- **99th Percentile:** Model: 60.2 min, Data: 72.7 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 39 minutes

*Analysis completed with 18 distribution models*
