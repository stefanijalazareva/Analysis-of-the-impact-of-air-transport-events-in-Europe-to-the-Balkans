# Comprehensive Distribution Analysis: Barcelona (LEBL)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 250,813 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -1275424 | -1275382 | 0.0132 | 0.0000 | No | 36.7 |
| 2 | Burr XII | 1804273 | 1804315 | 0.0153 | 0.0000 | No | 40.2 |
| 3 | F-Distribution | 1805307 | 1805348 | 0.0204 | 0.0000 | No | 40.8 |
| 4 | Generalized Gamma | 1807715 | 1807756 | 0.0227 | 0.0000 | No | 41.1 |
| 5 | Pareto | 1809642 | 1809674 | 0.0446 | 0.0000 | No | 42.1 |
| 6 | Lomax | 1809792 | 1809823 | 0.0470 | 0.0000 | No | 41.7 |
| 7 | Weibull | 1811748 | 1811779 | 0.0309 | 0.0000 | No | 40.5 |
| 8 | Exponential | 1811805 | 1811826 | 0.0308 | 0.0000 | No | 40.8 |
| 9 | Log-Normal | 1812815 | 1812846 | 0.0264 | 0.0000 | No | 42.4 |
| 10 | Inverse Gaussian | 1814156 | 1814187 | 0.0268 | 0.0000 | No | 42.1 |
| 11 | Log-Logistic | 1817873 | 1817904 | 0.0266 | 0.0000 | No | 48.9 |
| 12 | T-Distribution | 1922954 | 1922985 | 0.1463 | 0.0000 | No | 28.2 |
| 13 | Laplace | 1955276 | 1955297 | 0.1747 | 0.0000 | No | 30.4 |
| 14 | Logistic | 1976387 | 1976408 | 0.1532 | 0.0000 | No | 30.6 |
| 15 | Normal | 2087954 | 2087975 | 0.1903 | 0.0000 | No | 39.2 |
| 16 | Chi-Square | 2671679 | 2671711 | 0.6273 | 0.0000 | No | 8.4 |
| 17 | Uniform | 3088629 | 3088650 | 0.8682 | 0.0000 | No | 448.5 |
| 18 | Gamma | 4220062 | 4220093 | 0.8651 | 0.0000 | No | 1.7 |

## Best Distribution Details
**Beta Distribution**
- AIC: -1275423.59
- BIC: -1275381.86
- KS Statistic: 0.0132
- p-value: 0.000000
- Parameters: (np.float64(1.1852496427208319), np.float64(1466643368016494.2), np.float64(0.000483365433788134), np.float64(33807675689921.86))

## Predictions vs Actual Data
- **90th Percentile:** Model: 28.7 min, Data: 28.8 min
- **95th Percentile:** Model: 36.7 min, Data: 39.6 min  
- **99th Percentile:** Model: 54.8 min, Data: 73.1 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 37 minutes

*Analysis completed with 18 distribution models*
