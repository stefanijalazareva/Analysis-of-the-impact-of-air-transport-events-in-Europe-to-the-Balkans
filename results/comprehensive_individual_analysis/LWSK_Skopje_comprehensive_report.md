# Comprehensive Distribution Analysis: Skopje (LWSK)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 11,374 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -59425 | -59396 | 0.0249 | 0.0000 | No | 34.3 |
| 2 | Burr XII | 78697 | 78726 | 0.0198 | 0.0003 | No | 33.7 |
| 3 | F-Distribution | 78752 | 78782 | 0.0233 | 0.0000 | No | 34.2 |
| 4 | Generalized Gamma | 78830 | 78859 | 0.0219 | 0.0000 | No | 34.3 |
| 5 | Weibull | 78909 | 78931 | 0.0253 | 0.0000 | No | 34.0 |
| 6 | Pareto | 78979 | 79001 | 0.0493 | 0.0000 | No | 35.8 |
| 7 | Lomax | 78979 | 79001 | 0.0493 | 0.0000 | No | 35.8 |
| 8 | Exponential | 78989 | 79003 | 0.0452 | 0.0000 | No | 35.5 |
| 9 | Log-Normal | 79180 | 79202 | 0.0351 | 0.0000 | No | 35.5 |
| 10 | Inverse Gaussian | 79200 | 79222 | 0.0347 | 0.0000 | No | 35.1 |
| 11 | Log-Logistic | 79500 | 79522 | 0.0335 | 0.0000 | No | 40.9 |
| 12 | T-Distribution | 83653 | 83675 | 0.1336 | 0.0000 | No | 24.8 |
| 13 | Laplace | 84756 | 84771 | 0.1555 | 0.0000 | No | 26.5 |
| 14 | Logistic | 85167 | 85181 | 0.1369 | 0.0000 | No | 26.3 |
| 15 | Normal | 89830 | 89845 | 0.1726 | 0.0000 | No | 32.5 |
| 16 | Chi-Square | 111712 | 111734 | 0.6033 | 0.0000 | No | 8.6 |
| 17 | Uniform | 138274 | 138289 | 0.8820 | 0.0000 | No | 414.5 |
| 18 | Gamma | 194047 | 194069 | 0.9060 | 0.0000 | No | 0.8 |

## Best Distribution Details
**Beta Distribution**
- AIC: -59425.04
- BIC: -59395.68
- KS Statistic: 0.0249
- p-value: 0.000001
- Parameters: (np.float64(1.1381440739817306), np.float64(2181987901.042654), np.float64(0.0004894921025881827), np.float64(52340457.3193118))

## Predictions vs Actual Data
- **90th Percentile:** Model: 26.8 min, Data: 24.7 min
- **95th Percentile:** Model: 34.3 min, Data: 32.4 min  
- **99th Percentile:** Model: 51.7 min, Data: 57.5 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 34 minutes

*Analysis completed with 18 distribution models*
