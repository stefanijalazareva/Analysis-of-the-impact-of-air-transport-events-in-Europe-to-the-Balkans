# Comprehensive Distribution Analysis: Rome Fiumicino (LIRF)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 153,381 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -831710 | -831670 | 0.0299 | 0.0000 | No | 36.6 |
| 2 | Burr XII | 1064510 | 1064550 | 0.0127 | 0.0000 | No | 36.6 |
| 3 | F-Distribution | 1064893 | 1064933 | 0.0166 | 0.0000 | No | 36.9 |
| 4 | Pareto | 1065996 | 1066025 | 0.0310 | 0.0000 | No | 37.7 |
| 5 | Lomax | 1065996 | 1066025 | 0.0310 | 0.0000 | No | 37.7 |
| 6 | Generalized Gamma | 1066405 | 1066444 | 0.0198 | 0.0000 | No | 37.4 |
| 7 | Weibull | 1068094 | 1068123 | 0.0289 | 0.0000 | No | 37.0 |
| 8 | Exponential | 1068912 | 1068932 | 0.0284 | 0.0000 | No | 36.0 |
| 9 | Log-Normal | 1070663 | 1070693 | 0.0288 | 0.0000 | No | 39.2 |
| 10 | Inverse Gaussian | 1071550 | 1071579 | 0.0291 | 0.0000 | No | 38.6 |
| 11 | Log-Logistic | 1074070 | 1074099 | 0.0283 | 0.0000 | No | 46.4 |
| 12 | T-Distribution | 1145552 | 1145581 | 0.1576 | 0.0000 | No | 25.1 |
| 13 | Laplace | 1167706 | 1167726 | 0.1890 | 0.0000 | No | 27.1 |
| 14 | Logistic | 1181771 | 1181791 | 0.1647 | 0.0000 | No | 27.4 |
| 15 | Normal | 1257637 | 1257657 | 0.2056 | 0.0000 | No | 36.0 |
| 16 | Chi-Square | 1503310 | 1503340 | 0.5775 | 0.0000 | No | 8.5 |
| 17 | Uniform | 1900421 | 1900441 | 0.8808 | 0.0000 | No | 465.8 |
| 18 | Gamma | 2483603 | 2483633 | 0.8695 | 0.0000 | No | 1.3 |

## Best Distribution Details
**Beta Distribution**
- AIC: -831710.14
- BIC: -831670.38
- KS Statistic: 0.0299
- p-value: 0.000000
- Parameters: (np.float64(0.9602989213909369), np.float64(125.11144295078327), np.float64(0.0004999999999999999), np.float64(3.2128869698483316))

## Predictions vs Actual Data
- **90th Percentile:** Model: 28.1 min, Data: 25.8 min
- **95th Percentile:** Model: 36.6 min, Data: 35.5 min  
- **99th Percentile:** Model: 56.1 min, Data: 67.7 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 37 minutes

*Analysis completed with 18 distribution models*
