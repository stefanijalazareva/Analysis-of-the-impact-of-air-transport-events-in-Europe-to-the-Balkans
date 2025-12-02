# Comprehensive Distribution Analysis: Madrid Barajas (LEMD)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 318,721 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -2084923 | -2084881 | 0.1218 | 0.0000 | No | 34.0 |
| 2 | Burr XII | 2287644 | 2287687 | 0.0162 | 0.0000 | No | 39.1 |
| 3 | F-Distribution | 2289445 | 2289487 | 0.0220 | 0.0000 | No | 39.7 |
| 4 | Generalized Gamma | 2293321 | 2293364 | 0.0240 | 0.0000 | No | 40.2 |
| 5 | Pareto | 2296980 | 2297012 | 0.0514 | 0.0000 | No | 41.6 |
| 6 | Lomax | 2296980 | 2297012 | 0.0514 | 0.0000 | No | 41.6 |
| 7 | Log-Normal | 2299060 | 2299092 | 0.0279 | 0.0000 | No | 41.0 |
| 8 | Exponential | 2300126 | 2300147 | 0.0394 | 0.0000 | No | 40.7 |
| 9 | Inverse Gaussian | 2301795 | 2301827 | 0.0279 | 0.0000 | No | 41.0 |
| 10 | Log-Logistic | 2305925 | 2305957 | 0.0279 | 0.0000 | No | 46.9 |
| 11 | T-Distribution | 2429148 | 2429180 | 0.1383 | 0.0000 | No | 28.1 |
| 12 | Laplace | 2470173 | 2470195 | 0.1658 | 0.0000 | No | 30.2 |
| 13 | Logistic | 2491815 | 2491836 | 0.1451 | 0.0000 | No | 30.2 |
| 14 | Normal | 2726967 | 2726989 | 0.2182 | 0.0000 | No | 42.3 |
| 15 | Weibull | 3033380 | 3033412 | 0.6360 | 0.0000 | No | 34.4 |
| 16 | Chi-Square | 3386255 | 3386287 | 0.6322 | 0.0000 | No | 8.5 |
| 17 | Uniform | 4645830 | 4645851 | 0.9447 | 0.0000 | No | 1389.9 |
| 18 | Gamma | 7543076 | 7543108 | 0.9865 | 0.0000 | No | 0.0 |

## Best Distribution Details
**Beta Distribution**
- AIC: -2084923.39
- BIC: -2084880.70
- KS Statistic: 0.1218
- p-value: 0.000000
- Parameters: (np.float64(65.02642455537071), np.float64(17615288527169.86), np.float64(-0.054438265782868894), np.float64(17353674602.344166))

## Predictions vs Actual Data
- **90th Percentile:** Model: 29.3 min, Data: 28.2 min
- **95th Percentile:** Model: 34.0 min, Data: 37.6 min  
- **99th Percentile:** Model: 43.2 min, Data: 68.8 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 34 minutes

*Analysis completed with 18 distribution models*
