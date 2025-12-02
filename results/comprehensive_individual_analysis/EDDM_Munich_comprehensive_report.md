# Comprehensive Distribution Analysis: Munich (EDDM)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 246,287 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -1412111 | -1412069 | 0.0247 | 0.0000 | No | 35.2 |
| 2 | Burr XII | 1705541 | 1705582 | 0.0131 | 0.0000 | No | 35.2 |
| 3 | F-Distribution | 1706304 | 1706346 | 0.0175 | 0.0000 | No | 35.6 |
| 4 | Generalized Gamma | 1708303 | 1708344 | 0.0198 | 0.0000 | No | 35.9 |
| 5 | Pareto | 1709008 | 1709039 | 0.0371 | 0.0000 | No | 36.6 |
| 6 | Lomax | 1709015 | 1709047 | 0.0367 | 0.0000 | No | 36.7 |
| 7 | Weibull | 1710761 | 1710792 | 0.0281 | 0.0000 | No | 35.9 |
| 8 | Exponential | 1710840 | 1710861 | 0.0244 | 0.0000 | No | 35.5 |
| 9 | Log-Normal | 1715616 | 1715648 | 0.0290 | 0.0000 | No | 37.5 |
| 10 | Inverse Gaussian | 1716412 | 1716443 | 0.0290 | 0.0000 | No | 36.9 |
| 11 | Log-Logistic | 1721736 | 1721767 | 0.0291 | 0.0000 | No | 44.0 |
| 12 | T-Distribution | 1827549 | 1827581 | 0.1489 | 0.0000 | No | 24.9 |
| 13 | Laplace | 1856205 | 1856226 | 0.1760 | 0.0000 | No | 26.7 |
| 14 | Logistic | 1873262 | 1873282 | 0.1536 | 0.0000 | No | 26.8 |
| 15 | Normal | 1979099 | 1979120 | 0.1889 | 0.0000 | No | 34.0 |
| 16 | Chi-Square | 2412800 | 2412831 | 0.5888 | 0.0000 | No | 8.5 |
| 17 | Uniform | 3122448 | 3122469 | 0.8961 | 0.0000 | No | 538.0 |
| 18 | Gamma | 3941459 | 3941491 | 0.8655 | 0.0000 | No | 1.5 |

## Best Distribution Details
**Beta Distribution**
- AIC: -1412110.83
- BIC: -1412069.17
- KS Statistic: 0.0247
- p-value: 0.000000
- Parameters: (np.float64(1.0436197823993156), np.float64(317.72265632345756), np.float64(0.0004987191744279334), np.float64(6.397226777228026))

## Predictions vs Actual Data
- **90th Percentile:** Model: 27.3 min, Data: 25.4 min
- **95th Percentile:** Model: 35.2 min, Data: 34.2 min  
- **99th Percentile:** Model: 53.5 min, Data: 62.1 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 35 minutes

*Analysis completed with 18 distribution models*
