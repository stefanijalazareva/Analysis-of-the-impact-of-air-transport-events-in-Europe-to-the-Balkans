# Comprehensive Distribution Analysis: Frankfurt (EDDF)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 311,706 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -1884374 | -1884331 | 0.0294 | 0.0000 | No | 44.5 |
| 2 | Burr XII | 2286294 | 2286336 | 0.0102 | 0.0000 | No | 44.8 |
| 3 | F-Distribution | 2286888 | 2286930 | 0.0137 | 0.0000 | No | 45.2 |
| 4 | Pareto | 2289254 | 2289286 | 0.0296 | 0.0000 | No | 46.0 |
| 5 | Lomax | 2289254 | 2289286 | 0.0296 | 0.0000 | No | 46.0 |
| 6 | Generalized Gamma | 2289456 | 2289499 | 0.0174 | 0.0000 | No | 45.7 |
| 7 | Weibull | 2293915 | 2293947 | 0.0265 | 0.0000 | No | 44.3 |
| 8 | Exponential | 2295021 | 2295042 | 0.0298 | 0.0000 | No | 43.8 |
| 9 | Log-Normal | 2297726 | 2297758 | 0.0257 | 0.0000 | No | 48.1 |
| 10 | Inverse Gaussian | 2299099 | 2299131 | 0.0261 | 0.0000 | No | 47.2 |
| 11 | Log-Logistic | 2304570 | 2304602 | 0.0261 | 0.0000 | No | 57.0 |
| 12 | T-Distribution | 2454835 | 2454867 | 0.1610 | 0.0000 | No | 30.5 |
| 13 | Laplace | 2498556 | 2498577 | 0.1927 | 0.0000 | No | 33.0 |
| 14 | Logistic | 2529100 | 2529121 | 0.1673 | 0.0000 | No | 33.5 |
| 15 | Normal | 2671292 | 2671313 | 0.2029 | 0.0000 | No | 43.5 |
| 16 | Chi-Square | 3435244 | 3435276 | 0.6239 | 0.0000 | No | 8.4 |
| 17 | Uniform | 4178724 | 4178746 | 0.9033 | 0.0000 | No | 774.1 |
| 18 | Gamma | 5592399 | 5592431 | 0.8790 | 0.0000 | No | 1.3 |

## Best Distribution Details
**Beta Distribution**
- AIC: -1884373.61
- BIC: -1884331.01
- KS Statistic: 0.0294
- p-value: 0.000000
- Parameters: (np.float64(0.9712261380801487), np.float64(276.4640569359377), np.float64(0.0004999999999999999), np.float64(5.110290032870343))

## Predictions vs Actual Data
- **90th Percentile:** Model: 34.2 min, Data: 32.0 min
- **95th Percentile:** Model: 44.5 min, Data: 44.1 min  
- **99th Percentile:** Model: 68.2 min, Data: 82.8 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 44 minutes

*Analysis completed with 18 distribution models*
