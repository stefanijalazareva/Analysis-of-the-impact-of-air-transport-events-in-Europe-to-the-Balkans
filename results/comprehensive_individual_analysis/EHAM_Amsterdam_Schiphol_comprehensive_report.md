# Comprehensive Distribution Analysis: Amsterdam Schiphol (EHAM)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 455,426 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -2356634 | -2356589 | 0.4304 | 0.0000 | No | 74.1 |
| 2 | Burr XII | 3449005 | 3449049 | 0.0151 | 0.0000 | No | 48.3 |
| 3 | F-Distribution | 3451775 | 3451819 | 0.0211 | 0.0000 | No | 49.1 |
| 4 | Generalized Gamma | 3456823 | 3456867 | 0.0235 | 0.0000 | No | 49.6 |
| 5 | Log-Normal | 3460344 | 3460377 | 0.0230 | 0.0000 | No | 50.4 |
| 6 | Inverse Gaussian | 3464304 | 3464338 | 0.0243 | 0.0000 | No | 50.6 |
| 7 | Log-Logistic | 3466106 | 3466139 | 0.0226 | 0.0000 | No | 56.3 |
| 8 | Pareto | 3466760 | 3466793 | 0.0572 | 0.0000 | No | 51.4 |
| 9 | Lomax | 3466760 | 3466793 | 0.0572 | 0.0000 | No | 51.4 |
| 10 | Weibull | 3470367 | 3470400 | 0.0399 | 0.0000 | No | 48.9 |
| 11 | Exponential | 3471034 | 3471056 | 0.0426 | 0.0000 | No | 49.8 |
| 12 | T-Distribution | 3647802 | 3647835 | 0.1406 | 0.0000 | No | 33.9 |
| 13 | Laplace | 3715328 | 3715350 | 0.1712 | 0.0000 | No | 36.7 |
| 14 | Logistic | 3761378 | 3761400 | 0.1506 | 0.0000 | No | 36.9 |
| 15 | Normal | 3980318 | 3980340 | 0.1924 | 0.0000 | No | 48.1 |
| 16 | Chi-Square | 5470773 | 5470806 | 0.6829 | 0.0000 | No | 8.4 |
| 17 | Uniform | 6352679 | 6352702 | 0.9141 | 0.0000 | No | 1015.5 |
| 18 | Gamma | 8918076 | 8918109 | 0.9044 | 0.0000 | No | 1.2 |

## Best Distribution Details
**Beta Distribution**
- AIC: -2356633.58
- BIC: -2356589.46
- KS Statistic: 0.4304
- p-value: 0.000000
- Parameters: (np.float64(0.2141407354486642), np.float64(31.501786288099417), np.float64(0.0004999999999999999), np.float64(2.0098670600333195))

## Predictions vs Actual Data
- **90th Percentile:** Model: 44.8 min, Data: 34.2 min
- **95th Percentile:** Model: 74.1 min, Data: 47.7 min  
- **99th Percentile:** Model: 151.8 min, Data: 96.6 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 74 minutes

*Analysis completed with 18 distribution models*
