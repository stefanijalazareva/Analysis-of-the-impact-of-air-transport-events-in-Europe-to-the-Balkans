# Comprehensive Distribution Analysis: Sofia (LBSF)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 33,755 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -162121 | -162088 | 0.0213 | 0.0000 | No | 32.7 |
| 2 | Burr XII | 229724 | 229758 | 0.0140 | 0.0000 | No | 32.8 |
| 3 | F-Distribution | 229819 | 229853 | 0.0183 | 0.0000 | No | 33.1 |
| 4 | Generalized Gamma | 230020 | 230054 | 0.0191 | 0.0000 | No | 33.3 |
| 5 | Pareto | 230184 | 230209 | 0.0371 | 0.0000 | No | 34.1 |
| 6 | Lomax | 230184 | 230209 | 0.0371 | 0.0000 | No | 34.1 |
| 7 | Weibull | 230247 | 230272 | 0.0233 | 0.0000 | No | 32.8 |
| 8 | Exponential | 230281 | 230298 | 0.0290 | 0.0000 | No | 33.4 |
| 9 | Log-Normal | 231184 | 231210 | 0.0297 | 0.0000 | No | 34.9 |
| 10 | Inverse Gaussian | 231207 | 231232 | 0.0289 | 0.0000 | No | 34.2 |
| 11 | Log-Logistic | 232115 | 232141 | 0.0301 | 0.0000 | No | 41.1 |
| 12 | T-Distribution | 246231 | 246256 | 0.1460 | 0.0000 | No | 23.5 |
| 13 | Laplace | 249597 | 249614 | 0.1716 | 0.0000 | No | 25.0 |
| 14 | Logistic | 251445 | 251462 | 0.1489 | 0.0000 | No | 25.2 |
| 15 | Normal | 263707 | 263724 | 0.1771 | 0.0000 | No | 30.9 |
| 16 | Chi-Square | 319271 | 319297 | 0.5773 | 0.0000 | No | 8.5 |
| 17 | Uniform | 392243 | 392260 | 0.8557 | 0.0000 | No | 317.0 |
| 18 | Gamma | 493935 | 493960 | 0.8272 | 0.0000 | No | 2.1 |

## Best Distribution Details
**Beta Distribution**
- AIC: -162121.40
- BIC: -162087.69
- KS Statistic: 0.0213
- p-value: 0.000000
- Parameters: (np.float64(1.0760426024278669), np.float64(18501.504012823378), np.float64(0.0004954593802764149), np.float64(573.2256641893914))

## Predictions vs Actual Data
- **90th Percentile:** Model: 25.3 min, Data: 23.9 min
- **95th Percentile:** Model: 32.7 min, Data: 32.2 min  
- **99th Percentile:** Model: 49.6 min, Data: 55.5 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 33 minutes

*Analysis completed with 18 distribution models*
