# Comprehensive Distribution Analysis: Tirana (LATI)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 22,958 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -94378 | -94346 | 0.0202 | 0.0000 | No | 31.5 |
| 2 | Burr XII | 156597 | 156629 | 0.0190 | 0.0000 | No | 31.3 |
| 3 | F-Distribution | 156748 | 156780 | 0.0240 | 0.0000 | No | 31.9 |
| 4 | Generalized Gamma | 156937 | 156969 | 0.0262 | 0.0000 | No | 32.1 |
| 5 | Weibull | 156974 | 156998 | 0.0243 | 0.0000 | No | 31.6 |
| 6 | Exponential | 157449 | 157465 | 0.0617 | 0.0000 | No | 34.0 |
| 7 | Pareto | 157451 | 157475 | 0.0617 | 0.0000 | No | 34.0 |
| 8 | Lomax | 157451 | 157475 | 0.0613 | 0.0000 | No | 34.1 |
| 9 | Log-Normal | 157535 | 157560 | 0.0315 | 0.0000 | No | 32.7 |
| 10 | Inverse Gaussian | 157550 | 157575 | 0.0315 | 0.0000 | No | 32.4 |
| 11 | Log-Logistic | 158209 | 158233 | 0.0320 | 0.0000 | No | 37.1 |
| 12 | T-Distribution | 165641 | 165665 | 0.1233 | 0.0000 | No | 23.7 |
| 13 | Laplace | 167616 | 167632 | 0.1446 | 0.0000 | No | 25.1 |
| 14 | Logistic | 168002 | 168018 | 0.1255 | 0.0000 | No | 24.8 |
| 15 | Normal | 175576 | 175592 | 0.1528 | 0.0000 | No | 29.6 |
| 16 | Chi-Square | 221188 | 221212 | 0.6069 | 0.0000 | No | 8.6 |
| 17 | Uniform | 251217 | 251233 | 0.8258 | 0.0000 | No | 225.9 |
| 18 | Gamma | 338855 | 338879 | 0.8357 | 0.0000 | No | 2.3 |

## Best Distribution Details
**Beta Distribution**
- AIC: -94378.41
- BIC: -94346.24
- KS Statistic: 0.0202
- p-value: 0.000000
- Parameters: (np.float64(1.2477990476771605), np.float64(42478489.40474805), np.float64(0.00044455043182134793), np.float64(1620501.987026577))

## Predictions vs Actual Data
- **90th Percentile:** Model: 24.8 min, Data: 23.1 min
- **95th Percentile:** Model: 31.5 min, Data: 29.7 min  
- **99th Percentile:** Model: 46.8 min, Data: 50.7 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 32 minutes

*Analysis completed with 18 distribution models*
