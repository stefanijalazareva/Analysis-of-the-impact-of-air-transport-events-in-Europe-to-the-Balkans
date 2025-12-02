# Comprehensive Distribution Analysis: Zagreb (LDZA)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 29,373 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -103820 | -103787 | 0.0796 | 0.0000 | No | 23.9 |
| 2 | F-Distribution | 198192 | 198225 | 0.0203 | 0.0000 | No | 31.6 |
| 3 | Generalized Gamma | 198295 | 198329 | 0.0199 | 0.0000 | No | 31.5 |
| 4 | Weibull | 198375 | 198400 | 0.0189 | 0.0000 | No | 31.3 |
| 5 | Burr XII | 198376 | 198410 | 0.0182 | 0.0000 | No | 31.2 |
| 6 | Pareto | 198499 | 198524 | 0.0394 | 0.0000 | No | 32.5 |
| 7 | Lomax | 198499 | 198524 | 0.0394 | 0.0000 | No | 32.5 |
| 8 | Exponential | 198501 | 198517 | 0.0376 | 0.0000 | No | 32.3 |
| 9 | Inverse Gaussian | 199477 | 199502 | 0.0295 | 0.0000 | No | 32.5 |
| 10 | Log-Normal | 199526 | 199551 | 0.0295 | 0.0000 | No | 33.2 |
| 11 | Log-Logistic | 200415 | 200440 | 0.0310 | 0.0000 | No | 38.9 |
| 12 | T-Distribution | 211965 | 211990 | 0.1407 | 0.0000 | No | 22.9 |
| 13 | Laplace | 214356 | 214372 | 0.1643 | 0.0000 | No | 24.2 |
| 14 | Logistic | 215483 | 215500 | 0.1420 | 0.0000 | No | 24.2 |
| 15 | Normal | 223999 | 224015 | 0.1623 | 0.0000 | No | 28.8 |
| 16 | Chi-Square | 272697 | 272722 | 0.5742 | 0.0000 | No | 8.6 |
| 17 | Uniform | 313299 | 313315 | 0.8035 | 0.0000 | No | 196.7 |
| 18 | Gamma | 384159 | 384184 | 0.7608 | 0.0000 | No | 3.4 |

## Best Distribution Details
**Beta Distribution**
- AIC: -103819.81
- BIC: -103786.66
- KS Statistic: 0.0796
- p-value: 0.000000
- Parameters: (np.float64(5.801485091034847), np.float64(373173427202589.0), np.float64(-0.03654485022723274), np.float64(5536371297494.603))

## Predictions vs Actual Data
- **90th Percentile:** Model: 20.2 min, Data: 23.1 min
- **95th Percentile:** Model: 23.9 min, Data: 30.5 min  
- **99th Percentile:** Model: 31.8 min, Data: 51.1 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 24 minutes

*Analysis completed with 18 distribution models*
