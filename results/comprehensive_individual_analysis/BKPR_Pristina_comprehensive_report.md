# Comprehensive Distribution Analysis: Pristina (BKPR)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 13,072 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -40956 | -40926 | 0.0223 | 0.0000 | No | 33.9 |
| 2 | Burr XII | 90919 | 90949 | 0.0162 | 0.0021 | No | 33.7 |
| 3 | F-Distribution | 90978 | 91008 | 0.0220 | 0.0000 | No | 34.1 |
| 4 | Chi-Square | 91009 | 91032 | 0.0198 | 0.0001 | No | 33.8 |
| 5 | Weibull | 91042 | 91064 | 0.0208 | 0.0000 | No | 33.7 |
| 6 | Generalized Gamma | 91105 | 91135 | 0.0250 | 0.0000 | No | 34.6 |
| 7 | Exponential | 91289 | 91304 | 0.0594 | 0.0000 | No | 36.2 |
| 8 | Pareto | 91291 | 91313 | 0.0594 | 0.0000 | No | 36.2 |
| 9 | Lomax | 91292 | 91314 | 0.0620 | 0.0000 | No | 35.9 |
| 10 | Inverse Gaussian | 91457 | 91480 | 0.0274 | 0.0000 | No | 34.8 |
| 11 | Log-Normal | 91486 | 91508 | 0.0277 | 0.0000 | No | 35.3 |
| 12 | Log-Logistic | 91891 | 91914 | 0.0315 | 0.0000 | No | 40.5 |
| 13 | T-Distribution | 96377 | 96400 | 0.1278 | 0.0000 | No | 25.4 |
| 14 | Laplace | 97329 | 97344 | 0.1503 | 0.0000 | No | 26.7 |
| 15 | Logistic | 97573 | 97588 | 0.1287 | 0.0000 | No | 26.6 |
| 16 | Normal | 100900 | 100915 | 0.1462 | 0.0000 | No | 31.0 |
| 17 | Uniform | 132019 | 132034 | 0.7464 | 0.0000 | No | 148.2 |
| 18 | Gamma | 174973 | 174995 | 0.7533 | 0.0000 | No | 4.2 |

## Best Distribution Details
**Beta Distribution**
- AIC: -40955.94
- BIC: -40926.03
- KS Statistic: 0.0223
- p-value: 0.000005
- Parameters: (np.float64(1.1534533757767267), np.float64(31.175122997543202), np.float64(0.0004814551448385481), np.float64(2.174458580225798))

## Predictions vs Actual Data
- **90th Percentile:** Model: 26.8 min, Data: 25.2 min
- **95th Percentile:** Model: 33.9 min, Data: 32.3 min  
- **99th Percentile:** Model: 49.7 min, Data: 54.9 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 34 minutes

*Analysis completed with 18 distribution models*
