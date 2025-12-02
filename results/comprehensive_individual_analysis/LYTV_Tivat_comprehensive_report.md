# Comprehensive Distribution Analysis: Tivat (LYTV)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0003)  
**Sample Size:** 8,053 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -23546 | -23518 | 0.0232 | 0.0003 | No | 34.6 |
| 2 | Burr XII | 55883 | 55911 | 0.0140 | 0.0829 | Yes | 34.7 |
| 3 | F-Distribution | 55898 | 55926 | 0.0167 | 0.0221 | No | 35.0 |
| 4 | Generalized Gamma | 55918 | 55946 | 0.0199 | 0.0033 | No | 34.9 |
| 5 | Chi-Square | 55925 | 55946 | 0.0225 | 0.0006 | No | 34.6 |
| 6 | Weibull | 55936 | 55957 | 0.0241 | 0.0002 | No | 34.6 |
| 7 | Exponential | 55955 | 55969 | 0.0301 | 0.0000 | No | 35.6 |
| 8 | Pareto | 55956 | 55977 | 0.0317 | 0.0000 | No | 35.7 |
| 9 | Lomax | 55956 | 55977 | 0.0300 | 0.0000 | No | 35.9 |
| 10 | Inverse Gaussian | 56238 | 56259 | 0.0323 | 0.0000 | No | 36.1 |
| 11 | Log-Normal | 56261 | 56282 | 0.0325 | 0.0000 | No | 37.1 |
| 12 | Log-Logistic | 56507 | 56528 | 0.0318 | 0.0000 | No | 43.9 |
| 13 | T-Distribution | 59821 | 59842 | 0.1445 | 0.0000 | No | 25.2 |
| 14 | Laplace | 60486 | 60500 | 0.1678 | 0.0000 | No | 26.7 |
| 15 | Logistic | 60826 | 60840 | 0.1458 | 0.0000 | No | 26.8 |
| 16 | Normal | 62893 | 62907 | 0.1615 | 0.0000 | No | 31.6 |
| 17 | Uniform | 79467 | 79481 | 0.7176 | 0.0000 | No | 132.0 |
| 18 | Gamma | 102739 | 102760 | 0.7154 | 0.0000 | No | 4.6 |

## Best Distribution Details
**Beta Distribution**
- AIC: -23545.67
- BIC: -23517.69
- KS Statistic: 0.0232
- p-value: 0.000331
- Parameters: (np.float64(1.0787667161830414), np.float64(170.75525529949988), np.float64(0.0004879435508326706), np.float64(13.606314003199106))

## Predictions vs Actual Data
- **90th Percentile:** Model: 26.9 min, Data: 25.6 min
- **95th Percentile:** Model: 34.6 min, Data: 35.0 min  
- **99th Percentile:** Model: 52.3 min, Data: 57.1 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 35 minutes

*Analysis completed with 18 distribution models*
