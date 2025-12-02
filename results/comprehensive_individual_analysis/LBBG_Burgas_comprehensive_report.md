# Comprehensive Distribution Analysis: Burgas (LBBG)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 10,507 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -58111 | -58082 | 0.0262 | 0.0000 | No | 35.8 |
| 2 | Burr XII | 72509 | 72538 | 0.0127 | 0.0677 | Yes | 35.4 |
| 3 | F-Distribution | 72540 | 72569 | 0.0174 | 0.0034 | No | 35.8 |
| 4 | Pareto | 72622 | 72644 | 0.0345 | 0.0000 | No | 36.7 |
| 5 | Lomax | 72627 | 72649 | 0.0386 | 0.0000 | No | 35.8 |
| 6 | Generalized Gamma | 72674 | 72703 | 0.0216 | 0.0001 | No | 36.5 |
| 7 | Exponential | 72864 | 72879 | 0.0250 | 0.0000 | No | 35.3 |
| 8 | Log-Normal | 72966 | 72988 | 0.0295 | 0.0000 | No | 38.0 |
| 9 | Inverse Gaussian | 73056 | 73078 | 0.0298 | 0.0000 | No | 37.6 |
| 10 | Log-Logistic | 73232 | 73254 | 0.0319 | 0.0000 | No | 45.3 |
| 11 | T-Distribution | 78039 | 78061 | 0.1551 | 0.0000 | No | 24.7 |
| 12 | Laplace | 79527 | 79541 | 0.1865 | 0.0000 | No | 26.6 |
| 13 | Logistic | 80351 | 80366 | 0.1612 | 0.0000 | No | 27.0 |
| 14 | Weibull | 85930 | 85952 | 0.4853 | 0.0000 | No | 29.0 |
| 15 | Normal | 88144 | 88159 | 0.2312 | 0.0000 | No | 38.2 |
| 16 | Chi-Square | 102139 | 102161 | 0.5723 | 0.0000 | No | 8.5 |
| 17 | Uniform | 130972 | 130986 | 0.8900 | 0.0000 | No | 483.6 |
| 18 | Gamma | 207181 | 207203 | 0.9602 | 0.0000 | No | 0.0 |

## Best Distribution Details
**Beta Distribution**
- AIC: -58110.81
- BIC: -58081.78
- KS Statistic: 0.0262
- p-value: 0.000001
- Parameters: (np.float64(0.9805093623591699), np.float64(381.5720645262437), np.float64(0.0004999999999999999), np.float64(9.048045369330346))

## Predictions vs Actual Data
- **90th Percentile:** Model: 27.6 min, Data: 25.5 min
- **95th Percentile:** Model: 35.8 min, Data: 33.6 min  
- **99th Percentile:** Model: 55.0 min, Data: 60.6 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 36 minutes

*Analysis completed with 18 distribution models*
