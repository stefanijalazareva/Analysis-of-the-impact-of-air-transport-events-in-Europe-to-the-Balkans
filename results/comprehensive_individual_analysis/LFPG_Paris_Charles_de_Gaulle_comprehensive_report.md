# Comprehensive Distribution Analysis: Paris Charles de Gaulle (LFPG)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 409,104 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -2796122 | -2796078 | 0.0641 | 0.0000 | No | 32.6 |
| 2 | Burr XII | 3014492 | 3014535 | 0.0131 | 0.0000 | No | 43.5 |
| 3 | F-Distribution | 3016205 | 3016249 | 0.0184 | 0.0000 | No | 44.0 |
| 4 | Generalized Gamma | 3019934 | 3019977 | 0.0208 | 0.0000 | No | 44.4 |
| 5 | Pareto | 3024354 | 3024386 | 0.0468 | 0.0000 | No | 45.7 |
| 6 | Lomax | 3024354 | 3024386 | 0.0468 | 0.0000 | No | 45.7 |
| 7 | Weibull | 3026390 | 3026423 | 0.0287 | 0.0000 | No | 43.8 |
| 8 | Exponential | 3026927 | 3026948 | 0.0360 | 0.0000 | No | 44.6 |
| 9 | Log-Normal | 3028197 | 3028229 | 0.0243 | 0.0000 | No | 45.8 |
| 10 | Inverse Gaussian | 3030123 | 3030156 | 0.0248 | 0.0000 | No | 45.4 |
| 11 | Log-Logistic | 3036932 | 3036965 | 0.0256 | 0.0000 | No | 52.6 |
| 12 | T-Distribution | 3205314 | 3205347 | 0.1440 | 0.0000 | No | 30.9 |
| 13 | Laplace | 3253906 | 3253928 | 0.1718 | 0.0000 | No | 33.1 |
| 14 | Logistic | 3284639 | 3284661 | 0.1496 | 0.0000 | No | 33.3 |
| 15 | Normal | 3473846 | 3473868 | 0.1893 | 0.0000 | No | 42.7 |
| 16 | Chi-Square | 4580884 | 4580917 | 0.6513 | 0.0000 | No | 8.5 |
| 17 | Uniform | 5948623 | 5948645 | 0.9387 | 0.0000 | No | 1365.2 |
| 18 | Gamma | 8098815 | 8098848 | 0.9290 | 0.0000 | No | 0.5 |

## Best Distribution Details
**Beta Distribution**
- AIC: -2796121.99
- BIC: -2796078.30
- KS Statistic: 0.0641
- p-value: 0.000000
- Parameters: (np.float64(3.2073162210514194), np.float64(18276400639846.6), np.float64(-0.002316812925462839), np.float64(69133384564.36896))

## Predictions vs Actual Data
- **90th Percentile:** Model: 27.2 min, Data: 31.6 min
- **95th Percentile:** Model: 32.6 min, Data: 42.7 min  
- **99th Percentile:** Model: 44.3 min, Data: 77.3 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 33 minutes

*Analysis completed with 18 distribution models*
