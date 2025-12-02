# Comprehensive Distribution Analysis: Dubrovnik (LDDU)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 13,236 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -56348 | -56318 | 0.0301 | 0.0000 | No | 26.1 |
| 2 | Burr XII | 90016 | 90046 | 0.0153 | 0.0041 | No | 32.3 |
| 3 | F-Distribution | 90072 | 90102 | 0.0207 | 0.0000 | No | 32.7 |
| 4 | Generalized Gamma | 90175 | 90205 | 0.0220 | 0.0000 | No | 32.9 |
| 5 | Lomax | 90332 | 90355 | 0.0457 | 0.0000 | No | 34.1 |
| 6 | Pareto | 90332 | 90355 | 0.0457 | 0.0000 | No | 34.1 |
| 7 | Exponential | 90368 | 90383 | 0.0378 | 0.0000 | No | 33.5 |
| 8 | Log-Normal | 90521 | 90543 | 0.0311 | 0.0000 | No | 34.1 |
| 9 | Inverse Gaussian | 90558 | 90580 | 0.0303 | 0.0000 | No | 33.7 |
| 10 | Log-Logistic | 90864 | 90886 | 0.0302 | 0.0000 | No | 39.5 |
| 11 | Weibull | 90906 | 90928 | 0.0762 | 0.0000 | No | 38.2 |
| 12 | T-Distribution | 96061 | 96083 | 0.1393 | 0.0000 | No | 23.4 |
| 13 | Laplace | 97472 | 97487 | 0.1648 | 0.0000 | No | 24.9 |
| 14 | Logistic | 98164 | 98179 | 0.1434 | 0.0000 | No | 25.0 |
| 15 | Normal | 103582 | 103597 | 0.1780 | 0.0000 | No | 31.1 |
| 16 | Chi-Square | 125608 | 125630 | 0.5845 | 0.0000 | No | 8.6 |
| 17 | Uniform | 147646 | 147661 | 0.8329 | 0.0000 | No | 251.1 |
| 18 | Gamma | 201348 | 201370 | 0.8581 | 0.0000 | No | 1.7 |

## Best Distribution Details
**Beta Distribution**
- AIC: -56347.88
- BIC: -56317.92
- KS Statistic: 0.0301
- p-value: 0.000000
- Parameters: (np.float64(1.5947762661142122), np.float64(101560084710.55853), np.float64(-0.0004110340999882327), np.float64(2471925167.4773345))

## Predictions vs Actual Data
- **90th Percentile:** Model: 21.0 min, Data: 23.4 min
- **95th Percentile:** Model: 26.1 min, Data: 31.3 min  
- **99th Percentile:** Model: 37.6 min, Data: 55.9 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 26 minutes

*Analysis completed with 18 distribution models*
