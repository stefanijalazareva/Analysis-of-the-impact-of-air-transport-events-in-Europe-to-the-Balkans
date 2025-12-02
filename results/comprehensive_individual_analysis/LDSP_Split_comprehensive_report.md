# Comprehensive Distribution Analysis: Split (LDSP)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 18,956 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -121178 | -121147 | 0.1522 | 0.0000 | No | 30.8 |
| 2 | Burr XII | 130154 | 130186 | 0.0175 | 0.0000 | No | 33.0 |
| 3 | F-Distribution | 130240 | 130272 | 0.0235 | 0.0000 | No | 33.5 |
| 4 | Generalized Gamma | 130416 | 130447 | 0.0252 | 0.0000 | No | 33.8 |
| 5 | Pareto | 130602 | 130625 | 0.0505 | 0.0000 | No | 35.1 |
| 6 | Lomax | 130602 | 130625 | 0.0505 | 0.0000 | No | 35.1 |
| 7 | Exponential | 130697 | 130712 | 0.0433 | 0.0000 | No | 34.6 |
| 8 | Log-Normal | 130958 | 130982 | 0.0269 | 0.0000 | No | 34.8 |
| 9 | Inverse Gaussian | 131036 | 131059 | 0.0282 | 0.0000 | No | 34.5 |
| 10 | Log-Logistic | 131474 | 131498 | 0.0300 | 0.0000 | No | 40.1 |
| 11 | T-Distribution | 138649 | 138673 | 0.1368 | 0.0000 | No | 24.2 |
| 12 | Laplace | 140528 | 140543 | 0.1621 | 0.0000 | No | 25.7 |
| 13 | Logistic | 141421 | 141436 | 0.1403 | 0.0000 | No | 25.7 |
| 14 | Normal | 155548 | 155564 | 0.2150 | 0.0000 | No | 35.7 |
| 15 | Weibull | 181009 | 181033 | 0.6618 | 0.0000 | No | 40.8 |
| 16 | Chi-Square | 183826 | 183849 | 0.5984 | 0.0000 | No | 8.6 |
| 17 | Uniform | 270172 | 270187 | 0.9462 | 0.0000 | No | 1181.9 |
| 18 | Gamma | 446967 | 446990 | 0.9919 | 0.0000 | No | 0.0 |

## Best Distribution Details
**Beta Distribution**
- AIC: -121178.25
- BIC: -121146.85
- KS Statistic: 0.1522
- p-value: 0.000000
- Parameters: (np.float64(792.1485220090801), np.float64(277903848.6517961), np.float64(-0.243841409085224), np.float64(88964.52850805614))

## Predictions vs Actual Data
- **90th Percentile:** Model: 26.6 min, Data: 24.3 min
- **95th Percentile:** Model: 30.8 min, Data: 32.1 min  
- **99th Percentile:** Model: 38.8 min, Data: 55.5 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 31 minutes

*Analysis completed with 18 distribution models*
