# Comprehensive Distribution Analysis: London Gatwick (EGKK)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0000)  
**Sample Size:** 239,448 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -1292483 | -1292442 | 0.0221 | 0.0000 | No | 52.9 |
| 2 | Burr XII | 1876767 | 1876808 | 0.0152 | 0.0000 | No | 52.9 |
| 3 | F-Distribution | 1878088 | 1878130 | 0.0209 | 0.0000 | No | 53.7 |
| 4 | Generalized Gamma | 1879683 | 1879725 | 0.0214 | 0.0000 | No | 53.8 |
| 5 | Weibull | 1882223 | 1882255 | 0.0266 | 0.0000 | No | 53.2 |
| 6 | Log-Normal | 1884626 | 1884657 | 0.0237 | 0.0000 | No | 55.2 |
| 7 | Inverse Gaussian | 1885152 | 1885183 | 0.0246 | 0.0000 | No | 54.8 |
| 8 | Pareto | 1885618 | 1885650 | 0.0587 | 0.0000 | No | 56.7 |
| 9 | Lomax | 1885618 | 1885650 | 0.0587 | 0.0000 | No | 56.7 |
| 10 | Exponential | 1885646 | 1885667 | 0.0573 | 0.0000 | No | 56.5 |
| 11 | Log-Logistic | 1890303 | 1890334 | 0.0266 | 0.0000 | No | 62.4 |
| 12 | T-Distribution | 1975798 | 1975829 | 0.1303 | 0.0000 | No | 39.1 |
| 13 | Laplace | 1997427 | 1997448 | 0.1538 | 0.0000 | No | 41.5 |
| 14 | Logistic | 2008615 | 2008636 | 0.1333 | 0.0000 | No | 41.5 |
| 15 | Normal | 2094954 | 2094975 | 0.1630 | 0.0000 | No | 50.5 |
| 16 | Chi-Square | 3116199 | 3116230 | 0.7187 | 0.0000 | No | 8.2 |
| 17 | Uniform | 3172804 | 3172825 | 0.8876 | 0.0000 | No | 716.2 |
| 18 | Gamma | 4996493 | 4996524 | 0.9103 | 0.0000 | No | 1.2 |

## Best Distribution Details
**Beta Distribution**
- AIC: -1292483.27
- BIC: -1292441.73
- KS Statistic: 0.0221
- p-value: 0.000000
- Parameters: (np.float64(1.2365631830500758), np.float64(21893.93575296976), np.float64(0.0004835624018185798), np.float64(443.12297273745776))

## Predictions vs Actual Data
- **90th Percentile:** Model: 41.6 min, Data: 39.2 min
- **95th Percentile:** Model: 52.9 min, Data: 51.7 min  
- **99th Percentile:** Model: 78.6 min, Data: 89.8 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 53 minutes

*Analysis completed with 18 distribution models*
