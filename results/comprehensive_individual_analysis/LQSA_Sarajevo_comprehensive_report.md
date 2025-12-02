# Comprehensive Distribution Analysis: Sarajevo (LQSA)

## Executive Summary
**Recommended Distribution:** Beta  
**Evidence Strength:** Very Strong  
**Statistical Significance:** Fail (p = 0.0002)  
**Sample Size:** 9,457 positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
| 1 | Beta | -33476 | -33447 | 0.0220 | 0.0002 | No | 36.3 |
| 2 | Burr XII | 66629 | 66658 | 0.0160 | 0.0153 | No | 35.5 |
| 3 | F-Distribution | 66668 | 66696 | 0.0216 | 0.0003 | No | 36.0 |
| 4 | Generalized Gamma | 66697 | 66726 | 0.0213 | 0.0004 | No | 35.9 |
| 5 | Weibull | 66726 | 66748 | 0.0233 | 0.0001 | No | 35.5 |
| 6 | Exponential | 66859 | 66874 | 0.0569 | 0.0000 | No | 37.8 |
| 7 | Pareto | 66861 | 66883 | 0.0569 | 0.0000 | No | 37.8 |
| 8 | Lomax | 66918 | 66940 | 0.0763 | 0.0000 | No | 35.0 |
| 9 | Inverse Gaussian | 67018 | 67040 | 0.0270 | 0.0000 | No | 36.7 |
| 10 | Log-Normal | 67035 | 67057 | 0.0274 | 0.0000 | No | 37.3 |
| 11 | Log-Logistic | 67319 | 67340 | 0.0334 | 0.0000 | No | 42.9 |
| 12 | T-Distribution | 70714 | 70736 | 0.1318 | 0.0000 | No | 26.5 |
| 13 | Laplace | 71419 | 71434 | 0.1547 | 0.0000 | No | 27.9 |
| 14 | Logistic | 71679 | 71694 | 0.1327 | 0.0000 | No | 27.9 |
| 15 | Normal | 74255 | 74269 | 0.1519 | 0.0000 | No | 32.8 |
| 16 | Chi-Square | 96600 | 96621 | 0.6277 | 0.0000 | No | 8.6 |
| 17 | Uniform | 100166 | 100180 | 0.7775 | 0.0000 | No | 189.5 |
| 18 | Gamma | 134968 | 134990 | 0.7882 | 0.0000 | No | 3.5 |

## Best Distribution Details
**Beta Distribution**
- AIC: -33475.66
- BIC: -33447.04
- KS Statistic: 0.0220
- p-value: 0.000213
- Parameters: (np.float64(1.16545928411955), np.float64(5561454.6263036085), np.float64(0.000481779499270476), np.float64(304543.01775898796))

## Predictions vs Actual Data
- **90th Percentile:** Model: 28.3 min, Data: 26.5 min
- **95th Percentile:** Model: 36.3 min, Data: 34.9 min  
- **99th Percentile:** Model: 54.4 min, Data: 56.1 min

## Recommendations
- **Primary Model:** Use Beta for delay modeling
- **Confidence Level:** Low
- **Operational Planning:** Plan for 95th percentile delays of 36 minutes

*Analysis completed with 18 distribution models*
