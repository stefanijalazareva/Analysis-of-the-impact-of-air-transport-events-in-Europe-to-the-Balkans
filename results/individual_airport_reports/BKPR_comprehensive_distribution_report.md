# Comprehensive Distribution Analysis Report: Pristina (BKPR)

**Generated:** 2025-12-02 15:40:14  
**Region:** Balkans  
**Airport Type:** Balkan Regional Airport
**Analysis Type:** Positive and Negative Delays

---

## Executive Summary

### Recommended Distributions
**Positive Delays:** Burr XII is optimal for modeling late arrivals at Pristina.
**Negative Delays:** Burr XII is optimal for modeling early arrivals at Pristina.

**Positive Delay Analysis:**
- **Statistical Evidence:** KS Statistic = 0.0108
- **AIC Score:** 90816.74
- **P-value:** 0.093660
- **Sample Size:** 13,072 late arrivals

**Negative Delay Analysis:**
- **Statistical Evidence:** KS Statistic = 0.0108
- **AIC Score:** 48030.11
- **P-value:** 0.296977
- **Sample Size:** 8,059 early arrivals

### Operational Performance Summary
- **Late Arrivals:** 12.1 min average delay (95th percentile: 32.4 min)
- **Early Arrivals:** 7.5 min average early (95th percentile: 19.5 min)
- **Asymmetry Ratio:** 1.62 (late vs early magnitude)

### Key Findings
- **Best Positive Model:** Burr XII (Significant)
- **Best Negative Model:** Burr XII (Significant)
- **Model Consistency:** Same distribution family
- **Operational Balance:** 61.7% early vs 100.0% late arrivals

---

## Detailed Analysis Results

### All Distributions Tested

| Rank | Distribution | KS Statistic | p-value | AIC | BIC | P95 (min) |
|------|--------------|--------------|---------|-----|-----|-----------|
| 1 | Burr XII | 0.0108 | 0.0937 | 90816.74 | 90846.66 | 32.4 |
| 2 | Burr XII | 0.0108 | 0.2970 | 48030.11 | 48058.09 | 19.5 |
| 3 | Beta | 0.0191 | 0.0001 | -40936.88 | -40906.97 | 0.2 |
| 4 | Chi-Square | 0.0198 | 0.0001 | 91009.46 | 91031.89 | 33.8 |
| 5 | Weibull Min | 0.0208 | 0.0000 | 91041.59 | 91064.03 | 33.7 |
| 6 | F-Distribution | 0.0220 | 0.0000 | 90978.47 | 91008.38 | 34.1 |
| 7 | Weibull Min | 0.0248 | 0.0001 | 48312.12 | 48333.11 | 20.6 |
| 8 | Generalized Gamma | 0.0250 | 0.0000 | 91105.08 | 91134.99 | 34.6 |
| 9 | Generalized Gamma | 0.0273 | 0.0000 | 48306.96 | 48334.94 | 20.7 |
| 10 | Inverse Gaussian | 0.0274 | 0.0000 | 91457.19 | 91479.62 | 34.8 |
| 11 | Log-Normal | 0.0277 | 0.0000 | 91485.55 | 91507.99 | 35.3 |
| 12 | Beta | 0.0297 | 0.0000 | -40507.74 | -40479.77 | 0.1 |
| 13 | Noncentral-t | 0.0305 | 0.0000 | 91896.24 | 91926.15 | 34.8 |
| 14 | Log-Logistic (Fisk) | 0.0315 | 0.0000 | 91891.31 | 91913.75 | 40.5 |
| 15 | F-Distribution | 0.0331 | 0.0000 | 48287.98 | 48315.95 | 20.9 |
| 16 | Log-Logistic (Fisk) | 0.0421 | 0.0000 | 49064.52 | 49085.51 | 25.2 |
| 17 | Log-Normal | 0.0437 | 0.0000 | 48727.97 | 48748.96 | 21.5 |
| 18 | Inverse Gaussian | 0.0438 | 0.0000 | 48705.02 | 48726.00 | 21.2 |
| 19 | Noncentral-t | 0.0446 | 0.0000 | 49011.36 | 49039.33 | 20.8 |
| 20 | Lomax | 0.0550 | 0.0000 | 48487.13 | 48508.11 | 22.5 |
| 21 | Exponential | 0.0569 | 0.0000 | 48484.80 | 48498.79 | 22.3 |
| 22 | Pareto | 0.0569 | 0.0000 | 48486.80 | 48507.78 | 22.3 |
| 23 | Exponential | 0.0594 | 0.0000 | 91288.77 | 91303.72 | 36.2 |
| 24 | Pareto | 0.0594 | 0.0000 | 91290.77 | 91313.20 | 36.2 |
| 25 | Lomax | 0.0620 | 0.0000 | 91291.97 | 91314.40 | 35.9 |
| 26 | T-Distribution | 0.1198 | 0.0000 | 51474.39 | 51495.38 | 15.9 |
| 27 | T-Distribution | 0.1278 | 0.0000 | 96377.33 | 96399.76 | 25.4 |
| 28 | Normal | 0.1462 | 0.0000 | 100899.63 | 100914.59 | 31.0 |
| 29 | Normal | 0.1535 | 0.0000 | 54896.61 | 54910.60 | 19.5 |
| 30 | Chi-Square | 0.2697 | 0.0000 | 51419.57 | 51440.55 | 20.0 |
| 31 | Gamma | 0.7533 | 0.0000 | 174972.66 | 174995.10 | 4.2 |
| 32 | Weibull Max | 0.8675 | 0.0000 | 212190.00 | 212212.43 | 156.0 |
| 33 | Weibull Max | 0.8833 | 0.0000 | 140529.15 | 140550.13 | 247.2 |
| 34 | Gamma | 0.8961 | 0.0000 | 112863.61 | 112884.59 | 0.7 |

### Distribution Performance Analysis

**Kolmogorov-Smirnov Test Results:**
- Best performing: Burr XII (KS = 0.0108)
- Statistical significance: PASSED at α = 0.05
- Model reliability: MODERATE

**Information Criterion Analysis:**
- AIC winner: Beta
- BIC winner: Beta
- Model complexity: Balanced between fit quality and parameter parsimony

### Extreme Value Predictions

**Burr XII Distribution Predictions:**
- **90th Percentile:** 24.8 minutes
- **95th Percentile:** 32.4 minutes  
- **99th Percentile:** 56.5 minutes

**Actual Data Percentiles:**
- **90th Percentile:** 25.2 minutes
- **95th Percentile:** 32.3 minutes
- **99th Percentile:** 54.9 minutes

**Prediction Accuracy:**
- P90 Error: 0.3 minutes
- P95 Error: 0.1 minutes
- P99 Error: 1.5 minutes

---

## Statistical Quality Assessment

### Data Characteristics
- **Sample Size:** 13,072 positive delay observations
- **Mean Delay:** 12.10 minutes
- **Median Delay:** 9.17 minutes  
- **Standard Deviation:** 11.48 minutes
- **Skewness:** High (right-tailed)

### Model Quality Indicators
- **Log-likelihood:** -45404.37
- **Number of parameters:** 4
- **Convergence:** Successful
- **Overfitting risk:** Low

### Distribution Parameters
**Burr XII Parameters:**
- Parameter 1: 3.046938
- Parameter 2: 0.322209
- Parameter 3: 0.016667
- Parameter 4: 18.181670

---

## Operational Recommendations

### For Air Traffic Management
1. **Primary Model:** Use Burr XII distribution for delay forecasting
2. **Capacity Planning:** Plan for 95th percentile delays of ~32 minutes
3. **Extreme Events:** Prepare for 99th percentile delays up to 56 minutes

### For Airline Operations
- **Schedule Buffering:** Add 32-minute buffer for on-time performance
- **Passenger Communication:** Use distribution percentiles for delay probability estimates
- **Resource Allocation:** Base crew and aircraft planning on statistical delay patterns

### For Airport Operations
- **Gate Management:** Account for 32-minute delay distributions in scheduling
- **Ground Handling:** Scale operations for predicted delay volumes
- **Passenger Services:** Implement delay management based on statistical predictions

### Model Validation Recommendations
- **Monitoring:** High priority - validate monthly
- **Recalibration:** Annual review sufficient
- **Alternative Models:** Single model recommended

---

## Technical Validation

### Statistical Tests Passed
✓ Sample size adequacy (>13072 observations)  
✓ Kolmogorov-Smirnov goodness-of-fit test  
✓ Parameter estimation convergence  
✓ Numerical stability validation  

### Data Quality Checks
✓ No missing values in delay measurements  
✓ Positive delay filter applied correctly  
✓ Outlier analysis completed  
✓ Temporal consistency verified  

### Model Assumptions
✓ Independent observations assumed  
✓ Stationary process assumed  
✓ Large sample approximation valid  
✓ Maximum likelihood estimation appropriate  

---

## Appendix: Full Distribution Comparison

### Performance Rankings

**By KS_Statistic:**
1. Burr XII: 0.0108
2. Burr XII: 0.0108
3. Beta: 0.0191
4. Chi-Square: 0.0198
5. Weibull Min: 0.0208

**By AIC:**
1. Beta: -40936.8793
2. Beta: -40507.7447
3. Burr XII: 48030.1109
4. F-Distribution: 48287.9758
5. Generalized Gamma: 48306.9602

**By BIC:**
1. Beta: -40906.9663
2. Beta: -40479.7666
3. Burr XII: 48058.0891
4. F-Distribution: 48315.9540
5. Weibull Min: 48333.1082

### Methodology Notes
- **Fitting Method:** Maximum Likelihood Estimation
- **Goodness-of-Fit:** Kolmogorov-Smirnov test
- **Model Selection:** Akaike Information Criterion (AIC)
- **Significance Level:** α = 0.05
- **Software:** SciPy statistical distributions

---

*Report generated automatically by comprehensive distribution analysis system*  
*For questions about methodology, consult the statistical documentation*  
*Individual airport analysis completed: 2025-12-02 15:40:14*
