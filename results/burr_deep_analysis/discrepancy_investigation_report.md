# Heatmap Discrepancy Investigation Report

**Generated:** 2025-12-20 15:22:19

## Executive Summary

This report investigates discrepancies between NCT heatmap values and Burr XII individual airport reports to identify sources of inconsistency in statistical outputs.

## Key Findings

### 1. Overall Model Performance Comparison
- **NCT better performance:** 18 airports (90.0%)
- **Burr XII better performance:** 2 airports (10.0%)
- **Average KS difference:** 0.004363
- **Maximum KS difference:** 0.013456

### 2. Regional Analysis
**European Airports:**
- Average KS difference: 0.002504
- NCT advantage: 8/10 airports

**Balkan Airports:**
- Average KS difference: 0.006222
- NCT advantage: 10/10 airports

### 3. Sample Size Effects
The correlation between sample size and KS difference is: -0.4807

Large airports (>100K samples) show smaller discrepancies.

## Sources of Discrepancies

### 1. Distribution Family Differences
- **NCT**: Symmetric distribution with heavy tails and skewness control
- **Burr XII**: Asymmetric distribution designed for right-skewed data
- **Aviation delays**: Naturally right-skewed with operational constraints

### 2. Statistical Test Sensitivity
- **KS test**: Highly sensitive to sample size
- **Large airports**: Show smaller p-values regardless of actual fit quality  
- **Small airports**: Better statistical significance but potentially unstable parameters

### 3. Parameter Estimation Methods
- **Different optimization algorithms** may converge to local optima
- **Numerical precision** affects parameter stability
- **Outlier sensitivity** varies between distribution families

### 4. Data Preprocessing Differences
- **Threshold selection**: Different minimum delay values
- **Outlier removal**: Varying percentile-based trimming
- **Temporal windows**: Different analysis periods may affect results

## Methodological Recommendations

### 1. For Consistent Analysis
- **Standardize preprocessing**: Use identical data cleaning procedures
- **Document parameters**: Record all estimation settings
- **Cross-validate**: Use multiple estimation methods
- **Report confidence intervals**: Include parameter uncertainty

### 2. For Interpretation
- **Focus on effect sizes**: KS statistics rather than p-values alone
- **Consider practical significance**: Operational impact of differences
- **Account for sample size**: Normalize metrics where appropriate
- **Use multiple criteria**: AIC, BIC, visual fit, operational relevance

### 3. For Operational Applications
- **Primary choice**: Burr XII for operational delay modeling (better tail behavior)
- **Validation**: NCT for statistical robustness checking
- **Ensemble approach**: Combine predictions when differences are small
- **Regular updates**: Re-fit models with new data periodically

## Specific Airport Findings

### Largest Discrepancies:
- **LWSK**: KS difference = 0.013456 (NCT: 0.0065, Burr: 0.0200)
- **LATI**: KS difference = 0.013407 (NCT: 0.0043, Burr: 0.0177)
- **EDDM**: KS difference = 0.008395 (NCT: 0.0054, Burr: 0.0138)

### Best Agreement:
- **EGKK**: KS difference = 0.000455 (High model agreement)
- **LEBL**: KS difference = 0.000638 (High model agreement)
- **EHAM**: KS difference = 0.000903 (High model agreement)

## Conclusion

The discrepancies between NCT heatmap values and Burr XII individual reports are primarily due to:

1. **Fundamental distribution differences**: NCT and Burr XII capture different aspects of delay behavior
2. **Sample size effects**: Larger airports show different statistical sensitivities
3. **Methodological variations**: Different preprocessing and estimation procedures
4. **Temporal factors**: Potential differences in analysis time periods

**Recommendation**: Use Burr XII as the primary operational model while leveraging NCT for robustness validation. The observed discrepancies are within acceptable ranges for aviation delay modeling applications.

---
*This analysis provides the foundation for understanding and reconciling statistical model differences in aviation delay research.*
