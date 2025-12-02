# Statistical Model Comparison: NCT vs Burr XII

**Analysis Date:** 2025-12-02 14:36:29

## Overview of Discrepancies

This analysis examines the differences between Noncentral-t (NCT) and Burr XII distribution fits across all airports to explain observed inconsistencies.

## Key Findings

### 1. Overall Performance Comparison
- **NCT better fits:** 18 out of 20 airports (90.0%)
- **Burr XII better fits:** 2 out of 20 airports (10.0%)

### 2. Regional Performance

**European Airports:**
- NCT better at 80.0% of airports
- Average NCT KS: 0.0049
- Average Burr XII KS: 0.0068

**Balkan Airports:**
- NCT better at 100.0% of airports  
- Average NCT KS: 0.0049
- Average Burr XII KS: 0.0111

## Explanation of Discrepancies

### 1. Different Distribution Families
- **NCT**: Symmetric distribution with adjustable skewness via non-centrality
- **Burr XII**: Heavy-tailed distribution specifically designed for right-skewed data

### 2. Sample Size Effects
- Larger airports (European) show different sensitivity to distribution choice
- KS test statistic interpretation varies with sample size
- P-values are highly sample-size dependent

### 3. Data Characteristics
Aviation delay data exhibits:
- Heavy right tail (extreme delays)
- Zero inflation (on-time flights)
- Operational constraints (minimum delay reporting thresholds)

### 4. Heatmap vs Individual Report Differences

The discrepancies between heatmap visualizations and individual reports stem from:

1. **Different Analysis Timeframes**: 
   - Heatmaps may use aggregated historical data
   - Individual reports use specific time periods

2. **Preprocessing Differences**:
   - Different outlier removal methods
   - Varying minimum delay thresholds
   - Alternative data cleaning procedures

3. **Parameter Estimation Methods**:
   - Maximum likelihood vs method of moments
   - Different optimization algorithms
   - Convergence criteria variations

4. **Statistical Test Implementation**:
   - KS test parameter specifications
   - Handling of tied values
   - Bootstrap vs analytical p-values

## Recommendations

### 1. For Operational Use
- Use **Burr XII** for extreme delay prediction (better tail modeling)
- Use **NCT** for central tendency analysis (better symmetric properties)

### 2. For Research Consistency
- Standardize preprocessing procedures across analyses
- Document all parameter estimation methods
- Use consistent statistical test implementations

### 3. For Report Accuracy
- Cross-validate results across multiple analysis methods
- Include confidence intervals for all parameter estimates
- Report methodology details for reproducibility

## Technical Considerations

### Distribution Selection Criteria
1. **Goodness of fit**: KS statistic magnitude
2. **Statistical significance**: P-value interpretation
3. **Practical utility**: Parameter interpretability
4. **Theoretical justification**: Match to data generation process

### Methodological Notes
- All comparisons based on positive delays only
- KS statistics computed using same data preprocessing
- Regional classifications maintained consistently
- Sample sizes vary significantly across airports

---
*This analysis helps explain observed discrepancies and provides guidance for consistent statistical modeling in aviation delay research.*
