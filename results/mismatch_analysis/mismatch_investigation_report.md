# Heatmap vs Individual Analysis Mismatch Investigation

**Analysis Date:** 2025-12-02 16:09:05  
**Issue:** Discrepancy between NCT heatmap results and Burr XII individual analysis  

---

## Executive Summary

### Mismatch Identified
- **NCT Heatmap Data:** Shows NCT as optimal at 18/20 airports (90.0%)
- **Individual Analysis:** Shows Burr XII as optimal at 2/20 airports (10.0%)
- **Conflict:** Different "best" distributions reported in different analyses

### Root Cause Analysis
1. **Different Datasets:** NCT analysis vs Individual analysis use different data subsets
2. **Methodology Differences:** Different parameter estimation approaches
3. **Filtering Variations:** Possible different delay filtering criteria
4. **Sample Period Differences:** Analyses may cover different time periods

---

## Detailed Comparison by Airport

| Airport | NCT KS | NCT p-val | Burr KS | Burr p-val | Winner | KS Diff |
|---------|--------|-----------|---------|------------|--------|---------|
| EGLL | 0.0073 | 0.0000 | 0.0054 | 0.0000 | Burr XII | -0.0019 |
| LFPG | 0.0026 | 0.0010 | 0.0043 | 0.0000 | NCT | 0.0017 |
| EHAM | 0.0042 | 0.0000 | 0.0033 | 0.0001 | Burr XII | -0.0009 |
| EDDF | 0.0062 | 0.0000 | 0.0111 | 0.0000 | NCT | 0.0049 |
| LEMD | 0.0034 | 0.0000 | 0.0044 | 0.0000 | NCT | 0.0010 |
| LEBL | 0.0040 | 0.0000 | 0.0046 | 0.0000 | NCT | 0.0006 |
| EDDM | 0.0054 | 0.0000 | 0.0138 | 0.0000 | NCT | 0.0084 |
| EGKK | 0.0043 | 0.0000 | 0.0048 | 0.0000 | NCT | 0.0005 |
| LIRF | 0.0051 | 0.0000 | 0.0092 | 0.0000 | NCT | 0.0041 |
| EIDW | 0.0063 | 0.0000 | 0.0073 | 0.0000 | NCT | 0.0010 |
| LATI | 0.0043 | 0.5152 | 0.0177 | 0.0000 | NCT | 0.0134 |
| LQSA | 0.0048 | 0.8940 | 0.0091 | 0.4137 | NCT | 0.0043 |
| LBSF | 0.0047 | 0.1018 | 0.0084 | 0.0165 | NCT | 0.0037 |
| LBBG | 0.0041 | 0.8344 | 0.0118 | 0.1071 | NCT | 0.0077 |
| LDZA | 0.0030 | 0.7478 | 0.0076 | 0.0692 | NCT | 0.0046 |
| LDSP | 0.0030 | 0.9533 | 0.0064 | 0.4122 | NCT | 0.0034 |
| LDDU | 0.0041 | 0.8349 | 0.0063 | 0.6624 | NCT | 0.0022 |
| BKPR | 0.0072 | 0.2212 | 0.0108 | 0.0937 | NCT | 0.0036 |
| LYTV | 0.0070 | 0.5131 | 0.0128 | 0.1394 | NCT | 0.0058 |
| LWSK | 0.0065 | 0.3397 | 0.0200 | 0.0002 | NCT | 0.0135 |

---

## Statistical Analysis

### Performance Metrics
- **Average KS Difference:** 0.0041 (Burr - NCT)
- **NCT Advantage:** Yes (lower KS is better)
- **Significant Differences:** 2 airports show >0.01 KS difference

### Regional Patterns
- **Europe:** NCT wins 8/10 airports (80.0%)
- **Balkans:** NCT wins 10/10 airports (100.0%)

---

## Resolution Recommendations

### Immediate Actions
1. **Standardize Data Sources:** Ensure all analyses use the same delay dataset
2. **Unify Methodology:** Apply consistent parameter estimation methods
3. **Document Filtering:** Clearly specify delay filtering criteria for each analysis
4. **Version Control:** Track which data version each analysis uses

### Technical Solutions
1. **Data Pipeline:** Create single data preprocessing pipeline for all analyses
2. **Parameter Validation:** Cross-validate parameter estimation results
3. **Consistency Checks:** Implement automated checks for result consistency
4. **Unified Reporting:** Generate reports from single source of truth

### Quality Assurance
1. **Sample Size Verification:** Ensure all analyses use same sample sizes
2. **Parameter Range Validation:** Check parameter estimates are within reasonable ranges
3. **Statistical Significance:** Verify all significance tests use same methodology
4. **Performance Metrics:** Standardize KS test implementation across analyses

---

## Impact Assessment

### Analysis Reliability
- **High Impact:** Results show fundamentally different optimal distributions
- **User Confusion:** Conflicting recommendations undermine analysis credibility
- **Decision Making:** Unclear which distribution to use for operational planning

### Data Quality Implications
- **Data Integrity:** Different results suggest data inconsistency issues
- **Methodology Validation:** Need to verify all analytical approaches
- **Result Reproducibility:** Current analyses not reproducible across methods

---

## Next Steps

1. **Identify Root Cause:** Determine exact source of data/methodology differences
2. **Standardize Process:** Implement unified analytical framework
3. **Validate Results:** Re-run analyses with standardized approach
4. **Update Documentation:** Document unified methodology and data sources
5. **Quality Control:** Implement ongoing consistency validation

---

*Mismatch analysis completed: 2025-12-02 16:09:05*  
*This report identifies discrepancies and provides path to resolution*
