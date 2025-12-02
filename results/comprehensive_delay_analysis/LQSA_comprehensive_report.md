# Comprehensive Delay Analysis Report: LQSA
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 12.6 ± 12.3 minutes
- **Early Arrivals (Negative):** Average 8.1 ± 8.0 minutes early
- **Asymmetry Index:** 1.56 (more late delays)
- **Operational Efficiency:** 39.0% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0091, p = 0.413661
- **Negative Delay Fit:** KS = 0.0286, p = 0.000460

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 9,457 observations

**Burr XII Parameters:**
- Shape c: 2.937
- Shape d: 0.331
- Scale: 18.74
- Location: 0.017

**Performance Metrics:**
- Mean: 12.6 minutes
- Standard Deviation: 12.3 minutes
- 90th percentile: 26.5 minutes
- 95th percentile: 34.9 minutes

**Model Quality:**
- KS Statistic: 0.0091
- P-value: 0.413661
- AIC: 66567.81

### Negative Delays (Early Arrivals) 
**Sample:** 5,107 observations

**Burr XII Parameters:**
- Shape c: 2.789
- Shape d: 0.324
- Scale: 11.73
- Location: 0.017

**Performance Metrics:**
- Mean: 8.1 minutes (early)
- Standard Deviation: 8.0 minutes
- 90th percentile: 17.4 minutes (early)
- 95th percentile: 21.7 minutes (early)

**Model Quality:**
- KS Statistic: 0.0286
- P-value: 0.000460
- AIC: 31410.32

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.56
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 26.5 minutes
- 5% of flights delayed > 34.9 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 17.4 minutes early
- 5% of flights arrive > 21.7 minutes early

### Operational Recommendations

- **Balance**: Good operational balance between early and late operations
- **Maintain**: Continue current operational procedures
- **Optimize**: Fine-tune scheduling for marginal improvements

### Regional Context
This airport's performance aligns with typical Balkans patterns:
- Balkans airports show lower operational variability
- Delay asymmetry is atypical for the region
- Model fit quality is consistent with regional standards

---

## Technical Notes
- Analysis includes complete delay spectrum (positive and negative)
- Burr XII distribution provides optimal fit for both delay types
- Statistical significance assessed using Kolmogorov-Smirnov tests
- Regional classifications based on operational similarities

**Data Quality:**
- Positive delay samples: 9,457
- Negative delay samples: 5,107
- Combined coverage: 14,564 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
