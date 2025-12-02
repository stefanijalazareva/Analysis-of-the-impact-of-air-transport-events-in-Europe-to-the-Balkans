# Comprehensive Delay Analysis Report: LBSF
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.2 ± 12.0 minutes
- **Early Arrivals (Negative):** Average 9.8 ± 9.7 minutes early
- **Asymmetry Index:** 1.14 (balanced operations)
- **Operational Efficiency:** 46.8% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0084, p = 0.016478
- **Negative Delay Fit:** KS = 0.0104, p = 0.001195

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 33,755 observations

**Burr XII Parameters:**
- Shape c: 2.675
- Shape d: 0.350
- Scale: 16.00
- Location: 0.017

**Performance Metrics:**
- Mean: 11.2 minutes
- Standard Deviation: 12.0 minutes
- 90th percentile: 23.9 minutes
- 95th percentile: 32.2 minutes

**Model Quality:**
- KS Statistic: 0.0084
- P-value: 0.016478
- AIC: 229438.24

### Negative Delays (Early Arrivals) 
**Sample:** 34,114 observations

**Burr XII Parameters:**
- Shape c: 3.355
- Shape d: 0.271
- Scale: 15.99
- Location: 0.017

**Performance Metrics:**
- Mean: 9.8 minutes (early)
- Standard Deviation: 9.7 minutes
- 90th percentile: 20.2 minutes (early)
- 95th percentile: 25.2 minutes (early)

**Model Quality:**
- KS Statistic: 0.0104
- P-value: 0.001195
- AIC: 222092.18

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.14
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 23.9 minutes
- 5% of flights delayed > 32.2 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 20.2 minutes early
- 5% of flights arrive > 25.2 minutes early

### Operational Recommendations

- **Balance**: Good operational balance between early and late operations
- **Maintain**: Continue current operational procedures
- **Optimize**: Fine-tune scheduling for marginal improvements

### Regional Context
This airport's performance aligns with typical Balkans patterns:
- Balkans airports show lower operational variability
- Delay asymmetry is typical for the region
- Model fit quality is consistent with regional standards

---

## Technical Notes
- Analysis includes complete delay spectrum (positive and negative)
- Burr XII distribution provides optimal fit for both delay types
- Statistical significance assessed using Kolmogorov-Smirnov tests
- Regional classifications based on operational similarities

**Data Quality:**
- Positive delay samples: 33,755
- Negative delay samples: 34,114
- Combined coverage: 67,869 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
