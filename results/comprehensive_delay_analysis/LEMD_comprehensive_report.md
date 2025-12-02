# Comprehensive Delay Analysis Report: LEMD
**Generated:** 2025-12-02 14:50:52
**Region:** Europe
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 13.6 ± 17.4 minutes
- **Early Arrivals (Negative):** Average 7.4 ± 9.9 minutes early
- **Asymmetry Index:** 1.83 (more late delays)
- **Operational Efficiency:** 35.3% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0044, p = 0.000009
- **Negative Delay Fit:** KS = 0.0255, p = 0.000000

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 318,721 observations

**Burr XII Parameters:**
- Shape c: 2.696
- Shape d: 0.364
- Scale: 18.77
- Location: 0.017

**Performance Metrics:**
- Mean: 13.6 minutes
- Standard Deviation: 17.4 minutes
- 90th percentile: 28.2 minutes
- 95th percentile: 37.6 minutes

**Model Quality:**
- KS Statistic: 0.0044
- P-value: 0.000009
- AIC: 2284126.58

### Negative Delays (Early Arrivals) 
**Sample:** 157,813 observations

**Burr XII Parameters:**
- Shape c: 3.551
- Shape d: 0.247
- Scale: 12.67
- Location: 0.017

**Performance Metrics:**
- Mean: 7.4 minutes (early)
- Standard Deviation: 9.9 minutes
- 90th percentile: 15.6 minutes (early)
- 95th percentile: 19.8 minutes (early)

**Model Quality:**
- KS Statistic: 0.0255
- P-value: 0.000000
- AIC: 939524.48

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.83
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 28.2 minutes
- 5% of flights delayed > 37.6 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 15.6 minutes early
- 5% of flights arrive > 19.8 minutes early

### Operational Recommendations

- **Balance**: Good operational balance between early and late operations
- **Maintain**: Continue current operational procedures
- **Optimize**: Fine-tune scheduling for marginal improvements

### Regional Context
This airport's performance aligns with typical Europe patterns:
- Europe airports show higher operational variability
- Delay asymmetry is typical for the region
- Model fit quality is challenging with regional standards

---

## Technical Notes
- Analysis includes complete delay spectrum (positive and negative)
- Burr XII distribution provides optimal fit for both delay types
- Statistical significance assessed using Kolmogorov-Smirnov tests
- Regional classifications based on operational similarities

**Data Quality:**
- Positive delay samples: 318,721
- Negative delay samples: 157,813
- Combined coverage: 476,534 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
