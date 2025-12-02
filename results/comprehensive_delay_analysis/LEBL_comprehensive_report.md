# Comprehensive Delay Analysis Report: LEBL
**Generated:** 2025-12-02 14:50:52
**Region:** Europe
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 13.6 ± 15.5 minutes
- **Early Arrivals (Negative):** Average 7.0 ± 7.5 minutes early
- **Asymmetry Index:** 1.94 (more late delays)
- **Operational Efficiency:** 34.1% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0046, p = 0.000041
- **Negative Delay Fit:** KS = 0.0133, p = 0.000000

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 250,813 observations

**Burr XII Parameters:**
- Shape c: 2.497
- Shape d: 0.392
- Scale: 17.93
- Location: 0.017

**Performance Metrics:**
- Mean: 13.6 minutes
- Standard Deviation: 15.5 minutes
- 90th percentile: 28.8 minutes
- 95th percentile: 39.6 minutes

**Model Quality:**
- KS Statistic: 0.0046
- P-value: 0.000041
- AIC: 1802157.52

### Negative Delays (Early Arrivals) 
**Sample:** 131,394 observations

**Burr XII Parameters:**
- Shape c: 3.316
- Shape d: 0.262
- Scale: 11.75
- Location: 0.017

**Performance Metrics:**
- Mean: 7.0 minutes (early)
- Standard Deviation: 7.5 minutes
- 90th percentile: 14.8 minutes (early)
- 95th percentile: 18.5 minutes (early)

**Model Quality:**
- KS Statistic: 0.0133
- P-value: 0.000000
- AIC: 768164.44

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.94
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 28.8 minutes
- 5% of flights delayed > 39.6 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 14.8 minutes early
- 5% of flights arrive > 18.5 minutes early

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
- Positive delay samples: 250,813
- Negative delay samples: 131,394
- Combined coverage: 382,207 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
