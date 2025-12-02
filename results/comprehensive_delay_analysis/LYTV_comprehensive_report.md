# Comprehensive Delay Analysis Report: LYTV
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.9 ± 12.0 minutes
- **Early Arrivals (Negative):** Average 7.9 ± 7.9 minutes early
- **Asymmetry Index:** 1.51 (more late delays)
- **Operational Efficiency:** 39.8% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0128, p = 0.139353
- **Negative Delay Fit:** KS = 0.0244, p = 0.002080

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 8,053 observations

**Burr XII Parameters:**
- Shape c: 2.663
- Shape d: 0.354
- Scale: 16.81
- Location: 0.017

**Performance Metrics:**
- Mean: 11.9 minutes
- Standard Deviation: 12.0 minutes
- 90th percentile: 25.6 minutes
- 95th percentile: 35.0 minutes

**Model Quality:**
- KS Statistic: 0.0128
- P-value: 0.139353
- AIC: 55832.94

### Negative Delays (Early Arrivals) 
**Sample:** 5,777 observations

**Burr XII Parameters:**
- Shape c: 4.219
- Shape d: 0.210
- Scale: 14.35
- Location: 0.017

**Performance Metrics:**
- Mean: 7.9 minutes (early)
- Standard Deviation: 7.9 minutes
- 90th percentile: 16.0 minutes (early)
- 95th percentile: 19.5 minutes (early)

**Model Quality:**
- KS Statistic: 0.0244
- P-value: 0.002080
- AIC: 34911.48

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.51
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 25.6 minutes
- 5% of flights delayed > 35.0 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 16.0 minutes early
- 5% of flights arrive > 19.5 minutes early

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
- Positive delay samples: 8,053
- Negative delay samples: 5,777
- Combined coverage: 13,830 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
