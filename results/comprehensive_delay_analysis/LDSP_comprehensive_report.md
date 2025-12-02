# Comprehensive Delay Analysis Report: LDSP
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.6 ± 14.6 minutes
- **Early Arrivals (Negative):** Average 7.2 ± 6.9 minutes early
- **Asymmetry Index:** 1.61 (more late delays)
- **Operational Efficiency:** 38.3% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0064, p = 0.412181
- **Negative Delay Fit:** KS = 0.0184, p = 0.001359

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 18,956 observations

**Burr XII Parameters:**
- Shape c: 2.793
- Shape d: 0.348
- Scale: 16.47
- Location: 0.017

**Performance Metrics:**
- Mean: 11.6 minutes
- Standard Deviation: 14.6 minutes
- 90th percentile: 24.3 minutes
- 95th percentile: 32.1 minutes

**Model Quality:**
- KS Statistic: 0.0064
- P-value: 0.412181
- AIC: 129997.40

### Negative Delays (Early Arrivals) 
**Sample:** 10,701 observations

**Burr XII Parameters:**
- Shape c: 3.333
- Shape d: 0.250
- Scale: 12.43
- Location: 0.017

**Performance Metrics:**
- Mean: 7.2 minutes (early)
- Standard Deviation: 6.9 minutes
- 90th percentile: 15.4 minutes (early)
- 95th percentile: 19.7 minutes (early)

**Model Quality:**
- KS Statistic: 0.0184
- P-value: 0.001359
- AIC: 63304.22

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.61
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 24.3 minutes
- 5% of flights delayed > 32.1 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 15.4 minutes early
- 5% of flights arrive > 19.7 minutes early

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
- Positive delay samples: 18,956
- Negative delay samples: 10,701
- Combined coverage: 29,657 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
