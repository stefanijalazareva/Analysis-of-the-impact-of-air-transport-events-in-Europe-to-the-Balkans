# Comprehensive Delay Analysis Report: LDDU
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.2 ± 12.1 minutes
- **Early Arrivals (Negative):** Average 7.8 ± 7.5 minutes early
- **Asymmetry Index:** 1.43 (balanced operations)
- **Operational Efficiency:** 41.1% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0063, p = 0.662403
- **Negative Delay Fit:** KS = 0.0194, p = 0.001789

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 13,236 observations

**Burr XII Parameters:**
- Shape c: 2.723
- Shape d: 0.356
- Scale: 15.73
- Location: 0.017

**Performance Metrics:**
- Mean: 11.2 minutes
- Standard Deviation: 12.1 minutes
- 90th percentile: 23.4 minutes
- 95th percentile: 31.3 minutes

**Model Quality:**
- KS Statistic: 0.0063
- P-value: 0.662403
- AIC: 89910.54

### Negative Delays (Early Arrivals) 
**Sample:** 9,273 observations

**Burr XII Parameters:**
- Shape c: 3.509
- Shape d: 0.240
- Scale: 13.57
- Location: 0.017

**Performance Metrics:**
- Mean: 7.8 minutes (early)
- Standard Deviation: 7.5 minutes
- 90th percentile: 16.6 minutes (early)
- 95th percentile: 20.7 minutes (early)

**Model Quality:**
- KS Statistic: 0.0194
- P-value: 0.001789
- AIC: 56267.78

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.43
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 23.4 minutes
- 5% of flights delayed > 31.3 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 16.6 minutes early
- 5% of flights arrive > 20.7 minutes early

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
- Positive delay samples: 13,236
- Negative delay samples: 9,273
- Combined coverage: 22,509 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
