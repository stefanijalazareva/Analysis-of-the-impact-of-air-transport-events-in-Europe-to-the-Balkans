# Comprehensive Delay Analysis Report: BKPR
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 12.1 ± 11.5 minutes
- **Early Arrivals (Negative):** Average 7.5 ± 7.3 minutes early
- **Asymmetry Index:** 1.62 (more late delays)
- **Operational Efficiency:** 38.2% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0108, p = 0.093660
- **Negative Delay Fit:** KS = 0.0108, p = 0.296977

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 13,072 observations

**Burr XII Parameters:**
- Shape c: 3.047
- Shape d: 0.322
- Scale: 18.18
- Location: 0.017

**Performance Metrics:**
- Mean: 12.1 minutes
- Standard Deviation: 11.5 minutes
- 90th percentile: 25.2 minutes
- 95th percentile: 32.3 minutes

**Model Quality:**
- KS Statistic: 0.0108
- P-value: 0.093660
- AIC: 90816.74

### Negative Delays (Early Arrivals) 
**Sample:** 8,059 observations

**Burr XII Parameters:**
- Shape c: 3.464
- Shape d: 0.264
- Scale: 12.50
- Location: 0.017

**Performance Metrics:**
- Mean: 7.5 minutes (early)
- Standard Deviation: 7.3 minutes
- 90th percentile: 15.4 minutes (early)
- 95th percentile: 19.1 minutes (early)

**Model Quality:**
- KS Statistic: 0.0108
- P-value: 0.296977
- AIC: 48030.11

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.62
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 25.2 minutes
- 5% of flights delayed > 32.3 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 15.4 minutes early
- 5% of flights arrive > 19.1 minutes early

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
- Positive delay samples: 13,072
- Negative delay samples: 8,059
- Combined coverage: 21,131 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
