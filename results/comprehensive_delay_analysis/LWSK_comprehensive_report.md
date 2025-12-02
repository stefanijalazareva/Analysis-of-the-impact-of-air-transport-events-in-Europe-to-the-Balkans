# Comprehensive Delay Analysis Report: LWSK
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.9 ± 12.6 minutes
- **Early Arrivals (Negative):** Average 9.4 ± 8.1 minutes early
- **Asymmetry Index:** 1.26 (balanced operations)
- **Operational Efficiency:** 44.2% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0200, p = 0.000229
- **Negative Delay Fit:** KS = 0.0235, p = 0.000049

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 11,374 observations

**Burr XII Parameters:**
- Shape c: 3.069
- Shape d: 0.315
- Scale: 18.04
- Location: 0.017

**Performance Metrics:**
- Mean: 11.9 minutes
- Standard Deviation: 12.6 minutes
- 90th percentile: 24.7 minutes
- 95th percentile: 32.4 minutes

**Model Quality:**
- KS Statistic: 0.0200
- P-value: 0.000229
- AIC: 78567.98

### Negative Delays (Early Arrivals) 
**Sample:** 9,559 observations

**Burr XII Parameters:**
- Shape c: 3.815
- Shape d: 0.232
- Scale: 16.96
- Location: 0.017

**Performance Metrics:**
- Mean: 9.4 minutes (early)
- Standard Deviation: 8.1 minutes
- 90th percentile: 19.8 minutes (early)
- 95th percentile: 24.5 minutes (early)

**Model Quality:**
- KS Statistic: 0.0235
- P-value: 0.000049
- AIC: 61451.78

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.26
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 24.7 minutes
- 5% of flights delayed > 32.4 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 19.8 minutes early
- 5% of flights arrive > 24.5 minutes early

### Operational Recommendations

- **Balance**: Good operational balance between early and late operations
- **Maintain**: Continue current operational procedures
- **Optimize**: Fine-tune scheduling for marginal improvements

### Regional Context
This airport's performance aligns with typical Balkans patterns:
- Balkans airports show lower operational variability
- Delay asymmetry is typical for the region
- Model fit quality is challenging with regional standards

---

## Technical Notes
- Analysis includes complete delay spectrum (positive and negative)
- Burr XII distribution provides optimal fit for both delay types
- Statistical significance assessed using Kolmogorov-Smirnov tests
- Regional classifications based on operational similarities

**Data Quality:**
- Positive delay samples: 11,374
- Negative delay samples: 9,559
- Combined coverage: 20,933 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
