# Comprehensive Delay Analysis Report: LDZA
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 10.8 ± 11.0 minutes
- **Early Arrivals (Negative):** Average 7.8 ± 7.8 minutes early
- **Asymmetry Index:** 1.39 (balanced operations)
- **Operational Efficiency:** 41.8% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0076, p = 0.069192
- **Negative Delay Fit:** KS = 0.0305, p = 0.000000

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 29,373 observations

**Burr XII Parameters:**
- Shape c: 2.722
- Shape d: 0.350
- Scale: 15.63
- Location: 0.017

**Performance Metrics:**
- Mean: 10.8 minutes
- Standard Deviation: 11.0 minutes
- 90th percentile: 23.1 minutes
- 95th percentile: 30.5 minutes

**Model Quality:**
- KS Statistic: 0.0076
- P-value: 0.069192
- AIC: 197868.19

### Negative Delays (Early Arrivals) 
**Sample:** 21,709 observations

**Burr XII Parameters:**
- Shape c: 3.836
- Shape d: 0.234
- Scale: 13.74
- Location: 0.017

**Performance Metrics:**
- Mean: 7.8 minutes (early)
- Standard Deviation: 7.8 minutes
- 90th percentile: 16.2 minutes (early)
- 95th percentile: 20.3 minutes (early)

**Model Quality:**
- KS Statistic: 0.0305
- P-value: 0.000000
- AIC: 131335.23

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.39
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 23.1 minutes
- 5% of flights delayed > 30.5 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 16.2 minutes early
- 5% of flights arrive > 20.3 minutes early

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
- Positive delay samples: 29,373
- Negative delay samples: 21,709
- Combined coverage: 51,082 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
