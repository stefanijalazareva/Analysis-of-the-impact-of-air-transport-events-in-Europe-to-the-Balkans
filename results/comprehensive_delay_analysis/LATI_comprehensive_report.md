# Comprehensive Delay Analysis Report: LATI
**Generated:** 2025-12-02 14:50:52
**Region:** Balkans
**Analysis Type:** Comprehensive (Positive + Negative Delays)

---

## Executive Summary

### Operational Performance Overview
- **Late Delays (Positive):** Average 11.4 ± 11.1 minutes
- **Early Arrivals (Negative):** Average 8.0 ± 8.1 minutes early
- **Asymmetry Index:** 1.41 (balanced operations)
- **Operational Efficiency:** 41.4% (early arrival rate)

### Distribution Model Performance
- **Best Model:** Burr XII (both positive and negative delays)
- **Positive Delay Fit:** KS = 0.0177, p = 0.000001
- **Negative Delay Fit:** KS = 0.0126, p = 0.030155

---

## Detailed Statistical Analysis

### Positive Delays (Late Arrivals)
**Sample:** 22,958 observations

**Burr XII Parameters:**
- Shape c: 3.724
- Shape d: 0.251
- Scale: 18.21
- Location: 0.017

**Performance Metrics:**
- Mean: 11.4 minutes
- Standard Deviation: 11.1 minutes
- 90th percentile: 23.1 minutes
- 95th percentile: 29.7 minutes

**Model Quality:**
- KS Statistic: 0.0177
- P-value: 0.000001
- AIC: 156415.87

### Negative Delays (Early Arrivals) 
**Sample:** 13,284 observations

**Burr XII Parameters:**
- Shape c: 2.714
- Shape d: 0.353
- Scale: 11.70
- Location: 0.017

**Performance Metrics:**
- Mean: 8.0 minutes (early)
- Standard Deviation: 8.1 minutes
- 90th percentile: 16.9 minutes (early)
- 95th percentile: 21.8 minutes (early)

**Model Quality:**
- KS Statistic: 0.0126
- P-value: 0.030155
- AIC: 81551.39

---

## Operational Insights

### Delay Balance Analysis
- **Asymmetry Index:** 1.41
  - Values < 1.0: More early arrivals than late delays
  - Values = 1.0: Perfectly balanced operations  
  - Values > 1.0: More late delays than early arrivals

### Risk Assessment
**Late Arrival Risks:**
- 10% of flights delayed > 23.1 minutes
- 5% of flights delayed > 29.7 minutes

**Early Arrival Impacts:**
- 10% of flights arrive > 16.9 minutes early
- 5% of flights arrive > 21.8 minutes early

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
- Positive delay samples: 22,958
- Negative delay samples: 13,284
- Combined coverage: 36,242 total observations

---
*Comprehensive delay analysis incorporating full operational spectrum for complete understanding of airport performance.*
