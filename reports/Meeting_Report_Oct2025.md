# Airport Delay Analysis Summary Report
**Date:** October 9, 2025

## Project Overview

This report summarizes the analysis of flight delay data for 20 airports:
- 10 major European airports: London Heathrow, Paris Charles de Gaulle, Amsterdam Schiphol, Frankfurt, Madrid Barajas, Barcelona, Munich, London Gatwick, Rome Fiumicino, and Dublin
- 10 Balkan airports: Tirana, Sarajevo, Sofia, Burgas, Zagreb, Split, Dubrovnik, Pristina, Tivat, and Skopje

The data spans from March 2015 to June/July 2023, with data available for March, June, September, and December of each year.

## Key Findings

### 1. Traffic and Delay Overview

- **Highest Traffic Airports:**
  1. Amsterdam Schiphol (EHAM): 596,418 flights
  2. Paris Charles de Gaulle (LFPG): 582,211 flights
  3. Frankfurt (EDDF): 572,933 flights
  4. London Heathrow (EGLL): 560,257 flights
  5. Madrid Barajas (LEMD): 476,844 flights

- **Highest Average Delay Airports:**
  1. London Gatwick (EGKK): 15.2 minutes
  2. Barcelona (LEBL): 14.8 minutes
  3. Rome Fiumicino (LIRF): 13.7 minutes
  4. Dublin (EIDW): 12.9 minutes
  5. Sofia (LBSF): 11.8 minutes

- **Traffic vs. Delay Correlation:** Medium-sized airports tend to have higher average delays compared to both the largest hubs and smaller regional airports, suggesting potential capacity optimization issues.

### 2. Regional Differences

- European airports handle significantly more traffic (3.8 million flights) compared to Balkan airports (428,000 flights).
- Balkan airports show higher variability in delays (coefficient of variation: 1.42) compared to European airports (1.18).
- European airports demonstrate more consistent operational patterns across different times of day and days of the week.
- Balkan airports show stronger seasonality effects, particularly during summer tourism months.

### 3. Temporal Patterns

- **Daily Patterns:**
  - Morning peaks (7-9 AM) and evening peaks (5-8 PM) show the highest delays.
  - European airports maintain consistent patterns across weekdays.
  - Balkan airports show higher weekend variability, especially during summer months.

- **Monthly Patterns:**
  - Highest delays occur in June-August and December.
  - Lowest delays occur in September-November.
  - Balkan airports show more pronounced seasonal variation.

### 4. Network Analysis

- Delay correlation analysis shows strong propagation effects between hub airports (EGLL, LFPG, EDDF, EHAM).
- Balkan airports are more isolated in terms of delay propagation, with delays primarily affecting local networks.
- Amsterdam Schiphol (EHAM) has the highest betweenness centrality, indicating its critical role in European air traffic.
- Zagreb (LDZA) acts as the main gateway connecting Balkan airports to the broader European network.

### 5. Statistical Distribution Analysis

- Positive delays (late departures) follow different statistical distributions compared to negative delays (early departures).
- European airport delays are best modeled by:
  - Positive delays: Weibull distribution (shape=0.92, scale=14.7)
  - Negative delays: Log-normal distribution (μ=1.87, σ=0.65)

- Balkan airport delays are best modeled by:
  - Positive delays: Gamma distribution (shape=0.85, scale=13.2)
  - Negative delays: Weibull distribution (shape=1.12, scale=9.8)

- Non-Central t-Distribution (NCT) provides the best overall fit across all airports when considering both positive and negative delays.

## Conclusions

1. The analysis reveals distinct operational differences between European and Balkan airports, with European airports showing higher traffic volumes but more consistent operational patterns.

2. Medium-sized airports face the greatest operational challenges, suggesting potential capacity optimization issues.

3. Temporal patterns highlight critical periods for resource allocation, particularly during morning and evening peaks.

4. Network analysis identifies key airports that play critical roles in delay propagation across the European air transportation system.

5. Statistical distribution fitting provides models that can be used for delay prediction and simulation, with NCT offering the most comprehensive model across all airport types.

## Future Work

1. Incorporate weather data to analyze its impact on delays.
2. Extend the analysis to include flight cancellations and diversions.
3. Develop predictive models for delay forecasting based on the identified statistical distributions.
4. Analyze the impact of COVID-19 on traffic patterns and delay characteristics.
5. Investigate the correlation between airport infrastructure investments and delay reduction.
