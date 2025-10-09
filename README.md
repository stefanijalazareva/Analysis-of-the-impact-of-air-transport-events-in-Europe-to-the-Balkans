# Air Transport Events Analysis

This repository contains the analysis of flight delay data for European and Balkan airports from March 2015 to July 2023.

## Project Overview

This analysis examines and compares flight delays between 10 major European airports and 10 Balkan airports:

**European Airports**:
- London Heathrow (EGLL)
- Paris Charles de Gaulle (LFPG)
- Amsterdam Schiphol (EHAM)
- Frankfurt (EDDF)
- Madrid Barajas (LEMD)
- Barcelona (LEBL)
- Munich (EDDM)
- London Gatwick (EGKK)
- Rome Fiumicino (LIRF)
- Dublin (EIDW)

**Balkan Airports**:
- Tirana (LATI)
- Sarajevo (LQSA)
- Sofia (LBSF)
- Burgas (LBBG)
- Zagreb (LDZA)
- Split (LDSP)
- Dubrovnik (LDDU)
- Pristina (BKPR)
- Tivat (LYTV)
- Skopje (LWSK)

## Analysis Components

The analysis covers:
1. **Traffic and Delay Overview** - Basic statistics and visualizations of traffic volumes and delays
2. **Regional Differences** - Comparison of European vs. Balkan airport performance
3. **Temporal Patterns** - Analysis of hourly, daily, and monthly delay patterns
4. **Network Analysis** - Delay correlation and propagation between airports
5. **Statistical Distribution Analysis** - Fitting of delay data to statistical distributions

## Repository Structure

### Scripts
- `validate_data.py` - Data validation and initial processing
- `analyze_airports.py` - Detailed airport analysis with visualizations
- `network_analysis.py` - Network structure and delay correlation analysis
- `distribution_fitting.py` - Statistical distribution fitting for delays
- `fast_distribution_fitting.py` - Specialized distribution fitting

### Supporting Modules
- `load_data.py` & `load_dataframe.py` - Data loading utilities
- `delaynet.py` - Network analysis classes and functions
- `network_metrics.py` - Network metrics calculations
- `detrend_timeseries.py` - Time series detrending functions
- `build_timeseries.py` - Time series construction
- `eda_analysis.py` - Exploratory data analysis functions

### Results
- `results/` - Key results and visualizations from the analysis

## Key Findings

For detailed findings, see [Meeting_Report_Oct2025.md](reports/Meeting_Report_Oct2025.md).

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- NetworkX
- SciPy
- Seaborn

## Usage

```python
# Example to run the complete analysis pipeline
python validate_data.py
python analyze_airports.py
python network_analysis.py
python distribution_fitting.py
```

## Visualizations

The analysis includes various visualizations:
- Traffic vs. Delay comparison plots
- Airport network visualizations
- Delay correlation heatmaps
- Delay distribution histograms
- Temporal pattern analysis

All visualization results are stored in the `results/` directory.
