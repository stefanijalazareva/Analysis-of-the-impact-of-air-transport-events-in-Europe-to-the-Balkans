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

The project is organized into the following modules:

### `/distribution_analysis`
Statistical distribution fitting and analysis scripts:
- Distribution fitting (Burr, Weibull, GenGamma, Non-central t)
- Comprehensive distribution testing and comparison
- Multi-distribution analysis
- Statistical distribution analysis

### `/statistical_analysis`
Statistical tests and advanced analyses:
- Tail analysis and stability testing
- Confidence interval computation
- Comparative statistical analysis (Burr vs NCT)
- Kolmogorov-Smirnov test visualization
- Integrated statistical analysis

### `/visualization`
Plotting and visualization scripts:
- QQ plots and comparison plots
- Parameter visualizations
- Region comparison plots
- Confidence interval visualizations
- Heatmap and panel generation

### `/data_processing`
Data loading, validation, and preprocessing:
- `data_loader.py` - Main data loading utilities
- `validate_data.py` - Data validation
- `download_data.py` - Data download utilities
- `build_timeseries.py` - Time series construction
- `detrend_timeseries.py` - Time series detrending

### `/analysis`
Airport-specific and general analyses:
- Airport analysis and comparisons
- Comprehensive delay analysis
- Individual airport analysis
- Enhanced airport analysis
- Exploratory data analysis (EDA)

### `/network_analysis`
Network structure and correlation analysis:
- Network metrics and calculations
- Delay correlation analysis
- Network propagation studies

### `/utils`
Utility scripts and helpers:
- Airport grouping definitions
- Report generators
- Analysis runners
- Bootstrap and parallel processing utilities

### `/data`
Data directories:
- `RawData/` - Original flight delay data
- `ProcessedData/` - Cleaned and processed data
- `TimeSeries/` - Time series data
- `Analysis/` - Analysis outputs
- `DistributionFitting/` - Distribution fitting results
- `NonCentralT/` - NCT-specific results

### `/reports`
Analysis reports and documentation

### `/results`
Generated visualizations and analysis outputs

### Core Module
- `delaynet.py` - Main network analysis classes and functions

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
# Data processing and validation
from data_processing.data_loader import DataLoader
from data_processing.validate_data import validate

# Run comprehensive analysis
from analysis.analyze_airports import analyze
from analysis.eda_analysis import exploratory_analysis

# Distribution fitting
from distribution_analysis.distribution_fitting import fit_distributions
from distribution_analysis.burr_distribution_analysis import analyze_burr

# Statistical analysis
from statistical_analysis.integrated_statistical_analysis import run_tests

# Network analysis
from network_analysis.network_analysis import analyze_network

# Visualization
from visualization.enhanced_qq_plots import generate_qq_plots
```

**Quick Start:**
```bash
# 1. Validate and load data
python data_processing/validate_data.py

# 2. Run exploratory analysis
python analysis/eda_analysis.py

# 3. Fit distributions
python distribution_analysis/distribution_fitting.py

# 4. Run network analysis
python network_analysis/network_analysis.py
```

## Visualizations

The analysis includes various visualizations:
- Traffic vs. Delay comparison plots
- Airport network visualizations
- Delay correlation heatmaps
- Delay distribution histograms
- Temporal pattern analysis

All visualization results are stored in the `results/` directory.
