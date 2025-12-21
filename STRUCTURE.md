# Project Structure Refactoring

**Date:** December 21, 2025

## Overview
The project has been reorganized into a modular structure for better navigation and maintainability.

## New Directory Structure

```
AirTransportEvents/
├── distribution_analysis/      # Statistical distribution fitting
│   ├── burr_distribution_analysis.py
│   ├── burr_deep_analysis.py
│   ├── weibull_distribution_analysis.py
│   ├── gengamma_distribution_analysis.py
│   ├── noncentral_t_fitting.py
│   ├── distribution_fitting.py
│   ├── fast_distribution_fitting.py
│   ├── extended_distribution_fitting.py
│   ├── complete_distribution_testing.py
│   ├── delay_distribution_analysis.py
│   ├── comprehensive_distribution_analysis.py
│   ├── comprehensive_new_distributions_analysis.py
│   ├── simple_new_distributions_analysis.py
│   ├── multi_distribution_analysis.py
│   ├── fit_new_distributions.py
│   ├── full_normal_distribution_analysis.py
│   └── statistical_distribution_analysis.py
│
├── statistical_analysis/       # Statistical tests & tail analysis
│   ├── advanced_tail_and_stability_analysis.py
│   ├── tail_comparison_burr_vs_nct.py
│   ├── compute_burr_confidence_intervals.py
│   ├── enhanced_confidence_analysis.py
│   ├── compare_burr_vs_nct_ci.py
│   ├── ks_test_visualization.py
│   └── integrated_statistical_analysis.py
│
├── visualization/               # Plotting & visualization
│   ├── enhanced_qq_plots.py
│   ├── qq_plots_comparison.py
│   ├── clear_qq_plots.py
│   ├── fix_qq_plot_scale.py
│   ├── NCT_parameter_visualizations.py
│   ├── nct_region_comparison_plot.py
│   ├── noncentral_t_summary_plot.py
│   ├── noncentral_t_visual_fit.py
│   ├── visualization_CI.py
│   ├── extract_burr_panels.py
│   └── fix_heatmap_mismatch.py
│
├── data_processing/             # Data loading & preprocessing
│   ├── data_loader.py
│   ├── validate_data.py
│   ├── download_data.py
│   ├── build_timeseries.py
│   └── detrend_timeseries.py
│
├── analysis/                    # Airport & delay analysis
│   ├── analyze_airports.py
│   ├── comprehensive_delay_analysis.py
│   ├── comprehensive_individual_airport_analysis.py
│   ├── individual_airport_analysis.py
│   ├── individual_airport_analysis_complete.py
│   ├── enhanced_clean_airport_analysis.py
│   └── eda_analysis.py
│
├── network_analysis/            # Network metrics & correlation
│   ├── network_analysis.py
│   └── network_metrics.py
│
├── utils/                       # Utilities & helpers
│   ├── airport_groups.py
│   ├── run_analysis.py
│   ├── fast_analysis.py
│   ├── generate_comprehensive_report.py
│   ├── generate_detailed_reports.py
│   ├── generate_distribution_report.py
│   ├── combine_eu_balkan_fits.py
│   └── nct_bootstrap_parallel.py
│
├── data/                        # Data files
│   ├── RawData/
│   ├── ProcessedData/
│   ├── TimeSeries/
│   ├── Analysis/
│   ├── DistributionFitting/
│   └── NonCentralT/
│
├── reports/                     # Analysis reports
├── results/                     # Generated outputs
├── depricated/                  # Deprecated code
├── delaynet.py                     # Core network module
└── README.md                       # Main documentation
```

## Module Descriptions

### Distribution Analysis
Contains all scripts related to fitting and analyzing statistical distributions (Burr, Weibull, GenGamma, Non-central t) to flight delay data.

### Statistical Analysis
Houses statistical testing, tail analysis, confidence interval computation, and comparative analyses between different distributions.

### Visualization
All plotting and visualization scripts including QQ plots, parameter visualizations, heatmaps, and comparative plots.

### Data Processing
Data loading, validation, downloading, time series building, and preprocessing utilities.

### Analysis
General analysis scripts including airport-specific analyses, comprehensive delay analysis, and exploratory data analysis (EDA).

### Network Analysis
Network structure analysis, delay correlation analysis, and network metrics calculations.

### Utils
Utility scripts, helper functions, report generators, and supporting tools for the analysis pipeline.