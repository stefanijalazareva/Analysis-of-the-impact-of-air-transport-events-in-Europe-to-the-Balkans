"""
Statistical Analysis of Air Transport Delays Distribution Fits

This script implements the analysis requirements from the November 2025 meeting:
- Handles KS test sensitivity to data quantity
- Extracts confidence intervals for fits
- Compares multiple distributions
- Generates comprehensive comparison tables and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (norm, nct, lognorm, gamma, weibull_min,
                        expon, ks_2samp, kstest)
from pathlib import Path
import json
from data_loader import DataLoader
from airport_groups import EU_AIRPORTS, BALKAN_AIRPORTS
import warnings
from tqdm import tqdm
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from scipy.stats import t as student_t

class DistributionAnalyzer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.results_dir = Path("results/distribution_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Use predefined airport groups
        self.eu_airports = EU_AIRPORTS
        self.balkan_airports = BALKAN_AIRPORTS

        # Configure distributions to test
        self.distributions = {
            'norm': {'dist': norm, 'name': 'Normal', 'color': 'blue'},
            'nct': {'dist': nct, 'name': 'Noncentral t', 'color': 'red'},
            'lognorm': {'dist': lognorm, 'name': 'Log-Normal', 'color': 'green'},
            'gamma': {'dist': gamma, 'name': 'Gamma', 'color': 'purple'}
        }

    def fit_with_confidence(self, data: np.ndarray, dist_name: str,
                          bootstrap_samples: int = 1000) -> Dict:
        """
        Fit distribution with bootstrap confidence intervals.
        Addresses the data quantity sensitivity issue through bootstrapping.
        """
        dist = self.distributions[dist_name]['dist']
        n = len(data)

        # For large datasets, use stratified sampling
        if n > 10000:
            # Stratify by percentiles to maintain distribution shape
            percentiles = np.percentile(data, np.linspace(0, 100, 21))
            sample_size = 10000
            stratified_sample = []

            for i in range(len(percentiles)-1):
                mask = (data >= percentiles[i]) & (data < percentiles[i+1])
                stratum = data[mask]
                if len(stratum) > 0:
                    n_samples = int(sample_size * len(stratum) / n)
                    stratified_sample.extend(
                        np.random.choice(stratum, size=n_samples, replace=False))

            data = np.array(stratified_sample)

        # Fit to full dataset
        params = dist.fit(data)

        # Bootstrap for confidence intervals
        bootstrap_params = []
        for _ in range(bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            try:
                sample_params = dist.fit(sample)
                bootstrap_params.append(sample_params)
            except:
                continue

        bootstrap_params = np.array(bootstrap_params)

        # Calculate confidence intervals (95%)
        ci_lower = np.percentile(bootstrap_params, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_params, 97.5, axis=0)

        # Calculate KS statistic with sample size consideration
        if len(data) > 1000:
            test_data = np.random.choice(data, size=1000, replace=False)
        else:
            test_data = data
        ks_stat = kstest(test_data, dist_name, params).statistic

        # Prepare parameter names
        if dist_name == 'norm':
            param_names = ['loc', 'scale']
        else:
            param_names = dist.shapes.split(',') + ['loc', 'scale']

        # Package results
        results = {
            'distribution': dist_name,
            'parameters': dict(zip(param_names, params)),
            'confidence_intervals': {
                name: {'lower': l, 'upper': u}
                for name, l, u in zip(param_names, ci_lower, ci_upper)
            },
            'ks_statistic': ks_stat,
            'sample_size': len(data)
        }

        return results

    def analyze_by_region(self, df: pd.DataFrame, max_samples: int = 100000) -> Dict:
        """Analyze delays separately for EU and Balkan airports."""
        print("\nAnalyzing delays by region...")
        results = {
            'eu': {'airports': self.eu_airports,
                  'data': df[df['arr'].isin(self.eu_airports)]},
            'balkan': {'airports': self.balkan_airports,
                      'data': df[df['arr'].isin(self.balkan_airports)]}
        }

        for region, data in results.items():
            print(f"\nProcessing {region.upper()} region...")
            n_flights = len(data['data'])
            print(f"Found {len(data['airports'])} airports with {n_flights} records")

            # Sample data if needed
            delays = data['data']['delay_s'].values
            if len(delays) > max_samples:
                print(f"Sampling {max_samples} records for analysis...")
                delays = np.random.choice(delays, size=max_samples, replace=False)

            fits = {}
            for dist_name, config in self.distributions.items():
                print(f"Fitting {config['name']} distribution...")
                fits[dist_name] = self.fit_with_confidence(delays, dist_name,
                                                         bootstrap_samples=500)
                print(f"KS statistic: {fits[dist_name]['ks_statistic']:.4f}")

            data['fits'] = fits
            self.generate_comparison_plots(delays, fits, f"{region}_combined")

        return results

    def generate_regional_summary(self, region_results: Dict):
        """Generate summary statistics and visualizations for regional comparison."""
        summary_dir = self.results_dir / "regional_comparison"
        summary_dir.mkdir(exist_ok=True)

        # Create summary tables
        summary_data = []
        for region in ['eu', 'balkan']:
            data = region_results[region]['data']
            summary = {
                'Region': region.upper(),
                'Airports': len(region_results[region]['airports']),
                'Total_Flights': len(data),
                'Mean_Delay': data['delay_s'].mean(),
                'Median_Delay': data['delay_s'].median(),
                'Std_Delay': data['delay_s'].std(),
                'Skewness': data['delay_s'].skew(),
                'Kurtosis': data['delay_s'].kurtosis()
            }
            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_dir / 'regional_summary.csv', index=False)

        # Generate comparison plots
        plt.figure(figsize=(12, 6))
        for region in ['eu', 'balkan']:
            data = region_results[region]['data']['delay_s']
            plt.hist(data, bins=50, alpha=0.5, density=True,
                    label=region.upper())
        plt.xlabel('Delay (seconds)')
        plt.ylabel('Density')
        plt.title('Delay Distribution Comparison: EU vs Balkans')
        plt.legend()
        plt.savefig(summary_dir / 'region_comparison_hist.png')
        plt.close()

        return summary_df

    def run_analysis(self, max_samples: int = 100000):
        """Run the complete analysis pipeline."""
        print("Loading data...")
        df = self.data_loader.load_processed_data()

        print("\nAnalyzing regional patterns...")
        region_results = self.analyze_by_region(df, max_samples)

        print("\nGenerating regional summary...")
        summary_df = self.generate_regional_summary(region_results)

        print("\nGenerating detailed reports...")
        self.generate_region_comparison_report(region_results)

        print(f"\nAnalysis complete! Results saved in: {self.results_dir}")
        print("\nRegional Summary:")
        print(summary_df.to_string(index=False))

        return region_results
