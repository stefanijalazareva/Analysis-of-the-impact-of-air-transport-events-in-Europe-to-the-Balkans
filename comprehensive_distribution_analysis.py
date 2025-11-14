"""
Comprehensive Distribution Analysis for Air Transport Delays
"""

import sys
import os
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
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure plotting style
sns.set_theme(style="whitegrid")  # Use seaborn's whitegrid style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.figsize': (10, 6)
})

class DistributionAnalyzer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.results_dir = Path('results/distribution_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.distributions = {
            'normal': {'dist': norm, 'name': 'Normal'},
            'nct': {'dist': nct, 'name': 'Noncentral t'},
            'lognorm': {'dist': lognorm, 'name': 'Log-Normal'},
            'gamma': {'dist': gamma, 'name': 'Gamma'}
        }

    def _get_param_names(self, dist_name: str) -> List[str]:
        """Get parameter names for each distribution"""
        param_names = {
            'normal': ['loc', 'scale'],  # mean, std
            'nct': ['df', 'nc', 'loc', 'scale'],  # degrees of freedom, noncentrality, location, scale
            'lognorm': ['s', 'loc', 'scale'],  # shape, location, scale
            'gamma': ['a', 'loc', 'scale']  # shape, location, scale
        }
        return param_names.get(dist_name, [])

    def _stratified_sample(self, data: np.ndarray, size: int) -> np.ndarray:
        """Create a stratified sample of the data to maintain distribution characteristics"""
        percentiles = np.percentile(data, np.linspace(0, 100, 11))  # 10 strata
        strata = []
        stratum_size = size // 10

        for i in range(len(percentiles)-1):
            stratum = data[(data >= percentiles[i]) & (data < percentiles[i+1])]
            if len(stratum) > 0:
                sample = np.random.choice(stratum, size=min(stratum_size, len(stratum)), replace=True)
                strata.append(sample)

        return np.concatenate(strata)

    def calculate_confidence_intervals(self, data: np.ndarray, dist_name: str,
                                    n_bootstrap: int = 200) -> Dict:
        """Calculate confidence intervals for distribution parameters using bootstrapping"""
        dist = self.distributions[dist_name]['dist']
        bootstrap_params = []

        # Use smaller samples for bootstrapping to improve speed
        sample_size = min(len(data), 2000)  # Further reduced sample size for nct

        # Reduce bootstrap samples for nct distribution
        if dist_name == 'nct':
            n_bootstrap = 50  # Reduced bootstraps for nct due to computational intensity

        for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap {dist_name}"):
            try:
                # Use stratified sampling to maintain distribution characteristics
                sample = self._stratified_sample(data, sample_size)

                if dist_name == 'nct':
                    # For nct, use moment-based initial guesses
                    loc_guess = float(np.mean(sample))
                    scale_guess = float(np.std(sample))
                    skew = float(stats.skew(sample))
                    kurtosis = float(stats.kurtosis(sample))

                    # Estimate df based on kurtosis (heavy tails)
                    df_guess = max(2.1, 6.0 / (kurtosis + 2))  # ensure df > 2
                    # Estimate nc based on skewness
                    nc_guess = np.sign(skew) * min(abs(skew), 2.0)

                    params = dist.fit(sample, f0=[df_guess, nc_guess, loc_guess, scale_guess])
                else:
                    params = dist.fit(sample)

                bootstrap_params.append(params)
            except Exception as e:
                print(f"Warning: Bootstrap iteration failed for {dist_name}: {str(e)}")
                continue

        if not bootstrap_params:
            print(f"Warning: No successful bootstrap iterations for {dist_name}")
            return None

        bootstrap_params = np.array(bootstrap_params)
        ci_lower = np.percentile(bootstrap_params, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_params, 97.5, axis=0)

        param_names = self._get_param_names(dist_name)
        return {
            'parameters': param_names,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_params': np.mean(bootstrap_params, axis=0)
        }

    def analyze_ks_sensitivity(self, data: np.ndarray, dist_name: str) -> pd.DataFrame:
        """Analyze KS test sensitivity to sample size"""
        sample_sizes = [100, 500, 1000, min(5000, len(data))]
        results = []

        dist = self.distributions[dist_name]['dist']

        for size in sample_sizes:
            # Take multiple random samples to get confidence in the results
            ks_stats = []
            p_values = []
            for _ in range(5):  # Reduced from 10 to 5 repetitions for efficiency
                sample = np.random.choice(data, size=size, replace=False)
                try:
                    params = dist.fit(sample)
                    # Create a frozen distribution with fitted parameters
                    fitted_dist = dist(*params)
                    ks_stat, p_value = kstest(sample, fitted_dist.cdf)
                    ks_stats.append(ks_stat)
                    p_values.append(p_value)
                except:
                    continue

            if ks_stats:  # Only add results if we got valid statistics
                results.append({
                    'sample_size': size,
                    'mean_ks_statistic': np.mean(ks_stats),
                    'std_ks_statistic': np.std(ks_stats),
                    'mean_p_value': np.mean(p_values),
                    'std_p_value': np.std(p_values)
                })

        return pd.DataFrame(results)

    def compare_distributions(self, data: np.ndarray) -> pd.DataFrame:
        """Compare fit quality of different distributions"""
        results = []
        for name, dist_info in self.distributions.items():
            dist = dist_info['dist']
            try:
                params = dist.fit(data)

                # Create a frozen distribution with fitted parameters
                fitted_dist = dist(*params)

                # Calculate AIC
                log_likelihood = np.sum(fitted_dist.logpdf(data))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood

                # KS test using the frozen distribution
                ks_stat, p_value = kstest(data, fitted_dist.cdf)

                results.append({
                    'distribution': name,
                    'aic': aic,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'parameters': params
                })
            except Exception as e:
                print(f"Error fitting {name} distribution: {str(e)}")
                continue

        return pd.DataFrame(results)

    def generate_qq_plot(self, data: np.ndarray, airport: str):
        """Generate QQ plots for normal and noncentral-t distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Normal Q-Q plot
        stats.probplot(data, dist="norm", plot=ax1)
        ax1.set_title(f'Normal Q-Q Plot - {airport}')

        # Noncentral t Q-Q plot
        nct_params = stats.nct.fit(data)
        theoretical_quantiles = stats.nct.ppf(
            np.linspace(0.01, 0.99, len(data)),
            *nct_params
        )
        ax2.scatter(np.sort(theoretical_quantiles), np.sort(data))
        ax2.plot([data.min(), data.max()], [data.min(), data.max()], 'r--')
        ax2.set_title(f'Noncentral t Q-Q Plot - {airport}')

        plt.tight_layout()
        plt.savefig(self.results_dir / f'qq_plots_{airport}.png')
        plt.close()

    def analyze_airport(self, airport_code: str):
        """Complete analysis for one airport"""
        # Load raw delay data for the airport
        filepath = Path('data/RawData') / f'Delays_{airport_code}.npy'
        data = np.load(filepath, allow_pickle=True)
        delays = data[:, 3].astype(float)  # Extract delays from the 4th column

        print(f"\nAnalyzing airport: {airport_code}")
        print(f"Number of delay records: {len(delays)}")

        results = {
            'airport': airport_code,
            'distribution_comparison': self.compare_distributions(delays),
            'confidence_intervals': {},
            'ks_sensitivity': {}
        }

        # Calculate confidence intervals and KS sensitivity for each distribution
        for dist_name in self.distributions:
            print(f"\nAnalyzing {dist_name} distribution...")
            ci_results = self.calculate_confidence_intervals(delays, dist_name)
            if ci_results is not None:
                results['confidence_intervals'][dist_name] = ci_results

            ks_results = self.analyze_ks_sensitivity(delays, dist_name)
            if not ks_results.empty:
                results['ks_sensitivity'][dist_name] = ks_results

        # Generate QQ plot
        self.generate_qq_plot(delays, airport_code)

        return results

    def save_results(self, results: Dict, airport_code: str):
        """Save analysis results"""
        airport_dir = self.results_dir / airport_code
        airport_dir.mkdir(exist_ok=True)

        # Save distribution comparison
        results['distribution_comparison'].to_csv(
            airport_dir / 'distribution_comparison.csv'
        )

        # Save confidence intervals
        pd.DataFrame(results['confidence_intervals']).to_json(
            airport_dir / 'confidence_intervals.json'
        )

        # Save KS sensitivity analysis
        for dist_name, sensitivity in results['ks_sensitivity'].items():
            sensitivity.to_csv(
                airport_dir / f'ks_sensitivity_{dist_name}.csv'
            )

def main():
    analyzer = DistributionAnalyzer()

    # Get list of airports from data directory
    data_dir = Path('data/RawData')
    airports = [f.stem.replace('Delays_', '') for f in data_dir.glob('Delays_*.npy')]

    all_results = []
    for airport in tqdm(airports, desc="Analyzing airports"):
        results = analyzer.analyze_airport(airport)
        analyzer.save_results(results, airport)
        all_results.append(results)

    # Generate summary report
    summary_dir = analyzer.results_dir / 'summary'
    summary_dir.mkdir(exist_ok=True)

    # Combine all distribution comparisons
    all_comparisons = pd.concat([
        r['distribution_comparison'].assign(airport=r['airport'])
        for r in all_results
    ])
    all_comparisons.to_csv(summary_dir / 'all_distribution_comparisons.csv')

if __name__ == "__main__":
    main()
