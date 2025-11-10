"""
Multi-Distribution Analysis for Air Transport Delays
Compares multiple theoretical distributions and ranks them by goodness of fit
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
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn')
plt.rcParams.update({'font.size': 12})

class MultiDistributionAnalyzer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.distributions = {
            'Normal': norm,
            'Noncentral-t': nct,
            'Log-normal': lognorm,
            'Gamma': gamma,
            'Weibull': weibull_min,
            'Exponential': expon
        }
        self.results = {}

    def fit_all_distributions(self, data: np.ndarray) -> Dict:
        """Fit all distributions and compute goodness of fit metrics"""
        results = {}

        for dist_name, dist in self.distributions.items():
            try:
                # Fit distribution
                params = dist.fit(data)

                # Calculate metrics
                ks_stat, p_value = kstest(data, dist.name, params)

                # Calculate AIC and BIC
                log_likelihood = np.sum(dist.logpdf(data, *params))
                n_params = len(params)
                n_samples = len(data)
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_samples) - 2 * log_likelihood

                results[dist_name] = {
                    'params': params,
                    'ks_stat': ks_stat,
                    'p_value': p_value,
                    'aic': aic,
                    'bic': bic
                }
            except Exception as e:
                print(f"Failed to fit {dist_name}: {str(e)}")

        return results

    def compare_distributions(self, airport_code: str) -> None:
        """Compare different distributions for a given airport"""
        data = self.data_loader.load_airport_data(airport_code)

        # Fit all distributions
        self.results[airport_code] = self.fit_all_distributions(data)

        # Create comparison plots
        self._plot_distribution_comparison(data, airport_code)
        self._plot_metrics_comparison(airport_code)

    def _plot_distribution_comparison(self, data: np.ndarray, airport_code: str) -> None:
        """Create PDF comparison plot for all distributions"""
        plt.figure(figsize=(12, 8))

        # Plot histogram of actual data
        plt.hist(data, bins=50, density=True, alpha=0.6, label='Data')

        # Plot fitted distributions
        x = np.linspace(min(data), max(data), 1000)
        for dist_name, result in self.results[airport_code].items():
            try:
                dist = self.distributions[dist_name]
                params = result['params']
                plt.plot(x, dist.pdf(x, *params), label=dist_name)
            except Exception:
                continue

        plt.title(f'Distribution Comparison - {airport_code}')
        plt.xlabel('Delay (minutes)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'results/distribution_analysis/{airport_code}_distribution_comparison.png')
        plt.close()

    def _plot_metrics_comparison(self, airport_code: str) -> None:
        """Create comparison plot for goodness of fit metrics"""
        metrics = ['ks_stat', 'p_value', 'aic', 'bic']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            values = [result[metric] for result in self.results[airport_code].values()]
            names = list(self.results[airport_code].keys())

            axes[idx].bar(names, values)
            axes[idx].set_title(f'{metric.upper()}')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'results/distribution_analysis/{airport_code}_metrics_comparison.png')
        plt.close()

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of best distributions for each airport"""
        summary_data = []

        for airport_code in self.results:
            best_dist = min(self.results[airport_code].items(),
                          key=lambda x: x[1]['aic'])

            summary_data.append({
                'Airport': airport_code,
                'Best Distribution': best_dist[0],
                'AIC': best_dist[1]['aic'],
                'BIC': best_dist[1]['bic'],
                'KS Statistic': best_dist[1]['ks_stat'],
                'P-value': best_dist[1]['p_value']
            })

        return pd.DataFrame(summary_data)

if __name__ == "__main__":
    data_loader = DataLoader()
    analyzer = MultiDistributionAnalyzer(data_loader)

    # Analyze all airports
    for airport in data_loader.get_airport_list():
        print(f"Analyzing {airport}...")
        analyzer.compare_distributions(airport)

    # Generate and save summary
    summary = analyzer.generate_summary_table()
    summary.to_csv('results/distribution_analysis/distribution_comparison_summary.csv', index=False)

    print("Analysis complete. Results saved in results/distribution_analysis/")
