"""
Kolmogorov-Smirnov Test Results Visualization

This script creates visualizations for KS test results across different airports
and distributions to help understand the goodness of fit.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def load_all_distribution_results():
    """Load distribution comparison results from all airports."""
    all_results = []

    # Get all distribution comparison files
    files = glob.glob('results/distribution_analysis/*/distribution_comparison.csv')

    for file in files:
        airport = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df['airport'] = airport
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)

def create_ks_heatmap(data):
    """Create a heatmap of KS statistics across airports and distributions."""
    pivot_ks = data.pivot(index='airport', columns='distribution', values='ks_statistic')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_ks, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('KS Statistics Across Airports and Distributions')
    plt.tight_layout()
    plt.savefig('results/distribution_analysis/ks_test_heatmap.png')
    plt.close()

def plot_ks_comparison(data):
    """Create a box plot comparing KS statistics across distributions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='distribution', y='ks_statistic', data=data)
    plt.xticks(rotation=45)
    plt.title('Distribution of KS Statistics by Distribution Type')
    plt.tight_layout()
    plt.savefig('results/distribution_analysis/ks_test_boxplot.png')
    plt.close()

def generate_summary_table(data):
    """Generate a summary table of KS test results."""
    summary = data.groupby('distribution').agg({
        'ks_statistic': ['mean', 'std', 'min', 'max'],
        'p_value': ['mean', 'max']
    }).round(4)

    summary.to_csv('results/distribution_analysis/ks_test_summary.csv')
    return summary

def main():
    # Load all results
    results = load_all_distribution_results()

    # Create visualizations
    create_ks_heatmap(results)
    plot_ks_comparison(results)

    # Generate and save summary
    summary = generate_summary_table(results)
    print("\nKS Test Summary Statistics:")
    print(summary)

if __name__ == "__main__":
    main()
