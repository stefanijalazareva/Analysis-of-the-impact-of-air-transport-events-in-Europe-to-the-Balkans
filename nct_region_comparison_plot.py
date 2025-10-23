# nct_region_comparison_plot.py
# Description: Compare mean (loc) and scale (std) parameters of Noncentral t fits
# between European and Balkan airports.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_nct_region_comparison(param_csv, output_dir='results/Combined_Comparison'):
    """Plot comparison of Noncentral t fit parameters between Europe and Balkans."""
    df = pd.read_csv(param_csv)

    if df.empty:
        print(" Parameter CSV is empty or not found.")
        return

    # Extract only needed columns
    df = df[['Airport', 'Region', 'loc (mean)', 'scale (std)']]

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Average loc (mean delay) per region ---
    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df,
        x='Region',
        y='loc (mean)',
        palette='coolwarm',
        ci='sd'
    )
    plt.title('Average Mean Delay (loc parameter)\nEurope vs Balkans')
    plt.ylabel('Mean Delay (minutes)')
    plt.xlabel('')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    mean_path = os.path.join(output_dir, 'region_mean_delay_comparison.png')
    plt.savefig(mean_path, dpi=300)
    plt.close()
    print(f" Saved mean comparison plot to: {mean_path}")

    # --- Average scale (standard deviation) per region ---
    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df,
        x='Region',
        y='scale (std)',
        palette='viridis',
        ci='sd'
    )
    plt.title('Average Delay Variability (scale parameter)\nEurope vs Balkans')
    plt.ylabel('Standard Deviation (minutes)')
    plt.xlabel('')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    scale_path = os.path.join(output_dir, 'region_scale_delay_comparison.png')
    plt.savefig(scale_path, dpi=300)
    plt.close()
    print(f" Saved scale comparison plot to: {scale_path}")

if __name__ == "__main__":
    # Use your current Noncentral t parameters file
    param_csv = os.path.join('data', 'NonCentralT', 'noncentral_t_parameters.csv')
    plot_nct_region_comparison(param_csv)
