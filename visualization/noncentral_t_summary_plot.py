# noncentral_t_summary_plot.py
# Description: Creates combined Noncentral Student’s t-distribution plots for EU vs Balkan airports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nct
import seaborn as sns

def plot_combined_nct_fits(param_csv, output_dir='results/NonCentralT_Fits'):
    """Plot combined NCT fits for Europe and Balkan airports."""
    # Load parameter table generated previously
    df = pd.read_csv(param_csv)

    if df.empty:
        print(" Parameter CSV is empty or not found.")
        return

    # Separate by region
    df_eu = df[df['Region'] == 'Europe']
    df_balkan = df[df['Region'] == 'Balkans']

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Generate x-axis
    x = np.linspace(-60, 120, 1000)

    # --- Europe ---
    for _, row in df_eu.iterrows():
        params = (row['df'], row['nc'], row['loc (mean)'], row['scale (std)'])
        y = nct.pdf(x, *params)
        plt.plot(x, y, alpha=0.6, lw=1.8, label=f"{row['Airport']} (EU)")

    # --- Balkans ---
    for _, row in df_balkan.iterrows():
        params = (row['df'], row['nc'], row['loc (mean)'], row['scale (std)'])
        y = nct.pdf(x, *params)
        plt.plot(x, y, alpha=0.8, lw=1.8, linestyle="--", label=f"{row['Airport']} (Balkans)")

    # --- Styling ---
    plt.title("Comparison of Noncentral t-distribution Fits\nEurope vs Balkans Airports", fontsize=15)
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Density")
    plt.legend(fontsize=8, ncol=2)
    plt.axvline(x=0, color='green', linestyle='--', lw=1, label="On-time")
    plt.grid(alpha=0.3)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'combined_nct_europe_vs_balkan.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()

    print(f" Combined summary plot saved to: {summary_path}")

if __name__ == "__main__":
    # Correct path to your CSV file in data/NonCentralT/
    param_csv = os.path.join('data', 'NonCentralT', 'noncentral_t_parameters.csv')
    plot_combined_nct_fits(param_csv)
