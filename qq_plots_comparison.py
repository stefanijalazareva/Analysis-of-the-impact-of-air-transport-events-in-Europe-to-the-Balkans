# qq_plots_comparison.py
# Description: Generate Q-Q plots for Normal and Noncentral t fits for EU vs Balkans

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, nct, probplot

def load_fit_results():
    """Load fit parameters from the previous script."""
    path = os.path.join('results', 'Combined_Comparison', 'combined_fits_comparison.csv')
    return pd.read_csv(path)

def load_raw_delays():
    """Optionally reload combined delays (to generate Q-Q data)."""
    from combine_eu_balkan_fits import combine_all_delays
    return combine_all_delays()

def generate_individual_qq_plots(df, eu_delays, balkan_delays, output_dir):
    """Generate separate Q-Q plots for all fits."""
    for region, data in [('Europe', eu_delays), ('Balkans', balkan_delays)]:
        for dist in ['Normal', 'Noncentral t']:
            params = df[(df['Region'] == region) & (df['Distribution'] == dist)].iloc[0]

            plt.figure(figsize=(6,6))
            if dist == 'Normal':
                mu, sigma = params['loc'], params['scale']
                probplot(data, dist="norm", sparams=(mu, sigma), plot=plt)
            else:
                df_t, nc, loc, scale = params[['df','nc','loc','scale']]
                probplot(data, dist=nct, sparams=(df_t, nc, loc, scale), plot=plt)

            plt.title(f"Q-Q Plot: {dist} fit – {region}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            filename = f"qq_{region}_{dist.replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()
            print(f"✅ Saved {filename}")

def generate_combined_normal_qq(df, eu_delays, balkan_delays, output_dir):
    """Generate a single Q-Q plot comparing Europe vs Balkans under Normal fit with improved scaling."""
    plt.figure(figsize=(8,8))  # Increased figure size for better visibility

    # --- Europe fit ---
    params_eu = df[(df['Region'] == 'Europe') & (df['Distribution'] == 'Normal')].iloc[0]
    mu_eu, sigma_eu = params_eu['loc'], params_eu['scale']
    (osm_eu, osr_eu), _ = probplot(eu_delays, dist="norm", sparams=(mu_eu, sigma_eu))

    # --- Balkans fit ---
    params_bk = df[(df['Region'] == 'Balkans') & (df['Distribution'] == 'Normal')].iloc[0]
    mu_bk, sigma_bk = params_bk['loc'], params_bk['scale']
    (osm_bk, osr_bk), _ = probplot(balkan_delays, dist="norm", sparams=(mu_bk, sigma_bk))

    # Calculate plot limits based on data
    xlim = np.percentile(np.concatenate([osm_eu, osm_bk]), [1, 99])
    ylim = np.percentile(np.concatenate([osr_eu, osr_bk]), [1, 99])

    # Plot with improved visibility
    plt.scatter(osm_eu, osr_eu, s=15, color='blue', alpha=0.4, label='Europe Normal')
    plt.scatter(osm_bk, osr_bk, s=15, color='green', alpha=0.4, label='Balkans Normal')

    # Reference line using the calculated limits
    plt.plot(xlim, ylim, color='red', lw=1.5, linestyle='--', label='Reference')

    # Set axis limits to focus on the meaningful part of the data
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.title("Combined Q-Q Plot: Normal Fit\nEurope vs Balkans", pad=20)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    # Improve grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left')

    # Add minor gridlines for better readability
    plt.grid(True, which='minor', alpha=0.1)
    plt.minorticks_on()

    plt.tight_layout()

    filename = os.path.join(output_dir, "qq_combined_normal.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved combined Normal Q-Q plot → {filename}")

def generate_qq_plots():
    df = load_fit_results()
    eu_delays, balkan_delays = load_raw_delays()

    output_dir = os.path.join('results', 'Combined_Comparison', 'QQ_Plots')
    os.makedirs(output_dir, exist_ok=True)

    # Generate all individual Q-Q plots
    generate_individual_qq_plots(df, eu_delays, balkan_delays, output_dir)

    # Generate combined Q-Q plot for Normal distributions
    generate_combined_normal_qq(df, eu_delays, balkan_delays, output_dir)

    print(f"\n✅ All Q-Q plots saved in: {output_dir}")

if __name__ == "__main__":
    generate_qq_plots()
