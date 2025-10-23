# combine_eu_balkan_fits.py
# Description: Combine all EU and Balkan delays, fit Normal and Noncentral Student's t distributions,
# compare their parameters and goodness-of-fit, and save results + visualization.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, nct, kstest
import warnings
warnings.filterwarnings("ignore")

def load_airport_data(airport_code):
    """Load delay data for a specific airport."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')
    if not os.path.exists(filepath):
        print(f" File not found for {airport_code}")
        return None
    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'Timestamp', 'Delay'])
    df['Delay'] = df['Delay'].astype(float) / 60.0  # Convert to minutes
    df = df[df['Delay'] != 0]  # remove on-time flights
    return df

def fit_distribution(data, dist_type="normal"):
    """Fit Normal or Noncentral t-distribution and return parameters + KS test results."""
    if dist_type == "normal":
        mu, sigma = norm.fit(data)
        ks_stat, p_val = kstest(data, 'norm', args=(mu, sigma))
        return {'Distribution': 'Normal', 'df': np.nan, 'nc': np.nan, 'loc': mu, 'scale': sigma,
                'KS': ks_stat, 'p-value': p_val}

    elif dist_type == "nct":
        params = nct.fit(data)
        df_val, nc, loc, scale = params
        ks_stat, p_val = kstest(data, 'nct', args=params)
        return {'Distribution': 'Noncentral t', 'df': df_val, 'nc': nc, 'loc': loc, 'scale': scale,
                'KS': ks_stat, 'p-value': p_val}

def combine_all_delays():
    """Combine all delays from EU and Balkan airports."""
    europe_airports = ['EGLL','LFPG','EHAM','EDDF','LEMD','LEBL','EDDM','EGKK','LIRF','EIDW']
    balkans_airports = ['LATI','LQSA','LBSF','LBBG','LDZA','LDSP','LDDU','BKPR','LYTV','LWSK']

    eu_delays, balkan_delays = [], []

    print("🔹 Loading data for Europe...")
    for code in europe_airports:
        df = load_airport_data(code)
        if df is not None:
            eu_delays.extend(df['Delay'].values)

    print("🔹 Loading data for Balkans...")
    for code in balkans_airports:
        df = load_airport_data(code)
        if df is not None:
            balkan_delays.extend(df['Delay'].values)

    return np.array(eu_delays), np.array(balkan_delays)

def analyze_and_compare():
    """Perform both fits and create comparison table + plot."""
    eu_delays, balkan_delays = combine_all_delays()

    output_dir = os.path.join('results', 'Combined_Comparison')
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # --- Fit Normal distribution ---
    print("\n📈 Fitting Normal distributions...")
    results.append({'Region': 'Europe', **fit_distribution(eu_delays, 'normal')})
    results.append({'Region': 'Balkans', **fit_distribution(balkan_delays, 'normal')})

    # --- Fit Noncentral t-distribution ---
    print("\n📈 Fitting Noncentral t distributions...")
    results.append({'Region': 'Europe', **fit_distribution(eu_delays, 'nct')})
    results.append({'Region': 'Balkans', **fit_distribution(balkan_delays, 'nct')})

    # --- Save results to CSV ---
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'combined_fits_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n📄 Saved fit comparison table to: {csv_path}")

    # --- Plot comparison of PDFs ---
    x = np.linspace(-60, 180, 1000)
    plt.figure(figsize=(10, 6))

    # Normal fits
    mu_eu, sigma_eu = df_results.loc[(df_results['Region'] == 'Europe') & (df_results['Distribution'] == 'Normal'), ['loc', 'scale']].values[0]
    mu_bk, sigma_bk = df_results.loc[(df_results['Region'] == 'Balkans') & (df_results['Distribution'] == 'Normal'), ['loc', 'scale']].values[0]
    plt.plot(x, norm.pdf(x, mu_eu, sigma_eu), 'b-', lw=2, label='Europe - Normal')
    plt.plot(x, norm.pdf(x, mu_bk, sigma_bk), 'g--', lw=2, label='Balkans - Normal')

    # Noncentral t fits
    df_eu, nc_eu, loc_eu, scale_eu = df_results.loc[(df_results['Region'] == 'Europe') & (df_results['Distribution'] == 'Noncentral t'),
                                                    ['df', 'nc', 'loc', 'scale']].values[0]
    df_bk, nc_bk, loc_bk, scale_bk = df_results.loc[(df_results['Region'] == 'Balkans') & (df_results['Distribution'] == 'Noncentral t'),
                                                    ['df', 'nc', 'loc', 'scale']].values[0]
    plt.plot(x, nct.pdf(x, df_eu, nc_eu, loc_eu, scale_eu), 'r-', lw=2, label='Europe - Noncentral t')
    plt.plot(x, nct.pdf(x, df_bk, nc_bk, loc_bk, scale_bk), 'orange', lw=2, linestyle='--', label='Balkans - Noncentral t')

    plt.title("Comparison of Normal vs Noncentral t Fits\nEurope vs Balkans (Combined Data)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Density")
    plt.axvline(x=0, color='gray', linestyle='--', label='On-time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'combined_normal_vs_nct.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f" Saved comparison plot to: {output_path}")
    print("\n Analysis complete! Results and visualization are in:", output_dir)

if __name__ == "__main__":
    analyze_and_compare()
