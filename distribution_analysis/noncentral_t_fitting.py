# noncentral_t_analysis.py
# Description: Fits Noncentral Student's t-distribution to flight delays for each airport

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nct, kstest
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def convert_timestamp(ts):
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))

def load_airport_data(airport_code):
    """Load and prepare delay data for a specific airport."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')

    if not os.path.exists(filepath):
        print(f" Data file for {airport_code} not found, skipping.")
        return None

    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'Timestamp', 'Delay'])
    df['Delay'] = df['Delay'].astype(float)
    df['DelayMinutes'] = df['Delay'] / 60.0  # convert seconds to minutes
    df = df[df['DelayMinutes'] != 0]  # remove exact on-time flights
    return df

def fit_noncentral_t(delays):
    """Fit a noncentral t-distribution and compute KS goodness-of-fit."""
    delays = np.array(delays)
    if len(delays) < 100:
        return None  # skip small datasets

    # Fit the distribution
    params = nct.fit(delays)
    df, nc, loc, scale = params

    # Goodness-of-fit test
    ks_stat, p_val = kstest(delays, 'nct', args=params)
    return df, nc, loc, scale, ks_stat, p_val

# ---------------------------
# Main analysis
# ---------------------------

def analyze_noncentral_t():
    """Run noncentral Student's t fitting for all airports."""
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris CDG', 'EHAM': 'Amsterdam Schiphol', 'EDDF': 'Frankfurt',
        'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich', 'EGKK': 'London Gatwick',
        'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas', 'LDZA': 'Zagreb',
        'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina', 'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    output_dir = os.path.join('data', 'NonCentralT')
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(" Starting Noncentral Student's t-distribution fitting...\n")

    for code in europe_airports + balkans_airports:
        df = load_airport_data(code)
        if df is None:
            continue

        result = fit_noncentral_t(df['DelayMinutes'])
        if result is None:
            print(f" Skipping {code} (not enough data)")
            continue

        df_val, nc, loc, scale, ks, p = result
        results.append({
            'Airport': code,
            'Airport Name': airport_names.get(code, code),
            'Region': 'Europe' if code in europe_airports else 'Balkans',
            'df': round(df_val, 3),
            'nc': round(nc, 3),
            'loc (mean)': round(loc, 3),
            'scale (std)': round(scale, 3),
            'KS Statistic': round(ks, 4),
            'p-value': round(p, 4)
        })

        print(f" {airport_names.get(code, code)} ({code}) fitted successfully.")
        print(f"    df={df_val:.3f}, nc={nc:.3f}, loc={loc:.3f}, scale={scale:.3f}, KS={ks:.4f}, p={p:.4f}")

    # ---------------------------
    # Save and visualize results
    # ---------------------------
    if not results:
        print(" No valid results found.")
        return

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'noncentral_t_parameters.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n Saved parameters table to: {csv_path}")

    # Plot: Mean (loc) and Std (scale)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=results_df, x='Airport', y='loc (mean)', hue='Region', palette='Blues', alpha=0.8)
    sns.barplot(data=results_df, x='Airport', y='scale (std)', hue='Region', palette='Oranges', alpha=0.5)
    plt.xticks(rotation=90)
    plt.title("Noncentral t-distribution Parameters (Mean and Std per Airport)")
    plt.ylabel("Minutes")
    plt.xlabel("Airport Code")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noncentral_t_parameters.png'))
    plt.close()
    print(f" Saved bar chart to: {os.path.join(output_dir, 'noncentral_t_parameters.png')}")

    # ---------------------------
    # Compute simple summary by region
    # ---------------------------
    summary = results_df.groupby('Region')[['loc (mean)', 'scale (std)']].mean()
    print("\n Average parameters by region:")
    print(summary)

    return results_df

if __name__ == "__main__":
    analyze_noncentral_t()
