# noncentral_t_visual_fit.py
# Description: Fits Noncentral Student’s t-distribution for each airport and saves visual fit plots only.

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
    """Load delay data for a specific airport."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')
    if not os.path.exists(filepath):
        print(f" File not found for {airport_code}")
        return None

    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'Timestamp', 'Delay'])
    df['Delay'] = df['Delay'].astype(float) / 60.0  # Convert to minutes
    df = df[df['Delay'] != 0]  # remove exact on-time flights
    return df

def fit_and_plot_nct(df, airport_code, airport_name, output_dir):
    """Fit Noncentral t-distribution and create a visual plot for one airport."""
    delays = df['Delay'].values

    if len(delays) < 100:
        print(f" Not enough data for {airport_code}")
        return

    # Fit noncentral t distribution
    params = nct.fit(delays)
    df_val, nc, loc, scale = params

    # KS test (goodness of fit)
    ks_stat, p_val = kstest(delays, 'nct', args=params)

    # Generate fitted PDF
    x = np.linspace(min(delays), max(delays), 500)
    pdf = nct.pdf(x, *params)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(delays, bins=40, stat='density', color='skyblue', label='Data', alpha=0.6)
    plt.plot(x, pdf, 'r-', lw=2, label=f'NCT Fit (df={df_val:.2f}, nc={nc:.2f}, μ={loc:.2f}, σ={scale:.2f})')

    # Mark on-time line
    plt.axvline(x=0, color='g', linestyle='--', label='On-time')

    # Labels and aesthetics
    plt.title(f"Noncentral t-Distribution Fit for Delays at {airport_name} ({airport_code})")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    # Add KS test result text
    plt.text(0.05, 0.02, f"K-S test: {ks_stat:.4f} (p={p_val:.4f})", transform=plt.gca().transAxes)

    # Save plot
    filename = f"nct_fit_{airport_code}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

    print(f" Saved plot for {airport_name} ({airport_code})")

def analyze_all_airports():
    """Run analysis for all airports and save individual plots."""
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris CDG', 'EHAM': 'Amsterdam Schiphol', 'EDDF': 'Frankfurt',
        'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich', 'EGKK': 'London Gatwick',
        'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik',
        'BKPR': 'Pristina', 'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    output_dir = os.path.join('results', 'NonCentralT_Fits')
    os.makedirs(output_dir, exist_ok=True)

    print(" Starting Noncentral Student’s t visual fitting...\n")

    for code in europe_airports + balkans_airports:
        df = load_airport_data(code)
        if df is not None:
            fit_and_plot_nct(df, code, airport_names.get(code, code), output_dir)

    print("\n All visual fits saved in:", output_dir)

if __name__ == "__main__":
    analyze_all_airports()
