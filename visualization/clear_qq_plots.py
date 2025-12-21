import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

"""
Set larger font sizes globally for all matplotlib visualizations.
This ensures consistency and readability across all figures.
"""

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18
})


def convert_timestamp(ts):
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))


def load_airport_data(airport_code):
    """
    Load data for a specific airport and convert it into a DataFrame.
    Includes timestamp conversion, delay normalization, and datetime field creation.
    """
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')

    if not os.path.exists(filepath):
        print(f"Data file for {airport_code} not found.")
        return None

    data = np.load(filepath, allow_pickle=True)

    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'ScheduledTimestamp', 'Delay'])

    df['ScheduledTimestamp'] = df['ScheduledTimestamp'].astype(float)
    df['Delay'] = df['Delay'].astype(float)

    df['ScheduledTime'] = df['ScheduledTimestamp'].apply(convert_timestamp)

    return df


def create_custom_qq_plots():
    """
    Create Q-Q plots comparing European and Balkan airport delays.
    The function loads airport data, filters outliers, computes quantiles,
    fits regression lines, calculates R² statistics, and saves both raw and
    standardized Q-Q plots as high-resolution images.
    """
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    airport_names = {
        'EGLL': 'London Heathrow',
        'LFPG': 'Paris Charles de Gaulle',
        'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt',
        'LEMD': 'Madrid Barajas',
        'LEBL': 'Barcelona',
        'EDDM': 'Munich',
        'EGKK': 'London Gatwick',
        'LIRF': 'Rome Fiumicino',
        'EIDW': 'Dublin',
        'LATI': 'Tirana',
        'LQSA': 'Sarajevo',
        'LBSF': 'Sofia',
        'LBBG': 'Burgas',
        'LDZA': 'Zagreb',
        'LDSP': 'Split',
        'LDDU': 'Dubrovnik',
        'BKPR': 'Pristina',
        'LYTV': 'Tivat',
        'LWSK': 'Skopje'
    }

    output_dir = os.path.join('results', 'qq_plots')
    os.makedirs(output_dir, exist_ok=True)

    europe_delays = []
    balkans_delays = []

    print("Loading data for airports...")

    for code in europe_airports:
        print(f"Loading {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)
        if df is not None:
            # Convert to minutes and get a sample to avoid memory issues
            sample_size = min(5000, len(df))
            delays_minutes = df['Delay'].sample(sample_size, random_state=42) / 60
            europe_delays.extend(delays_minutes.tolist())

    for code in balkans_airports:
        print(f"Loading {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)
        if df is not None:
            # Convert to minutes and get a sample to avoid memory issues
            sample_size = min(5000, len(df))
            delays_minutes = df['Delay'].sample(sample_size, random_state=42) / 60
            balkans_delays.extend(delays_minutes.tolist())

    europe_delays = np.array(europe_delays)
    balkans_delays = np.array(balkans_delays)

    """
    Filter extreme outliers using 1st and 99th percentiles to improve visualization.
    This avoids distortion from exceptionally large or small delay values.
    """

    europe_max = np.percentile(europe_delays, 99)
    europe_min = np.percentile(europe_delays, 1)
    balkans_max = np.percentile(balkans_delays, 99)
    balkans_min = np.percentile(balkans_delays, 1)

    europe_delays_filtered = europe_delays[(europe_delays <= europe_max) & (europe_delays >= europe_min)]
    balkans_delays_filtered = balkans_delays[(balkans_delays <= balkans_max) & (balkans_delays >= balkans_min)]

    print(f"Loaded {len(europe_delays_filtered)} filtered delay samples for European airports")
    print(f"Loaded {len(balkans_delays_filtered)} filtered delay samples for Balkan airports")

    europe_mean = np.mean(europe_delays_filtered)
    europe_std = np.std(europe_delays_filtered)
    balkans_mean = np.mean(balkans_delays_filtered)
    balkans_std = np.std(balkans_delays_filtered)

    fig, ax = plt.subplots(figsize=(18, 14))

    europe_quantiles = np.percentile(europe_delays_filtered, np.linspace(0.5, 99.5, 100))
    balkans_quantiles = np.percentile(balkans_delays_filtered, np.linspace(0.5, 99.5, 100))

    theoretical_quantiles = stats.norm.ppf(np.linspace(0.005, 0.995, 100))

    europe_color = '#0066cc'  # darker blue for better visibility
    ax.scatter(theoretical_quantiles, europe_quantiles, s=150, color=europe_color,
               alpha=0.7, label='European Airports', edgecolors='white', linewidths=0.5)

    slope_europe, intercept_europe, r_europe, p_europe, stderr_europe = stats.linregress(
        theoretical_quantiles, europe_quantiles)
    line_europe = slope_europe * theoretical_quantiles + intercept_europe
    ax.plot(theoretical_quantiles, line_europe, color=europe_color, linewidth=4)

    balkans_color = '#ff6600'  # darker orange for better visibility
    ax.scatter(theoretical_quantiles, balkans_quantiles, s=150, color=balkans_color,
               alpha=0.7, label='Balkan Airports', edgecolors='white', linewidths=0.5)

    slope_balkans, intercept_balkans, r_balkans, p_balkans, stderr_balkans = stats.linregress(
        theoretical_quantiles, balkans_quantiles)
    line_balkans = slope_balkans * theoretical_quantiles + intercept_balkans
    ax.plot(theoretical_quantiles, line_balkans, color=balkans_color, linewidth=4)

    ax.plot([-4, 4], [-4 * 1, 4 * 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Normal')

    r_squared_europe = r_europe ** 2
    r_squared_balkans = r_balkans ** 2

    ax.legend(title=f'R²: Europe={r_squared_europe:.4f}, Balkans={r_squared_balkans:.4f}',
              fontsize=18, title_fontsize=18, loc='upper left', framealpha=0.9)

    ax.set_title('Q-Q Plot Comparison: European vs Balkan Airports', fontsize=26, pad=20)
    ax.set_xlabel('Theoretical Quantiles', fontsize=22, labelpad=15)
    ax.set_ylabel('Sample Quantiles (Delay Minutes)', fontsize=22, labelpad=15)

    ax.grid(True, alpha=0.3)

    ax.set_aspect('equal')

    fig.text(0.15, 0.03, f'European: μ={europe_mean:.2f}, σ={europe_std:.2f} minutes',
             color=europe_color, fontsize=20, ha='left', weight='bold')
    fig.text(0.55, 0.03, f'Balkan: μ={balkans_mean:.2f}, σ={balkans_std:.2f} minutes',
             color=balkans_color, fontsize=20, ha='left', weight='bold')

    output_path = os.path.join(output_dir, 'clear_qq_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"Clear Q-Q plot saved to {output_path}")

    fig, ax = plt.subplots(figsize=(18, 14))

    europe_delays_std = (europe_delays_filtered - europe_mean) / europe_std
    balkans_delays_std = (balkans_delays_filtered - balkans_mean) / balkans_std

    europe_quantiles_std = np.percentile(europe_delays_std, np.linspace(0.5, 99.5, 100))
    balkans_quantiles_std = np.percentile(balkans_delays_std, np.linspace(0.5, 99.5, 100))

    ax.scatter(theoretical_quantiles, europe_quantiles_std, s=150, color=europe_color,
               alpha=0.7, label='European Airports (Standardized)', edgecolors='white', linewidths=0.5)

    slope_europe_std, intercept_europe_std, r_europe_std, _, _ = stats.linregress(
        theoretical_quantiles, europe_quantiles_std)
    line_europe_std = slope_europe_std * theoretical_quantiles + intercept_europe_std
    ax.plot(theoretical_quantiles, line_europe_std, color=europe_color, linewidth=4)

    ax.scatter(theoretical_quantiles, balkans_quantiles_std, s=150, color=balkans_color,
               alpha=0.7, label='Balkan Airports (Standardized)', edgecolors='white', linewidths=0.5)

    slope_balkans_std, intercept_balkans_std, r_balkans_std, _, _ = stats.linregress(
        theoretical_quantiles, balkans_quantiles_std)
    line_balkans_std = slope_balkans_std * theoretical_quantiles + intercept_balkans_std
    ax.plot(theoretical_quantiles, line_balkans_std, color=balkans_color, linewidth=4)

    ax.plot([-4, 4], [-4 * 1, 4 * 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Normal')

    r_squared_europe_std = r_europe_std ** 2
    r_squared_balkans_std = r_balkans_std ** 2

    ax.legend(title=f'R²: Europe={r_squared_europe_std:.4f}, Balkans={r_squared_balkans_std:.4f}',
              fontsize=18, title_fontsize=18, loc='upper left', framealpha=0.9)

    ax.set_title('Standardized Q-Q Plot: European vs Balkan Airports', fontsize=26, pad=20)
    ax.set_xlabel('Theoretical Quantiles', fontsize=22, labelpad=15)
    ax.set_ylabel('Standardized Sample Quantiles', fontsize=22, labelpad=15)

    ax.grid(True, alpha=0.3)

    ax.set_aspect('equal')

    fig.text(0.15, 0.03, 'Standardized data shows shape differences regardless of scale',
             fontsize=20, ha='left', weight='bold')

    output_path_std = os.path.join(output_dir, 'clear_standardized_qq_plot.png')
    plt.tight_layout()
    plt.savefig(output_path_std, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"Clear standardized Q-Q plot saved to {output_path_std}")


if __name__ == '__main__':
    print("Creating clear, readable Q-Q plots...")
    create_custom_qq_plots()
    print("Analysis complete!")
