import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')  # Suppress fit warnings

# Set larger font sizes globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
    'legend.title_fontsize': 16
})

def convert_timestamp(ts):
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))

def load_airport_data(airport_code):
    """Load data for a specific airport and convert to DataFrame."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')

    if not os.path.exists(filepath):
        print(f"Data file for {airport_code} not found.")
        return None

    # Load raw data
    data = np.load(filepath, allow_pickle=True)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'ScheduledTimestamp', 'Delay'])

    # Convert timestamp and delay to appropriate types
    df['ScheduledTimestamp'] = df['ScheduledTimestamp'].astype(float)
    df['Delay'] = df['Delay'].astype(float)

    # Add datetime columns
    df['ScheduledTime'] = df['ScheduledTimestamp'].apply(convert_timestamp)

    return df

def create_comparative_qq_plot():
    """Create a single Q-Q plot comparing European and Balkan airports with larger elements."""
    # Define airport groups
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    # Airport names mapping
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

    # Collect all delays by region
    europe_delays = []
    balkans_delays = []

    print("Loading data for airports...")

    # Load European airports data
    for code in europe_airports:
        print(f"Loading {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)
        if df is not None:
            # Convert to minutes and get a sample to avoid memory issues
            sample_size = min(10000, len(df))
            delays_minutes = df['Delay'].sample(sample_size, random_state=42) / 60
            europe_delays.extend(delays_minutes.tolist())

    # Load Balkan airports data
    for code in balkans_airports:
        print(f"Loading {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)
        if df is not None:
            # Convert to minutes and get a sample to avoid memory issues
            sample_size = min(10000, len(df))
            delays_minutes = df['Delay'].sample(sample_size, random_state=42) / 60
            balkans_delays.extend(delays_minutes.tolist())

    # Convert to numpy arrays
    europe_delays = np.array(europe_delays)
    balkans_delays = np.array(balkans_delays)

    print(f"Loaded {len(europe_delays)} delay samples for European airports")
    print(f"Loaded {len(balkans_delays)} delay samples for Balkan airports")

    # Create a combined Q-Q plot - LARGER VERSION
    plt.figure(figsize=(15, 12))

    # Calculate Q-Q plot data for Europe
    (osm_europe, osr_europe), (slope_europe, intercept_europe, r_europe) = stats.probplot(europe_delays, dist="norm", fit=True, plot=None)

    # Calculate Q-Q plot data for Balkans
    (osm_balkans, osr_balkans), (slope_balkans, intercept_balkans, r_balkans) = stats.probplot(balkans_delays, dist="norm", fit=True, plot=None)

    # Plot Europe data - larger points and thicker lines
    europe_color = '#1f77b4'  # blue
    plt.scatter(osm_europe, osr_europe, color=europe_color, alpha=0.7, s=30, label='European Airports')

    # Plot Europe fit line
    line_europe = slope_europe * osm_europe + intercept_europe
    plt.plot(osm_europe, line_europe, linestyle='-', linewidth=3, color=europe_color)

    # Plot Balkans data
    balkans_color = '#ff7f0e'  # orange
    plt.scatter(osm_balkans, osr_balkans, color=balkans_color, alpha=0.7, s=30, label='Balkan Airports')

    # Plot Balkans fit line
    line_balkans = slope_balkans * osm_balkans + intercept_balkans
    plt.plot(osm_balkans, line_balkans, linestyle='-', linewidth=3, color=balkans_color)

    # Add reference line for perfect normal - thicker line
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    lims = [min(x_min, y_min), max(x_max, y_max)]
    plt.plot(lims, lims, 'k-', alpha=0.6, zorder=0, linewidth=2, label='Perfect Normal')

    # Add R² values to the legend - larger font
    plt.legend(title=f'R²: Europe={r_europe**2:.4f}, Balkans={r_balkans**2:.4f}',
               fontsize=16, title_fontsize=16, loc='best')

    # Add details - larger fonts
    plt.title('Q-Q Plot Comparison: European vs Balkan Airports', fontsize=22, pad=20)
    plt.xlabel('Theoretical Quantiles', fontsize=18, labelpad=15)
    plt.ylabel('Sample Quantiles (Delay Minutes)', fontsize=18, labelpad=15)
    plt.grid(alpha=0.3)

    # Add mean and std as text - larger, bolder text
    europe_mean = np.mean(europe_delays)
    europe_std = np.std(europe_delays)
    balkans_mean = np.mean(balkans_delays)
    balkans_std = np.std(balkans_delays)

    plt.figtext(0.15, 0.05, f'European: μ={europe_mean:.2f}, σ={europe_std:.2f} minutes',
                color=europe_color, fontsize=16, ha='left', weight='bold')
    plt.figtext(0.55, 0.05, f'Balkan: μ={balkans_mean:.2f}, σ={balkans_std:.2f} minutes',
                color=balkans_color, fontsize=16, ha='left', weight='bold')

    # Ensure tight layout with more padding
    plt.tight_layout(pad=3.0)

    # Save the figure at higher resolution
    output_path = os.path.join(output_dir, 'europe_vs_balkans_qq_plot_large.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"Comparative Q-Q plot saved to {output_path}")

    # Create a second version with standardized data for shape comparison - LARGER VERSION
    plt.figure(figsize=(15, 12))

    # Standardize the data
    europe_delays_std = (europe_delays - europe_mean) / europe_std
    balkans_delays_std = (balkans_delays - balkans_mean) / balkans_std

    # Calculate Q-Q plot data for standardized Europe
    (osm_europe_std, osr_europe_std), (slope_europe_std, intercept_europe_std, r_europe_std) = stats.probplot(europe_delays_std, dist="norm", fit=True, plot=None)

    # Calculate Q-Q plot data for standardized Balkans
    (osm_balkans_std, osr_balkans_std), (slope_balkans_std, intercept_balkans_std, r_balkans_std) = stats.probplot(balkans_delays_std, dist="norm", fit=True, plot=None)

    # Plot standardized Europe data - larger points
    plt.scatter(osm_europe_std, osr_europe_std, color=europe_color, alpha=0.7, s=30, label='European Airports (Standardized)')

    # Plot standardized Europe fit line - thicker line
    line_europe_std = slope_europe_std * osm_europe_std + intercept_europe_std
    plt.plot(osm_europe_std, line_europe_std, linestyle='-', linewidth=3, color=europe_color)

    # Plot standardized Balkans data - larger points
    plt.scatter(osm_balkans_std, osr_balkans_std, color=balkans_color, alpha=0.7, s=30, label='Balkan Airports (Standardized)')

    # Plot standardized Balkans fit line - thicker line
    line_balkans_std = slope_balkans_std * osm_balkans_std + intercept_balkans_std
    plt.plot(osm_balkans_std, line_balkans_std, linestyle='-', linewidth=3, color=balkans_color)

    # Add reference line for perfect normal - thicker line
    plt.plot(lims, lims, 'k-', alpha=0.6, zorder=0, linewidth=2, label='Perfect Normal')

    # Add R² values to the legend - larger font
    plt.legend(title=f'R²: Europe={r_europe_std**2:.4f}, Balkans={r_balkans_std**2:.4f}',
               fontsize=16, title_fontsize=16, loc='best')

    # Add details - larger fonts
    plt.title('Standardized Q-Q Plot: European vs Balkan Airports', fontsize=22, pad=20)
    plt.xlabel('Theoretical Quantiles', fontsize=18, labelpad=15)
    plt.ylabel('Standardized Sample Quantiles', fontsize=18, labelpad=15)
    plt.grid(alpha=0.3)

    # Add explanation text - larger font
    plt.figtext(0.15, 0.05, 'Standardized data shows shape differences regardless of scale',
                fontsize=16, ha='left', weight='bold')

    # Ensure tight layout with more padding
    plt.tight_layout(pad=3.0)

    # Save the standardized figure at higher resolution
    output_path_std = os.path.join(output_dir, 'europe_vs_balkans_standardized_qq_plot_large.png')
    plt.savefig(output_path_std, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"Standardized comparative Q-Q plot saved to {output_path_std}")

if __name__ == '__main__':
    print("Starting comparative Q-Q plot analysis with larger, more visible elements...")
    create_comparative_qq_plot()
    print("Analysis complete!")
