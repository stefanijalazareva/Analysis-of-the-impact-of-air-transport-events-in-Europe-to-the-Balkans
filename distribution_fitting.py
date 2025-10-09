import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import norm, lognorm, gamma, weibull_min, expon, pearson3, genextreme
import warnings
warnings.filterwarnings('ignore')  # Suppress fit warnings

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

    # Separate positive and negative delays
    df['PositiveDelay'] = df['Delay'].clip(lower=0)  # Only positive delays
    df['NegativeDelay'] = (-df['Delay']).clip(lower=0)  # Convert negative delays to positive values

    return df

def fit_distributions(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Fit various distributions to delay data and determine best fit."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter out zeros for better fitting
    delays_nonzero = delays[delays > 0]

    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    # Convert to minutes for better interpretability
    delays_minutes = delays_nonzero / 60

    # Define distributions to test
    distributions = [
        ('Normal', norm),
        ('Lognormal', lognorm),
        ('Gamma', gamma),
        ('Weibull', weibull_min),
        ('Exponential', expon),
        ('Pearson Type III', pearson3),
        ('Generalized Extreme Value', genextreme)
    ]

    # Fit distributions and calculate goodness-of-fit
    results = []

    for dist_name, distribution in distributions:
        try:
            # Fit distribution
            params = distribution.fit(delays_minutes)

            # Calculate Kolmogorov-Smirnov test statistic and p-value
            ks_statistic, p_value = stats.kstest(delays_minutes, distribution.name, params)

            # Calculate AIC
            log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))
            k = len(params)
            n = len(delays_minutes)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            results.append({
                'Distribution': dist_name,
                'Parameters': params,
                'KS Statistic': ks_statistic,
                'P-value': p_value,
                'AIC': aic,
                'BIC': bic
            })

        except Exception as e:
            print(f"Error fitting {dist_name} distribution for {airport_code}: {e}")

    if not results:
        print(f"Could not fit any distributions for {airport_code}")
        return None

    # Convert results to DataFrame and sort by AIC (lower is better)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC')

    # Save results
    results_csv = os.path.join(output_dir, f"distribution_fit_{delay_type}_{airport_code}.csv")
    results_df.to_csv(results_csv, index=False)

    # Get the best distribution
    best_dist_name = results_df.iloc[0]['Distribution']
    best_dist_params = results_df.iloc[0]['Parameters']
    best_distribution = [d for d_name, d in distributions if d_name == best_dist_name][0]

    # Plot data and best-fitting distribution
    plt.figure(figsize=(12, 8))

    # Plot histogram
    sns.histplot(delays_minutes, bins=50, stat='density', alpha=0.6, label='Data')

    # Plot PDF of best-fitting distribution
    x = np.linspace(min(delays_minutes), min(np.percentile(delays_minutes, 99.5), max(delays_minutes)), 1000)
    pdf = best_distribution.pdf(x, *best_dist_params)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Best fit: {best_dist_name}')

    # Plot top 3 distributions if available
    colors = ['g', 'b', 'm']
    for i in range(1, min(3, len(results_df))):
        dist_name = results_df.iloc[i]['Distribution']
        dist_params = results_df.iloc[i]['Parameters']
        distribution = [d for d_name, d in distributions if d_name == dist_name][0]
        pdf = distribution.pdf(x, *dist_params)
        plt.plot(x, pdf, color=colors[i-1], linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'{dist_name} (AIC: {results_df.iloc[i]["AIC"]:.2f})')

    # Add plot details
    plt.title(f'Distribution Fitting for {delay_type.capitalize()} Delays at {airport_name} ({airport_code})\n'
              f'Best fit: {best_dist_name} (AIC: {results_df.iloc[0]["AIC"]:.2f})')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"distribution_fit_{delay_type}_{airport_code}.png"))
    plt.close()

    return results_df

def analyze_distributions(airport_codes=None):
    """Analyze delay distributions for specified airports."""
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

    # Use all airports if none specified
    if airport_codes is None:
        airport_codes = europe_airports + balkans_airports

    output_dir = os.path.join('data', 'DistributionFitting')
    os.makedirs(output_dir, exist_ok=True)

    # Track best distributions for each airport
    best_distributions_positive = {}
    best_distributions_negative = {}

    for code in airport_codes:
        print(f"\nProcessing {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Fit distributions to positive delays
            print(f"Fitting distributions to positive delays...")
            pos_results = fit_distributions(
                df['PositiveDelay'],
                code,
                airport_names.get(code, code),
                output_dir,
                'positive'
            )

            if pos_results is not None:
                best_distributions_positive[code] = pos_results.iloc[0]['Distribution']

            # Fit distributions to negative delays (early arrivals)
            print(f"Fitting distributions to negative delays (early arrivals)...")
            neg_results = fit_distributions(
                df['NegativeDelay'],
                code,
                airport_names.get(code, code),
                output_dir,
                'negative'
            )

            if neg_results is not None:
                best_distributions_negative[code] = neg_results.iloc[0]['Distribution']

    # Summarize findings
    summary = []
    for code in airport_codes:
        summary.append({
            'Airport': code,
            'Airport Name': airport_names.get(code, ''),
            'Region': 'Europe' if code in europe_airports else 'Balkans',
            'Best Positive Delay Distribution': best_distributions_positive.get(code, 'N/A'),
            'Best Negative Delay Distribution': best_distributions_negative.get(code, 'N/A')
        })

    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'distribution_summary.csv'), index=False)

    # Create a summary visualization
    plt.figure(figsize=(15, 10))

    # Count distribution types for positive delays
    pos_dist_counts = pd.Series(best_distributions_positive.values()).value_counts()

    ax1 = plt.subplot(2, 2, 1)
    pos_dist_counts.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Best Distributions for Positive Delays (Across All Airports)')
    ax1.set_ylabel('Number of Airports')
    ax1.tick_params(axis='x', rotation=45)

    # Count distribution types for negative delays
    neg_dist_counts = pd.Series(best_distributions_negative.values()).value_counts()

    ax2 = plt.subplot(2, 2, 2)
    neg_dist_counts.plot(kind='bar', ax=ax2, color='salmon')
    ax2.set_title('Best Distributions for Negative Delays (Across All Airports)')
    ax2.set_ylabel('Number of Airports')
    ax2.tick_params(axis='x', rotation=45)

    # Compare Europe vs Balkans for positive delays
    europe_pos_dists = {k: v for k, v in best_distributions_positive.items() if k in europe_airports}
    balkans_pos_dists = {k: v for k, v in best_distributions_positive.items() if k in balkans_airports}

    europe_pos_counts = pd.Series(europe_pos_dists.values()).value_counts()
    balkans_pos_counts = pd.Series(balkans_pos_dists.values()).value_counts()

    # Combine into a DataFrame
    region_pos_df = pd.DataFrame({
        'Europe': europe_pos_counts,
        'Balkans': balkans_pos_counts
    }).fillna(0)

    ax3 = plt.subplot(2, 2, 3)
    region_pos_df.plot(kind='bar', ax=ax3)
    ax3.set_title('Best Positive Delay Distributions by Region')
    ax3.set_ylabel('Number of Airports')
    ax3.tick_params(axis='x', rotation=45)

    # Create a detailed table for reference
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    headers = ['Airport', 'Region', 'Positive Delays', 'Negative Delays']

    for code in airport_codes:
        region = 'Europe' if code in europe_airports else 'Balkans'
        pos_dist = best_distributions_positive.get(code, 'N/A')
        neg_dist = best_distributions_negative.get(code, 'N/A')
        table_data.append([code, region, pos_dist, neg_dist])

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.15, 0.35, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.suptitle('Summary of Delay Distribution Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'distribution_summary.png'))
    plt.close()

    print(f"\nDistribution fitting analysis complete. Results saved to {output_dir}")
    return summary_df

if __name__ == "__main__":
    print("Starting distribution analysis...")
    summary = analyze_distributions()
    print("Analysis complete.")
