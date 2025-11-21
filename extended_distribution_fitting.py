import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import (norm, lognorm, gamma, weibull_min, expon, pearson3, genextreme,
                        loglogistic, burr, gengamma, dweibull)
import warnings
warnings.filterwarnings('ignore')

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
    df['PositiveDelay'] = df['Delay'].clip(lower=0)
    df['NegativeDelay'] = (-df['Delay']).clip(lower=0)

    return df

def fit_extended_distributions(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Fit extended set of distributions to delay data including new distributions."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter out zeros for better fitting
    delays_nonzero = delays[delays > 0]

    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    # Convert to minutes for better interpretability
    delays_minutes = delays_nonzero / 60

    # Define extended distributions to test including new ones
    distributions = [
        ('Normal', norm),
        ('Lognormal', lognorm),
        ('Gamma', gamma),
        ('Weibull', weibull_min),
        ('Exponential', expon),
        ('Pearson Type III', pearson3),
        ('Generalized Extreme Value', genextreme),
        ('Log-Logistic', loglogistic),
        ('Burr XII', burr),
        ('Generalized Gamma', gengamma),
        ('Double Weibull', dweibull)
    ]

    # Fit distributions and calculate goodness-of-fit
    results = []
    failed_fits = []

    for dist_name, distribution in distributions:
        try:
            print(f"  Fitting {dist_name}...")

            # Fit distribution
            params = distribution.fit(delays_minutes)

            # Calculate Kolmogorov-Smirnov test statistic and p-value
            ks_statistic, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)

            # Calculate log-likelihood
            log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))

            # Check for invalid log-likelihood
            if not np.isfinite(log_likelihood):
                raise ValueError(f"Invalid log-likelihood for {dist_name}")

            # Calculate AIC and BIC
            k = len(params)
            n = len(delays_minutes)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            results.append({
                'Distribution': dist_name,
                'Parameters': params,
                'KS Statistic': ks_statistic,
                'P-value': p_value,
                'Log-Likelihood': log_likelihood,
                'AIC': aic,
                'BIC': bic,
                'Num_Parameters': k
            })

        except Exception as e:
            failed_fits.append(f"{dist_name}: {str(e)}")
            print(f"    Error fitting {dist_name}: {e}")

    if failed_fits:
        print(f"Failed to fit distributions: {', '.join(failed_fits)}")

    if not results:
        print(f"Could not fit any distributions for {airport_code}")
        return None

    # Convert results to DataFrame and sort by AIC (lower is better)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC')

    # Save results
    results_csv = os.path.join(output_dir, f"extended_distribution_fit_{delay_type}_{airport_code}.csv")
    results_df.to_csv(results_csv, index=False)

    # Get the best distribution
    best_dist_name = results_df.iloc[0]['Distribution']
    best_dist_params = results_df.iloc[0]['Parameters']
    best_distribution = [d for d_name, d in distributions if d_name == best_dist_name][0]

    # Create comprehensive visualization
    create_extended_distribution_plots(delays_minutes, results_df, distributions,
                                     airport_code, airport_name, delay_type, output_dir)

    return results_df

def create_extended_distribution_plots(delays_minutes, results_df, distributions,
                                     airport_code, airport_name, delay_type, output_dir):
    """Create comprehensive plots for distribution fitting results."""

    # Main distribution plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Data histogram with best fits
    ax1.hist(delays_minutes, bins=50, density=True, alpha=0.6, color='lightblue', label='Data')

    x = np.linspace(min(delays_minutes), np.percentile(delays_minutes, 99.5), 1000)
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Plot top 5 distributions
    for i in range(min(5, len(results_df))):
        dist_name = results_df.iloc[i]['Distribution']
        dist_params = results_df.iloc[i]['Parameters']
        distribution = [d for d_name, d in distributions if d_name == dist_name][0]

        try:
            pdf = distribution.pdf(x, *dist_params)
            ax1.plot(x, pdf, color=colors[i], linewidth=2 if i == 0 else 1.5,
                    label=f'{dist_name} (AIC: {results_df.iloc[i]["AIC"]:.1f})')
        except:
            continue

    ax1.set_title(f'{delay_type.capitalize()} Delays - {airport_name} ({airport_code})')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: AIC comparison
    top_dists = results_df.head(8)
    ax2.barh(range(len(top_dists)), top_dists['AIC'], color='skyblue')
    ax2.set_yticks(range(len(top_dists)))
    ax2.set_yticklabels(top_dists['Distribution'])
    ax2.set_xlabel('AIC (lower is better)')
    ax2.set_title('AIC Comparison')
    ax2.grid(alpha=0.3)

    # Plot 3: Q-Q plot for best distribution
    best_dist_name = results_df.iloc[0]['Distribution']
    best_dist_params = results_df.iloc[0]['Parameters']
    best_distribution = [d for d_name, d in distributions if d_name == best_dist_name][0]

    theoretical_quantiles = best_distribution.ppf(np.linspace(0.01, 0.99, len(delays_minutes)), *best_dist_params)
    sample_quantiles = np.sort(delays_minutes)

    ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=1)
    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    ax3.set_title(f'Q-Q Plot: {best_dist_name}')
    ax3.grid(alpha=0.3)

    # Plot 4: P-P plot for best distribution
    sample_cdf = np.arange(1, len(delays_minutes) + 1) / len(delays_minutes)
    theoretical_cdf = best_distribution.cdf(sample_quantiles, *best_dist_params)

    ax4.scatter(theoretical_cdf, sample_cdf, alpha=0.6, s=1)
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax4.set_xlabel('Theoretical CDF')
    ax4.set_ylabel('Sample CDF')
    ax4.set_title(f'P-P Plot: {best_dist_name}')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"extended_distribution_analysis_{delay_type}_{airport_code}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_extended_distributions(airport_codes=None):
    """Analyze delay distributions for specified airports with extended distribution set."""
    # Define airport groups
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    # Airport names mapping
    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    # Use all airports if none specified
    if airport_codes is None:
        airport_codes = europe_airports + balkans_airports

    output_dir = os.path.join('results', 'extended_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Track best distributions for each airport
    all_results = []

    for code in airport_codes:
        print(f"\nProcessing {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Fit distributions to positive delays
            print(f"Fitting distributions to positive delays...")
            pos_results = fit_extended_distributions(
                df['PositiveDelay'],
                code,
                airport_names.get(code, code),
                output_dir,
                delay_type='positive'
            )

            if pos_results is not None:
                pos_results['Airport'] = code
                pos_results['Airport_Name'] = airport_names.get(code, code)
                pos_results['Delay_Type'] = 'positive'
                all_results.append(pos_results)

            # Fit distributions to negative delays (converted to positive)
            print(f"Fitting distributions to negative delays...")
            neg_results = fit_extended_distributions(
                df['NegativeDelay'],
                code,
                airport_names.get(code, code),
                output_dir,
                delay_type='negative'
            )

            if neg_results is not None:
                neg_results['Airport'] = code
                neg_results['Airport_Name'] = airport_names.get(code, code)
                neg_results['Delay_Type'] = 'negative'
                all_results.append(neg_results)

    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(os.path.join(output_dir, 'extended_distribution_summary.csv'), index=False)

        # Create summary analysis
        create_summary_analysis(combined_results, output_dir)

        print(f"\nExtended distribution analysis complete!")
        print(f"Results saved to: {output_dir}")
        return combined_results
    else:
        print("No results to analyze.")
        return None

def create_summary_analysis(combined_results, output_dir):
    """Create summary analysis of distribution fitting results."""

    # Best distribution by airport
    best_by_airport = combined_results.groupby(['Airport', 'Delay_Type']).first().reset_index()

    # Distribution popularity
    dist_popularity = combined_results.groupby(['Distribution', 'Delay_Type']).size().reset_index(name='Count')

    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Distribution popularity for positive delays
    pos_pop = dist_popularity[dist_popularity['Delay_Type'] == 'positive'].sort_values('Count', ascending=True)
    ax1.barh(pos_pop['Distribution'], pos_pop['Count'], color='lightcoral')
    ax1.set_title('Distribution Popularity - Positive Delays')
    ax1.set_xlabel('Number of Airports (Best Fit)')

    # Plot 2: Distribution popularity for negative delays
    neg_pop = dist_popularity[dist_popularity['Delay_Type'] == 'negative'].sort_values('Count', ascending=True)
    ax2.barh(neg_pop['Distribution'], neg_pop['Count'], color='lightblue')
    ax2.set_title('Distribution Popularity - Negative Delays')
    ax2.set_xlabel('Number of Airports (Best Fit)')

    # Plot 3: AIC distribution by distribution type (positive delays)
    pos_results = combined_results[combined_results['Delay_Type'] == 'positive']
    ax3.boxplot([pos_results[pos_results['Distribution'] == dist]['AIC'].values
                for dist in pos_results['Distribution'].unique()],
               labels=pos_results['Distribution'].unique())
    ax3.set_title('AIC Distribution by Distribution Type - Positive Delays')
    ax3.set_ylabel('AIC')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Regional comparison
    europe_codes = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_codes = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    best_by_airport['Region'] = best_by_airport['Airport'].apply(
        lambda x: 'Europe' if x in europe_codes else 'Balkans'
    )

    regional_summary = best_by_airport[best_by_airport['Delay_Type'] == 'positive'].groupby(['Region', 'Distribution']).size().unstack(fill_value=0)
    regional_summary.plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Best Distribution by Region - Positive Delays')
    ax4.set_ylabel('Number of Airports')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extended_distribution_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary tables
    best_by_airport.to_csv(os.path.join(output_dir, 'best_distributions_by_airport.csv'), index=False)
    dist_popularity.to_csv(os.path.join(output_dir, 'distribution_popularity.csv'), index=False)

if __name__ == "__main__":
    # Run extended distribution analysis
    results = analyze_extended_distributions()

    if results is not None:
        print(f"\nAnalysis completed successfully!")
        print(f"Best distributions summary:")
        print(results.groupby(['Distribution', 'Delay_Type']).size().sort_values(ascending=False))
