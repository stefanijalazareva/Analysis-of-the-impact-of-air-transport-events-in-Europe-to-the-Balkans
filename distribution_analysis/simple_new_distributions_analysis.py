import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisk, burr, gengamma, weibull_min, gamma, lognorm
import warnings
warnings.filterwarnings('ignore')

def load_airport_data(airport_code):
    """Load data for a specific airport and convert to DataFrame."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')

    if not os.path.exists(filepath):
        print(f"Data file for {airport_code} not found.")
        return None

    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'ScheduledTimestamp', 'Delay'])

    df['ScheduledTimestamp'] = df['ScheduledTimestamp'].astype(float)
    df['Delay'] = df['Delay'].astype(float)
    df['PositiveDelay'] = df['Delay'].clip(lower=0)
    df['NegativeDelay'] = (-df['Delay']).clip(lower=0)

    return df

def fit_new_distributions(delays, airport_code, airport_name, delay_type='positive'):
    """Fit all new distributions and return results."""
    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    # Define distributions to test
    distributions = [
        ('Log-Logistic', fisk),
        ('Burr XII', burr),
        ('Generalized Gamma', gengamma),
        ('Weibull', weibull_min),
        ('Gamma', gamma),
        ('Log-Normal', lognorm)
    ]

    results = []

    for dist_name, distribution in distributions:
        try:
            print(f"  Fitting {dist_name}...")

            # Fit distribution
            params = distribution.fit(delays_minutes)

            # Calculate goodness of fit metrics
            ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)
            log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))

            n = len(delays_minutes)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Calculate percentiles
            p90 = distribution.ppf(0.90, *params)
            p95 = distribution.ppf(0.95, *params)
            p99 = distribution.ppf(0.99, *params)

            results.append({
                'Airport': airport_code,
                'Airport_Name': airport_name,
                'Delay_Type': delay_type,
                'Distribution': dist_name,
                'KS_Statistic': ks_stat,
                'P_value': p_value,
                'AIC': aic,
                'BIC': bic,
                'P90': p90,
                'P95': p95,
                'P99': p99,
                'Sample_Size': n,
                'Data_P90': np.percentile(delays_minutes, 90),
                'Data_P95': np.percentile(delays_minutes, 95),
                'Data_P99': np.percentile(delays_minutes, 99)
            })

        except Exception as e:
            print(f"    Error fitting {dist_name}: {e}")

    return results

def analyze_single_airport(airport_code, create_plots=True):
    """Analyze distributions for a single airport."""
    # Airport names mapping
    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    print(f"\nAnalyzing {airport_names.get(airport_code, airport_code)} ({airport_code})...")
    df = load_airport_data(airport_code)

    if df is None:
        return None

    # Analyze positive delays
    pos_results = fit_new_distributions(
        df['PositiveDelay'], airport_code, airport_names.get(airport_code, airport_code), 'positive'
    )

    # Analyze negative delays
    neg_results = fit_new_distributions(
        df['NegativeDelay'], airport_code, airport_names.get(airport_code, airport_code), 'negative'
    )

    all_results = []
    if pos_results:
        all_results.extend(pos_results)
    if neg_results:
        all_results.extend(neg_results)

    if create_plots and all_results:
        create_airport_summary_plot(df, all_results, airport_code, airport_names.get(airport_code, airport_code))

    return all_results

def create_airport_summary_plot(df, results, airport_code, airport_name):
    """Create summary plot for a single airport."""
    output_dir = os.path.join('results', 'new_distributions_analysis')
    os.makedirs(output_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Separate positive and negative results
    pos_results = [r for r in results if r['Delay_Type'] == 'positive']
    neg_results = [r for r in results if r['Delay_Type'] == 'negative']

    # Plot 1: Positive delays AIC comparison
    if pos_results:
        pos_df = pd.DataFrame(pos_results).sort_values('AIC')
        ax1.bar(range(len(pos_df)), pos_df['AIC'], color='lightblue', alpha=0.7)
        ax1.set_xticks(range(len(pos_df)))
        ax1.set_xticklabels(pos_df['Distribution'], rotation=45)
        ax1.set_title(f'Positive Delays - AIC Comparison\n{airport_name}')
        ax1.set_ylabel('AIC (lower is better)')
        ax1.grid(alpha=0.3)

        # Annotate best
        best_idx = pos_df['AIC'].idxmin()
        best_dist = pos_df.loc[best_idx, 'Distribution']
        ax1.annotate(f'Best: {best_dist}',
                    xy=(0, pos_df.iloc[0]['AIC']),
                    xytext=(0.5, 0.9), textcoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Negative delays AIC comparison
    if neg_results:
        neg_df = pd.DataFrame(neg_results).sort_values('AIC')
        ax2.bar(range(len(neg_df)), neg_df['AIC'], color='lightcoral', alpha=0.7)
        ax2.set_xticks(range(len(neg_df)))
        ax2.set_xticklabels(neg_df['Distribution'], rotation=45)
        ax2.set_title(f'Negative Delays - AIC Comparison\n{airport_name}')
        ax2.set_ylabel('AIC (lower is better)')
        ax2.grid(alpha=0.3)

    # Plot 3: Delay distributions
    delays_pos = df['PositiveDelay'][df['PositiveDelay'] > 0] / 60
    delays_neg = df['NegativeDelay'][df['NegativeDelay'] > 0] / 60

    ax3.hist(delays_pos, bins=50, alpha=0.6, label='Positive Delays', color='blue', density=True)
    ax3.hist(delays_neg, bins=50, alpha=0.6, label='Negative Delays', color='red', density=True)
    ax3.set_xlabel('Delay (minutes)')
    ax3.set_ylabel('Density')
    ax3.set_title('Delay Distribution Comparison')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Extreme percentiles comparison
    if pos_results and neg_results:
        percentiles = ['P90', 'P95', 'P99']

        # Get best distribution for each type
        best_pos = min(pos_results, key=lambda x: x['AIC'])
        best_neg = min(neg_results, key=lambda x: x['AIC'])

        pos_vals = [best_pos[p] for p in percentiles]
        neg_vals = [best_neg[p] for p in percentiles]

        x = np.arange(len(percentiles))
        width = 0.35

        ax4.bar(x - width/2, pos_vals, width, label=f'Positive ({best_pos["Distribution"]})', alpha=0.7)
        ax4.bar(x + width/2, neg_vals, width, label=f'Negative ({best_neg["Distribution"]})', alpha=0.7)

        ax4.set_xlabel('Percentiles')
        ax4.set_ylabel('Delay (minutes)')
        ax4.set_title('Extreme Percentiles - Best Distributions')
        ax4.set_xticks(x)
        ax4.set_xticklabels(percentiles)
        ax4.legend()
        ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{airport_code}_new_distributions_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_airports_simple():
    """Simple analysis of all airports with new distributions."""
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    all_airports = europe_airports + balkans_airports
    all_results = []

    for airport_code in all_airports:
        results = analyze_single_airport(airport_code, create_plots=False)
        if results:
            all_results.extend(results)

    if all_results:
        # Save comprehensive results
        results_df = pd.DataFrame(all_results)
        output_dir = os.path.join('results', 'new_distributions_analysis')
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'all_airports_new_distributions.csv'), index=False)

        # Create summary analysis
        create_overall_summary(results_df, output_dir)

        return results_df

    return None

def create_overall_summary(results_df, output_dir):
    """Create overall summary plots and analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Separate positive and negative delays
    pos_results = results_df[results_df['Delay_Type'] == 'positive']

    # Find best distribution for each airport
    best_by_airport = pos_results.groupby('Airport')['AIC'].idxmin()
    best_distributions = pos_results.loc[best_by_airport]

    # Plot 1: Distribution popularity
    dist_counts = best_distributions['Distribution'].value_counts()
    ax1.pie(dist_counts.values, labels=dist_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Most Popular Distributions\n(Positive Delays - Best AIC)')

    # Plot 2: AIC by distribution type
    ax2.boxplot([pos_results[pos_results['Distribution'] == dist]['AIC'].values
                for dist in pos_results['Distribution'].unique()],
               labels=pos_results['Distribution'].unique())
    ax2.set_title('AIC Distribution by Type')
    ax2.set_ylabel('AIC')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3)

    # Plot 3: Regional comparison
    europe_codes = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    best_distributions['Region'] = best_distributions['Airport'].apply(
        lambda x: 'Europe' if x in europe_codes else 'Balkans'
    )

    regional_dist = pd.crosstab(best_distributions['Region'], best_distributions['Distribution'])
    regional_dist.plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('Best Distribution by Region')
    ax3.set_ylabel('Number of Airports')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=0)

    # Plot 4: Performance metrics
    ax4.scatter(best_distributions['KS_Statistic'], best_distributions['AIC'],
               alpha=0.7, s=60, c='blue')
    ax4.set_xlabel('KS Statistic')
    ax4.set_ylabel('AIC')
    ax4.set_title('Model Performance Overview')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'new_distributions_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"\n" + "="*60)
    print("NEW DISTRIBUTIONS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total analyses: {len(results_df)}")
    print(f"Airports analyzed: {len(pos_results['Airport'].unique())}")
    print(f"Distributions tested: {', '.join(pos_results['Distribution'].unique())}")

    print(f"\nMost popular distributions (positive delays):")
    for dist, count in dist_counts.items():
        pct = (count / len(best_distributions)) * 100
        print(f"  {dist}: {count} airports ({pct:.1f}%)")

    # Best overall performance
    best_overall = pos_results.loc[pos_results['AIC'].idxmin()]
    print(f"\nBest overall fit:")
    print(f"  {best_overall['Distribution']} at {best_overall['Airport_Name']}")
    print(f"  AIC: {best_overall['AIC']:.2f}")
    print(f"  KS p-value: {best_overall['P_value']:.4f}")

if __name__ == "__main__":
    print("Starting simplified new distributions analysis...")

    # Test with a single airport first
    print("\nTesting with Frankfurt (EDDF)...")
    test_results = analyze_single_airport('EDDF', create_plots=True)

    if test_results:
        print("Single airport test successful!")

        # Run full analysis
        print("\nRunning analysis for all airports...")
        results = analyze_all_airports_simple()

        if results is not None:
            print(f"\nAnalysis complete! Results saved to: results/new_distributions_analysis/")
        else:
            print("Analysis failed.")
    else:
        print("Single airport test failed - check data files.")
