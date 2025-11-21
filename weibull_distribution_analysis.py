import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import weibull_min, weibull_max, dweibull
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

def fit_weibull_distributions(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Detailed analysis of Weibull distribution variants."""
    os.makedirs(output_dir, exist_ok=True)

    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    # Define Weibull variants to test
    weibull_variants = [
        ('Weibull Min', weibull_min),
        ('Weibull Max', weibull_max),
        ('Double Weibull', dweibull)
    ]

    all_results = []

    for dist_name, distribution in weibull_variants:
        try:
            print(f"Fitting {dist_name} distribution for {airport_code}...")
            params = distribution.fit(delays_minutes)

            # Calculate goodness of fit metrics
            ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)
            log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))

            n = len(delays_minutes)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Calculate statistics
            try:
                mean_est = distribution.mean(*params)
                var_est = distribution.var(*params)
                median_est = distribution.median(*params)
            except:
                mean_est = np.nan
                var_est = np.nan
                median_est = distribution.ppf(0.5, *params)

            # Calculate percentiles
            p90 = distribution.ppf(0.90, *params)
            p95 = distribution.ppf(0.95, *params)
            p99 = distribution.ppf(0.99, *params)

            # Extract shape and scale parameters based on distribution
            if dist_name == 'Double Weibull':
                shape1, shape2, loc, scale = params
                result = {
                    'Airport': airport_code,
                    'Airport_Name': airport_name,
                    'Delay_Type': delay_type,
                    'Distribution': dist_name,
                    'Shape1': shape1,
                    'Shape2': shape2,
                    'Location': loc,
                    'Scale': scale,
                }
            else:  # Weibull Min/Max
                shape, loc, scale = params
                result = {
                    'Airport': airport_code,
                    'Airport_Name': airport_name,
                    'Delay_Type': delay_type,
                    'Distribution': dist_name,
                    'Shape': shape,
                    'Location': loc,
                    'Scale': scale,
                }

            # Add common metrics
            result.update({
                'Mean': mean_est,
                'Variance': var_est,
                'Median': median_est,
                'P90': p90,
                'P95': p95,
                'P99': p99,
                'KS_Statistic': ks_stat,
                'P_value': p_value,
                'Log_Likelihood': log_likelihood,
                'AIC': aic,
                'BIC': bic,
                'Sample_Size': n,
                'Data_Mean': np.mean(delays_minutes),
                'Data_Std': np.std(delays_minutes),
                'Data_Median': np.median(delays_minutes),
                'Data_P90': np.percentile(delays_minutes, 90),
                'Data_P95': np.percentile(delays_minutes, 95),
                'Data_P99': np.percentile(delays_minutes, 99)
            })

            all_results.append(result)

        except Exception as e:
            print(f"Error fitting {dist_name} for {airport_code}: {e}")

    if not all_results:
        return None

    # Create visualization comparing all Weibull variants
    create_weibull_comparison_plot(delays_minutes, all_results, weibull_variants,
                                 airport_code, airport_name, delay_type, output_dir)

    return all_results

def create_weibull_comparison_plot(delays_minutes, results, weibull_variants,
                                 airport_code, airport_name, delay_type, output_dir):
    """Create comprehensive Weibull comparison plots."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: PDF comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(delays_minutes, bins=50, density=True, alpha=0.6, color='lightgray', label='Data')

    colors = ['red', 'blue', 'green']
    x = np.linspace(0, np.percentile(delays_minutes, 99.5), 1000)

    for i, (result, (dist_name, distribution)) in enumerate(zip(results, weibull_variants)):
        if dist_name == 'Double Weibull':
            params = (result['Shape1'], result['Shape2'], result['Location'], result['Scale'])
        else:
            params = (result['Shape'], result['Location'], result['Scale'])

        try:
            pdf = distribution.pdf(x, *params)
            ax1.plot(x, pdf, color=colors[i], linewidth=2,
                    label=f'{dist_name} (AIC: {result["AIC"]:.1f})')
        except:
            continue

    ax1.set_title(f'Weibull Variants PDF Comparison\n{airport_name} ({airport_code}) - {delay_type}')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Log-scale PDF (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(delays_minutes, bins=50, density=True, alpha=0.6, color='lightgray', label='Data')

    for i, (result, (dist_name, distribution)) in enumerate(zip(results, weibull_variants)):
        if dist_name == 'Double Weibull':
            params = (result['Shape1'], result['Shape2'], result['Location'], result['Scale'])
        else:
            params = (result['Shape'], result['Location'], result['Scale'])

        try:
            pdf = distribution.pdf(x, *params)
            ax2.plot(x, pdf, color=colors[i], linewidth=2, label=f'{dist_name}')
        except:
            continue

    ax2.set_yscale('log')
    ax2.set_title('PDF Comparison (Log Scale)')
    ax2.set_xlabel('Delay (minutes)')
    ax2.set_ylabel('Density (log)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: AIC comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    dist_names = [r['Distribution'] for r in results]
    aics = [r['AIC'] for r in results]
    ax3.bar(dist_names, aics, color=colors[:len(results)])
    ax3.set_title('AIC Comparison')
    ax3.set_ylabel('AIC (lower is better)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3)

    # Find best distribution
    best_idx = np.argmin(aics)
    best_result = results[best_idx]
    best_dist_name, best_distribution = weibull_variants[best_idx]

    if best_dist_name == 'Double Weibull':
        best_params = (best_result['Shape1'], best_result['Shape2'], best_result['Location'], best_result['Scale'])
    else:
        best_params = (best_result['Shape'], best_result['Location'], best_result['Scale'])

    # Plot 4: Q-Q plot for best distribution (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    theoretical_quantiles = best_distribution.ppf(np.linspace(0.01, 0.99, len(delays_minutes)), *best_params)
    sample_quantiles = np.sort(delays_minutes)

    ax4.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=2, color=colors[best_idx])
    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')
    ax4.set_title(f'Q-Q Plot: {best_dist_name}')
    ax4.grid(alpha=0.3)

    # Plot 5: Survival function comparison (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    sample_sorted = np.sort(delays_minutes)
    empirical_sf = 1 - (np.arange(1, len(sample_sorted) + 1) / len(sample_sorted))

    ax5.plot(sample_sorted, empirical_sf, 'k-', alpha=0.8, linewidth=2, label='Empirical')

    for i, (result, (dist_name, distribution)) in enumerate(zip(results, weibull_variants)):
        if dist_name == 'Double Weibull':
            params = (result['Shape1'], result['Shape2'], result['Location'], result['Scale'])
        else:
            params = (result['Shape'], result['Location'], result['Scale'])

        try:
            theoretical_sf = distribution.sf(sample_sorted, *params)
            ax5.plot(sample_sorted, theoretical_sf, color=colors[i],
                    linewidth=2, alpha=0.7, label=f'{dist_name}')
        except:
            continue

    ax5.set_yscale('log')
    ax5.set_title('Survival Functions')
    ax5.set_xlabel('Delay (minutes)')
    ax5.set_ylabel('P(X > x)')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Plot 6: Hazard function for best distribution (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        # Calculate hazard function (pdf / survival function)
        x_hazard = np.linspace(0.1, np.percentile(delays_minutes, 95), 200)
        pdf_vals = best_distribution.pdf(x_hazard, *best_params)
        sf_vals = best_distribution.sf(x_hazard, *best_params)
        hazard = pdf_vals / sf_vals

        ax6.plot(x_hazard, hazard, color=colors[best_idx], linewidth=2)
        ax6.set_title(f'Hazard Function: {best_dist_name}')
        ax6.set_xlabel('Delay (minutes)')
        ax6.set_ylabel('Hazard Rate')
        ax6.grid(alpha=0.3)
    except:
        ax6.text(0.5, 0.5, 'Hazard function\ncalculation failed',
                ha='center', va='center', transform=ax6.transAxes)

    # Plot 7-9: Parameter interpretation (bottom row)
    ax7 = fig.add_subplot(gs[2, :])

    # Create parameter summary table
    table_data = []
    for result in results:
        dist_name = result['Distribution']
        if dist_name == 'Double Weibull':
            params_str = f"k1={result['Shape1']:.3f}, k2={result['Shape2']:.3f}, λ={result['Scale']:.3f}"
        else:
            params_str = f"k={result['Shape']:.3f}, λ={result['Scale']:.3f}"

        table_data.append([
            dist_name,
            params_str,
            f"{result['AIC']:.2f}",
            f"{result['KS_Statistic']:.4f}",
            f"{result['P_value']:.4f}",
            f"{result['P95']:.2f}"
        ])

    table = ax7.table(cellText=table_data,
                     colLabels=['Distribution', 'Parameters', 'AIC', 'KS Stat', 'p-value', '95th Percentile'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax7.axis('off')
    ax7.set_title('Parameter Summary', y=0.8, fontsize=12, fontweight='bold')

    plt.savefig(os.path.join(output_dir, f"weibull_variants_analysis_{delay_type}_{airport_code}.png"),
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_weibull_all_airports(airport_codes=None):
    """Analyze Weibull distribution variants for all airports."""
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    if airport_codes is None:
        airport_codes = europe_airports + balkans_airports

    output_dir = os.path.join('results', 'weibull_analysis')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for code in airport_codes:
        print(f"\nAnalyzing Weibull variants for {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Positive delays
            pos_results = fit_weibull_distributions(
                df['PositiveDelay'], code, airport_names.get(code, code), output_dir, 'positive'
            )
            if pos_results:
                all_results.extend(pos_results)

            # Negative delays
            neg_results = fit_weibull_distributions(
                df['NegativeDelay'], code, airport_names.get(code, code), output_dir, 'negative'
            )
            if neg_results:
                all_results.extend(neg_results)

    if all_results:
        # Save comprehensive results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'weibull_analysis_summary.csv'), index=False)

        # Create summary comparison plots
        create_weibull_summary_plots(results_df, output_dir)

        print(f"\nWeibull variants analysis complete! Results saved to: {output_dir}")
        return results_df
    else:
        print("No successful fits found.")
        return None

def create_weibull_summary_plots(results_df, output_dir):
    """Create summary comparison plots for Weibull variants across all airports."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Distribution popularity
    pos_results = results_df[results_df['Delay_Type'] == 'positive']
    best_by_airport = pos_results.groupby('Airport')['AIC'].idxmin()
    best_distributions = pos_results.loc[best_by_airport]

    dist_counts = best_distributions['Distribution'].value_counts()
    ax1.pie(dist_counts.values, labels=dist_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Best Weibull Variant by Airport\n(Positive Delays)')

    # Plot 2: AIC comparison across variants
    ax2.boxplot([pos_results[pos_results['Distribution'] == dist]['AIC'].values
                for dist in pos_results['Distribution'].unique()],
               labels=pos_results['Distribution'].unique())
    ax2.set_title('AIC Distribution by Weibull Variant')
    ax2.set_ylabel('AIC')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(alpha=0.3)

    # Plot 3: Shape parameter comparison for Weibull Min
    weibull_min_results = pos_results[pos_results['Distribution'] == 'Weibull Min']
    if len(weibull_min_results) > 0:
        ax3.scatter(weibull_min_results['Airport'], weibull_min_results['Shape'],
                   alpha=0.7, s=60, c='blue')
        ax3.set_title('Weibull Min Shape Parameter by Airport')
        ax3.set_ylabel('Shape Parameter')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(alpha=0.3)

    # Plot 4: Regional comparison
    europe_codes = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    best_distributions['Region'] = best_distributions['Airport'].apply(
        lambda x: 'Europe' if x in europe_codes else 'Balkans'
    )

    regional_dist = pd.crosstab(best_distributions['Region'], best_distributions['Distribution'])
    regional_dist.plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('Best Weibull Variant by Region')
    ax4.set_ylabel('Number of Airports')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weibull_summary_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = analyze_weibull_all_airports()
    if results is not None:
        print("\nWeibull Distribution Analysis Summary:")
        print(f"Analyzed {len(results)} airport-delay type-distribution combinations")

        pos_results = results[results['Delay_Type'] == 'positive']
        best_by_airport = pos_results.groupby('Airport')['AIC'].idxmin()
        best_distributions = pos_results.loc[best_by_airport]

        print("\nBest Weibull variant by popularity:")
        print(best_distributions['Distribution'].value_counts())

        print(f"\nOverall best AIC: {pos_results['AIC'].min():.2f}")
        best_overall = pos_results.loc[pos_results['AIC'].idxmin()]
        print(f"Best fit: {best_overall['Distribution']} at {best_overall['Airport_Name']}")
