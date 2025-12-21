import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gengamma
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

def fit_gengamma_distribution(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Detailed analysis of Generalized Gamma distribution fitting."""
    os.makedirs(output_dir, exist_ok=True)

    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    try:
        # Fit Generalized Gamma distribution
        print(f"Fitting Generalized Gamma distribution for {airport_code}...")
        params = gengamma.fit(delays_minutes)
        a, c, loc, scale = params  # a=shape1, c=shape2, loc=location, scale=scale

        # Calculate goodness of fit metrics
        ks_stat, p_value = stats.kstest(delays_minutes, gengamma.cdf, args=params)
        log_likelihood = np.sum(gengamma.logpdf(delays_minutes, *params))

        n = len(delays_minutes)
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Calculate statistics
        try:
            mean_est = gengamma.mean(*params)
            var_est = gengamma.var(*params)
            median_est = gengamma.median(*params)
        except:
            mean_est = np.nan
            var_est = np.nan
            median_est = gengamma.ppf(0.5, *params)

        # Calculate percentiles
        p10 = gengamma.ppf(0.10, *params)
        p25 = gengamma.ppf(0.25, *params)
        p75 = gengamma.ppf(0.75, *params)
        p90 = gengamma.ppf(0.90, *params)
        p95 = gengamma.ppf(0.95, *params)
        p99 = gengamma.ppf(0.99, *params)

        # Calculate skewness and kurtosis
        try:
            skewness = stats.skew(delays_minutes)
            kurtosis = stats.kurtosis(delays_minutes)
        except:
            skewness = np.nan
            kurtosis = np.nan

        results = {
            'Airport': airport_code,
            'Airport_Name': airport_name,
            'Delay_Type': delay_type,
            'Distribution': 'Generalized Gamma',
            'Shape_a': a,
            'Shape_c': c,
            'Location': loc,
            'Scale': scale,
            'Mean': mean_est,
            'Variance': var_est,
            'Median': median_est,
            'P10': p10,
            'P25': p25,
            'P75': p75,
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
            'Data_Skewness': skewness,
            'Data_Kurtosis': kurtosis,
            'Data_P90': np.percentile(delays_minutes, 90),
            'Data_P95': np.percentile(delays_minutes, 95),
            'Data_P99': np.percentile(delays_minutes, 99)
        }

        # Create detailed visualization
        create_gengamma_detailed_plot(delays_minutes, results, params,
                                    airport_code, airport_name, delay_type, output_dir)

        return results

    except Exception as e:
        print(f"Error fitting Generalized Gamma distribution for {airport_code}: {e}")
        return None

def create_gengamma_detailed_plot(delays_minutes, results, params,
                                airport_code, airport_name, delay_type, output_dir):
    """Create comprehensive Generalized Gamma analysis plots."""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    a, c, loc, scale = params

    # Plot 1: PDF comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(delays_minutes, bins=50, density=True, alpha=0.7, color='lightcyan', label='Data')
    x = np.linspace(0, np.percentile(delays_minutes, 99.5), 1000)
    pdf = gengamma.pdf(x, *params)
    ax1.plot(x, pdf, 'r-', linewidth=2, label='Generalized Gamma PDF')
    ax1.set_title(f'Generalized Gamma Distribution Fit\n{airport_name} ({airport_code})')
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Log-scale PDF (top center left)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(delays_minutes, bins=50, density=True, alpha=0.7, color='lightcyan', label='Data')
    ax2.plot(x, pdf, 'r-', linewidth=2, label='Generalized Gamma PDF')
    ax2.set_yscale('log')
    ax2.set_title('PDF (Log Scale)')
    ax2.set_xlabel('Delay (minutes)')
    ax2.set_ylabel('Density (log)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: CDF comparison (top center right)
    ax3 = fig.add_subplot(gs[0, 2])
    sample_sorted = np.sort(delays_minutes)
    empirical_cdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)
    theoretical_cdf = gengamma.cdf(sample_sorted, *params)

    ax3.plot(sample_sorted, empirical_cdf, 'b-', alpha=0.8, linewidth=2, label='Empirical CDF')
    ax3.plot(sample_sorted, theoretical_cdf, 'r-', linewidth=2, label='Generalized Gamma CDF')
    ax3.set_title('CDF Comparison')
    ax3.set_xlabel('Delay (minutes)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Parameter space visualization (top right)
    ax4 = fig.add_subplot(gs[0, 3])
    # Create parameter interpretation plot
    shape_info = f'Shape a = {a:.3f}\nShape c = {c:.3f}\nScale = {scale:.3f}'

    # Show how parameters affect distribution shape
    x_param = np.linspace(0, np.percentile(delays_minutes, 95), 100)

    # Compare with different parameter values
    pdf_current = gengamma.pdf(x_param, a, c, loc, scale)
    pdf_alt1 = gengamma.pdf(x_param, a*1.2, c, loc, scale)
    pdf_alt2 = gengamma.pdf(x_param, a, c*1.2, loc, scale)

    ax4.plot(x_param, pdf_current, 'r-', linewidth=2, label=f'Fitted (a={a:.2f}, c={c:.2f})')
    ax4.plot(x_param, pdf_alt1, 'g--', linewidth=1.5, label=f'a+20% (a={a*1.2:.2f}, c={c:.2f})')
    ax4.plot(x_param, pdf_alt2, 'b--', linewidth=1.5, label=f'c+20% (a={a:.2f}, c={c*1.2:.2f})')
    ax4.set_title('Parameter Sensitivity')
    ax4.set_xlabel('Delay (minutes)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # Plot 5: Q-Q plot (middle left)
    ax5 = fig.add_subplot(gs[1, 0])
    theoretical_quantiles = gengamma.ppf(np.linspace(0.01, 0.99, len(delays_minutes)), *params)
    sample_quantiles = np.sort(delays_minutes)

    ax5.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=2, c='purple')
    min_val = min(min(theoretical_quantiles), min(sample_quantiles))
    max_val = max(max(theoretical_quantiles), max(sample_quantiles))
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    ax5.set_xlabel('Theoretical Quantiles')
    ax5.set_ylabel('Sample Quantiles')
    ax5.set_title(f'Q-Q Plot (KS: {results["KS_Statistic"]:.4f})')
    ax5.grid(alpha=0.3)

    # Plot 6: P-P plot (middle center left)
    ax6 = fig.add_subplot(gs[1, 1])
    sample_cdf = np.arange(1, len(delays_minutes) + 1) / len(delays_minutes)
    theoretical_cdf_sorted = gengamma.cdf(sample_quantiles, *params)

    ax6.scatter(theoretical_cdf_sorted, sample_cdf, alpha=0.6, s=2, c='orange')
    ax6.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax6.set_xlabel('Theoretical CDF')
    ax6.set_ylabel('Sample CDF')
    ax6.set_title('P-P Plot')
    ax6.grid(alpha=0.3)

    # Plot 7: Survival function (middle center right)
    ax7 = fig.add_subplot(gs[1, 2])
    empirical_sf = 1 - empirical_cdf
    theoretical_sf = gengamma.sf(sample_sorted, *params)

    ax7.plot(sample_sorted, empirical_sf, 'b-', alpha=0.8, linewidth=2, label='Empirical')
    ax7.plot(sample_sorted, theoretical_sf, 'r-', linewidth=2, label='Generalized Gamma')
    ax7.set_yscale('log')
    ax7.set_title('Survival Function (Log Scale)')
    ax7.set_xlabel('Delay (minutes)')
    ax7.set_ylabel('P(X > x)')
    ax7.legend()
    ax7.grid(alpha=0.3)

    # Plot 8: Residuals analysis (middle right)
    ax8 = fig.add_subplot(gs[1, 3])
    residuals = empirical_cdf - theoretical_cdf
    ax8.plot(sample_sorted, residuals, 'g-', alpha=0.7, linewidth=1)
    ax8.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax8.fill_between(sample_sorted, residuals, alpha=0.3, color='green')
    ax8.set_title('CDF Residuals')
    ax8.set_xlabel('Delay (minutes)')
    ax8.set_ylabel('Empirical - Theoretical')
    ax8.grid(alpha=0.3)

    # Plot 9: Percentile comparison (bottom left)
    ax9 = fig.add_subplot(gs[2, 0])
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    data_percentiles = [np.percentile(delays_minutes, p) for p in percentiles]
    model_percentiles = [results[f'P{p}'] if f'P{p}' in results else gengamma.ppf(p/100, *params)
                        for p in percentiles]

    ax9.scatter(data_percentiles, model_percentiles, s=60, alpha=0.7, c='red')
    ax9.plot([min(data_percentiles), max(data_percentiles)],
             [min(data_percentiles), max(data_percentiles)], 'k--', linewidth=2)
    ax9.set_xlabel('Data Percentiles')
    ax9.set_ylabel('Model Percentiles')
    ax9.set_title('Percentile Comparison')
    ax9.grid(alpha=0.3)

    # Plot 10: Distribution properties (bottom center left)
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.axis('off')

    properties_text = f"""
    GENERALIZED GAMMA PARAMETERS
    Shape a: {a:.4f}
    Shape c: {c:.4f}
    Location: {loc:.4f}
    Scale: {scale:.4f}
    
    DISTRIBUTION PROPERTIES
    Mean: {results['Mean']:.3f} min
    Median: {results['Median']:.3f} min
    Std Dev: {np.sqrt(results['Variance']) if not np.isnan(results['Variance']) else 'N/A'}
    
    FIT QUALITY
    AIC: {results['AIC']:.2f}
    BIC: {results['BIC']:.2f}
    KS Statistic: {results['KS_Statistic']:.4f}
    p-value: {results['P_value']:.4f}
    
    EXTREME VALUES
    95th percentile: {results['P95']:.2f} min
    99th percentile: {results['P99']:.2f} min
    """

    ax10.text(0.05, 0.95, properties_text, transform=ax10.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Plot 11: Moment comparison (bottom center right)
    ax11 = fig.add_subplot(gs[2, 2])
    moments = ['Mean', 'Median', '90th %ile', '95th %ile']
    data_moments = [results['Data_Mean'], results['Data_Median'],
                   results['Data_P90'], results['Data_P95']]
    model_moments = [results['Mean'], results['Median'],
                    results['P90'], results['P95']]

    x_pos = np.arange(len(moments))
    width = 0.35

    ax11.bar(x_pos - width/2, data_moments, width, alpha=0.7, label='Data', color='blue')
    ax11.bar(x_pos + width/2, model_moments, width, alpha=0.7, label='Model', color='red')
    ax11.set_xlabel('Moments')
    ax11.set_ylabel('Value (minutes)')
    ax11.set_title('Moment Comparison')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(moments, rotation=45)
    ax11.legend()
    ax11.grid(alpha=0.3)

    # Plot 12: Special cases visualization (bottom right)
    ax12 = fig.add_subplot(gs[2, 3])

    # Show relationship to special cases
    ax12.axis('off')
    special_cases_text = f"""
    GENERALIZED GAMMA SPECIAL CASES
    
    When c = 1:
    → Gamma distribution
    
    When a = 1:
    → Weibull distribution
    
    When a = c = 1:
    → Exponential distribution
    
    Current parameters:
    a = {a:.3f} (shape parameter)
    c = {c:.3f} (power parameter)
    
    Closest to:
    {'Gamma' if abs(c-1) < 0.1 else 'Weibull' if abs(a-1) < 0.1 else 'General form'}
    """

    ax12.text(0.05, 0.95, special_cases_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Comprehensive Generalized Gamma Analysis - {delay_type.capitalize()} Delays',
                fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, f"gengamma_detailed_analysis_{delay_type}_{airport_code}.png"),
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_gengamma_all_airports(airport_codes=None):
    """Analyze Generalized Gamma distribution for all airports."""
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

    output_dir = os.path.join('results', 'gengamma_analysis')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for code in airport_codes:
        print(f"\nAnalyzing Generalized Gamma for {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Positive delays
            pos_result = fit_gengamma_distribution(
                df['PositiveDelay'], code, airport_names.get(code, code), output_dir, 'positive'
            )
            if pos_result:
                all_results.append(pos_result)

            # Negative delays
            neg_result = fit_gengamma_distribution(
                df['NegativeDelay'], code, airport_names.get(code, code), output_dir, 'negative'
            )
            if neg_result:
                all_results.append(neg_result)

    if all_results:
        # Save comprehensive results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'gengamma_analysis_summary.csv'), index=False)

        # Create summary plots
        create_gengamma_summary_plots(results_df, output_dir)

        print(f"\nGeneralized Gamma analysis complete! Results saved to: {output_dir}")
        return results_df
    else:
        print("No successful fits found.")
        return None

def create_gengamma_summary_plots(results_df, output_dir):
    """Create summary plots for Generalized Gamma analysis across all airports."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    pos_results = results_df[results_df['Delay_Type'] == 'positive']
    neg_results = results_df[results_df['Delay_Type'] == 'negative']

    # Plot 1: Shape parameter relationship
    ax1.scatter(pos_results['Shape_a'], pos_results['Shape_c'],
               alpha=0.7, s=80, c='red', label='Positive', edgecolor='black', linewidth=0.5)
    ax1.scatter(neg_results['Shape_a'], neg_results['Shape_c'],
               alpha=0.7, s=80, c='blue', label='Negative', edgecolor='black', linewidth=0.5)

    # Add special case lines
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='c=1 (Gamma)')
    ax1.axvline(x=1, color='orange', linestyle='--', alpha=0.7, label='a=1 (Weibull)')

    ax1.set_title('Generalized Gamma Shape Parameters')
    ax1.set_xlabel('Shape Parameter a')
    ax1.set_ylabel('Shape Parameter c')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: AIC comparison by region
    europe_codes = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    pos_results['Region'] = pos_results['Airport'].apply(lambda x: 'Europe' if x in europe_codes else 'Balkans')

    europe_aic = pos_results[pos_results['Region'] == 'Europe']['AIC']
    balkans_aic = pos_results[pos_results['Region'] == 'Balkans']['AIC']

    ax2.boxplot([europe_aic, balkans_aic], labels=['Europe', 'Balkans'])
    ax2.set_title('AIC Distribution by Region\n(Positive Delays)')
    ax2.set_ylabel('AIC')
    ax2.grid(alpha=0.3)

    # Plot 3: Scale parameter vs data variability
    ax3.scatter(pos_results['Data_Std'], pos_results['Scale'],
               alpha=0.7, s=80, c='purple', edgecolor='black', linewidth=0.5)
    ax3.set_title('Scale Parameter vs Data Standard Deviation')
    ax3.set_xlabel('Data Standard Deviation (minutes)')
    ax3.set_ylabel('Scale Parameter')
    ax3.grid(alpha=0.3)

    # Add correlation coefficient
    corr_coef = pos_results[['Data_Std', 'Scale']].corr().iloc[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}',
            transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Parameter stability across airports
    airports = pos_results['Airport'].tolist()
    x_pos = np.arange(len(airports))

    ax4_twin = ax4.twinx()

    bars1 = ax4.bar(x_pos - 0.2, pos_results['Shape_a'], 0.4, alpha=0.7, label='Shape a', color='red')
    bars2 = ax4_twin.bar(x_pos + 0.2, pos_results['Shape_c'], 0.4, alpha=0.7, label='Shape c', color='blue')

    ax4.set_title('Shape Parameters by Airport (Positive Delays)')
    ax4.set_ylabel('Shape Parameter a', color='red')
    ax4_twin.set_ylabel('Shape Parameter c', color='blue')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(airports, rotation=45)
    ax4.grid(alpha=0.3)

    # Add legends
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gengamma_summary_analysis.png'),
               dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = analyze_gengamma_all_airports()
    if results is not None:
        print("\nGeneralized Gamma Distribution Analysis Summary:")
        print(f"Analyzed {len(results)} airport-delay type combinations")

        pos_results = results[results['Delay_Type'] == 'positive']
        print(f"Mean shape parameter a (positive delays): {pos_results['Shape_a'].mean():.3f}")
        print(f"Mean shape parameter c (positive delays): {pos_results['Shape_c'].mean():.3f}")
        print(f"Mean AIC (positive delays): {pos_results['AIC'].mean():.2f}")

        # Check for special cases
        gamma_like = pos_results[abs(pos_results['Shape_c'] - 1) < 0.1]
        weibull_like = pos_results[abs(pos_results['Shape_a'] - 1) < 0.1]

        if len(gamma_like) > 0:
            print(f"\nAirports close to Gamma distribution (c≈1): {len(gamma_like)}")
        if len(weibull_like) > 0:
            print(f"Airports close to Weibull distribution (a≈1): {len(weibull_like)}")
