import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import burr
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

def fit_burr_distribution(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Detailed analysis of Burr XII distribution fitting."""
    os.makedirs(output_dir, exist_ok=True)

    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    try:
        # Fit Burr XII distribution
        print(f"Fitting Burr XII distribution for {airport_code}...")
        params = burr.fit(delays_minutes)
        c, d, loc, scale = params

        # Calculate goodness of fit metrics
        ks_stat, p_value = stats.kstest(delays_minutes, burr.cdf, args=params)
        log_likelihood = np.sum(burr.logpdf(delays_minutes, *params))

        n = len(delays_minutes)
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Calculate percentiles and statistics
        mean_est = burr.mean(*params) if np.isfinite(burr.mean(*params)) else np.nan
        var_est = burr.var(*params) if np.isfinite(burr.var(*params)) else np.nan
        median_est = burr.median(*params)

        # Calculate percentiles
        p25 = burr.ppf(0.25, *params)
        p75 = burr.ppf(0.75, *params)
        p90 = burr.ppf(0.90, *params)
        p95 = burr.ppf(0.95, *params)

        results = {
            'Airport': airport_code,
            'Airport_Name': airport_name,
            'Delay_Type': delay_type,
            'Distribution': 'Burr XII',
            'Shape_c': c,
            'Shape_d': d,
            'Location': loc,
            'Scale': scale,
            'Mean': mean_est,
            'Variance': var_est,
            'Median': median_est,
            'P25': p25,
            'P75': p75,
            'P90': p90,
            'P95': p95,
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
            'Data_P95': np.percentile(delays_minutes, 95)
        }

        # Create detailed visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: PDF comparison
        ax1.hist(delays_minutes, bins=50, density=True, alpha=0.7, color='lightgreen', label='Data')
        x = np.linspace(0, np.percentile(delays_minutes, 99), 1000)
        pdf = burr.pdf(x, *params)
        ax1.plot(x, pdf, 'r-', linewidth=2, label=f'Burr XII PDF')
        ax1.set_title(f'Burr XII Distribution Fit\n{airport_name} ({airport_code}) - {delay_type} delays')
        ax1.set_xlabel('Delay (minutes)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Log-scale PDF for heavy tails
        ax2.hist(delays_minutes, bins=50, density=True, alpha=0.7, color='lightgreen', label='Data')
        ax2.plot(x, pdf, 'r-', linewidth=2, label=f'Burr XII PDF')
        ax2.set_yscale('log')
        ax2.set_title('PDF Comparison (Log Scale)')
        ax2.set_xlabel('Delay (minutes)')
        ax2.set_ylabel('Density (log scale)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Q-Q plot
        theoretical_quantiles = burr.ppf(np.linspace(0.01, 0.99, len(delays_minutes)), *params)
        sample_quantiles = np.sort(delays_minutes)

        ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=2)
        min_val = min(min(theoretical_quantiles), min(sample_quantiles))
        max_val = max(max(theoretical_quantiles), max(sample_quantiles))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Sample Quantiles')
        ax3.set_title(f'Q-Q Plot (KS stat: {ks_stat:.4f})')
        ax3.grid(alpha=0.3)

        # Plot 4: Survival function (1-CDF) for tail analysis
        sample_sorted = np.sort(delays_minutes)
        empirical_sf = 1 - (np.arange(1, len(sample_sorted) + 1) / len(sample_sorted))
        theoretical_sf = burr.sf(sample_sorted, *params)

        ax4.plot(sample_sorted, empirical_sf, 'b-', alpha=0.7, label='Empirical SF', linewidth=2)
        ax4.plot(sample_sorted, theoretical_sf, 'r--', linewidth=2, label='Burr XII SF')
        ax4.set_yscale('log')
        ax4.set_title('Survival Function (Log Scale)')
        ax4.set_xlabel('Delay (minutes)')
        ax4.set_ylabel('P(X > x)')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # Add text box with parameter information
        textstr = f'Parameters:\nc (shape 1) = {c:.3f}\nd (shape 2) = {d:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\n\nFit Quality:\nAIC = {aic:.2f}\nKS p-value = {p_value:.4f}\n\nPercentiles:\n90th = {p90:.2f} min\n95th = {p95:.2f} min'
        props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"burr_analysis_{delay_type}_{airport_code}.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

        return results

    except Exception as e:
        print(f"Error fitting Burr XII distribution for {airport_code}: {e}")
        return None

def analyze_burr_all_airports(airport_codes=None):
    """Analyze Burr XII distribution for all airports."""
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

    output_dir = os.path.join('results', 'burr_analysis')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for code in airport_codes:
        print(f"\nAnalyzing Burr XII distribution for {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Positive delays
            pos_result = fit_burr_distribution(
                df['PositiveDelay'], code, airport_names.get(code, code), output_dir, 'positive'
            )
            if pos_result:
                all_results.append(pos_result)

            # Negative delays
            neg_result = fit_burr_distribution(
                df['NegativeDelay'], code, airport_names.get(code, code), output_dir, 'negative'
            )
            if neg_result:
                all_results.append(neg_result)

    if all_results:
        # Save comprehensive results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'burr_analysis_summary.csv'), index=False)

        # Create parameter comparison plot
        create_burr_comparison_plot(results_df, output_dir)

        print(f"\nBurr XII analysis complete! Results saved to: {output_dir}")
        return results_df
    else:
        print("No successful fits found.")
        return None

def create_burr_comparison_plot(results_df, output_dir):
    """Create Burr XII parameter comparison plots across airports."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Separate positive and negative delays
    pos_results = results_df[results_df['Delay_Type'] == 'positive']
    neg_results = results_df[results_df['Delay_Type'] == 'negative']

    # Plot 1: Shape parameters relationship
    ax1.scatter(pos_results['Shape_c'], pos_results['Shape_d'], alpha=0.7, label='Positive', s=60, c='red')
    ax1.scatter(neg_results['Shape_c'], neg_results['Shape_d'], alpha=0.7, label='Negative', s=60, c='blue')
    ax1.set_title('Burr XII Shape Parameters Relationship')
    ax1.set_xlabel('Shape Parameter c')
    ax1.set_ylabel('Shape Parameter d')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Scale parameter comparison
    airports_pos = pos_results['Airport'].tolist()
    scale_pos = pos_results['Scale'].tolist()
    airports_neg = neg_results['Airport'].tolist()
    scale_neg = neg_results['Scale'].tolist()

    x_pos = np.arange(len(airports_pos))
    x_neg = np.arange(len(airports_neg))

    ax2.bar(x_pos - 0.2, scale_pos, 0.4, alpha=0.7, label='Positive', color='red')
    ax2.bar(x_neg + 0.2, scale_neg, 0.4, alpha=0.7, label='Negative', color='blue')
    ax2.set_title('Scale Parameter by Airport')
    ax2.set_ylabel('Scale Parameter')
    ax2.set_xticks(range(len(set(airports_pos + airports_neg))))
    ax2.set_xticklabels(sorted(set(airports_pos + airports_neg)), rotation=45)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: 95th percentile comparison with data
    ax3.scatter(pos_results['Data_P95'], pos_results['P95'], alpha=0.7, label='Positive', s=60, c='red')
    ax3.scatter(neg_results['Data_P95'], neg_results['P95'], alpha=0.7, label='Negative', s=60, c='blue')

    # Add perfect fit line
    all_data_p95 = list(pos_results['Data_P95']) + list(neg_results['Data_P95'])
    all_model_p95 = list(pos_results['P95']) + list(neg_results['P95'])
    min_val = min(min(all_data_p95), min(all_model_p95))
    max_val = max(max(all_data_p95), max(all_model_p95))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)

    ax3.set_title('95th Percentile: Data vs Model')
    ax3.set_xlabel('Data 95th Percentile (minutes)')
    ax3.set_ylabel('Model 95th Percentile (minutes)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: AIC distribution by region
    europe_codes = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    pos_results['Region'] = pos_results['Airport'].apply(lambda x: 'Europe' if x in europe_codes else 'Balkans')

    europe_aic = pos_results[pos_results['Region'] == 'Europe']['AIC']
    balkans_aic = pos_results[pos_results['Region'] == 'Balkans']['AIC']

    ax4.boxplot([europe_aic, balkans_aic], labels=['Europe', 'Balkans'])
    ax4.set_title('AIC Distribution by Region (Positive Delays)')
    ax4.set_ylabel('AIC')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burr_parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create tail analysis plot
    create_tail_analysis_plot(results_df, output_dir)

def create_tail_analysis_plot(results_df, output_dir):
    """Create specialized tail analysis plot for Burr XII distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    pos_results = results_df[results_df['Delay_Type'] == 'positive']

    # Plot 1: Extreme percentiles comparison
    percentiles = ['P90', 'P95']
    data_percentiles = ['Data_P90', 'Data_P95']

    for i, (model_p, data_p) in enumerate(zip(percentiles, data_percentiles)):
        ax1.scatter(pos_results[data_p], pos_results[model_p],
                   alpha=0.7, s=60, label=f'{model_p}')

    # Perfect fit line
    all_data = list(pos_results['Data_P90']) + list(pos_results['Data_P95'])
    all_model = list(pos_results['P90']) + list(pos_results['P95'])
    min_val = min(min(all_data), min(all_model))
    max_val = max(max(all_data), max(all_model))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)

    ax1.set_title('Extreme Percentiles: Data vs Burr XII Model')
    ax1.set_xlabel('Data Percentiles (minutes)')
    ax1.set_ylabel('Model Percentiles (minutes)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Shape parameter influence on tail behavior
    ax2.scatter(pos_results['Shape_c'], pos_results['P95'],
               alpha=0.7, s=60, c=pos_results['Shape_d'], cmap='viridis')
    ax2.set_title('95th Percentile vs Shape Parameters')
    ax2.set_xlabel('Shape Parameter c')
    ax2.set_ylabel('95th Percentile (minutes)')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Shape Parameter d')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'burr_tail_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = analyze_burr_all_airports()
    if results is not None:
        print("\nBurr XII Distribution Analysis Summary:")
        print(f"Analyzed {len(results)} airport-delay type combinations")
        pos_results = results[results['Delay_Type'] == 'positive']
        print(f"Mean shape parameter c (positive delays): {pos_results['Shape_c'].mean():.3f}")
        print(f"Mean shape parameter d (positive delays): {pos_results['Shape_d'].mean():.3f}")
        print(f"Mean AIC (positive delays): {pos_results['AIC'].mean():.2f}")
        print(f"Best fit airport: {pos_results.loc[pos_results['AIC'].idxmin(), 'Airport_Name']}")
