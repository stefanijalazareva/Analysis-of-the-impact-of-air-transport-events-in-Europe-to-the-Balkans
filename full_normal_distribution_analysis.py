import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import norm
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

    return df

def fit_normal_distribution(delays, airport_code, airport_name):
    """Fit normal distribution to delay data and return parameters and fit metrics."""

    if len(delays) < 100:
        print(f"Not enough delay samples for {airport_code}")
        return None

    # Convert to minutes for better interpretability
    delays_minutes = delays / 60

    # Fit normal distribution
    params = norm.fit(delays_minutes)
    mean, std = params

    # Calculate Kolmogorov-Smirnov test statistic and p-value
    ks_statistic, p_value = stats.kstest(delays_minutes, 'norm', params)

    # Calculate R-squared (coefficient of determination)
    # For this, we compare the CDF of the fitted distribution with the empirical CDF
    empirical_cdf = np.arange(1, len(delays_minutes) + 1) / len(delays_minutes)
    sorted_data = np.sort(delays_minutes)
    fitted_cdf = norm.cdf(sorted_data, *params)
    r_squared = 1 - np.sum((empirical_cdf - fitted_cdf) ** 2) / np.sum((empirical_cdf - np.mean(empirical_cdf)) ** 2)

    # Calculate AIC and BIC
    log_likelihood = np.sum(norm.logpdf(delays_minutes, *params))
    k = len(params)  # number of parameters (mean and std for normal)
    n = len(delays_minutes)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    # Calculate RMSE between fitted PDF and histogram
    hist, bin_edges = np.histogram(delays_minutes, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf_values = norm.pdf(bin_centers, *params)
    # Only include bins where both hist and pdf are defined
    valid_indices = ~np.isnan(hist) & ~np.isnan(pdf_values)
    rmse = np.sqrt(np.mean((hist[valid_indices] - pdf_values[valid_indices]) ** 2))

    result = {
        'Airport': airport_code,
        'Airport Name': airport_name,
        'Mean (minutes)': mean,
        'Std (minutes)': std,
        'KS Statistic': ks_statistic,
        'P-value': p_value,
        'R-squared': r_squared,
        'AIC': aic,
        'BIC': bic,
        'RMSE': rmse,
        'Sample Size': len(delays_minutes)
    }

    return result

def create_normal_fit_plot(delays, params, airport_code, airport_name, output_dir):
    """Create plot showing histogram with normal distribution fit."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    delays_minutes = delays / 60

    mean, std = params

    # Plot histogram and fitted normal distribution
    plt.figure(figsize=(12, 7))

    # Plot histogram
    sns.histplot(delays_minutes, bins=50, stat='density', alpha=0.6, label='Data')

    # Plot fitted normal distribution
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    pdf = norm.pdf(x, mean, std)
    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Normal Distribution (μ={mean:.2f}, σ={std:.2f})')

    # Add a vertical line at x=0 to mark on-time arrivals
    plt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='On-time')

    # Add plot details
    plt.title(f'Full Normal Distribution Fit for Delays at {airport_name} ({airport_code})')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    # Add quality of fit metrics to the plot
    ks_statistic, p_value = stats.kstest(delays_minutes, 'norm', (mean, std))
    plt.figtext(0.15, 0.02, f'K-S test: {ks_statistic:.4f} (p={p_value:.4f})', ha='left')

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"full_normal_fit_{airport_code}.png"), bbox_inches='tight')
    plt.close()

    # Create QQ plot to assess normality
    plt.figure(figsize=(8, 8))
    stats.probplot(delays_minutes, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {airport_name} ({airport_code})')
    plt.savefig(os.path.join(output_dir, f"qq_plot_{airport_code}.png"), bbox_inches='tight')
    plt.close()

def analyze_full_normal_distributions():
    """Analyze normal distribution fits for all airports using full delay distributions."""
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

    # Use all airports
    airport_codes = europe_airports + balkans_airports

    output_dir = os.path.join('results', 'full_normal_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Results container
    results = []

    for code in airport_codes:
        print(f"\nProcessing {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)

        if df is not None:
            # Fit normal distribution to full delays (both positive and negative)
            print(f"Fitting normal distribution to full delays...")
            result = fit_normal_distribution(
                df['Delay'],
                code,
                airport_names.get(code, code)
            )

            if result:
                results.append(result)
                # Create plots
                create_normal_fit_plot(
                    df['Delay'],
                    (result['Mean (minutes)'], result['Std (minutes)']),
                    code,
                    airport_names.get(code, code),
                    output_dir
                )

    # Create result table
    results_df = pd.DataFrame(results)

    # Add region information
    results_df['Region'] = results_df['Airport'].apply(
        lambda x: 'Europe' if x in europe_airports else 'Balkans'
    )

    # Sort by region and R-squared
    results_df = results_df.sort_values(['Region', 'R-squared'], ascending=[True, False])

    # Save table
    results_df.to_csv(os.path.join(output_dir, 'full_normal_distribution_parameters.csv'), index=False)

    # Create parameter comparison charts
    create_parameter_comparison_charts(results_df, output_dir)

    print("\nAnalysis complete. Results saved in:", output_dir)
    return results_df

def create_parameter_comparison_charts(results_df, output_dir):
    """Create bar charts comparing means and standard deviations across airports."""

    # Create mean comparison chart
    plt.figure(figsize=(14, 8))

    # Group by region for better comparison
    europe_results = results_df[results_df['Region'] == 'Europe'].sort_values('Mean (minutes)')
    balkans_results = results_df[results_df['Region'] == 'Balkans'].sort_values('Mean (minutes)')

    # Create color palette
    europe_color = '#1f77b4'  # blue
    balkans_color = '#ff7f0e'  # orange

    # Plot Europe airports
    plt.bar(range(len(europe_results)), europe_results['Mean (minutes)'],
            yerr=europe_results['Std (minutes)'],
            capsize=5,
            color=europe_color,
            label='Europe')

    # Plot Balkans airports (offset by Europe count)
    offset = len(europe_results)
    plt.bar(range(offset, offset + len(balkans_results)), balkans_results['Mean (minutes)'],
            yerr=balkans_results['Std (minutes)'],
            capsize=5,
            color=balkans_color,
            label='Balkans')

    # Add airport codes as x-tick labels
    plt.xticks(range(len(results_df)),
              list(europe_results['Airport']) + list(balkans_results['Airport']),
              rotation=45)

    # Add a horizontal line at y=0 to mark on-time arrivals
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='On-time')

    plt.title('Mean and Standard Deviation of Delays by Airport (Full Normal Distribution)')
    plt.ylabel('Delay (minutes)')
    plt.xlabel('Airport')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Add a vertical line separating the regions
    plt.axvline(x=len(europe_results)-0.5, color='gray', linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_normal_parameters_comparison.png'))
    plt.close()

    # Create quality of fit visualization
    plt.figure(figsize=(14, 8))

    # Combine Europe and Balkans, but keep region info for color
    combined_results = pd.concat([
        europe_results.assign(Region='Europe'),
        balkans_results.assign(Region='Balkans')
    ]).sort_values('R-squared', ascending=False)

    # Create color map based on region
    colors = [europe_color if region == 'Europe' else balkans_color
              for region in combined_results['Region']]

    # Create bar chart of R-squared values
    plt.bar(range(len(combined_results)), combined_results['R-squared'], color=colors)

    # Add airport codes as x-tick labels
    plt.xticks(range(len(combined_results)), list(combined_results['Airport']), rotation=45)

    # Add a horizontal reference line at R² = 0.9
    plt.axhline(y=0.9, color='red', linestyle='--', label='R² = 0.9')

    # Add labels for Europe and Balkans regions in the legend
    europe_patch = plt.Rectangle((0, 0), 1, 1, fc=europe_color, label='Europe')
    balkans_patch = plt.Rectangle((0, 0), 1, 1, fc=balkans_color, label='Balkans')
    plt.legend(handles=[europe_patch, balkans_patch])

    plt.title('Quality of Normal Distribution Fit (R-squared) for Full Delays')
    plt.ylabel('R-squared')
    plt.xlabel('Airport')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'full_normal_fit_quality_comparison.png'))
    plt.close()

    # Create parameters table visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create data for table with selected columns
    table_data = results_df[['Airport', 'Airport Name', 'Mean (minutes)', 'Std (minutes)', 'R-squared', 'Sample Size', 'Region']]

    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    loc='center',
                    cellLoc='center',
                    bbox=[0, 0, 1, 1])

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color-code rows by region
    for i in range(len(results_df)):
        # +1 because row 0 is the header
        row_idx = i + 1
        if table_data.iloc[i]['Region'] == 'Europe':
            for j in range(len(table_data.columns)):
                table[(row_idx, j)].set_facecolor('#d6e4f0')  # light blue
        else:
            for j in range(len(table_data.columns)):
                table[(row_idx, j)].set_facecolor('#fae6d0')  # light orange

    plt.title('Normal Distribution Parameters for Airport Delays')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normal_parameters_table.png'), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    results = analyze_full_normal_distributions()
    print("\nAnalysis complete!")
