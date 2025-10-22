"""
Delay Distribution Analysis: Comparison between Skopje and Major European Airports

This script analyzes and compares flight delay distributions between Skopje Airport
and several major European airports (Frankfurt, Madrid, Paris). It visualizes
delay histograms, evaluates normal distribution fit, performs statistical tests,
and provides conclusions about similarities or differences.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = "results/delay_distribution"


def clean_data(data):
    """
    Clean and validate delay data for statistical analysis.

    Converts the input data to a NumPy array, ensures float type, and removes
    invalid (NaN or infinite) values.

    Args:
        data (array-like): Raw delay data.

    Returns:
        np.ndarray: Cleaned numeric delay values.

    Raises:
        ValueError: If no valid data points remain after cleaning.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    data = data.astype(float)

    data = data[np.isfinite(data)]

    if len(data) == 0:
        raise ValueError("No valid data points after cleaning")

    return data


def calculate_fit_quality(data, mu, sigma):
    """
    Calculate goodness-of-fit metrics for normal distribution.

    Computes various metrics to evaluate how well a normal distribution
    fits the data, including R² and Mean Squared Error (MSE).

    Args:
        data (np.ndarray): Delay data
        mu (float): Mean of fitted normal distribution
        sigma (float): Standard deviation of fitted normal distribution

    Returns:
        dict: Dictionary containing R², MSE, and other fit metrics
    """
    data = clean_data(data)

    # Calculate histogram for actual data (for comparison purposes)
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate PDF values for the fitted normal at bin centers
    pdf_fitted = stats.norm.pdf(bin_centers, mu, sigma)

    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((hist - np.mean(hist))**2)
    ss_residual = np.sum((hist - pdf_fitted)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Mean Squared Error
    mse = np.mean((hist - pdf_fitted)**2)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Akaike Information Criterion (AIC)
    n = len(data)
    k = 2  # number of parameters in normal distribution (mu and sigma)
    log_likelihood = np.sum(np.log(stats.norm.pdf(data, mu, sigma)))
    aic = 2 * k - 2 * log_likelihood

    # Calculate KS statistic
    ks_stat = stats.kstest(data, 'norm', args=(mu, sigma)).statistic

    return {
        'R²': r_squared,
        'MSE': mse,
        'RMSE': rmse,
        'AIC': aic,
        'KS': ks_stat
    }


def fit_and_plot_normal(data, airport_name, ax):
    """
    Fit a normal distribution to the delay data and visualize it.

    Args:
        data (np.ndarray): Delay data.
        airport_name (str): Airport name for labeling.
        ax (matplotlib.axes.Axes): Axis on which to draw the plot.

    Returns:
        tuple: Estimated mean (μ) and standard deviation (σ).
    """
    data = clean_data(data)

    mu, sigma = stats.norm.fit(data)

    sns.histplot(data, kde=True, stat="density", alpha=0.6, ax=ax, label="Actual delays")

    x = np.linspace(data.min(), data.max(), 1000)
    y = stats.norm.pdf(x, mu, sigma)

    ax.plot(x, y, 'r-', linewidth=2, label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_title(f"{airport_name} Delay Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Delay (minutes)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()

    return mu, sigma


def calculate_statistics(data):
    """
        Compute descriptive statistics for the given delay data.

        Args:
            data (np.ndarray): Delay data.

        Returns:
            dict: Mean, standard deviation, skewness, and kurtosis values.
        """
    data = clean_data(data)

    return {
        'Mean': np.mean(data),
        'Std Dev': np.std(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }


def test_normality(data):
    """
        Perform normality tests on the delay data.

        Includes:
            - Shapiro-Wilk test
            - Kolmogorov-Smirnov test (against fitted normal)

        Args:
            data (np.ndarray): Delay data.

        Returns:
            dict: Test statistics and p-values for each test.
        """
    data = clean_data(data)

    shapiro_test = stats.shapiro(data[:5000] if len(data) > 5000 else data)  # Shapiro-Wilk has sample size limitation

    mu, sigma = stats.norm.fit(data)
    ks_test = stats.kstest(data, 'norm', args=(mu, sigma))

    return {
        'Shapiro-Wilk': {'Statistic': shapiro_test[0], 'p-value': shapiro_test[1]},
        'Kolmogorov-Smirnov': {'Statistic': ks_test[0], 'p-value': ks_test[1]}
    }


def main():
    """
        Execute the full delay distribution analysis workflow.

        Steps:
            1. Load datasets for selected airports.
            2. Fit and visualize normal distributions.
            3. Calculate descriptive and normality statistics.
            4. Generate comparative visualizations (histograms, Q-Q plots, CDF).
            5. Print analytical conclusions about delay distributions.
    """
    print("\n===== DELAY DISTRIBUTION ANALYSIS: SKOPJE VS MAJOR EUROPEAN AIRPORTS =====\n")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    try:
        print("Loading airport summary data...")
        df = pd.read_csv("results/summary_data/airport_summary.csv")
    except FileNotFoundError:
        print("Using the provided CSV file path...")
        df = pd.read_csv("results/summary_data/airport_summary.csv")

    # Map of airport codes to names - expanded to include all airports in the dataset
    airports = {
        'BKPR': 'Pristina',      # Kosovo
        'EDDF': 'Frankfurt',     # Germany
        'EDDM': 'Munich',        # Germany
        'EGKK': 'London Gatwick', # UK
        'EGLL': 'London Heathrow', # UK
        'EHAM': 'Amsterdam',     # Netherlands
        'EIDW': 'Dublin',        # Ireland
        'LATI': 'Tirana',        # Albania
        'LBBG': 'Burgas',        # Bulgaria
        'LBSF': 'Sofia',         # Bulgaria
        'LDDU': 'Dubrovnik',     # Croatia
        'LDSP': 'Split',         # Croatia
        'LDZA': 'Zagreb',        # Croatia
        'LEBL': 'Barcelona',     # Spain
        'LEMD': 'Madrid',        # Spain
        'LFPG': 'Paris',         # France
        'LIRF': 'Rome Fiumicino', # Italy
        'LQSA': 'Sarajevo',      # Bosnia and Herzegovina
        'LWSK': 'Skopje',        # North Macedonia
        'LYTV': 'Tivat'          # Montenegro
    }

    delay_data = {}
    for code in airports.keys():
        try:
            # Load the numpy file which contains the delay data
            raw_data = np.load(f"data/RawData/Delays_{code}.npy", allow_pickle=True)
            print(f"Loaded raw data for {airports[code]} ({code}) - {len(raw_data)} records")

            # Extract just the delay values (fourth column, index 3)
            numeric_data = []
            for row in raw_data:
                try:
                    # Extract the delay value (fourth column)
                    delay_value = float(row[3])
                    if np.isfinite(delay_value):  # Skip NaN or inf values
                        numeric_data.append(delay_value)
                except (ValueError, TypeError, IndexError):
                    # Skip invalid values
                    continue

            if len(numeric_data) == 0:
                print(f"Warning: No valid numeric delay data found for {airports[code]} ({code})")
                continue

            delay_data[code] = np.array(numeric_data)
            print(f"Extracted {len(numeric_data)} valid delay values for {airports[code]} ({code})")
        except FileNotFoundError:
            print(f"Warning: Data file for airport {code} not found")
            continue

    all_stats = {}
    test_results = {}
    normal_params = {}
    fit_qualities = {}

    # Create a grid of subplots for distribution plots - dynamically calculated based on number of airports
    n_airports = len(delay_data)
    n_cols = 4  # 4 columns in the grid
    n_rows = (n_airports + n_cols - 1) // n_cols  # Calculate how many rows we need

    fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes_dist = axes_dist.flatten() if n_airports > 1 else [axes_dist]

    # Create grid for Q-Q plots
    fig_qq, axes_qq = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes_qq = axes_qq.flatten() if n_airports > 1 else [axes_qq]

    # Process each airport's data
    for i, (code, name) in enumerate(airports.items()):
        if code in delay_data:
            print(f"\nAnalyzing delays for {name} ({code})...")

            # Fit normal distribution and plot
            if i < len(axes_dist):  # Make sure we don't exceed the number of subplot axes
                mu, sigma = fit_and_plot_normal(delay_data[code], name, axes_dist[i])
                normal_params[code] = {'mu': mu, 'sigma': sigma}

                # Create Q-Q plot
                stats.probplot(delay_data[code], dist="norm", plot=axes_qq[i])
                axes_qq[i].set_title(f"Q-Q Plot: {name}", fontsize=14, fontweight='bold')

                # Calculate descriptive statistics
                all_stats[code] = calculate_statistics(delay_data[code])

                # Conduct normality tests
                test_results[code] = test_normality(delay_data[code])

                # Calculate goodness-of-fit metrics
                fit_qualities[code] = calculate_fit_quality(delay_data[code], mu, sigma)

                print(f"Mean: {all_stats[code]['Mean']:.2f} minutes")
                print(f"Standard Deviation: {all_stats[code]['Std Dev']:.2f} minutes")
                print(f"Goodness-of-Fit (R²): {fit_qualities[code]['R²']:.4f}")
                print(f"Mean Squared Error (MSE): {fit_qualities[code]['MSE']:.4f}")
                print(f"Root Mean Squared Error (RMSE): {fit_qualities[code]['RMSE']:.4f}")
            else:
                print(f"Warning: Too many airports to visualize all plots for {name}")

    for j in range(i+1, len(axes_dist)):
        axes_dist[j].set_visible(False)
    for j in range(i+1, len(axes_qq)):
        axes_qq[j].set_visible(False)

    plt.tight_layout()
    fig_dist.savefig(f"{OUTPUT_DIR}/delay_distribution_comparison.png", dpi=300)
    print(f"\nSaved delay distribution visualization to {OUTPUT_DIR}/delay_distribution_comparison.png")

    plt.tight_layout()
    fig_qq.savefig(f"{OUTPUT_DIR}/delay_qq_plots_comparison.png", dpi=300)
    print(f"\nSaved Q-Q plots visualization to {OUTPUT_DIR}/delay_qq_plots_comparison.png")

    print("\n===== NORMAL DISTRIBUTION PARAMETERS =====")
    params_table = []
    headers = ["Airport", "Mean (μ)", "Std Dev (σ)", "R²", "RMSE", "KS Statistic"]
    for code, name in airports.items():
        if code in normal_params:
            params_table.append([
                name,
                f"{normal_params[code]['mu']:.2f}",
                f"{normal_params[code]['sigma']:.2f}",
                f"{fit_qualities[code]['R²']:.4f}",
                f"{fit_qualities[code]['RMSE']:.4f}",
                f"{fit_qualities[code]['KS']:.4f}"
            ])
    print(tabulate(params_table, headers=headers, tablefmt="grid"))

    params_df = pd.DataFrame({
        'Airport': [airports[code] for code in normal_params.keys()],
        'Code': list(normal_params.keys()),
        'Mean': [normal_params[code]['mu'] for code in normal_params.keys()],
        'StdDev': [normal_params[code]['sigma'] for code in normal_params.keys()],
        'R_squared': [fit_qualities[code]['R²'] for code in normal_params.keys()],
        'RMSE': [fit_qualities[code]['RMSE'] for code in normal_params.keys()],
        'KS': [fit_qualities[code]['KS'] for code in normal_params.keys()]
    })
    params_df.to_csv(f"{OUTPUT_DIR}/normal_distribution_parameters.csv", index=False)
    print(f"Saved normal distribution parameters to {OUTPUT_DIR}/normal_distribution_parameters.csv")

    region_mapping = {
        'Balkans': ['BKPR', 'LATI', 'LBBG', 'LBSF', 'LDDU', 'LDSP', 'LDZA', 'LQSA', 'LWSK', 'LYTV'],
        'Western Europe': ['EDDF', 'EDDM', 'EGKK', 'EGLL', 'EHAM', 'EIDW', 'LEBL', 'LEMD', 'LFPG', 'LIRF']
    }

    balkan_data = []
    western_data = []

    for code in normal_params.keys():
        entry = {
            'Airport': airports[code],
            'Code': code,
            'Mean': normal_params[code]['mu'],
            'StdDev': normal_params[code]['sigma'],
            'R²': fit_qualities[code]['R²'],
            'RMSE': fit_qualities[code]['RMSE']
        }

        if code in region_mapping['Balkans']:
            balkan_data.append(entry)
        else:
            western_data.append(entry)

    balkan_df = pd.DataFrame(balkan_data)
    western_df = pd.DataFrame(western_data)

    balkan_df = balkan_df.sort_values(by='Mean')
    western_df = western_df.sort_values(by='Mean')

    plt.figure(figsize=(14, 8))

    bar_width = 0.4
    indices_balkan = np.arange(len(balkan_df))
    indices_western = np.arange(len(western_df))

    plt.subplot(2, 1, 1)
    balkan_bars = plt.bar(indices_balkan, balkan_df['Mean'], bar_width,
                         label='Balkan Airports', color='tab:blue', alpha=0.8)
    plt.xlabel('')
    plt.ylabel('Mean Delay (minutes)', fontsize=12)
    plt.title('Mean Delay by Airport (Balkan Region)', fontsize=14, fontweight='bold')
    plt.xticks(indices_balkan, balkan_df['Airport'], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels
    for i, bar in enumerate(balkan_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 1, 2)
    western_bars = plt.bar(indices_western, western_df['Mean'], bar_width,
                         label='Western European Airports', color='tab:red', alpha=0.8)
    plt.xlabel('Airport', fontsize=12)
    plt.ylabel('Mean Delay (minutes)', fontsize=12)
    plt.title('Mean Delay by Airport (Western Europe)', fontsize=14, fontweight='bold')
    plt.xticks(indices_western, western_df['Airport'], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels
    for i, bar in enumerate(western_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mean_delay_by_region.png", dpi=300)
    print(f"\nSaved mean delay comparison by region chart to {OUTPUT_DIR}/mean_delay_by_region.png")

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    balkan_bars = plt.bar(indices_balkan, balkan_df['StdDev'], bar_width,
                          label='Balkan Airports', color='tab:green', alpha=0.8)
    plt.xlabel('')
    plt.ylabel('Standard Deviation (minutes)', fontsize=12)
    plt.title('Delay Standard Deviation by Airport (Balkan Region)', fontsize=14, fontweight='bold')
    plt.xticks(indices_balkan, balkan_df['Airport'], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels
    for i, bar in enumerate(balkan_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 1, 2)
    western_bars = plt.bar(indices_western, western_df['StdDev'], bar_width,
                           label='Western European Airports', color='tab:purple', alpha=0.8)
    plt.xlabel('Airport', fontsize=12)
    plt.ylabel('Standard Deviation (minutes)', fontsize=12)
    plt.title('Delay Standard Deviation by Airport (Western Europe)', fontsize=14, fontweight='bold')
    plt.xticks(indices_western, western_df['Airport'], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(western_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/std_dev_by_region.png", dpi=300)
    print(f"Saved standard deviation comparison by region chart to {OUTPUT_DIR}/std_dev_by_region.png")

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    balkan_bars = plt.bar(indices_balkan, balkan_df['R²'], bar_width,
                         color='darkgreen', alpha=0.8)
    plt.xlabel('')
    plt.ylabel('R² Value', fontsize=12)
    plt.title('Normal Distribution Goodness-of-Fit (Balkan Region)', fontsize=14, fontweight='bold')
    plt.xticks(indices_balkan, balkan_df['Airport'], rotation=45, ha='right')
    plt.ylim(0.92, 1.0)  # Adjust y-axis to focus on the high R² values
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(balkan_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 1, 2)
    western_bars = plt.bar(indices_western, western_df['R²'], bar_width,
                          color='darkmagenta', alpha=0.8)
    plt.xlabel('Airport', fontsize=12)
    plt.ylabel('R² Value', fontsize=12)
    plt.title('Normal Distribution Goodness-of-Fit (Western Europe)', fontsize=14, fontweight='bold')
    plt.xticks(indices_western, western_df['Airport'], rotation=45, ha='right')
    plt.ylim(0.92, 1.0)  # Adjust y-axis to focus on the high R² values
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(western_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/r_squared_by_region.png", dpi=300)
    print(f"Saved goodness-of-fit comparison by region chart to {OUTPUT_DIR}/r_squared_by_region.png")

    print("\n===== DESCRIPTIVE STATISTICS =====")
    stats_table = []
    headers = ["Airport", "Mean", "Std Dev", "Skewness", "Kurtosis"]
    for code, name in airports.items():
        if code in all_stats:
            stats_table.append([
                name,
                f"{all_stats[code]['Mean']:.2f}",
                f"{all_stats[code]['Std Dev']:.2f}",
                f"{all_stats[code]['Skewness']:.2f}",
                f"{all_stats[code]['Kurtosis']:.2f}"
            ])
    print(tabulate(stats_table, headers=headers, tablefmt="grid"))

    print("\n===== NORMALITY TEST RESULTS =====")
    normality_table = []
    headers = ["Airport", "Shapiro-Wilk Stat", "Shapiro-Wilk p-value", "K-S Stat", "K-S p-value"]
    for code, name in airports.items():
        if code in test_results:
            normality_table.append([
                name,
                f"{test_results[code]['Shapiro-Wilk']['Statistic']:.4f}",
                f"{test_results[code]['Shapiro-Wilk']['p-value']:.8f}",
                f"{test_results[code]['Kolmogorov-Smirnov']['Statistic']:.4f}",
                f"{test_results[code]['Kolmogorov-Smirnov']['p-value']:.8f}"
            ])
    print(tabulate(normality_table, headers=headers, tablefmt="grid"))


    # Create CDF comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define region groups with different line styles/colors
    regions = {
        'Balkans': ['BKPR', 'LATI', 'LBBG', 'LBSF', 'LDDU', 'LDSP', 'LDZA', 'LQSA', 'LWSK', 'LYTV'],
        'Western Europe': ['EDDF', 'EDDM', 'EGKK', 'EGLL', 'EHAM', 'EIDW', 'LEBL', 'LEMD', 'LFPG', 'LIRF']
    }

    # Use different colors for different regions
    colors = {'Balkans': 'tab:blue', 'Western Europe': 'tab:red'}

    # Place legend outside of plot to avoid overcrowding
    balkan_line = None
    western_line = None

    for code, name in airports.items():
        if code in delay_data:
            # Sort data for empirical CDF
            sorted_data = np.sort(delay_data[code])
            # Calculate empirical CDF
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            # Determine region
            region = 'Western Europe'  # default
            for r, codes in regions.items():
                if code in codes:
                    region = r
                    break

            # Plot with thinner lines and region-based color
            line = ax.step(sorted_data, y, label=name,
                          linewidth=1.0 if region == 'Western Europe' else 1.5,
                          alpha=0.7 if region == 'Western Europe' else 0.8,
                          color=colors[region])

            # Keep track of one line from each region for the legend
            if region == 'Balkans' and balkan_line is None:
                balkan_line = line
            elif region == 'Western Europe' and western_line is None:
                western_line = line

    ax.set_title('Empirical Cumulative Distribution Functions (ECDFs) of Delays', fontsize=16, fontweight='bold')
    ax.set_xlabel('Delay (minutes)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)

    # Create custom legend for regions only, not individual airports
    custom_lines = []
    custom_labels = []

    if balkan_line is not None:
        custom_lines.append(plt.Line2D([0], [0], color=colors['Balkans'], lw=2))
        custom_labels.append('Balkan Airports')
    if western_line is not None:
        custom_lines.append(plt.Line2D([0], [0], color=colors['Western Europe'], lw=2))
        custom_labels.append('Western European Airports')

    ax.legend(custom_lines, custom_labels, fontsize=12, loc='best')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/delay_cdf_comparison.png", dpi=300)
    print(f"\nSaved CDF comparison visualization to {OUTPUT_DIR}/delay_cdf_comparison.png")

    print("\n===== CONCLUSION =====\n")

    skopje_normal = test_results.get('LWSK', {}).get('Shapiro-Wilk', {}).get('p-value', 0) > 0.05
    major_normal_count = sum(1 for code in ['EDDF', 'LEMD', 'LFPG'] if code in test_results and
                             test_results[code]['Shapiro-Wilk']['p-value'] > 0.05)

    print("STATISTICAL EVIDENCE SUMMARY:")
    print(
        f"1. According to the Shapiro-Wilk test, Skopje airport delays {'do' if skopje_normal else 'do not'} follow a normal distribution.")
    print(
        f"2. Among the three major airports analyzed, {major_normal_count} out of 3 have delays that follow a normal distribution.")

    # Add analysis of fit quality metrics
    print("\nNORMAL DISTRIBUTION FIT QUALITY:")
    if 'LWSK' in fit_qualities:
        skopje_r2 = fit_qualities['LWSK']['R²']
        major_r2_avg = np.mean([fit_qualities[code]['R²'] for code in ['EDDF', 'LEMD', 'LFPG'] if code in fit_qualities])

        print(f"1. Skopje's normal fit has an R² value of {skopje_r2:.4f}, while major airports average {major_r2_avg:.4f}.")
        if skopje_r2 > major_r2_avg:
            print("   This indicates the normal distribution explains more of the variation in Skopje's delays")
            print("   compared to major European airports.")
        else:
            print("   This indicates the normal distribution explains less of the variation in Skopje's delays")
            print("   compared to major European airports.")

        skopje_rmse = fit_qualities['LWSK']['RMSE']
        major_rmse_avg = np.mean([fit_qualities[code]['RMSE'] for code in ['EDDF', 'LEMD', 'LFPG'] if code in fit_qualities])

        print(f"2. Skopje's fit has a RMSE of {skopje_rmse:.4f}, while major airports average {major_rmse_avg:.4f}.")
        if skopje_rmse < major_rmse_avg:
            print("   This suggests a better fit for Skopje with fewer prediction errors compared to major airports.")
        else:
            print("   This suggests a worse fit for Skopje with more prediction errors compared to major airports.")

    if 'LWSK' in all_stats and all(['EDDF' in all_stats, 'LEMD' in all_stats, 'LFPG' in all_stats]):
        skopje_skew = all_stats['LWSK']['Skewness']
        major_skew_avg = np.mean(
            [all_stats['EDDF']['Skewness'], all_stats['LEMD']['Skewness'], all_stats['LFPG']['Skewness']])

        print("\nCOMPARATIVE ANALYSIS:")
        print(
            f"1. Skopje's delay distribution has a skewness of {skopje_skew:.2f}, while major airports average {major_skew_avg:.2f}.")

        if abs(skopje_skew) < abs(major_skew_avg):
            print("   This indicates Skopje's delay distribution is more symmetric than major airports on average.")
        else:
            print("   This indicates Skopje's delay distribution is less symmetric than major airports on average.")

        skopje_kurt = all_stats['LWSK']['Kurtosis']
        major_kurt_avg = np.mean(
            [all_stats['EDDF']['Kurtosis'], all_stats['LEMD']['Kurtosis'], all_stats['LFPG']['Kurtosis']])

        print(
            f"2. Skopje's kurtosis is {skopje_kurt:.2f}, compared to major airports' average of {major_kurt_avg:.2f}.")
        if skopje_kurt > major_kurt_avg:
            print("   This suggests Skopje has more extreme delays (heavier tails) than major airports.")
        else:
            print("   This suggests Skopje has fewer extreme delays (lighter tails) than major airports.")

    print("\nNORMAL DISTRIBUTION PARAMETERS COMPARISON:")
    if 'LWSK' in normal_params:
        skopje_mu = normal_params['LWSK']['mu']
        major_mu_avg = np.mean([normal_params[code]['mu'] for code in ['EDDF', 'LEMD', 'LFPG'] if code in normal_params])

        print(f"1. Skopje's fitted normal distribution has a mean of {skopje_mu:.2f} minutes,")
        print(f"   while major airports average {major_mu_avg:.2f} minutes.")

        if skopje_mu < major_mu_avg:
            print("   This suggests Skopje tends to have shorter delays on average than major European airports.")
        else:
            print("   This suggests Skopje tends to have longer delays on average than major European airports.")

        skopje_sigma = normal_params['LWSK']['sigma']
        major_sigma_avg = np.mean([normal_params[code]['sigma'] for code in ['EDDF', 'LEMD', 'LFPG'] if code in normal_params])

        print(f"2. Skopje's fitted standard deviation is {skopje_sigma:.2f} minutes,")
        print(f"   while major airports average {major_sigma_avg:.2f} minutes.")

        if skopje_sigma < major_sigma_avg:
            print("   This indicates Skopje has more consistent and predictable delay patterns")
            print("   compared to the more variable delays at major European airports.")
        else:
            print("   This indicates Skopje has less consistent and less predictable delay patterns")
            print("   compared to the more stable delays at major European airports.")

    print("\nPOSSIBLE REASONS FOR DIFFERENCES:")
    print("1. Traffic Volume: Major airports handle significantly more flights, which may lead to")
    print("   more complex delay patterns and different statistical distributions.")
    print("2. Infrastructure: Larger airports have more runways and facilities, possibly affecting")
    print("   how delays are distributed throughout operations.")
    print("3. Scheduling Practices: Smaller airports like Skopje may have different scheduling")
    print("   buffers compared to major hubs with congested schedules.")
    print("4. Weather Impact: Different geographical locations experience different weather patterns,")
    print("   potentially affecting the symmetry and extremity of delays.")
    print("5. Air Traffic Management: Major European hubs operate under different air traffic control")
    print("   constraints compared to Balkan regional airports.")

    print("\nFINAL ASSESSMENT:")
    if skopje_normal and major_normal_count >= 2:
        print("The normal distribution appears to be a reasonable fit for both Skopje and major European")
        print("airports, though with different parameters reflecting their operational differences.")

        if 'LWSK' in fit_qualities:
            best_airport = max(fit_qualities.items(), key=lambda x: x[1]['R²'])
            print(f"The best normal distribution fit is observed for {airports[best_airport[0]]}")
            print(f"with an R² value of {best_airport[1]['R²']:.4f}.")
    elif skopje_normal and major_normal_count < 2:
        print("Interestingly, Skopje's delays fit a normal distribution better than most major European")
        print("airports analyzed, possibly due to its more predictable and less complex operations.")
    elif not skopje_normal and major_normal_count >= 2:
        print("Unlike the major European airports, Skopje's delays do not follow a normal distribution,")
        print("suggesting unique operational characteristics at this Balkan regional airport.")
    else:
        print("Neither Skopje nor major European airports show delays that follow a normal distribution,")
        print("though the specific patterns and reasons for non-normality differ between them.")

        if 'LWSK' in fit_qualities:
            print("\nDespite the statistical rejection of normality (likely due to large sample sizes),")
            print("the normal distribution may still be a useful approximation. The R² values indicate")
            print(f"that normal distributions explain {fit_qualities['LWSK']['R²']:.1%} of the variation")
            print("in Skopje's delays and similar proportions for major airports.")

    print("\n===== SUGGESTIONS FOR FURTHER ANALYSIS =====")
    print("1. Consider testing additional distribution types (e.g., log-normal, exponential)")
    print("   that might better capture the observed delay patterns.")
    print("2. Segment the analysis by time of day, day of week, or season to identify")
    print("   specific patterns in the delay distributions.")
    print("3. Incorporate weather data to explore correlations between weather events")
    print("   and deviations from normal distribution patterns.")
    print("4. Compare results with additional Balkan regional airports to establish")
    print("   whether Skopje's patterns are typical for the region.")


if __name__ == "__main__":
    main()
