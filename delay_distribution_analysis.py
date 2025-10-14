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

    airports = {
        'LWSK': 'Skopje',
        'EDDF': 'Frankfurt',
        'LEMD': 'Madrid',  # Using Madrid instead of Munich as it's available in the data
        'LFPG': 'Paris'  # Using Paris instead of Heathrow as it's available in the data
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    all_stats = {}
    test_results = {}

    for i, (code, name) in enumerate(airports.items()):
        if code in delay_data:
            print(f"\nAnalyzing delays for {name} ({code})...")

            # Fit normal distribution and plot
            mu, sigma = fit_and_plot_normal(delay_data[code], name, axes[i])

            # Calculate descriptive statistics
            all_stats[code] = calculate_statistics(delay_data[code])

            # Conduct normality tests
            test_results[code] = test_normality(delay_data[code])

            print(f"Mean: {all_stats[code]['Mean']:.2f} minutes")
            print(f"Standard Deviation: {all_stats[code]['Std Dev']:.2f} minutes")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/delay_distribution_comparison.png", dpi=300)
    print(f"\nSaved delay distribution visualization to {OUTPUT_DIR}/delay_distribution_comparison.png")

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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (code, name) in enumerate(airports.items()):
        if code in delay_data:
            stats.probplot(delay_data[code], dist="norm", plot=axes[i])
            axes[i].set_title(f"Q-Q Plot: {name}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/delay_qq_plots_comparison.png", dpi=300)
    print(f"\nSaved Q-Q plots visualization to {OUTPUT_DIR}/delay_qq_plots_comparison.png")

    # Create CDF comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for code, name in airports.items():
        if code in delay_data:
            # Sort data for empirical CDF
            sorted_data = np.sort(delay_data[code])
            # Calculate empirical CDF
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            # Plot empirical CDF
            ax.step(sorted_data, y, label=name, linewidth=2)

    ax.set_title('Empirical Cumulative Distribution Functions (ECDFs) of Delays', fontsize=16, fontweight='bold')
    ax.set_xlabel('Delay (minutes)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.legend(fontsize=12)
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
    elif skopje_normal and major_normal_count < 2:
        print("Interestingly, Skopje's delays fit a normal distribution better than most major European")
        print("airports analyzed, possibly due to its more predictable and less complex operations.")
    elif not skopje_normal and major_normal_count >= 2:
        print("Unlike the major European airports, Skopje's delays do not follow a normal distribution,")
        print("suggesting unique operational characteristics at this Balkan regional airport.")
    else:
        print("Neither Skopje nor major European airports show delays that follow a normal distribution,")
        print("though the specific patterns and reasons for non-normality differ between them.")


if __name__ == "__main__":
    main()
