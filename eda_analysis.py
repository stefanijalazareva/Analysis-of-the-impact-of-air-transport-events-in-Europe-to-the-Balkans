"""
Exploratory Data Analysis (EDA) for Air Transport Delays

This script performs initial exploratory analysis on the cleaned delay data,
generating summary statistics and visualizations for each airport.

Outputs:
- Summary statistics CSV files
- Delay distribution histograms
- Time series plots of delays
- Traffic vs delay scatter plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from load_dataframe import load_cleaned_data

# Set style for better-looking plots
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme(style="whitegrid")  # Apply seaborn styling

# Create output directory for analysis results
output_dir = Path("data/Analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Load the cleaned data
print("Loading cleaned data...")
df = load_cleaned_data()

# 1. Generate summary statistics for each airport
print("\nGenerating airport summaries...")
summary = df.groupby('arr').agg(
    flights=('arr', 'count'),
    mean_delay_min=('delay_min', 'mean'),
    median_delay_min=('delay_min', 'median'),
    std_delay_min=('delay_min', 'std'),
    pct_delay_over_15=('delay_min', lambda x: (x > 15).mean() * 100)
).sort_values('flights', ascending=False)

# Save summary to CSV
summary.to_csv(output_dir / 'airport_summary.csv')
print(f"Airport summary saved to {output_dir / 'airport_summary.csv'}")

# 2. Generate flights per month/year summary
print("\nAnalyzing temporal patterns...")
df['year'] = df['sched_dt'].dt.year
df['month'] = df['sched_dt'].dt.month

temporal_summary = df.groupby(['year', 'month', 'arr']).agg(
    flights=('arr', 'count'),
    mean_delay=('delay_min', 'mean')
).reset_index()

# Save temporal summary
temporal_summary.to_csv(output_dir / 'temporal_summary.csv', index=False)
print(f"Temporal summary saved to {output_dir / 'temporal_summary.csv'}")

# 3. Generate visualizations
print("\nGenerating visualizations...")

# Delay distribution histogram for each airport
def plot_delay_histogram(data, airport):
    plt.figure(figsize=(10, 6))
    plt.hist(data['delay_min'].clip(-100, 300), bins=50, alpha=0.75)
    plt.title(f'Delay Distribution for {airport}')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'delay_hist_{airport}.png')
    plt.close()

# Time series of mean delay by hour
def plot_hourly_delays(data, airport):
    hourly = data.groupby(data['sched_dt'].dt.hour)['delay_min'].mean()
    plt.figure(figsize=(12, 6))
    hourly.plot(kind='line', marker='o')
    plt.title(f'Average Delay by Hour for {airport}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Delay (minutes)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'hourly_delay_{airport}.png')
    plt.close()

# Traffic vs Delay scatter plot
def plot_traffic_vs_delay(data):
    # Calculate daily traffic and mean delay for each airport
    daily_stats = data.groupby(['arr', data['sched_dt'].dt.date]).agg({
        'delay_min': 'mean',
        'delay_s': 'count'  # Using delay_s instead of arr for counting
    })
    daily_stats = daily_stats.reset_index()
    daily_stats.columns = ['airport', 'date', 'mean_delay', 'flights']

    plt.figure(figsize=(12, 8))
    for airport in daily_stats['airport'].unique():
        airport_data = daily_stats[daily_stats['airport'] == airport]
        plt.scatter(airport_data['flights'], airport_data['mean_delay'],
                   alpha=0.5, label=airport)

    plt.xlabel('Daily Flights')
    plt.ylabel('Mean Delay (minutes)')
    plt.title('Traffic vs Mean Delay by Airport')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'traffic_vs_delay.png')
    plt.close()

# Generate plots for each airport
for airport in df['arr'].unique():
    airport_data = df[df['arr'] == airport]
    print(f"Processing {airport}...")

    plot_delay_histogram(airport_data, airport)
    plot_hourly_delays(airport_data, airport)

# Generate overall traffic vs delay plot
plot_traffic_vs_delay(df)

print("\nAnalysis complete! Results saved in:", output_dir)

# Print some key findings
print("\nKey Findings:")
print("\nTop 5 airports by number of flights:")
print(summary.head().to_string())

print("\nTop 5 airports by mean delay:")
print(summary.sort_values('mean_delay_min', ascending=False).head().to_string())
