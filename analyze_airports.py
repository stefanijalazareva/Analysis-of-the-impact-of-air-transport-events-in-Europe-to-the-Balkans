import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats

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
    df['Year'] = df['ScheduledTime'].dt.year
    df['Month'] = df['ScheduledTime'].dt.month
    df['Day'] = df['ScheduledTime'].dt.day
    df['DayOfWeek'] = df['ScheduledTime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Hour'] = df['ScheduledTime'].dt.hour

    # Add delay categories
    df['DelayCategory'] = pd.cut(
        df['Delay']/60,  # Convert to minutes
        bins=[-float('inf'), -15, -5, 5, 15, 30, 60, float('inf')],
        labels=['Very Early (>15m)', 'Early (5-15m)', 'On Time (±5m)',
                'Slight Delay (5-15m)', 'Moderate Delay (15-30m)',
                'Significant Delay (30-60m)', 'Severe Delay (>60m)']
    )

    # Add delay status
    df['DelayStatus'] = np.where(df['Delay'] > 0, 'Delayed', np.where(df['Delay'] < 0, 'Early', 'On Time'))

    return df

def analyze_airports(airport_codes=None, output_dir='data/Analysis'):
    """Analyze airport delay data and generate visualizations."""
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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Store DataFrames for each airport
    airport_dfs = {}
    airport_summaries = []

    # Load and process data for each airport
    for code in airport_codes:
        print(f"Processing {airport_names.get(code, code)} ({code})...")
        df = load_airport_data(code)
        if df is not None:
            airport_dfs[code] = df

            # Calculate summary statistics
            total_flights = len(df)
            avg_delay = df['Delay'].mean() / 60  # minutes
            median_delay = df['Delay'].median() / 60  # minutes
            delay_std = df['Delay'].std() / 60  # minutes
            pct_delayed = (df['Delay'] > 0).mean() * 100
            pct_early = (df['Delay'] < 0).mean() * 100
            pct_on_time = (df['Delay'] == 0).mean() * 100

            # Find busiest months and days
            flights_by_month = df.groupby('Month').size()
            busiest_month = flights_by_month.idxmax()

            flights_by_dow = df.groupby('DayOfWeek').size()
            busiest_day = flights_by_dow.idxmax()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            busiest_day_name = day_names[busiest_day]

            # Add to summary
            airport_summaries.append({
                'Airport Code': code,
                'Airport Name': airport_names.get(code, code),
                'Region': 'Europe' if code in europe_airports else 'Balkans',
                'Total Flights': total_flights,
                'Average Delay (min)': round(avg_delay, 2),
                'Median Delay (min)': round(median_delay, 2),
                'Delay Std Dev (min)': round(delay_std, 2),
                'Delayed Flights (%)': round(pct_delayed, 2),
                'Early Flights (%)': round(pct_early, 2),
                'On-time Flights (%)': round(pct_on_time, 2),
                'Busiest Month': busiest_month,
                'Busiest Day': busiest_day_name
            })

            # Generate individual airport visualizations
            generate_airport_visualizations(df, code, airport_names.get(code, code), output_dir)

    # Create combined summary DataFrame
    summary_df = pd.DataFrame(airport_summaries)
    summary_df.to_csv(os.path.join(output_dir, 'airport_summary.csv'), index=False)

    # Generate cross-airport visualizations
    if len(airport_dfs) > 1:
        generate_comparative_visualizations(airport_dfs, airport_names, output_dir)

    return summary_df, airport_dfs

def generate_airport_visualizations(df, airport_code, airport_name, output_dir):
    """Generate visualizations for a single airport."""
    # Set style
    sns.set(style="whitegrid")

    # 1. Delay Distribution Histogram
    plt.figure(figsize=(12, 6))
    # Clip extreme values for better visualization
    delays_clipped = df['Delay'].clip(-3600, 3600) / 60  # Convert to minutes and clip ±60 minutes
    sns.histplot(delays_clipped, bins=50, kde=True)
    plt.title(f'Delay Distribution for {airport_name} ({airport_code})')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Number of Flights')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(os.path.join(output_dir, f'delay_hist_{airport_code}.png'))
    plt.close()

    # 2. Hourly Delay Patterns
    plt.figure(figsize=(12, 6))
    hourly_stats = df.groupby('Hour')['Delay'].agg(['mean', 'median', 'std']) / 60  # Convert to minutes
    hourly_stats['count'] = df.groupby('Hour').size()

    ax1 = plt.subplot(111)
    ax1.plot(hourly_stats.index, hourly_stats['mean'], 'o-', color='blue', label='Mean Delay')
    ax1.plot(hourly_stats.index, hourly_stats['median'], 'o-', color='green', label='Median Delay')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Delay (minutes)')
    ax1.set_xticks(range(24))
    ax1.axhline(y=0, color='r', linestyle='--')

    ax2 = ax1.twinx()
    ax2.bar(hourly_stats.index, hourly_stats['count'], alpha=0.2, color='gray', label='Flight Count')
    ax2.set_ylabel('Number of Flights')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Hourly Delay Patterns for {airport_name} ({airport_code})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hourly_delay_{airport_code}.png'))
    plt.close()

    # 3. Monthly patterns
    monthly_data = df.groupby(['Year', 'Month']).agg({
        'Delay': ['mean', 'median', 'count']
    })
    monthly_data.columns = ['Mean Delay', 'Median Delay', 'Flights']
    monthly_data['Mean Delay'] /= 60  # Convert to minutes
    monthly_data['Median Delay'] /= 60  # Convert to minutes

    # Create date index for better plotting
    dates = [datetime(year, month, 1) for year, month in monthly_data.index]
    monthly_data = monthly_data.reset_index()
    monthly_data['Date'] = dates

    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(111)
    ax1.plot(monthly_data['Date'], monthly_data['Mean Delay'], 'o-', color='blue', label='Mean Delay')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.set_ylabel('Delay (minutes)')
    ax1.axhline(y=0, color='r', linestyle='--')

    ax2 = ax1.twinx()
    ax2.bar(monthly_data['Date'], monthly_data['Flights'], alpha=0.2, color='gray', label='Flights')
    ax2.set_ylabel('Number of Flights')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Monthly Delay Trends for {airport_name} ({airport_code})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'monthly_trend_{airport_code}.png'))
    plt.close()

def generate_comparative_visualizations(airport_dfs, airport_names, output_dir):
    """Generate visualizations comparing multiple airports."""
    # Set style
    sns.set(style="whitegrid")

    # Prepare data for comparisons
    all_data = []
    for code, df in airport_dfs.items():
        # Sample for performance if datasets are very large
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        sample_df['Airport'] = code
        sample_df['Airport Name'] = airport_names.get(code, code)
        all_data.append(sample_df)

    combined_df = pd.concat(all_data)

    # 1. Boxplot comparison of delays by airport
    plt.figure(figsize=(15, 8))
    # Calculate mean delays for sorting
    mean_delays = {code: df['Delay'].mean() for code, df in airport_dfs.items()}
    airport_order = sorted(airport_dfs.keys(), key=lambda x: mean_delays[x], reverse=True)

    # Create boxplot with airports sorted by mean delay
    sns.boxplot(
        x='Airport',
        y=combined_df['Delay'].clip(-3600, 3600)/60,  # Convert to minutes, clip extreme values
        data=combined_df,
        order=airport_order
    )
    plt.title('Comparison of Delay Distributions Across Airports')
    plt.xlabel('Airport')
    plt.ylabel('Delay (minutes)')
    plt.xticks(rotation=90)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_comparison_boxplot.png'))
    plt.close()

    # 2. Day of week patterns
    dow_data = []
    for code, df in airport_dfs.items():
        daily_delays = df.groupby('DayOfWeek')['Delay'].mean() / 60  # Convert to minutes
        for day, delay in daily_delays.items():
            dow_data.append({
                'Airport': code,
                'Airport Name': airport_names.get(code, code),
                'Day of Week': day,
                'Day Name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day],
                'Mean Delay': delay
            })

    dow_df = pd.DataFrame(dow_data)
    dow_df.to_csv(os.path.join(output_dir, 'day_of_week_patterns.csv'), index=False)

    # Plot for selected major airports
    selected_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LBSF', 'LDZA', 'BKPR']  # Example selection
    selected_dow_df = dow_df[dow_df['Airport'].isin(selected_airports)]

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=selected_dow_df,
        x='Day Name',
        y='Mean Delay',
        hue='Airport Name',
        style='Airport Name',
        markers=True,
        dashes=False
    )
    plt.title('Average Delay by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Mean Delay (minutes)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'day_of_week_comparison.png'))
    plt.close()

    # 3. Hour patterns
    hour_data = []
    for code, df in airport_dfs.items():
        hourly_delays = df.groupby('Hour')['Delay'].mean() / 60  # Convert to minutes
        for hour, delay in hourly_delays.items():
            hour_data.append({
                'Airport': code,
                'Airport Name': airport_names.get(code, code),
                'Hour': hour,
                'Mean Delay': delay
            })

    hour_df = pd.DataFrame(hour_data)
    hour_df.to_csv(os.path.join(output_dir, 'hour_patterns.csv'), index=False)

    # Plot for selected major airports
    selected_hour_df = hour_df[hour_df['Airport'].isin(selected_airports)]

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=selected_hour_df,
        x='Hour',
        y='Mean Delay',
        hue='Airport Name',
        style='Airport Name',
        markers=True,
        dashes=False
    )
    plt.title('Average Delay by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Mean Delay (minutes)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hour_comparison.png'))
    plt.close()

    # 4. Month patterns
    month_data = []
    for code, df in airport_dfs.items():
        monthly_delays = df.groupby('Month')['Delay'].mean() / 60  # Convert to minutes
        for month, delay in monthly_delays.items():
            month_data.append({
                'Airport': code,
                'Airport Name': airport_names.get(code, code),
                'Month': month,
                'Mean Delay': delay
            })

    month_df = pd.DataFrame(month_data)
    month_df.to_csv(os.path.join(output_dir, 'month_patterns.csv'), index=False)

    # Plot for selected major airports
    selected_month_df = month_df[month_df['Airport'].isin(selected_airports)]

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=selected_month_df,
        x='Month',
        y='Mean Delay',
        hue='Airport Name',
        style='Airport Name',
        markers=True,
        dashes=False
    )
    plt.title('Average Delay by Month')
    plt.xlabel('Month')
    plt.ylabel('Mean Delay (minutes)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'month_comparison.png'))
    plt.close()

    # 5. Create a temporal summary
    temporal_summary = pd.DataFrame({
        'Monthly Variation': month_df.groupby('Airport')['Mean Delay'].std(),
        'Daily Variation': dow_df.groupby('Airport')['Mean Delay'].std(),
        'Hourly Variation': hour_df.groupby('Airport')['Mean Delay'].std()
    }).reset_index()

    # Add airport names
    temporal_summary['Airport Name'] = temporal_summary['Airport'].map(airport_names)
    temporal_summary.to_csv(os.path.join(output_dir, 'temporal_summary.csv'), index=False)

    # 6. Traffic vs. Delay scatter plot
    traffic_vs_delay = []
    for code, df in airport_dfs.items():
        traffic_vs_delay.append({
            'Airport': code,
            'Airport Name': airport_names.get(code, code),
            'Total Flights': len(df),
            'Mean Delay': df['Delay'].mean() / 60  # Convert to minutes
        })

    traffic_delay_df = pd.DataFrame(traffic_vs_delay)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=traffic_delay_df,
        x='Total Flights',
        y='Mean Delay',
        s=100,  # Marker size
        alpha=0.7
    )

    # Add text labels for each airport
    for _, row in traffic_delay_df.iterrows():
        plt.text(row['Total Flights'] * 1.02, row['Mean Delay'], row['Airport'],
                fontsize=9, alpha=0.8)

    plt.title('Relationship Between Airport Traffic and Average Delay')
    plt.xlabel('Number of Flights')
    plt.ylabel('Average Delay (minutes)')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'traffic_vs_delay.png'))
    plt.close()

if __name__ == "__main__":
    summary, _ = analyze_airports()
    print("Analysis complete. Check the data/Analysis directory for results.")
