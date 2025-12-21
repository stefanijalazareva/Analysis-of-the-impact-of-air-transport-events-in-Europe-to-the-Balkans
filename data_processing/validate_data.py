import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def convert_timestamp(ts):
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))

def load_and_validate_data():
    """Load and validate the airport delay data."""
    raw_data_dir = os.path.join('data', 'RawData')

    # Check if the directory exists
    if not os.path.exists(raw_data_dir):
        print(f"Directory {raw_data_dir} not found. Please run download_data.py first.")
        return

    # List of expected airports based on the email
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']
    all_airports = europe_airports + balkans_airports

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

    summary_data = []

    # Check available files
    files = os.listdir(raw_data_dir)
    delay_files = [f for f in files if f.startswith('Delays_') and f.endswith('.npy')]

    print(f"Found {len(delay_files)} delay files in {raw_data_dir}")

    # Validate and summarize each file
    for airport in all_airports:
        filename = f"Delays_{airport}.npy"
        filepath = os.path.join(raw_data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: Data for {airport_names.get(airport, airport)} ({filename}) not found.")
            continue

        try:
            # Load the data
            data = np.load(filepath, allow_pickle=True)

            # Basic validation
            if data.shape[1] != 4:
                print(f"Warning: Data format incorrect for {airport}. Expected 4 columns, got {data.shape[1]}")
                continue

            # Check if destination matches the file name
            if not all(row[1] == airport for row in data):
                mismatched = sum(1 for row in data if row[1] != airport)
                print(f"Warning: {mismatched} flights in {airport} file have mismatched destination.")

            # Calculate date range
            timestamps = [float(row[2]) for row in data]
            start_date = convert_timestamp(min(timestamps))
            end_date = convert_timestamp(max(timestamps))

            # Calculate delay statistics
            delays = [float(row[3]) for row in data]
            avg_delay = sum(delays) / len(delays)

            # Count positive and negative delays
            positive_delays = sum(1 for d in delays if d > 0)
            negative_delays = sum(1 for d in delays if d < 0)
            on_time = sum(1 for d in delays if d == 0)

            # Add to summary
            summary_data.append({
                'Airport Code': airport,
                'Airport Name': airport_names.get(airport, airport),
                'Total Flights': len(data),
                'Date Range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'Average Delay (min)': round(avg_delay / 60, 2),
                'Delayed Flights': positive_delays,
                'Early Flights': negative_delays,
                'On-time Flights': on_time
            })

            print(f"Successfully validated data for {airport_names.get(airport, airport)} ({airport}):")
            print(f"  - {len(data)} flights from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"  - Average delay: {round(avg_delay / 60, 2)} minutes")
            print(f"  - {positive_delays} delayed flights, {negative_delays} early arrivals, {on_time} on-time")

        except Exception as e:
            print(f"Error processing data for {airport}: {e}")

    if summary_data:
        # Convert to DataFrame for easier analysis
        summary_df = pd.DataFrame(summary_data)

        # Save summary to file
        summary_path = os.path.join('data', 'Analysis', 'data_validation_summary.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Plot some basic statistics
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Sort by total flights
        plot_data = summary_df.sort_values('Total Flights', ascending=False)

        # Plot total flights
        axs[0].bar(plot_data['Airport Code'], plot_data['Total Flights'])
        axs[0].set_title('Total Flights by Airport')
        axs[0].set_ylabel('Number of Flights')
        axs[0].tick_params(axis='x', rotation=90)

        # Plot average delays
        axs[1].bar(plot_data['Airport Code'], plot_data['Average Delay (min)'])
        axs[1].set_title('Average Delay by Airport (minutes)')
        axs[1].set_ylabel('Delay (minutes)')
        axs[1].tick_params(axis='x', rotation=90)
        axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join('data', 'Analysis', 'airports_overview.png'))
        print(f"Overview plot saved to {os.path.join('data', 'Analysis', 'airports_overview.png')}")

        return summary_df

    return None

if __name__ == "__main__":
    summary = load_and_validate_data()
    if summary is not None:
        # Display the top 5 airports by flight volume
        print("\nTop 5 airports by flight volume:")
        print(summary.sort_values('Total Flights', ascending=False).head(5)[['Airport Code', 'Airport Name', 'Total Flights']])
