import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from datetime import datetime, timedelta

def convert_timestamp(ts):
    """Convert Unix timestamp to datetime."""
    return datetime.fromtimestamp(float(ts))

def load_all_airports():
    """Load data for all airports and create a combined DataFrame."""
    # Define airport groups
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

    all_data = []

    for airport in all_airports:
        try:
            filepath = os.path.join('data', 'RawData', f'Delays_{airport}.npy')
            data = np.load(filepath, allow_pickle=True)

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['Origin', 'Destination', 'ScheduledTimestamp', 'Delay'])

            # Convert timestamp and delay to appropriate types
            df['ScheduledTimestamp'] = df['ScheduledTimestamp'].astype(float)
            df['Delay'] = df['Delay'].astype(float)

            # Add datetime column
            df['ScheduledTime'] = df['ScheduledTimestamp'].apply(convert_timestamp)
            df['Year'] = df['ScheduledTime'].dt.year
            df['Month'] = df['ScheduledTime'].dt.month
            df['Day'] = df['ScheduledTime'].dt.day

            # Add airport group labels
            df['DestGroup'] = 'Europe' if airport in europe_airports else 'Balkans'
            df['OriginGroup'] = df['Origin'].apply(
                lambda x: 'Europe' if x in europe_airports else
                         ('Balkans' if x in balkans_airports else 'Other')
            )

            all_data.append(df)

        except Exception as e:
            print(f"Error loading data for {airport}: {e}")

    combined_df = pd.concat(all_data)

    # Add airport names
    combined_df['DestName'] = combined_df['Destination'].map(airport_names)
    combined_df['OriginName'] = combined_df['Origin'].map(
        lambda x: airport_names.get(x, 'Unknown')
    )

    return combined_df, airport_names

def build_delay_network(df, min_flights=100, output_dir='data/NetworkAnalysis'):
    """Build and analyze a network of delay correlations between airports."""
    os.makedirs(output_dir, exist_ok=True)

    # Count flights between airports
    route_counts = df.groupby(['Origin', 'Destination']).size()

    # Filter significant routes (with enough flights)
    significant_routes = route_counts[route_counts >= min_flights].reset_index()
    significant_routes.columns = ['Origin', 'Destination', 'Flights']
    significant_routes.to_csv(os.path.join(output_dir, 'significant_routes.csv'), index=False)

    print(f"Found {len(significant_routes)} significant routes with at least {min_flights} flights")

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with flight count as weight
    for _, row in significant_routes.iterrows():
        G.add_edge(row['Origin'], row['Destination'], weight=row['Flights'])

    # Calculate network metrics
    print("Calculating network metrics...")
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Create a DataFrame with network metrics
    metrics_df = pd.DataFrame({
        'Airport': list(degree_centrality.keys()),
        'Degree Centrality': list(degree_centrality.values()),
        'In-Degree Centrality': [in_degree_centrality.get(k, 0) for k in degree_centrality.keys()],
        'Out-Degree Centrality': [out_degree_centrality.get(k, 0) for k in degree_centrality.keys()],
        'Betweenness Centrality': [betweenness_centrality.get(k, 0) for k in degree_centrality.keys()]
    })

    metrics_df.to_csv(os.path.join(output_dir, 'airport_network_metrics.csv'), index=False)

    # Visualize the network
    print("Generating network visualization...")
    plt.figure(figsize=(16, 12))

    # Node positions using Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G)

    # Get unique airports in Europe and Balkans
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']
    other_airports = [n for n in G.nodes if n not in europe_airports and n not in balkans_airports]

    # Get edge weights for line thickness
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    # Normalize edge thickness
    edge_widths = [w / max_weight * 5 for w in edge_weights]

    # Draw nodes by group
    nx.draw_networkx_nodes(G, pos, nodelist=europe_airports, node_color='blue',
                          node_size=800, alpha=0.8, label='Major European Airports')
    nx.draw_networkx_nodes(G, pos, nodelist=balkans_airports, node_color='green',
                          node_size=600, alpha=0.8, label='Balkan Airports')
    nx.draw_networkx_nodes(G, pos, nodelist=other_airports, node_color='gray',
                          node_size=400, alpha=0.6, label='Other Airports')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, arrows=True,
                          arrowsize=15, arrowstyle='->')

    # Draw labels with smaller font
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title('Airport Connection Network', fontsize=16)
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'airport_network.png'), dpi=300)
    plt.close()

    return G, metrics_df

def analyze_delay_propagation(df, output_dir='data/NetworkAnalysis'):
    """Analyze how delays propagate from origin to destination airports."""
    os.makedirs(output_dir, exist_ok=True)

    # Group data by origin airport and calculate average delay
    origin_delays = df.groupby('Origin')['Delay'].mean().reset_index()
    origin_delays.columns = ['Airport', 'Avg Origin Delay']

    # Group data by destination airport and calculate average delay
    dest_delays = df.groupby('Destination')['Delay'].mean().reset_index()
    dest_delays.columns = ['Airport', 'Avg Destination Delay']

    # Merge the two datasets
    delay_comparison = pd.merge(origin_delays, dest_delays, on='Airport', how='outer')
    delay_comparison.fillna(0, inplace=True)

    # Convert to minutes for better readability
    delay_comparison['Avg Origin Delay'] /= 60
    delay_comparison['Avg Destination Delay'] /= 60

    # Calculate correlation between origin and destination delays
    correlation = delay_comparison['Avg Origin Delay'].corr(delay_comparison['Avg Destination Delay'])
    print(f"Correlation between origin and destination delays: {correlation:.4f}")

    # Save the delay comparison data
    delay_comparison.to_csv(os.path.join(output_dir, 'delay_propagation.csv'), index=False)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=delay_comparison,
        x='Avg Origin Delay',
        y='Avg Destination Delay',
        s=80,
        alpha=0.7
    )

    # Add identity line
    min_val = min(delay_comparison['Avg Origin Delay'].min(), delay_comparison['Avg Destination Delay'].min())
    max_val = max(delay_comparison['Avg Origin Delay'].max(), delay_comparison['Avg Destination Delay'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    # Add labels for interesting airports
    for _, row in delay_comparison.iterrows():
        if abs(row['Avg Origin Delay'] - row['Avg Destination Delay']) > 3:  # Label airports with big differences
            plt.text(row['Avg Origin Delay'], row['Avg Destination Delay'], row['Airport'],
                    fontsize=9, alpha=0.8)

    plt.title(f'Delay Propagation Analysis (Correlation: {correlation:.4f})')
    plt.xlabel('Average Delay as Origin (minutes)')
    plt.ylabel('Average Delay as Destination (minutes)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_propagation.png'))
    plt.close()

    # Analyze regional delay propagation
    regional_delay_prop = df.groupby(['OriginGroup', 'DestGroup'])['Delay'].agg(['mean', 'median', 'count']).reset_index()
    regional_delay_prop['mean'] /= 60  # Convert to minutes
    regional_delay_prop['median'] /= 60  # Convert to minutes
    regional_delay_prop.to_csv(os.path.join(output_dir, 'regional_delay_propagation.csv'), index=False)

    # Create a bar chart for regional delay propagation
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=regional_delay_prop,
        x='OriginGroup',
        y='mean',
        hue='DestGroup',
        palette='viridis'
    )
    plt.title('Delay Propagation Between Regions')
    plt.xlabel('Origin Region')
    plt.ylabel('Average Delay (minutes)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_delay_propagation.png'))
    plt.close()

    return delay_comparison, regional_delay_prop

def analyze_delay_correlations(df, airport_names, output_dir='data/NetworkAnalysis'):
    """Analyze correlations between delays at different airports over time."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a daily average delay time series for each airport
    print("Computing daily average delays...")

    # Create a date column
    df['Date'] = df['ScheduledTime'].dt.date

    # Group by destination and date
    daily_delays = df.groupby(['Destination', 'Date'])['Delay'].mean().reset_index()

    # Pivot to get airports as columns and dates as rows
    delay_pivot = daily_delays.pivot(index='Date', columns='Destination', values='Delay')

    # Convert to minutes for better readability
    delay_pivot = delay_pivot / 60

    # Calculate correlation matrix
    corr_matrix = delay_pivot.corr(method='pearson')

    # Save the correlation matrix
    corr_matrix.to_csv(os.path.join(output_dir, 'delay_correlation_matrix.csv'))

    # Create a heatmap of correlations
    plt.figure(figsize=(16, 14))

    # Replace airport codes with names in the correlation matrix
    corr_matrix_named = corr_matrix.copy()
    corr_matrix_named.index = [f"{airport_names.get(idx, idx)}" for idx in corr_matrix_named.index]
    corr_matrix_named.columns = [f"{airport_names.get(col, col)}" for col in corr_matrix_named.columns]

    # Create the heatmap
    mask = np.triu(np.ones_like(corr_matrix_named, dtype=bool))  # Mask for upper triangle
    sns.heatmap(
        corr_matrix_named,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        fmt=".2f",
        annot_kws={"size": 8}
    )

    plt.title('Correlation of Daily Average Delays Between Airports', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_correlations.png'), dpi=300)
    plt.close()

    # Find the most correlated airport pairs
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            airport1 = corr_matrix.columns[i]
            airport2 = corr_matrix.columns[j]
            correlation = corr_matrix.iloc[i, j]
            if not np.isnan(correlation):  # Ignore NaN values
                corr_pairs.append((airport1, airport2, correlation))

    # Sort by absolute correlation value (descending)
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Create a dataframe with the most correlated pairs
    top_correlations = pd.DataFrame(
        corr_pairs[:20],  # Top 20 correlations
        columns=['Airport 1', 'Airport 2', 'Correlation']
    )

    # Add airport names
    top_correlations['Airport 1 Name'] = top_correlations['Airport 1'].map(airport_names)
    top_correlations['Airport 2 Name'] = top_correlations['Airport 2'].map(airport_names)

    # Save to CSV
    top_correlations.to_csv(os.path.join(output_dir, 'top_delay_correlations.csv'), index=False)

    print(f"Top correlations saved to {os.path.join(output_dir, 'top_delay_correlations.csv')}")

    # Create a network visualization of the top correlations
    plt.figure(figsize=(16, 12))

    # Create a graph
    G = nx.Graph()

    # Define node groups
    europe_airports = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
    balkans_airports = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

    # Add all airports as nodes
    for airport in set(top_correlations['Airport 1']).union(set(top_correlations['Airport 2'])):
        G.add_node(airport)

    # Add edges for top correlations
    for _, row in top_correlations.iterrows():
        G.add_edge(
            row['Airport 1'],
            row['Airport 2'],
            weight=abs(row['Correlation']),
            color='green' if row['Correlation'] > 0 else 'red'
        )

    # Define positions
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Get edge colors and widths
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n in europe_airports],
                          node_color='blue', node_size=700, alpha=0.8, label='Major European Airports')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n in balkans_airports],
                          node_color='green', node_size=500, alpha=0.8, label='Balkan Airports')
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if n not in europe_airports and n not in balkans_airports],
                          node_color='gray', node_size=300, alpha=0.6, label='Other Airports')

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=10)

    plt.title('Network of Highly Correlated Airport Delays', fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_network.png'), dpi=300)
    plt.close()

    return corr_matrix, top_correlations

if __name__ == "__main__":
    print("Loading airport data...")
    combined_df, airport_names = load_all_airports()
    print(f"Loaded data for {combined_df['Destination'].nunique()} destination airports")

    print("\nBuilding airport network...")
    G, metrics_df = build_delay_network(combined_df, min_flights=100)

    print("\nAnalyzing delay propagation...")
    delay_comparison, regional_delay_prop = analyze_delay_propagation(combined_df)

    print("\nAnalyzing delay correlations...")
    corr_matrix, top_correlations = analyze_delay_correlations(combined_df, airport_names)

    print("\nAnalysis complete. Check the data/NetworkAnalysis directory for results.")
