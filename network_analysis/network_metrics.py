"""
Network Metrics and Visualizations

This script analyzes the reconstructed networks from different connectivity measures,
calculates network metrics, and generates visualizations.

Metrics calculated:
- Network density
- Global efficiency
- Transitivity (clustering)
- Reciprocity
- Betweenness centrality
- Eigenvector centrality
- Identification of isolated nodes

Visualizations:
- Directed graphs with node size based on out-degree or centrality
- Edge colors based on lag values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_analysis.log'),
        logging.StreamHandler()
    ]
)

class NetworkMetricsAnalyzer:
    """Class for analyzing network metrics and creating visualizations."""

    def __init__(self, input_dir="data/NetworkAnalysis", output_dir="data/NetworkAnalysis/Metrics"):
        """Initialize the NetworkMetricsAnalyzer."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store network metrics for each connectivity measure
        self.network_metrics = {}
        self.node_metrics = {}
        self.graphs = {}

    def load_network_data(self, measure):
        """
        Load the network data for a specific connectivity measure.

        Args:
            measure: The connectivity measure (e.g., 'lc', 'gc', 'te')

        Returns:
            tuple: (weights, lags, significant, node_names)
        """
        try:
            # Load from NPZ file
            npz_path = self.input_dir / f'network_data_{measure}.npz'
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                return (data['weights'], data['lags'],
                        data['significant'], data['node_names'])

            # Alternative: Load from separate CSV files
            weights = pd.read_csv(self.input_dir / f'weights_{measure}.csv', index_col=0).values
            lags = pd.read_csv(self.input_dir / f'lags_{measure}.csv', index_col=0).values

            # Read node names from weights file
            node_names = pd.read_csv(self.input_dir / f'weights_{measure}.csv', index_col=0).index.tolist()

            # Read node statistics to infer significance
            node_stats = pd.read_csv(self.input_dir / f'node_statistics_{measure}.csv')

            # Create a significance matrix based on degrees (this is an approximation)
            n_nodes = len(node_names)
            significant = np.zeros((n_nodes, n_nodes), dtype=bool)

            # Use weights as a proxy for significance (lower p-value = more significant)
            threshold = 0.05  # Typical significance level
            significant = weights < threshold

            return weights, lags, significant, np.array(node_names)

        except Exception as e:
            logging.error(f"Error loading network data for {measure}: {str(e)}")
            return None, None, None, None

    def create_networkx_graph(self, weights, lags, significant, node_names):
        """
        Create a NetworkX directed graph from the network data.

        Args:
            weights: Weight matrix (p-values or connectivity strengths)
            lags: Lag matrix
            significant: Boolean mask of significant connections
            node_names: List of node names

        Returns:
            nx.DiGraph: Directed graph
        """
        # Create an empty directed graph
        G = nx.DiGraph()

        # Add nodes with names
        for i, name in enumerate(node_names):
            G.add_node(i, name=name)

        # Add edges for significant connections
        n_nodes = len(node_names)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if significant[i, j] and i != j:  # Exclude self-loops
                    # Convert p-value to strength (1-p for better visualization)
                    strength = 1.0 - weights[i, j]
                    G.add_edge(i, j, weight=strength, lag=lags[i, j])

        return G

    def calculate_network_metrics(self, G, measure_name):
        """
        Calculate network-level metrics for a graph.

        Args:
            G: NetworkX directed graph
            measure_name: Name of the connectivity measure

        Returns:
            dict: Dictionary of network metrics
        """
        metrics = {}

        # Basic metrics
        metrics['n_nodes'] = G.number_of_nodes()
        metrics['n_edges'] = G.number_of_edges()

        # Check if the graph is empty (no edges)
        is_empty = metrics['n_edges'] == 0

        # Density - defined for empty graphs as 0
        metrics['density'] = nx.density(G)

        # Transitivity (clustering coefficient) - set to 0 for empty graphs
        if is_empty:
            metrics['transitivity'] = 0.0
        else:
            metrics['transitivity'] = nx.transitivity(G)

        # Reciprocity - not defined for empty graphs
        if is_empty:
            metrics['reciprocity'] = 0.0
        else:
            try:
                metrics['reciprocity'] = nx.reciprocity(G)
            except Exception as e:
                logging.warning(f"Error calculating reciprocity: {str(e)}. Setting to 0.")
                metrics['reciprocity'] = 0.0

        # Global efficiency (handle disconnected graphs)
        if is_empty:
            metrics['global_efficiency'] = 0.0
        else:
            try:
                metrics['global_efficiency'] = nx.global_efficiency(G)
            except Exception as e:
                logging.warning(f"Error calculating global efficiency: {str(e)}. Setting to 0.")
                metrics['global_efficiency'] = 0.0

        # Isolated nodes
        metrics['isolated_nodes'] = list(nx.isolates(G))
        metrics['n_isolated'] = len(metrics['isolated_nodes'])

        # Average shortest path length (only for connected components)
        if is_empty or not nx.is_strongly_connected(G):
            # For empty or disconnected graphs
            metrics['avg_shortest_path'] = float('inf')

            # Try to calculate for the largest connected component if it exists
            try:
                largest_cc = max(nx.strongly_connected_components(G), key=len)
                if len(largest_cc) > 1:
                    sg = G.subgraph(largest_cc)
                    metrics['avg_shortest_path'] = nx.average_shortest_path_length(sg)
                    metrics['largest_component_size'] = len(largest_cc)
                else:
                    metrics['largest_component_size'] = 0
            except ValueError:  # Empty sequence
                metrics['largest_component_size'] = 0
        else:
            # For connected graphs
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
            metrics['largest_component_size'] = G.number_of_nodes()

        # Save the metrics
        self.network_metrics[measure_name] = metrics

        return metrics

    def calculate_node_metrics(self, G, measure_name):
        """
        Calculate node-level centrality metrics.

        Args:
            G: NetworkX directed graph
            measure_name: Name of the connectivity measure

        Returns:
            pd.DataFrame: DataFrame with node metrics
        """
        # Get node names
        node_names = [G.nodes[n]['name'] for n in G.nodes()]

        # Initialize metrics dictionary
        metrics = {
            'node_name': node_names,
            'in_degree': [],
            'out_degree': [],
            'betweenness': [],
            'eigenvector': [],
        }

        # Calculate in-degree and out-degree centrality
        in_degree = G.in_degree(weight='weight')
        out_degree = G.out_degree(weight='weight')
        metrics['in_degree'] = [in_degree[n] for n in G.nodes()]
        metrics['out_degree'] = [out_degree[n] for n in G.nodes()]

        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            metrics['betweenness'] = [betweenness[n] for n in G.nodes()]
        except:
            metrics['betweenness'] = [0.0] * len(node_names)

        # Calculate eigenvector centrality (with error handling for disconnected graphs)
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
            metrics['eigenvector'] = [eigenvector[n] for n in G.nodes()]
        except:
            try:
                # Try without weights if weighted version fails
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
                metrics['eigenvector'] = [eigenvector[n] for n in G.nodes()]
            except:
                metrics['eigenvector'] = [0.0] * len(node_names)

        # Create DataFrame
        df = pd.DataFrame(metrics)

        # Save to the node metrics dictionary
        self.node_metrics[measure_name] = df

        return df

    def visualize_network(self, G, measure_name, centrality_type='out_degree'):
        """
        Create a network visualization.

        Args:
            G: NetworkX directed graph
            measure_name: Name of the connectivity measure
            centrality_type: Type of centrality to use for node sizing ('out_degree', 'betweenness', 'eigenvector')
        """
        plt.figure(figsize=(14, 12))

        # Get node positions using a layout algorithm
        if G.number_of_nodes() > 20:
            pos = nx.spring_layout(G, k=1.5/np.sqrt(G.number_of_nodes()), iterations=100, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Get node sizes based on the specified centrality
        if centrality_type == 'out_degree':
            node_size = [G.out_degree(n, weight='weight') * 100 + 100 for n in G.nodes()]
        elif centrality_type == 'betweenness':
            betweenness = nx.betweenness_centrality(G, weight='weight')
            node_size = [betweenness[n] * 5000 + 100 for n in G.nodes()]
        elif centrality_type == 'eigenvector':
            try:
                eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                node_size = [eigenvector[n] * 5000 + 100 for n in G.nodes()]
            except:
                node_size = [100] * G.number_of_nodes()
        else:
            node_size = [300] * G.number_of_nodes()

        # Get edge colors based on lag values
        edge_lags = [G[u][v]['lag'] for u, v in G.edges()]

        if not edge_lags:
            edge_colors = []
        else:
            # Normalize lags for color mapping
            max_lag = max(edge_lags) if edge_lags else 1
            norm = mcolors.Normalize(vmin=0, vmax=max_lag)
            cmap = plt.cm.viridis
            edge_colors = [cmap(norm(lag)) for lag in edge_lags]

        # Draw nodes with size based on centrality
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='skyblue', alpha=0.8)

        # Draw edges with color based on lag
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.7, arrowsize=20)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['name'] for n in G.nodes()}, font_size=10)

        # Add a colorbar for lag values if we have edges
        if edge_lags:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Lag (time steps)')

        # Set title and adjust layout
        plt.title(f'Network Structure - {measure_name.upper()}\nNode Size: {centrality_type.replace("_", " ").title()}')
        plt.axis('off')
        plt.tight_layout()

        # Save the figure
        plt.savefig(self.output_dir / f'network_graph_{measure_name}_{centrality_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_metrics_comparison(self):
        """Create comparison visualizations for network metrics across measures."""
        if not self.network_metrics:
            logging.warning("No network metrics to compare.")
            return

        # Convert network metrics to DataFrame for easier plotting
        metrics_df = pd.DataFrame({measure: metrics for measure, metrics in self.network_metrics.items()})
        metrics_df = metrics_df.transpose()

        # Select numeric columns for comparison and ensure they are numeric
        numeric_cols = ['density', 'transitivity', 'reciprocity', 'global_efficiency', 'n_isolated']
        metrics_df = metrics_df[numeric_cols].astype(float)

        # Replace inf values with NaN for better visualization
        metrics_df = metrics_df.replace([float('inf'), -float('inf')], np.nan)

        # Create comparison bar plots
        plt.figure(figsize=(14, 10))
        metrics_df.plot(kind='bar', figsize=(14, 8))
        plt.title('Network Metrics Comparison Across Connectivity Measures')
        plt.ylabel('Metric Value')
        plt.xlabel('Connectivity Measure')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create heatmap for better visual comparison
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df, annot=True, cmap='viridis', fmt='.3f', linewidths=.5,
                   mask=metrics_df.isna())  # Mask NaN values
        plt.title('Network Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save the metrics to CSV
        metrics_df.to_csv(self.output_dir / 'network_metrics_comparison.csv')

        return metrics_df

    def create_node_metrics_comparison(self, top_n=10):
        """
        Create comparison of top nodes across different measures.

        Args:
            top_n: Number of top nodes to include
        """
        if not self.node_metrics:
            logging.warning("No node metrics to compare.")
            return

        # Create a DataFrame to store top nodes by different centrality measures
        top_nodes = {}

        for measure, df in self.node_metrics.items():
            # Get top nodes by out degree
            top_out = df.sort_values('out_degree', ascending=False).head(top_n)[['node_name', 'out_degree']]
            top_out.columns = ['node', f'{measure}_out_degree']

            # Get top nodes by betweenness
            top_btw = df.sort_values('betweenness', ascending=False).head(top_n)[['node_name', 'betweenness']]
            top_btw.columns = ['node', f'{measure}_betweenness']

            # Get top nodes by eigenvector
            top_eig = df.sort_values('eigenvector', ascending=False).head(top_n)[['node_name', 'eigenvector']]
            top_eig.columns = ['node', f'{measure}_eigenvector']

            top_nodes[f'{measure}_out'] = top_out
            top_nodes[f'{measure}_btw'] = top_btw
            top_nodes[f'{measure}_eig'] = top_eig

        # Save individual top node lists
        for name, df in top_nodes.items():
            df.to_csv(self.output_dir / f'top_nodes_{name}.csv', index=False)

        # Create a consolidated view of top nodes across all measures
        # Count how many times each node appears in any top list
        all_top_nodes = pd.concat([df['node'] for df in top_nodes.values()]).value_counts()
        top_consistent = all_top_nodes[all_top_nodes > 1].reset_index()
        top_consistent.columns = ['node', 'appearances']

        # Save the consolidated list
        if not top_consistent.empty:
            top_consistent.to_csv(self.output_dir / 'top_consistent_nodes.csv', index=False)

            # Create a bar plot of most consistent top nodes
            plt.figure(figsize=(12, 8))
            sns.barplot(x='node', y='appearances', data=top_consistent.head(20))
            plt.title('Most Consistent Top Nodes Across All Measures and Centrality Types')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'top_consistent_nodes.png', dpi=300, bbox_inches='tight')
            plt.close()

        return top_consistent

    def analyze_measure(self, measure):
        """
        Analyze a specific connectivity measure.

        Args:
            measure: Name of the connectivity measure (e.g., 'lc', 'gc')
        """
        logging.info(f"Analyzing {measure.upper()} network...")

        # Load the network data
        weights, lags, significant, node_names = self.load_network_data(measure)

        if weights is None:
            logging.error(f"Failed to load data for {measure}. Skipping analysis.")
            return

        # Create NetworkX graph
        G = self.create_networkx_graph(weights, lags, significant, node_names)
        self.graphs[measure] = G

        # Calculate network metrics
        network_metrics = self.calculate_network_metrics(G, measure)
        logging.info(f"Network metrics for {measure.upper()}: {network_metrics}")

        # Calculate node metrics
        node_metrics = self.calculate_node_metrics(G, measure)

        # Visualize network with different centrality measures
        for centrality in ['out_degree', 'betweenness', 'eigenvector']:
            self.visualize_network(G, measure, centrality_type=centrality)

        # Save node metrics to CSV
        node_metrics.to_csv(self.output_dir / f'node_metrics_{measure}.csv', index=False)

        # Save network metrics to JSON-friendly format
        network_metrics_clean = {k: (list(map(str, v)) if isinstance(v, list) else v)
                                for k, v in network_metrics.items()}
        pd.Series(network_metrics_clean).to_json(self.output_dir / f'network_metrics_{measure}.json')

        logging.info(f"Completed analysis for {measure.upper()}")

    def analyze_all_measures(self):
        """Analyze all available connectivity measures."""
        # Find all available measures based on NPZ files
        measures = []
        for file in self.input_dir.glob('network_data_*.npz'):
            measure = file.stem.replace('network_data_', '')
            measures.append(measure)

        if not measures:
            # Try finding by CSV files
            measures = []
            for file in self.input_dir.glob('weights_*.csv'):
                measure = file.stem.replace('weights_', '')
                measures.append(measure)

        if not measures:
            logging.error("No network data files found.")
            return

        logging.info(f"Found {len(measures)} connectivity measures: {', '.join(measures)}")

        # Analyze each measure
        for measure in measures:
            self.analyze_measure(measure)

        # Create comparison visualizations
        self.create_metrics_comparison()
        self.create_node_metrics_comparison()

        logging.info(f"All network analyses completed. Results saved to {os.path.abspath(self.output_dir)}")


def main():
    """Main function to execute the network metrics analysis."""
    try:
        logging.info("Starting network metrics analysis...")

        # Initialize analyzer
        analyzer = NetworkMetricsAnalyzer()

        # Analyze all available measures
        analyzer.analyze_all_measures()

        logging.info("Network metrics analysis completed successfully!")

    except Exception as e:
        logging.error(f"Error during network metrics analysis: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
