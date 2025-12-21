"""
Causality Network Analysis for Air Transport Delays

This script performs comprehensive network analysis on detrended delay time series:
1. Granger causality analysis
2. Transfer entropy calculations
3. Correlation-based network construction
4. Delay propagation detection

The analysis identifies how delays propagate between airports and constructs
directed networks showing causal relationships.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import warnings
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_analysis.log'),
        logging.StreamHandler()
    ]
)

EUROPE_AIRPORTS = ['EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW']
BALKANS_AIRPORTS = ['LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK']

AIRPORT_NAMES = {
    'EGLL': 'London Heathrow', 'LFPG': 'Paris CDG', 'EHAM': 'Amsterdam',
    'EDDF': 'Frankfurt', 'LEMD': 'Madrid', 'LEBL': 'Barcelona',
    'EDDM': 'Munich', 'EGKK': 'London Gatwick', 'LIRF': 'Rome',
    'EIDW': 'Dublin', 'LATI': 'Tirana', 'LQSA': 'Sarajevo',
    'LBSF': 'Sofia', 'LBBG': 'Burgas', 'LDZA': 'Zagreb',
    'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
    'LYTV': 'Tivat', 'LWSK': 'Skopje'
}


def load_detrended_data(method='zs'):
    """
    Load detrended time series data.
    
    Args:
        method: Detrending method ('delta', 'delta2', 'zs', or 'linear')
    
    Returns:
        pandas.DataFrame: Detrended time series with airports as columns
    """
    data_path = Path(f"data/DetrendedData/detrended_{method}.parquet")
    logging.info(f"Loading detrended data from {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run 'python -m data_processing.detrend_timeseries' first."
        )
    
    df = pd.read_parquet(data_path)
    logging.info(f"Loaded data shape: {df.shape}")
    logging.info(f"Airports: {df.columns.tolist()}")
    
    return df


def granger_causality_matrix(data, max_lag=24, alpha=0.05):
    """
    Compute Granger causality matrix between all airport pairs.
    
    Args:
        data: DataFrame with airports as columns, time as rows
        max_lag: Maximum lag to test (default 24 hours)
        alpha: Significance level
    
    Returns:
        DataFrame: Matrix where (i,j) indicates if airport i Granger-causes airport j
        DataFrame: Matrix of p-values
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    airports = data.columns.tolist()
    n_airports = len(airports)
    
    causality_matrix = pd.DataFrame(0, index=airports, columns=airports)
    pvalue_matrix = pd.DataFrame(1.0, index=airports, columns=airports)
    
    logging.info(f"Computing Granger causality for {n_airports} airports...")
    
    for i, source in enumerate(airports):
        for j, target in enumerate(airports):
            if source == target:
                continue
            
            try:
                # Prepare data for Granger test
                test_data = data[[target, source]].dropna()
                
                if len(test_data) < 100:  # Need sufficient data
                    continue
                
                # Perform Granger causality test
                result = grangercausalitytests(test_data, max_lag, verbose=False)
                
                # Extract minimum p-value across all lags
                p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                min_p_value = min(p_values)
                
                pvalue_matrix.loc[source, target] = min_p_value
                
                if min_p_value < alpha:
                    causality_matrix.loc[source, target] = 1
                    
            except Exception as e:
                logging.warning(f"Granger test failed for {source}->{target}: {str(e)}")
                continue
        
        if (i + 1) % 5 == 0:
            logging.info(f"Processed {i + 1}/{n_airports} airports")
    
    return causality_matrix, pvalue_matrix


def correlation_network(data, method='pearson', threshold=0.3):
    """
    Build correlation-based network.
    
    Args:
        data: DataFrame with airports as columns
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        threshold: Minimum correlation to include edge
    
    Returns:
        DataFrame: Correlation matrix
        DataFrame: Thresholded adjacency matrix
    """
    logging.info(f"Computing {method} correlation network...")
    
    corr_matrix = data.corr(method=method)
    
    adj_matrix = (corr_matrix.abs() >= threshold).astype(int)
    
    np.fill_diagonal(adj_matrix.values, 0)
    
    n_edges = adj_matrix.sum().sum() // 2 
    logging.info(f"Created network with {n_edges} edges (threshold={threshold})")
    
    return corr_matrix, adj_matrix


def transfer_entropy_matrix(data, k=1, l=1, subsample=10):
    """
    Compute transfer entropy matrix (simplified version for speed).
    
    Args:
        data: DataFrame with airports as columns
        k: History length for target
        l: History length for source
        subsample: Subsample factor to speed up computation
    
    Returns:
        DataFrame: Transfer entropy matrix
    """
    logging.info(f"Computing transfer entropy (k={k}, l={l}, subsample={subsample})...")
    
    airports = data.columns.tolist()
    n_airports = len(airports)
    
    data_sub = data.iloc[::subsample, :]
    
    te_matrix = pd.DataFrame(0.0, index=airports, columns=airports)
    
    for i, source in enumerate(airports):
        for j, target in enumerate(airports):
            if source == target:
                continue
            
            try:
                x = data_sub[source].values
                y = data_sub[target].values
                
                if len(y) > k + l:
                    y_past = y[:-k-l]
                    y_future = y[k+l:]
                    x_past = x[l:-k] if k > 0 else x[l:]
                    
                    bins = 10
                    y_past_disc = pd.cut(y_past, bins=bins, labels=False)
                    y_future_disc = pd.cut(y_future, bins=bins, labels=False)
                    x_past_disc = pd.cut(x_past, bins=bins, labels=False)
                    
                    te_value = mutual_information_conditional(y_future_disc, x_past_disc, y_past_disc)
                    te_matrix.loc[source, target] = te_value
                    
            except Exception as e:
                logging.warning(f"TE calculation failed for {source}->{target}: {str(e)}")
                continue
        
        if (i + 1) % 5 == 0:
            logging.info(f"Processed {i + 1}/{n_airports} airports")
    
    return te_matrix


def mutual_information_conditional(y_future, x_past, y_past):
    """
    Simplified conditional mutual information calculation.
    This is an approximation - consider using proper TE libraries for production.
    """
    try:
        mask = ~(np.isnan(y_future) | np.isnan(x_past) | np.isnan(y_past))
        y_f = y_future[mask]
        x_p = x_past[mask]
        y_p = y_past[mask]
        
        if len(y_f) < 50:
            return 0.0
        
        from sklearn.metrics import mutual_info_score
        
        mi_yx = mutual_info_score(y_f, x_p)
        mi_yy = mutual_info_score(y_f, y_p)
        
        return max(0, mi_yx - mi_yy * 0.5)
        
    except:
        return 0.0


def analyze_delay_propagation(causality_matrix, corr_matrix):
    """
    Analyze delay propagation patterns between Europe and Balkans.
    
    Args:
        causality_matrix: Granger causality adjacency matrix
        corr_matrix: Correlation matrix
    
    Returns:
        dict: Propagation statistics
    """
    logging.info("Analyzing delay propagation patterns...")
    
    results = {
        'europe_to_balkans': 0,
        'balkans_to_europe': 0,
        'within_europe': 0,
        'within_balkans': 0,
        'key_influencers': [],
        'key_receivers': []
    }
    
    for source in causality_matrix.index:
        for target in causality_matrix.columns:
            if causality_matrix.loc[source, target] == 1:
                source_group = 'Europe' if source in EUROPE_AIRPORTS else 'Balkans'
                target_group = 'Europe' if target in EUROPE_AIRPORTS else 'Balkans'
                
                if source_group == 'Europe' and target_group == 'Balkans':
                    results['europe_to_balkans'] += 1
                elif source_group == 'Balkans' and target_group == 'Europe':
                    results['balkans_to_europe'] += 1
                elif source_group == 'Europe' and target_group == 'Europe':
                    results['within_europe'] += 1
                elif source_group == 'Balkans' and target_group == 'Balkans':
                    results['within_balkans'] += 1
    
    out_degree = causality_matrix.sum(axis=1).sort_values(ascending=False)
    results['key_influencers'] = out_degree.head(5).to_dict()
    
    in_degree = causality_matrix.sum(axis=0).sort_values(ascending=False)
    results['key_receivers'] = in_degree.head(5).to_dict()
    
    return results


def visualize_network(adjacency_matrix, title, output_path, edge_weights=None):
    """
    Visualize network graph.
    
    Args:
        adjacency_matrix: Adjacency matrix
        title: Plot title
        output_path: Path to save figure
        edge_weights: Optional edge weights
    """
    logging.info(f"Creating network visualization: {title}")
    
    G = nx.DiGraph()
    
    for airport in adjacency_matrix.index:
        group = 'Europe' if airport in EUROPE_AIRPORTS else 'Balkans'
        G.add_node(airport, group=group, name=AIRPORT_NAMES.get(airport, airport))
    
    for source in adjacency_matrix.index:
        for target in adjacency_matrix.columns:
            if adjacency_matrix.loc[source, target] > 0:
                weight = edge_weights.loc[source, target] if edge_weights is not None else 1
                G.add_edge(source, target, weight=abs(weight))
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    node_colors = ['blue' if G.nodes[node]['group'] == 'Europe' else 'green' 
                   for node in G.nodes()]
    
    node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=15, 
                           edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Major European Airports'),
        Patch(facecolor='green', label='Balkan Airports')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved network visualization to {output_path}")


def main():
    """
    Main execution function for network analysis.
    """
    try:
        logging.info("Starting Causality Network Analysis")
        
        output_dir = Path("results/network_causality_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data = load_detrended_data(method='zs')
        
        logging.info("1. GRANGER CAUSALITY ANALYSIS")
        granger_matrix, granger_pvalues = granger_causality_matrix(data, max_lag=24, alpha=0.05)
        granger_matrix.to_csv(output_dir / 'granger_causality_matrix.csv')
        granger_pvalues.to_csv(output_dir / 'granger_pvalues.csv')
        logging.info(f"Total causal links: {granger_matrix.sum().sum()}")
        
        logging.info("2. CORRELATION NETWORK CONSTRUCTION")
        corr_matrix, corr_adj = correlation_network(data, method='pearson', threshold=0.3)
        corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
        corr_adj.to_csv(output_dir / 'correlation_adjacency.csv')
        
        logging.info("3. TRANSFER ENTROPY CALCULATION")
        te_matrix = transfer_entropy_matrix(data, k=1, l=1, subsample=10)
        te_matrix.to_csv(output_dir / 'transfer_entropy_matrix.csv')
        
        logging.info("4. DELAY PROPAGATION DETECTION")
        propagation_stats = analyze_delay_propagation(granger_matrix, corr_matrix)
        
        logging.info(f"\nPropagation Statistics:")
        logging.info(f"  Europe → Balkans: {propagation_stats['europe_to_balkans']} causal links")
        logging.info(f"  Balkans → Europe: {propagation_stats['balkans_to_europe']} causal links")
        logging.info(f"  Within Europe: {propagation_stats['within_europe']} causal links")
        logging.info(f"  Within Balkans: {propagation_stats['within_balkans']} causal links")
        logging.info(f"\nTop 5 Influencers (out-degree):")
        for airport, degree in propagation_stats['key_influencers'].items():
            logging.info(f"  {airport} ({AIRPORT_NAMES.get(airport, airport)}): {degree}")
        logging.info(f"\nTop 5 Receivers (in-degree):")
        for airport, degree in propagation_stats['key_receivers'].items():
            logging.info(f"  {airport} ({AIRPORT_NAMES.get(airport, airport)}): {degree}")
        
        import json
        with open(output_dir / 'propagation_statistics.json', 'w') as f:
            json.dump(propagation_stats, f, indent=2)
        
        logging.info("CREATING VISUALIZATIONS")
        
        visualize_network(
            granger_matrix, 
            'Granger Causality Network\n(Europe-Balkans Delay Propagation)',
            output_dir / 'granger_network.png'
        )
        
        visualize_network(
            corr_adj,
            'Correlation Network\n(Threshold = 0.3)',
            output_dir / 'correlation_network.png',
            edge_weights=corr_matrix
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(granger_matrix, cmap='YlOrRd', cbar_kws={'label': 'Causal Link'}, 
                    ax=axes[0], square=True)
        axes[0].set_title('Granger Causality Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target Airport')
        axes[0].set_ylabel('Source Airport')
        
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Correlation'}, ax=axes[1], square=True)
        axes[1].set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Airport')
        axes[1].set_ylabel('Airport')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'network_matrices_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logging.info(f"Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logging.error(f"Error during network analysis: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
