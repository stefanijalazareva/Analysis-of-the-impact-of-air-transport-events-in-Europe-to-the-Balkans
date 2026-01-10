import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import json
import logging
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


warnings.filterwarnings("ignore")

DETREND_METHOD = "z_score"
MAX_LAG = 8
BASE_ALPHA = 0.05
ALPHA = BASE_ALPHA / MAX_LAG #Bonferroni correction

MONTHS = [3, 6, 9, 12]
YEARS = range(2015, 2025)

ALL_AIRPORTS = [
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW',
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
]
EUROPE_AIRPORTS = {
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW'
}

BALKANS_AIRPORTS = {
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
}

OUTPUT_BASE = Path("results/full_monthly_networks")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_detrended_data():
    path = Path(f"data/Detrended/detrended_{DETREND_METHOD}.parquet")
    if not path.exists():
        raise FileNotFoundError("Detrended data not found.")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def granger_connectivity(df, airports):
    adj = pd.DataFrame(0, index=airports, columns=airports)
    edge_stats = []

    for src in airports:
        for tgt in airports:
            if src == tgt:
                continue

            data = df[[tgt, src]].dropna()
            if len(data) < 10 * MAX_LAG:
                continue

            try:
                res = grangercausalitytests(
                    data, maxlag=MAX_LAG, verbose=False
                )

                pvals = [res[l][0]["ssr_ftest"][1] for l in range(1, MAX_LAG + 1)]
                min_p = np.min(pvals)
                best_lag = pvals.index(min_p) + 1


                if min_p < ALPHA:
                    adj.loc[src, tgt] = 1


                edge_stats.append({
                    "source": src,
                    "target": tgt,
                    "best_lag": best_lag,
                    "p_value": min_p
                })

            except Exception:
                continue

    return adj, pd.DataFrame(edge_stats)

def run_and_save_network(subset, airports, out_dir, label, year, month):
    adj, edge_stats = granger_connectivity(subset, airports)

    out_dir.mkdir(parents=True, exist_ok=True)

    adj.to_csv(out_dir / "adjacency_matrix.csv")
    edge_stats.to_csv(out_dir / "edges_with_stats.csv", index=False)

    metrics = compute_network_metrics(adj)
    with open(out_dir / "network_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    visualize_network(
        adj,
        out_dir / "network.png",
        title=f"{label} Delay Propagation Network {year}-{month:02d}"
    )

    return metrics



def compute_network_metrics(adj_matrix):
    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    density = nx.density(G)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    mean_degree = np.mean(list(dict(G.degree()).values()))

    top_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:5]

    # Assortativity (degree correlation)
    try:
        assortativity = nx.algorithms.assortativity.degree_pearson_correlation_coefficient(G)
    except (ValueError, ZeroDivisionError):
        assortativity = None
    
    # Modularity (using Louvain communities)
    try:
        G_undirected = G.to_undirected()
        if G_undirected.number_of_edges() > 0:
            communities = nx.algorithms.community.louvain.louvain_communities(G_undirected)
            modularity = nx.algorithms.community.quality.modularity(G_undirected, communities)
            n_communities = len(communities)
        else:
            modularity = None
            n_communities = 0
    except Exception as e:
        logging.warning(f"Modularity calculation failed: {e}")
        modularity = None
        n_communities = None
    
    # Transitivity (global clustering coefficient)
    try:
        transitivity = nx.algorithms.cluster.transitivity(G)
    except:
        transitivity = 0.0
    
    # Global Efficiency
    try:
        global_efficiency = nx.algorithms.efficiency_measures.global_efficiency(G)
    except:
        global_efficiency = 0.0

    metrics = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "mean_degree": mean_degree,
        "assortativity": assortativity,
        "modularity": modularity,
        "n_communities": n_communities,
        "transitivity": transitivity,
        "global_efficiency": global_efficiency,
        "top_out_degree_hubs": dict(top_out),
        "top_in_degree_hubs": dict(top_in)
    }

    return metrics

def visualize_network(adj_matrix, output_path, title):
    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph)

    plt.figure(figsize=(14, 12))

    pos = nx.spring_layout(G, seed=42, k=1.2)

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        if node in EUROPE_AIRPORTS:
            node_colors.append("#1f77b4")  # blue
        else:
            node_colors.append("#2ca02c")  # green

        node_sizes.append(300 + 100 * G.degree(node))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85
    )

    edge_colors = []

    for u, v in G.edges():
        if u in EUROPE_AIRPORTS and v in BALKANS_AIRPORTS:
            edge_colors.append("red")
        elif u in BALKANS_AIRPORTS and v in EUROPE_AIRPORTS:
            edge_colors.append("blue")
        else:
            edge_colors.append("gray")

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        edge_color=edge_colors,
        alpha=0.85,
        width=2.5,
        min_source_margin=15,
        min_target_margin=15
    )

    # ---------- LEGEND ----------
    legend_elements = [
        # Nodes
        Patch(facecolor="#1f77b4", edgecolor="k", label="European airport"),
        Patch(facecolor="#2ca02c", edgecolor="k", label="Balkan airport"),

        # Edges
        Line2D([0], [0], color="red", lw=2, label="Europe → Balkans"),
        Line2D([0], [0], color="blue", lw=2, label="Balkans → Europe"),
        Line2D([0], [0], color="gray", lw=2, label="Within region"),
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        frameon=True,
        title="Legend",
        title_fontsize=10
    )

    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()

def save_temporal_csv(data, filename):
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("time")
    df.to_csv(OUTPUT_BASE / filename, index=False)


def main():
    temporal_full = []
    temporal_europe = []
    temporal_balkans = []

    logging.info("Starting FULL monthly connectivity network analysis")

    df = load_detrended_data()

    for year in YEARS:
        for month in MONTHS:
            logging.info(f"Processing {year}-{month:02d}")

            subset = df[
                (df.index.year == year) &
                (df.index.month == month)
            ]

            if len(subset) < 300:
                logging.warning("Not enough data, skipping")
                continue

            out_dir = OUTPUT_BASE / f"{year}_{month:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. FULL network
            full_dir = out_dir / "FULL"
            metrics_full = run_and_save_network(
                subset,
                ALL_AIRPORTS,
                full_dir,
                "FULL",
                year,
                month
            )

            # 2. EUROPE only network
            eu_dir = out_dir / "EUROPE"
            metrics_eu = run_and_save_network(
                subset,
                list(EUROPE_AIRPORTS),
                eu_dir,
                "EUROPE",
                year,
                month
            )

            # 3. BALKANS only network
            balkan_dir = out_dir / "BALKANS"
            metrics_balkan = run_and_save_network(
                subset,
                list(BALKANS_AIRPORTS),
                balkan_dir,
                "BALKANS",
                year,
                month
            )

            # FULL
            temporal_full.append({
                "year": year,
                "month": month,
                "edges": metrics_full["edges"],
                "density": metrics_full["density"],
                "mean_degree": metrics_full["mean_degree"]
            })

            # EUROPE
            temporal_europe.append({
                "year": year,
                "month": month,
                "edges": metrics_eu["edges"],
                "density": metrics_eu["density"],
                "mean_degree": metrics_eu["mean_degree"]
            })

            # BALKANS
            temporal_balkans.append({
                "year": year,
                "month": month,
                "edges": metrics_balkan["edges"],
                "density": metrics_balkan["density"],
                "mean_degree": metrics_balkan["mean_degree"]
            })

    save_temporal_csv(temporal_full, "temporal_metrics_FULL.csv")
    save_temporal_csv(temporal_europe, "temporal_metrics_EUROPE.csv")
    save_temporal_csv(temporal_balkans, "temporal_metrics_BALKANS.csv")


    logging.info("ALL FULL MONTHLY NETWORKS GENERATED SUCCESSFULLY")

if __name__ == "__main__":
    main()
