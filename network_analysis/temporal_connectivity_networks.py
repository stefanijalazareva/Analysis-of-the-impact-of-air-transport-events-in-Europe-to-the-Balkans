import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import json
import logging
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


DETREND_METHOD = "zs"
MAX_LAG = 24
ALPHA = 0.05

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
    path = Path(f"data/DetrendedData/detrended_{DETREND_METHOD}.parquet")
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
            if len(data) < 100:
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

    metrics = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "mean_degree": mean_degree,
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

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        edge_color="gray",
        alpha=0.4,
        width=1.5
    )

    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    temporal_metrics = []

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

            # FULL NETWORK
            adj, edge_stats = granger_connectivity(subset, ALL_AIRPORTS)
            adj.to_csv(out_dir / "adjacency_matrix.csv")
            edge_stats.to_csv(out_dir / "edges_with_stats.csv", index=False)

            top_edges = edge_stats.sort_values("p_value").head(20)
            top_edges.to_csv(out_dir / "top20_edges.csv", index=False)

            metrics = compute_network_metrics(adj)
            temporal_metrics.append({
                "year": year,
                "month": month,
                "edges": metrics["edges"],
                "density": metrics["density"],
                "mean_degree": metrics["mean_degree"]
            })

            with open(out_dir / "network_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            visualize_network(
                adj,
                out_dir / "network.png",
                title=f"Delay Propagation Network {year}-{month:02d}"
            )

    tm = pd.DataFrame(temporal_metrics)
    tm["time"] = pd.to_datetime(
        tm["year"].astype(str) + "-" +
        tm["month"].astype(str).str.zfill(2) + "-01"
    )
    tm = tm.sort_values("time")
    tm.to_csv(OUTPUT_BASE / "temporal_metrics.csv", index=False)

    # EDGE COUNT OVER TIME
    plt.figure(figsize=(10, 4))
    plt.plot(tm["time"], tm["edges"], marker="o")
    plt.xticks(rotation=90)
    plt.title("Number of edges over time")
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / "edges_over_time.png")
    plt.close()

    # DENSITY OVER TIME
    plt.figure(figsize=(10, 4))
    plt.plot(tm["time"], tm["density"], marker="o", color="darkred")
    plt.xticks(rotation=90)
    plt.title("Network density over time")
    plt.tight_layout()
    plt.savefig(OUTPUT_BASE / "density_over_time.png")
    plt.close()

    logging.info("ALL FULL MONTHLY NETWORKS GENERATED SUCCESSFULLY")

if __name__ == "__main__":
    main()
