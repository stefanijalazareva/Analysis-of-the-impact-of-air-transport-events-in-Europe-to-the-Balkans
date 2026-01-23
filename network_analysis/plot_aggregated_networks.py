import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D


# CONFIG
NETWORK_FILES = {
    "EUROPE": Path("../results/full_monthly_networks/Aggregated/aggregated_EUROPE.csv"),
    "BALKANS": Path("../results/full_monthly_networks/Aggregated/aggregated_BALKANS.csv"),
}

COORD_FILE = Path("../data/airport_coordinates.csv")
OUTPUT_DIR = Path("../results/full_monthly_networks/Aggregated")

MIN_LINEWIDTH = 0.4
MAX_LINEWIDTH = 4.0

MIN_ALPHA = 0.15
MAX_ALPHA = 0.75

NODE_SIZE = 90

# map extent (Europe + Balkans)
MAP_EXTENT = [-15, 35, 35, 60]


# LOAD COORDINATES
coords = pd.read_csv(COORD_FILE).set_index("airport")

def scale(v, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin)


# FUNCTION: plot one aggregated network
def plot_network(adj_path, title, out_name):
    # ---- load adjacency matrix
    adj = pd.read_csv(adj_path, index_col=0)

    # ---- adjacency → edge list
    edges = (
        adj.stack()
        .reset_index()
        .rename(columns={
            "level_0": "source",
            "level_1": "target",
            0: "weight"
        })
    )
    edges = edges[edges["weight"] > 0]

    # ---- normalize weights for visual scaling
    w_min = edges["weight"].min()
    w_max = edges["weight"].max()

    def scale(v, vmin, vmax, out_min, out_max):
        if vmax == vmin:
            return (out_min + out_max) / 2
        return out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin)

    # ---- setup map
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)

    # ---- draw edges
    for _, row in edges.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]

        if src not in coords.index or tgt not in coords.index:
            continue

        lw = scale(w, w_min, w_max, MIN_LINEWIDTH, MAX_LINEWIDTH)
        a = scale(w, w_min, w_max, MIN_ALPHA, MAX_ALPHA)

        ax.plot(
            [coords.loc[src, "lon"], coords.loc[tgt, "lon"]],
            [coords.loc[src, "lat"], coords.loc[tgt, "lat"]],
            color="royalblue",
            linewidth=lw,
            alpha=a,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    # ---- draw nodes
    for airport, row in coords.iterrows():
        if airport not in adj.index:
            continue

        ax.scatter(
            row["lon"], row["lat"],
            s=NODE_SIZE,
            color="darkred",
            edgecolor="white",
            zorder=3,
            transform=ccrs.PlateCarree()
        )

        ax.text(
            row["lon"] + 0.25,
            row["lat"] + 0.25,
            airport,
            fontsize=9,
            transform=ccrs.PlateCarree()
        )

    # ---- finalize
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / out_name)
    plt.close()
    print(f"Saved: {out_name}")

# RUN BOTH NETWORKS,creates separate visualization for Balkans networks, and separate one for Europe
plot_network(
    NETWORK_FILES["EUROPE"],
    "Aggregated Delay Propagation Network – EUROPE",
    "aggregated_EUROPE_map.pdf"
)

plot_network(
    NETWORK_FILES["BALKANS"],
    "Aggregated Delay Propagation Network – BALKANS",
    "aggregated_BALKANS_map.pdf"
)

def plot_overlay_networks(out_name="aggregated_EUROPE_BALKANS.pdf"):
    # --- load both adjacency matrices
    #creates one visualization where on the same map both networks will be shown Europe(blue color) and Balkan(orange color) network
    adj_eu = pd.read_csv(NETWORK_FILES["EUROPE"], index_col=0)
    adj_ba = pd.read_csv(NETWORK_FILES["BALKANS"], index_col=0)

    # --- adjacency → edge list
    def to_edges(adj):
        edges = (
            adj.stack()
            .reset_index()
            .rename(columns={"level_0": "source", "level_1": "target", 0: "weight"})
        )
        edges = edges[edges["weight"] > 0].copy()
        edges = edges[edges["source"] != edges["target"]]  # no self-loops
        return edges

    edges_eu = to_edges(adj_eu)
    edges_ba = to_edges(adj_ba)

    # --- compute min/max weights separately (so each region has visible thickness differences)
    eu_min, eu_max = edges_eu["weight"].min(), edges_eu["weight"].max()
    ba_min, ba_max = edges_ba["weight"].min(), edges_ba["weight"].max()

    # --- setup map
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)

    # --- draw EUROPE edges (blue)
    for _, row in edges_eu.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        if src not in coords.index or tgt not in coords.index:
            continue

        lw = scale(w, eu_min, eu_max, MIN_LINEWIDTH, MAX_LINEWIDTH)
        a  = scale(w, eu_min, eu_max, MIN_ALPHA, MAX_ALPHA)

        ax.plot(
            [coords.loc[src, "lon"], coords.loc[tgt, "lon"]],
            [coords.loc[src, "lat"], coords.loc[tgt, "lat"]],
            color="royalblue",
            linewidth=lw,
            alpha=a,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    # --- draw BALKANS edges (orange)
    for _, row in edges_ba.iterrows():
        src, tgt, w = row["source"], row["target"], row["weight"]
        if src not in coords.index or tgt not in coords.index:
            continue

        lw = scale(w, ba_min, ba_max, MIN_LINEWIDTH, MAX_LINEWIDTH)
        a  = scale(w, ba_min, ba_max, MIN_ALPHA, MAX_ALPHA)

        ax.plot(
            [coords.loc[src, "lon"], coords.loc[tgt, "lon"]],
            [coords.loc[src, "lat"], coords.loc[tgt, "lat"]],
            color="darkorange",
            linewidth=lw,
            alpha=a,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

    # --- draw nodes (all airports from both networks)
    all_nodes = sorted(set(adj_eu.index).union(set(adj_ba.index)))
    for airport in all_nodes:
        if airport not in coords.index:
            continue
        lon, lat = coords.loc[airport, "lon"], coords.loc[airport, "lat"]

        ax.scatter(
            lon, lat,
            s=NODE_SIZE,
            color="darkred",
            edgecolor="white",
            zorder=3,
            transform=ccrs.PlateCarree()
        )
        ax.text(
            lon + 0.25,
            lat + 0.25,
            airport,
            fontsize=9,
            transform=ccrs.PlateCarree()
        )

    # --- legend (so viewer knows which color is which)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="royalblue", lw=2, label="EUROPE network"),
        Line2D([0], [0], color="darkorange", lw=2, label="BALKANS network"),
    ]
    ax.legend(handles=legend_elems, loc="lower left")

    plt.title("Aggregated Delay Propagation Network – EUROPE + BALKANS", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / out_name)
    plt.close()
    print(f"Saved: {out_name}")

plot_overlay_networks("aggregated_EUROPE_BALKANS.pdf")
