import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


# CONFIG
NETWORK_FILES = {
    "EUROPE": Path("../results/full_monthly_networks/Aggregated/aggregated_EUROPE.csv"),
    "BALKANS": Path("../results/full_monthly_networks/Aggregated/aggregated_BALKANS.csv"),
}

COORD_FILE = Path("../data/airport_coordinates.csv")
OUTPUT_DIR = Path("../results/full_monthly_networks/Aggregated")

EDGE_WIDTH_SCALE = 0.08
NODE_SIZE = 90

# map extent (Europe + Balkans)
MAP_EXTENT = [-15, 35, 35, 60]


# LOAD COORDINATES
coords = pd.read_csv(COORD_FILE).set_index("airport")


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

        ax.plot(
            [coords.loc[src, "lon"], coords.loc[tgt, "lon"]],
            [coords.loc[src, "lat"], coords.loc[tgt, "lat"]],
            color="royalblue",
            linewidth=EDGE_WIDTH_SCALE * w,
            alpha=0.45,
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


# RUN BOTH NETWORKS
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
