import pandas as pd
import numpy as np
from pathlib import Path


# CONFIG
BASE_DIR = Path(__file__).resolve().parents[1]
NETWORK_DIR = BASE_DIR / "results" / "full_monthly_networks"
COORDS_FILE = BASE_DIR / "data" / "airport_coordinates.csv"

EU_AIRPORTS = {
    'EGLL','LFPG','EHAM','EDDF','LEMD',
    'LEBL','EDDM','EGKK','LIRF','EIDW'
}

BALKAN_AIRPORTS = {
    'LATI','LQSA','LBSF','LBBG','LDZA',
    'LDSP','LDDU','BKPR','LYTV','LWSK'
}

REGIONS = {
    "EUROPE": EU_AIRPORTS,
    "BALKANS": BALKAN_AIRPORTS
}


def aggregate_region(region_name, airports):
    agg = pd.DataFrame(0, index=airports, columns=airports)

    for ym_dir in NETWORK_DIR.glob("20??_??"):
        adj_file = ym_dir / "FULL" / "adjacency_matrix.csv"
        if not adj_file.exists():
            continue

        adj = pd.read_csv(adj_file, index_col=0)


        sub = adj.loc[airports, airports]
        agg += sub

    return agg


def main():
    out_dir = NETWORK_DIR / "Aggregated"
    out_dir.mkdir(exist_ok=True)

    for region, airports in REGIONS.items():
        print(f"Aggregating {region} network...")
        agg = aggregate_region(region, list(airports))
        agg.to_csv(out_dir / f"aggregated_{region}.csv")

    print("Aggregation complete.")

if __name__ == "__main__":
    main()
