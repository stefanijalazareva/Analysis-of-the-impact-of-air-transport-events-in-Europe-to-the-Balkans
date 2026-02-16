import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = Path("results/full_monthly_networks")
YEARS = range(2015, 2025)
MONTHS = [3, 6, 9, 12]

EUROPE_ORDER = [
    "EGLL", "LFPG", "EHAM", "EDDF", "LEMD",
    "LEBL", "EDDM", "EGKK", "LIRF", "EIDW"
]

BALKANS_ORDER = [
    "LBSF", "LDZA", "LATI", "LDSP", "LBBG",
    "LDDU", "BKPR", "LWSK", "LQSA", "LYTV"
]

EUROPE_AIRPORTS = set(EUROPE_ORDER)
BALKANS_AIRPORTS = set(BALKANS_ORDER)

EU_COLOR = "#1f77b4"   # blue
BA_COLOR = "#ff7f0e"   # orange

OUT_PDF = BASE_DIR / "total_outgoing_links_summed_across_months.pdf"


def main():
    order = EUROPE_ORDER + BALKANS_ORDER
    airports = order

    total_outgoing = {a: 0 for a in airports}
    months_used = 0

    for year in YEARS:
        for month in MONTHS:
            adj_path = BASE_DIR / f"{year}_{month:02d}" / "FULL" / "adjacency_matrix.csv"
            if not adj_path.exists():
                continue

            adj = pd.read_csv(adj_path, index_col=0)


            adj = adj.reindex(index=airports, columns=airports).fillna(0)


            out_deg_month = adj.sum(axis=1)

            for a in airports:
                total_outgoing[a] += int(out_deg_month.loc[a])

            months_used += 1

    if months_used == 0:
        raise RuntimeError("No monthly adjacency matrices found. Check BASE_DIR paths.")

    values = [total_outgoing[a] for a in order]
    colors = ([EU_COLOR] * len(EUROPE_ORDER)) + ([BA_COLOR] * len(BALKANS_ORDER))

    plt.figure(figsize=(16, 5))
    x = np.arange(len(order))
    plt.bar(x, values, color=colors, alpha=0.9)


    plt.axvline(len(EUROPE_ORDER) - 0.5, linestyle="--", color="gray", linewidth=1.5)

    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("Total outgoing links (sum across all months)")
    plt.title(f"Total outgoing connectivity centrality (summed across months)")
    plt.grid(axis="y", alpha=0.25)

    plt.legend(handles=[
        Patch(facecolor=EU_COLOR, label="Europe"),
        Patch(facecolor=BA_COLOR, label="Balkans"),
    ], loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.close()
    print("Saved:", OUT_PDF)


if __name__ == "__main__":
    main()
