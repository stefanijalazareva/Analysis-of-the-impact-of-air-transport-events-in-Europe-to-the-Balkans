import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# CONFIG
BASE_DIR = Path("../results/full_monthly_networks")

EU_AIRPORTS = {
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW'
}

BALKANS_AIRPORTS = {
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
}

OUTPUT_DIR = BASE_DIR / "directionality"
OUTPUT_DIR.mkdir(exist_ok=True)


# HELPERS
def count_directional_links(adj: pd.DataFrame):
    """Count EU→Balkans and Balkans→EU links from adjacency matrix."""
    eu_to_balkans = adj.loc[
        adj.index.intersection(EU_AIRPORTS),
        adj.columns.intersection(BALKANS_AIRPORTS)
    ].values.sum()

    balkans_to_eu = adj.loc[
        adj.index.intersection(BALKANS_AIRPORTS),
        adj.columns.intersection(EU_AIRPORTS)
    ].values.sum()

    return int(eu_to_balkans), int(balkans_to_eu)



# MAIN DATA COLLECTION
records = []

for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue


    try:
        year, month = map(int, folder.name.split("_"))
    except ValueError:
        continue

    adj_path = folder / "FULL" / "adjacency_matrix.csv"
    if not adj_path.exists():
        continue

    adj = pd.read_csv(adj_path, index_col=0)

    eu_balk, balk_eu = count_directional_links(adj)

    records.append({
        "year": year,
        "month": month,
        "time": datetime(year, month, 1),
        "EU_to_Balkans": eu_balk,
        "Balkans_to_EU": balk_eu
    })


df = pd.DataFrame(records).sort_values("time")

# Save CSV
df.to_csv(OUTPUT_DIR / "directional_links_over_time.csv", index=False)


# PLOTTING
plt.figure(figsize=(10, 5))

plt.plot(
    df["time"],
    df["EU_to_Balkans"],
    marker="o",
    linewidth=2,
    label="EU → Balkans"
)

plt.plot(
    df["time"],
    df["Balkans_to_EU"],
    marker="s",
    linewidth=2,
    label="Balkans → EU"
)

plt.xlabel("Time")
plt.ylabel("Number of significant directional links")
plt.title("Directional Delay Propagation Links Over Time")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "directional_links_over_time.pdf")
plt.show()

print("✔ Directional link analysis completed.")
print(f"✔ Results saved in: {OUTPUT_DIR.resolve()}")
