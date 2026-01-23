import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import re

# CONFIG
BASE_DIR = Path("results/full_monthly_networks")
OUTPUT_DIR = BASE_DIR / "airport_dynamics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIRPORTS_OF_INTEREST = ["EDDM", "EGLL", "LFPG"]  # Munich, London, Paris
TOP_K = 5  # top spreaders / receivers per year

EUROPE_AIRPORTS = {
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW'
}

BALKANS_AIRPORTS = {
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
}
AIRPORT_ORDER = [
    # EUROPE
    "EGLL","LFPG","EHAM","EDDF","LEMD",
    "LEBL","EDDM","EGKK","LIRF","EIDW",
    # BALKANS
    "LBSF","LDZA","LATI","LDSP","LBBG",
    "LDDU","BKPR","LWSK","LQSA","LYTV"
]


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

#HELPERS
def extract_year_month(path):
    match = re.match(r"(\d{4})_(\d{2})", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_graph(adj_path):
    adj = pd.read_csv(adj_path, index_col=0)
    return nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)

# DATA COLLECTION
records = []
yearly_spreaders = {}
yearly_receivers = {}
monthly_out_records = []
monthly_in_records = []


for ym_dir in sorted(BASE_DIR.iterdir()):
    if not ym_dir.is_dir():
        continue

    year, month = extract_year_month(ym_dir)
    if year is None:
        continue

    adj_path = ym_dir / "FULL" / "adjacency_matrix.csv"
    if not adj_path.exists():
        continue

    G = load_graph(adj_path)

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    t = pd.Timestamp(year=year, month=month, day=1)

    for a in G.nodes():
        monthly_out_records.append({
            "time": t,
            "airport": a,
            "out_degree": out_deg.get(a, 0)
        })
        monthly_in_records.append({
            "time": t,
            "airport": a,
            "in_degree": in_deg.get(a, 0)
        })

    # store per-airport temporal data
    for airport in AIRPORTS_OF_INTEREST:
        if airport in G:
            records.append({
                "year": year,
                "month": month,
                "time": pd.Timestamp(year=year, month=month, day=1),
                "airport": airport,
                "in_degree": in_deg.get(airport, 0),
                "out_degree": out_deg.get(airport, 0),
            })

    # yearly aggregation
    yearly_spreaders.setdefault(year, {})
    yearly_receivers.setdefault(year, {})

    for a in G.nodes():
        yearly_spreaders[year][a] = yearly_spreaders[year].get(a, 0) + out_deg.get(a, 0)
        yearly_receivers[year][a] = yearly_receivers[year].get(a, 0) + in_deg.get(a, 0)


# SAVE MONTHLY RECORDS TO CSV (DEBUG / VALIDATION)

df_monthly_out = pd.DataFrame(monthly_out_records)
df_monthly_in  = pd.DataFrame(monthly_in_records)

df_monthly_out.sort_values(["time", "airport"]).to_csv(
    OUTPUT_DIR / "monthly_out_degree_records.csv",
    index=False
)

df_monthly_in.sort_values(["time", "airport"]).to_csv(
    OUTPUT_DIR / "monthly_in_degree_records.csv",
    index=False
)

print("Saved monthly degree CSV files:")
print(" - monthly_out_degree_records.csv")
print(" - monthly_in_degree_records.csv")


#DATAFRAME
df = pd.DataFrame(records).sort_values("time")

#PLOT 1: IN / OUT DEGREE OVER TIME
for airport in AIRPORTS_OF_INTEREST:
    sub = df[df["airport"] == airport]

    plt.figure(figsize=(10, 5))
    plt.plot(sub["time"], sub["out_degree"], marker="o", label="Out-degree")
    plt.plot(sub["time"], sub["in_degree"], marker="o", label="In-degree")

    plt.title(f"{airport} – Delay Propagation Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Degree")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"{airport}_in_out_degree_over_time.pdf")
    plt.close()

#PLOT 2: TOP SPREADERS PER YEAR
for year, values in yearly_spreaders.items():
    top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    airports, scores = zip(*top)

    plt.figure(figsize=(8, 4))
    plt.bar(airports, scores)
    plt.title(f"Top Delay Spreaders in {year} (Out-Degree)")
    plt.ylabel("Total Out-Degree")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"top_spreaders_{year}.pdf")
    plt.close()

#PLOT 3: TOP RECEIVERS PER YEAR
for year, values in yearly_receivers.items():
    top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    airports, scores = zip(*top)

    plt.figure(figsize=(8, 4))
    plt.bar(airports, scores)
    plt.title(f"Top Delay Receivers in {year} (In-Degree)")
    plt.ylabel("Total In-Degree")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / f"top_receivers_{year}.pdf")
    plt.close()

#PLOT 4: HEATMAP (OUT-DEGREE OVER TIME MONTHLY)

hm_out = pd.DataFrame(monthly_out_records)

pivot_out = (hm_out
    .pivot_table(index="airport", columns="time", values="out_degree", aggfunc="sum")
    .fillna(0)
)

pivot_out = pivot_out.reindex(AIRPORT_ORDER)

plt.figure(figsize=(16, 9))
plt.imshow(pivot_out, aspect="auto")
plt.colorbar(label="Out-degree (monthly)")

# y labels
plt.yticks(range(len(pivot_out.index)), pivot_out.index)

times = pivot_out.columns.sort_values()
xticks = list(range(len(times)))
xticklabels = [t.strftime("%Y-%m") for t in times]
plt.xticks(xticks, xticklabels, rotation=90, fontsize=7)


plt.title("Delay Propagation Heatmap (Out-Degree, FULL Network)")
plt.xlabel("Month")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_out_degree_full_monthly.pdf")
plt.close()


#PLOT 6: HEATMAP (IN-DEGREE OVER TIME MONTHLY)

hm_in = pd.DataFrame(monthly_in_records)

pivot_in = (hm_in
    .pivot_table(index="airport", columns="time", values="in_degree", aggfunc="sum")
    .fillna(0)
)

pivot_in = pivot_in.reindex(AIRPORT_ORDER)

plt.figure(figsize=(16, 9))
plt.imshow(pivot_in, aspect="auto")
plt.colorbar(label="In-degree (monthly)")

plt.yticks(range(len(pivot_in.index)), pivot_in.index)

times = pivot_out.columns.sort_values()
xticks = list(range(len(times)))
xticklabels = [t.strftime("%Y-%m") for t in times]
plt.xticks(xticks, xticklabels, rotation=90, fontsize=7)

plt.title("Delay Reception Heatmap (In-Degree, FULL Network)")
plt.xlabel("Month")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_in_degree_full_monthly.pdf")
plt.close()

#PLOT 5: TOP SPREADERS THROUGH TIME (MONTHLY)

hm_df = pd.DataFrame(monthly_out_records)  # columns: time, airport, out_degree

# find global top K spreaders (over all months)
total_outdegree = hm_df.groupby("airport")["out_degree"].sum()
top_airports = total_outdegree.sort_values(ascending=False).head(TOP_K).index

plt.figure(figsize=(12, 5))

for airport in top_airports:
    series = (hm_df[hm_df["airport"] == airport]
              .groupby("time")["out_degree"].sum()
              .sort_index())
    plt.plot(series.index, series.values, marker="o", label=airport)

plt.title("Top Delay Spreaders Over Time (FULL Network, Monthly)")
plt.xlabel("Month")
plt.ylabel("Out-Degree")
plt.legend(title="Airport")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "top_spreaders_over_time_full_monthly.pdf")
plt.close()


#EU vs BALKANS RECEIVER DOMINANCE

records = []

for year, values in yearly_receivers.items():
    eu_in = sum(
        deg for airport, deg in values.items()
        if airport in EUROPE_AIRPORTS
    )
    balkans_in = sum(
        deg for airport, deg in values.items()
        if airport in BALKANS_AIRPORTS
    )

    records.append({
        "year": year,
        "EU_in_degree": eu_in,
        "BALKANS_in_degree": balkans_in
    })

df_region = pd.DataFrame(records).sort_values("year")

plt.figure(figsize=(10, 5))

plt.plot(
    df_region["year"],
    df_region["EU_in_degree"],
    marker="o",
    linewidth=2.5,
    label="Europe (Receivers)"
)

plt.plot(
    df_region["year"],
    df_region["BALKANS_in_degree"],
    marker="o",
    linewidth=2.5,
    label="Balkans (Receivers)"
)

plt.title("EU vs Balkans Delay Reception Dominance (In-Degree)")
plt.xlabel("Year")
plt.ylabel("Total In-Degree")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "EU_vs_BALKANS_receiver_dominance.pdf")
plt.close()


print("Airport dynamics analysis completed.")
print("Saved to:", OUTPUT_DIR.resolve())
