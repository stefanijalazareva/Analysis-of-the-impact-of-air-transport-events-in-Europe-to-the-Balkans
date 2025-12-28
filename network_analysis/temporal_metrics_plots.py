import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# CONFIG
BASE_DIR = Path("results/full_monthly_networks")
OUTPUT_DIR = BASE_DIR / "temporal_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "FULL": BASE_DIR / "temporal_metrics_FULL.csv",
    "EUROPE": BASE_DIR / "temporal_metrics_EUROPE.csv",
    "BALKANS": BASE_DIR / "temporal_metrics_BALKANS.csv",
}


# LOAD DATA
data = {}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time")
    data[name] = df


# COMMON PLOT SETTINGS
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


# 1. EDGES OVER TIME
plt.figure(figsize=(10, 5))

for name, df in data.items():
    plt.plot(df["time"], df["edges"], marker="o", label=name)


plt.title("Number of Edges Over Time")
plt.xlabel("Time")
plt.ylabel("Edges")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "edges_comparison.pdf")
plt.close()


# 2. DENSITY OVER TIME
plt.figure(figsize=(10, 5))

for name, df in data.items():
    plt.plot(df["time"], df["density"], marker="o", label=name)


plt.title("Network Density Over Time")
plt.xlabel("Time")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "density_comparison.pdf")
plt.close()


# 3. MEAN DEGREE OVER TIME
plt.figure(figsize=(10, 5))

for name, df in data.items():
    plt.plot(df["time"], df["mean_degree"], marker="o", label=name)


plt.title("Mean Degree Over Time")
plt.xlabel("Time")
plt.ylabel("Mean Degree")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "mean_degree_comparison.pdf")
plt.close()


print("All temporal metric plots successfully saved to:")
print(OUTPUT_DIR.resolve())
