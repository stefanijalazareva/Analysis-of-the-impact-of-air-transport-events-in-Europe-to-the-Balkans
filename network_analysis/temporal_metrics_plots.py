import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


# ASSORTATIVITY OVER TIME
if "assortativity" in data["FULL"].columns:
    plt.figure(figsize=(10, 5))
    
    for name, df in data.items():
        valid_data = df[df["assortativity"].notna()]
        if len(valid_data) > 0:
            plt.plot(valid_data["time"], valid_data["assortativity"], marker="o", label=name)
    
    plt.title("Assortativity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Assortativity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "assortativity_comparison.pdf")
    plt.close()


# MODULARITY OVER TIME
if "modularity" in data["FULL"].columns:
    plt.figure(figsize=(10, 5))
    
    for name, df in data.items():
        valid_data = df[df["modularity"].notna()]
        if len(valid_data) > 0:
            plt.plot(valid_data["time"], valid_data["modularity"], marker="o", label=name)
    
    plt.title("Modularity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Modularity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "modularity_comparison.pdf")
    plt.close()


# TRANSITIVITY OVER TIME
if "transitivity" in data["FULL"].columns:
    plt.figure(figsize=(10, 5))
    
    for name, df in data.items():
        plt.plot(df["time"], df["transitivity"], marker="o", label=name)
    
    plt.title("Transitivity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Transitivity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "transitivity_comparison.pdf")
    plt.close()


# GLOBAL EFFICIENCY OVER TIME
if "global_efficiency" in data["FULL"].columns:
    plt.figure(figsize=(10, 5))
    
    for name, df in data.items():
        plt.plot(df["time"], df["global_efficiency"], marker="o", label=name)
    
    plt.title("Global Efficiency Over Time")
    plt.xlabel("Time")
    plt.ylabel("Global Efficiency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "global_efficiency_comparison.pdf")
    plt.close()


# CORRELATION SCATTER PLOTS (as requested in email)
# For each metric, plot EU vs Balkans values
if "EUROPE" in data and "BALKANS" in data:
    eu_df = data["EUROPE"]
    balkans_df = data["BALKANS"]
    
    merged = pd.merge(eu_df, balkans_df, on="time", suffixes=("_EU", "_BALKANS"))
    
    metrics_to_plot = ["mean_degree", "assortativity", "modularity", "transitivity", "global_efficiency"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        eu_col = f"{metric}_EU"
        balkans_col = f"{metric}_BALKANS"
        
        if eu_col in merged.columns and balkans_col in merged.columns:
            valid_data = merged[[eu_col, balkans_col]].dropna()
            
            if len(valid_data) > 0:
                ax.scatter(valid_data[eu_col], valid_data[balkans_col], alpha=0.6, s=60)

                min_val = min(valid_data[eu_col].min(), valid_data[balkans_col].min())
                max_val = max(valid_data[eu_col].max(), valid_data[balkans_col].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
                
                if len(valid_data) > 1:
                    corr = valid_data[eu_col].corr(valid_data[balkans_col])
                    ax.set_title(f"{metric.replace('_', ' ').title()}\n(r={corr:.3f})")
                else:
                    ax.set_title(metric.replace('_', ' ').title())
                
                ax.set_xlabel("Europe")
                ax.set_ylabel("Balkans")
                ax.grid(alpha=0.3)
                ax.legend()
    
    fig.delaxes(axes[5])
    
    plt.suptitle("EU vs Balkans Metric Correlations", fontsize=14, y=1.00)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "eu_balkans_correlation_scatter.pdf")
    plt.close()


print("All temporal metric plots successfully saved to:")
print(OUTPUT_DIR.resolve())
