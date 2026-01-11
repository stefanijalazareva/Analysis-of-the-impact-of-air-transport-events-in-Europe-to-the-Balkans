"""
Burr XII vs NCT – Confidence Interval Comparison Script
-------------------------------------------------------
Generates:
✔ Direct CI width comparison
✔ Europe vs Balkans CI comparison
✔ Parameter stability ranking
✔ Integrated visualizations (boxplots, barplots)

Input:
- results/burr_analysis/burr_ci_summary.csv
- results/NCT_confidence_intervals_results/bootstrap_CI_all_airports.csv

Output Folder:
- results/NCT_vs_Burr_CI_comparison/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp


# Load Burr XII CI summary

burr_path = "../results/burr_analysis/burr_ci_summary.csv"
nct_path = "../results/NCT_confidence_intervals_results/bootstrap_CI_all_airports.csv"

df_burr = pd.read_csv(burr_path)
df_nct = pd.read_csv(nct_path)

# Output directory
outdir = Path("../results/NCT_vs_Burr_CI_comparison")
outdir.mkdir(parents=True, exist_ok=True)

# Regions
EUROPE = ["EGLL","LFPG","EHAM","EDDF","LEMD","LEBL","EDDM","EGKK","LIRF","EIDW"]
BALKANS = ["LATI","LQSA","LBSF","LBBG","LDZA","LDSP","LDDU","BKPR","LYTV","LWSK"]


# 1. Preprocess Burr XII (positive delays only)

burr = df_burr[df_burr["Delay_Type"] == "positive"].copy()
burr["CI_width"] = burr["CI_upper"] - burr["CI_lower"]
burr["Region"] = burr["Airport"].apply(lambda x: "Europe" if x in EUROPE else "Balkans")

# Pivot: Airport rows, parameters columns
burr_wide = burr.pivot(index="Airport", columns="Param", values="CI_width").reset_index()


# 2. Preprocess NCT CI

nct = df_nct.copy()
nct["df_width"] = nct["df_high"] - nct["df_low"]
nct["nc_width"] = nct["nc_high"] - nct["nc_low"]
nct["loc_width"] = nct["loc_high"] - nct["loc_low"]
nct["scale_width"] = nct["scale_high"] - nct["scale_low"]

nct["Region"] = nct["airport"].apply(lambda x: "Europe" if x in EUROPE else "Balkans")


# 3. Combine Burr + NCT CI widths

combined = pd.DataFrame({
    "Airport": nct["airport"],
    "Region": nct["Region"],
    "Burr_c": burr_wide["c"].values,
    "Burr_d": burr_wide["d"].values,
    "Burr_loc": burr_wide["loc"].values,
    "Burr_scale": burr_wide["scale"].values,
    "NCT_df": nct["df_width"].values,
    "NCT_nc": nct["nc_width"].values,
    "NCT_loc": nct["loc_width"].values,
    "NCT_scale": nct["scale_width"].values,
})

combined.to_csv(outdir / "burr_vs_nct_ci_widths.csv", index=False)


# 4. Visualization 1: Mean CI Width Barplot


ci_means = combined.mean(numeric_only=True)

plt.figure(figsize=(12,6))
ci_means.plot(kind="bar", color="steelblue")
plt.title("Mean CI Width Comparison: Burr XII vs NCT")
plt.ylabel("Average CI Width")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outdir / "ci_width_barplot.png", dpi=300)
plt.close()


# 5. Visualization 2: CI Width Distribution (Boxplot)


plot_data = combined.melt(id_vars=["Airport"], value_vars=[
    "Burr_c","Burr_d","Burr_loc","Burr_scale",
    "NCT_df","NCT_nc","NCT_loc","NCT_scale"
], var_name="Parameter", value_name="CI_Width")

plt.figure(figsize=(14,7))
sns.boxplot(data=plot_data, x="Parameter", y="CI_Width")
plt.title("CI Width Distribution: Burr XII vs NCT")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outdir / "ci_width_boxplot.png", dpi=300)
plt.close()


# 6. Visualization 3: Europe vs Balkans Comparison
# 6. UPDATED Visualization: Faceted CI Width Comparison (PDF)

parameters = [
    "Burr_c", "Burr_d", "Burr_loc", "Burr_scale",
    "NCT_df", "NCT_nc", "NCT_loc", "NCT_scale"
]

n_cols = 2
n_rows = int(np.ceil(len(parameters) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
axes = axes.flatten()

for ax, param in zip(axes, parameters):
    sns.boxplot(
        data=combined,
        x="Region",
        y=param,
        hue="Region",
        palette=["steelblue", "darkorange"],
        legend=False,
        ax=ax
    )
    ax.set_title(param)
    ax.set_ylabel("CI Width")
    ax.set_xlabel("")

for i in range(len(parameters), len(axes)):
    fig.delaxes(axes[i])

fig.suptitle(
    "Europe vs Balkans – CI Width Comparison per Parameter\n(Burr XII & NCT)",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = outdir / "ci_europe_vs_balkans.pdf"
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close()


# 7. Visualization 4: Parameter Stability Ranking

stability = pd.DataFrame({
    "Parameter": ci_means.index,
    "Mean_CI_Width": ci_means.values
}).sort_values("Mean_CI_Width")

plt.figure(figsize=(10,5))
sns.barplot(data=stability, x="Parameter", y="Mean_CI_Width", palette="viridis")
plt.title("Parameter Stability Ranking (Lower CI Width = More Stable)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outdir / "parameter_stability_ranking.png", dpi=300)
plt.close()

print("\n✔ All comparison visualizations generated in:")
print(f"  → {outdir}\n")

# 8. Violin Plots – Burr XII vs NCT per parameter


plt.figure(figsize=(14, 7))
sns.violinplot(
    data=plot_data,
    x="Parameter",
    y="CI_Width",
    inner="quartile",
    palette="Set2"
)
plt.title("Violin Plot – CI Width Distribution (Burr XII vs NCT)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outdir / "violin_ci_width_burr_vs_nct.png", dpi=300)
plt.close()


# 9. KS Test (Burr vs NCT) for Each Parameter

ks_results = []

parameter_pairs = {
    "c": ("Burr_c",),
    "d": ("Burr_d",),
    "loc": ("Burr_loc",),
    "scale": ("Burr_scale",),
    "df": ("NCT_df",),
    "nc": ("NCT_nc",),
    "nct_loc": ("NCT_loc",),
    "nct_scale": ("NCT_scale",)
}

# Compare only Burr vs NCT where parameter types are comparable
comparable_parameters = [
    ("Burr_c", "NCT_df"),
    ("Burr_d", "NCT_nc"),
    ("Burr_loc", "NCT_loc"),
    ("Burr_scale", "NCT_scale")
]

for burr_param, nct_param in comparable_parameters:
    stat, p = ks_2samp(combined[burr_param], combined[nct_param])
    ks_results.append([burr_param, nct_param, stat, p])

ks_df = pd.DataFrame(ks_results, columns=["Burr_Param", "NCT_Param", "KS_Statistic", "p_value"])
ks_df.to_csv(outdir / "KS_test_burr_vs_nct.csv", index=False)

print("\n✔ KS Test Results Saved → KS_test_burr_vs_nct.csv")

ks_path = "results/NCT_vs_Burr_CI_comparison/KS_test_burr_vs_nct.csv"
df = pd.read_csv(ks_path)


# 10. HEATMAP – KS Statistics

# Create matrix for heatmap
ks_matrix = df.pivot(index="Burr_Param", columns="NCT_Param", values="KS_Statistic")

plt.figure(figsize=(8,6))
sns.heatmap(ks_matrix, annot=True, cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.5)
plt.title("KS Statistic Heatmap – Burr XII vs NCT")
plt.tight_layout()
plt.savefig(outdir / "ks_heatmap.png", dpi=300)
plt.close()

