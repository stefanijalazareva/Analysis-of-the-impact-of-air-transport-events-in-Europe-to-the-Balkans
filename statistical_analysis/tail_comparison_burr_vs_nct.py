"""
Tail Comparison: Burr XII vs Noncentral t (NCT)
------------------------------------------------
Input:
- results/burr_analysis/burr_analysis_summary.csv
    (има Data_P90, Data_P95, P90, P95, Shape_c, Shape_d, Scale, Location ...)
- data/NonCentralT/noncentral_t_parameters.csv
    (има df, nc, loc, scale по аеродром)

Output (се снима во results/Tail_Comparison_Burr_vs_NCT/):
- tail_comparison_burr_vs_nct.csv
- tail_scatter_p90_p95.png
- tail_abs_error_boxplot.png
- tail_ratio_boxplot.png
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import nct

sns.set_theme(style="whitegrid")


BURR_PATH = "results/burr_analysis/burr_analysis_summary.csv"
NCT_PATH = "data/NonCentralT/noncentral_t_parameters.csv"

OUTDIR = Path("results/Tail_Comparison_Burr_vs_NCT")
OUTDIR.mkdir(parents=True, exist_ok=True)


EUROPE = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LEBL", "EDDM", "EGKK", "LIRF", "EIDW"]
BALKANS = ["LATI", "LQSA", "LBSF", "LBBG", "LDZA", "LDSP", "LDDU", "BKPR", "LYTV", "LWSK"]


burr_df = pd.read_csv(BURR_PATH)


burr_pos = burr_df[burr_df["Delay_Type"] == "positive"].copy()
required_burr_cols = ["Airport", "Data_P90", "Data_P95", "P90", "P95"]
for col in required_burr_cols:
    if col not in burr_pos.columns:
        raise ValueError(f"Колоната '{col}' не постои во burr_analysis_summary.csv. Провери ги имињата.")


burr_pos["Region"] = burr_pos["Airport"].apply(
    lambda x: "Europe" if x in EUROPE else ("Balkans" if x in BALKANS else "Other")
)

nct_df = pd.read_csv(NCT_PATH)


if "airport" in nct_df.columns:
    nct_df = nct_df.rename(columns={"airport": "Airport"})
elif "Airport" not in nct_df.columns:
    raise ValueError("Во noncentral_t_parameters.csv мора да постои колоната 'airport' или 'Airport'.")

required_nct_cols = ["Airport", "df", "nc", "loc (mean)", "scale (std)"]
for col in required_nct_cols:
    if col not in nct_df.columns:
        raise ValueError(f"Колоната '{col}' не постои во noncentral_t_parameters.csv.")


def compute_nct_percentiles(row, probs=(0.9, 0.95)):
    df = row["df"]
    nc = row["nc"]
    loc = row["loc (mean)"]
    scale = row["scale (std)"]

    dist = nct(df, nc, loc=loc, scale=scale)
    return pd.Series(dist.ppf(probs), index=[f"NCT_P{int(p*100)}" for p in probs])

nct_percentiles = nct_df.apply(compute_nct_percentiles, axis=1)
nct_df = pd.concat([nct_df, nct_percentiles], axis=1)


# 3. Merge Burr + NCT по Airport

merged = pd.merge(
    burr_pos[
        ["Airport", "Region", "Data_P90", "Data_P95", "P90", "P95"]
    ],
    nct_df[["Airport", "NCT_P90", "NCT_P95"]],
    on="Airport",
    how="inner",
)


merged = merged.rename(
    columns={
        "P90": "Burr_P90",
        "P95": "Burr_P95",
    }
)

for p in [90, 95]:
    data_col = f"Data_P{p}"
    burr_col = f"Burr_P{p}"
    nct_col = f"NCT_P{p}"

    merged[f"Burr_abs_err_P{p}"] = (merged[burr_col] - merged[data_col]).abs()
    merged[f"NCT_abs_err_P{p}"] = (merged[nct_col] - merged[data_col]).abs()

    merged[f"Burr_ratio_P{p}"] = merged[burr_col] / merged[data_col]
    merged[f"NCT_ratio_P{p}"] = merged[nct_col] / merged[data_col]


merged.to_csv(OUTDIR / "tail_comparison_burr_vs_nct.csv", index=False)


# 5. Visualization 1 – Data vs Model (P90, P95)

plt.figure(figsize=(14, 6))

# P90 subplot
plt.subplot(1, 2, 1)
plt.scatter(merged["Data_P90"], merged["Burr_P90"], label="Burr XII", alpha=0.7)
plt.scatter(merged["Data_P90"], merged["NCT_P90"], label="NCT", alpha=0.7, marker="s")

min_p90 = min(merged["Data_P90"].min(), merged["Burr_P90"].min(), merged["NCT_P90"].min())
max_p90 = max(merged["Data_P90"].max(), merged["Burr_P90"].max(), merged["NCT_P90"].max())
plt.plot([min_p90, max_p90], [min_p90, max_p90], "k--", alpha=0.5)

plt.title("P90: Data vs Model (Burr XII & NCT)")
plt.xlabel("Data P90 (minutes)")
plt.ylabel("Model P90 (minutes)")
plt.legend()
plt.grid(alpha=0.3)

# P95 subplot
plt.subplot(1, 2, 2)
plt.scatter(merged["Data_P95"], merged["Burr_P95"], label="Burr XII", alpha=0.7)
plt.scatter(merged["Data_P95"], merged["NCT_P95"], label="NCT", alpha=0.7, marker="s")

min_p95 = min(merged["Data_P95"].min(), merged["Burr_P95"].min(), merged["NCT_P95"].min())
max_p95 = max(merged["Data_P95"].max(), merged["Burr_P95"].max(), merged["NCT_P95"].max())
plt.plot([min_p95, max_p95], [min_p95, max_p95], "k--", alpha=0.5)

plt.title("P95: Data vs Model (Burr XII & NCT)")
plt.xlabel("Data P95 (minutes)")
plt.ylabel("Model P95 (minutes)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / "tail_scatter_p90_p95.png", dpi=300)
plt.close()


# 6. Visualization 2 – Apsolute error

err_rows = []

for p in [90, 95]:
    for model in ["Burr", "NCT"]:
        col = f"{model}_abs_err_P{p}"
        for value in merged[col]:
            err_rows.append({
                "Percentile": f"P{p}",
                "Model": model,
                "Abs_Error": value
            })

err_df = pd.DataFrame(err_rows)

plt.figure(figsize=(10, 6))
sns.boxplot(data=err_df, x="Percentile", y="Abs_Error", hue="Model")
plt.title("Absolute Tail Error |Model - Data| (Burr XII vs NCT)")
plt.ylabel("Absolute Error (minutes)")
plt.xlabel("Percentile")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(OUTDIR / "tail_abs_error_boxplot.png", dpi=300)
plt.close()


# 7. Visualization 3 – Ratio Model/Data (boxplot)

ratio_rows = []

for p in [90, 95]:
    for model in ["Burr", "NCT"]:
        col = f"{model}_ratio_P{p}"
        for value in merged[col]:
            ratio_rows.append({
                "Percentile": f"P{p}",
                "Model": model,
                "Ratio": value
            })

ratio_df = pd.DataFrame(ratio_rows)

plt.figure(figsize=(10, 6))
sns.boxplot(data=ratio_df, x="Percentile", y="Ratio", hue="Model")
plt.axhline(1.0, color="k", linestyle="--", alpha=0.6)
plt.title("Tail Ratio Model/Data (Burr XII vs NCT)")
plt.ylabel("Model / Data")
plt.xlabel("Percentile")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(OUTDIR / "tail_ratio_boxplot.png", dpi=300)
plt.close()

print("\n✔ Tail comparison Burr XII vs NCT is finished.")

