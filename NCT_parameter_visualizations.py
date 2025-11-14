import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import probplot


# CONFIGURATION

INPUT_FILE = "data/NonCentralT/noncentral_t_parameters.csv"
OUTPUT_DIR = "results/NCT_parameters_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load table
df = pd.read_csv(INPUT_FILE)

# Parameters to visualize
param_cols = ["df", "nc", "loc (mean)", "scale (std)"]

# Airport grouping
europe = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LEBL", "EDDM", "EGKK", "LIRF", "EIDW"]
balkans = ["LATI", "LQSA", "LBSF", "LBBG", "LDZA", "LDSP", "LDDU", "BKPR", "LYTV", "LWSK"]

df_eu = df[df["Airport"].isin(europe)]
df_ba = df[df["Airport"].isin(balkans)]


#  BARPLOT — Graphical comparison of fitted NCT parameters

plt.figure(figsize=(18, 12))

for i, param in enumerate(param_cols):
    plt.subplot(2, 2, i+1)
    sns.barplot(data=df, x="Airport", y=param, hue="Region", palette="Set2")
    plt.title(f"NCT Parameter Comparison: {param}", fontsize=14)
    plt.xticks(rotation=70)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/barplot_parameter_comparison.png", dpi=300)
plt.close()

#  PAIRPLOT — Joint graphical comparison of parameters

sns.pairplot(
    df,
    vars=param_cols,
    hue="Region",
    diag_kind="kde",
    corner=True,
    palette="Set2"
)
plt.suptitle("Pairplot of NCT Parameters Across All Airports", y=1.02)
plt.savefig(f"{OUTPUT_DIR}/pairplot_parameters.png", dpi=300)
plt.close()


#  HEATMAP — Easy-to-compare matrix of parameters

df_hm = df.set_index("Airport")[param_cols]

plt.figure(figsize=(12, 10))
sns.heatmap(df_hm, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Fitted NCT Parameters per Airport", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_parameters.png", dpi=300)
plt.close()

#  VIOLIN PLOTS — Europe vs Balkans per parameter

plt.figure(figsize=(16, 10))

for i, param in enumerate(param_cols):
    plt.subplot(2, 2, i+1)
    sns.violinplot(data=df, x="Region", y=param, hue="Region",
                   palette="Set2", inner="box", legend=False)
    plt.title(f"Violin Plot of {param} (Europe vs Balkans)")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/violin_region_parameters.png", dpi=300)
plt.close()


# Q–Q PLOTS: Europe vs Balkans — parameter distributions

QQ_DIR = os.path.join(OUTPUT_DIR, "QQ_parameter_plots")
os.makedirs(QQ_DIR, exist_ok=True)

print("\nGenerating Q–Q plots for parameter distributions (Europe vs Balkans)...")

for param in param_cols:
    plt.figure(figsize=(6,6))

    # Europe Q–Q plot
    probplot(df_eu[param].values, dist="norm", plot=plt)
    # Balkans Q–Q overlaid
    probplot(df_ba[param].values, dist="norm", plot=plt)

    plt.title(f"Q–Q Plot: Europe vs Balkans — {param}")
    plt.legend(["Europe", "Line", "Balkans", "Line"])
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{QQ_DIR}/QQ_{param.replace(' ','_')}.png", dpi=300)
    plt.close()

print(f"Q–Q plots saved in: {QQ_DIR}")
print("\n All parameter comparison visualizations created successfully in:")
print(f"➡ {OUTPUT_DIR}")
