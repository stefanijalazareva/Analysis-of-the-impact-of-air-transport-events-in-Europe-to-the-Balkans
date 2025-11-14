import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Create output folder
output_dir = "results/NCT_confidence_intervals_results"
os.makedirs(output_dir, exist_ok=True)


# Load the CI table you generated with multiprocessing
df = pd.read_csv("results/NCT_confidence_intervals_results/bootstrap_CI_all_airports.csv")


# Define airport groups
europe = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LEBL", "EDDM", "EGKK", "LIRF", "EIDW"]
balkans = ["LATI", "LQSA", "LBSF", "LBBG", "LDZA", "LDSP", "LDDU", "BKPR", "LYTV", "LWSK"]

df_eu = df[df["airport"].isin(europe)]
df_ba = df[df["airport"].isin(balkans)]

params = ["df", "nc", "loc", "scale"]


# Function to generate whisker + boxplot visualizations
def plot_whisker_box(df_region, region_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(df_region))

    for i, param in enumerate(params):
        ax = axes[i // 2, i % 2]

        est = df_region[param].values
        low = df_region[param + "_low"].values
        high = df_region[param + "_high"].values

        # 1) Whisker plot (TRUE CI)
        ax.errorbar(
            x, est,
            yerr=[est - low, high - est],
            fmt='o',
            capsize=5,
            color='black',
            markersize=6,
            label="95% CI"
        )

        # 2) Boxplot directly from CI (no bootstrap samples needed)
        ci_lists = []
        for l, e, h in zip(low, est, high):
            ci_lists.append([l, e, h])   # each CI becomes its own "box"

        ax.boxplot(
            ci_lists,
            positions=x,
            widths=0.45,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.5)
        )

        ax.set_title(f"{param.upper()} — {region_name}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df_region["airport"], rotation=45)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Generate both final figures
plot_whisker_box(
    df_eu,
    "Europe",
    f"{output_dir}/Europe_CI_whisker_boxplot.png"
)

plot_whisker_box(
    df_ba,
    "Balkans",
    f"{output_dir}/Balkans_CI_whisker_boxplot.png"
)

print("\n Visualizations successfully created:")
print(f" - {output_dir}/Europe_CI_whisker_boxplot.png")
print(f" - {output_dir}/Balkans_CI_whisker_boxplot.png")
