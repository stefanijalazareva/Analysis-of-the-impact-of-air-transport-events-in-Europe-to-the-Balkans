import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, nct, lognorm, gamma, burr12, gengamma, lomax, invgauss, expon


AIRPORTS = ["EGLL", "LBSF"]

DIST_MAP = {
    "normal": norm,
    "nct": nct,
    "lognorm": lognorm,
    "gamma": gamma,
    "burr12": burr12,
    "gengamma": gengamma,
    "lomax": lomax,
    "invgauss": invgauss,
    "expon": expon,
}

RESULTS_DIR = Path("../results/distribution_analysis")


x = np.linspace(0.001, 100, 2000)


# PLOT
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

Y_LIMITS = {
    "EGLL": 0.125,
    "LBSF": 0.08
}


for i, (ax, airport) in enumerate(zip(axes, AIRPORTS)):
    df = pd.read_csv(RESULTS_DIR / airport / "distribution_comparison_new.csv")

    y_max = 0.0

    for _, row in df.iterrows():
        name = row["distribution"]
        if name not in DIST_MAP:
            continue

        params = eval(row["parameters"])
        dist = DIST_MAP[name]

        y = dist.pdf(x, *params)
        y_max = max(y_max, np.nanmax(y))

        ax.plot(x, y, lw=1.6, label=name.capitalize())

    # Titles and labels
    ax.set_xlabel("Delay x (minutes)")
    ax.set_title(airport)

    ax.set_ylim(0, Y_LIMITS[airport])
    ax.grid(alpha=0.25)

    # Panel labels: a) / b)
    panel = "a)" if i == 0 else "b)"
    ax.text(
        0.02, 0.95, f"{panel} {airport}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )

axes[0].set_ylabel("Probability p(x)")
axes[1].legend(fontsize=8, ncol=2, loc="upper right")

plt.tight_layout()
plt.savefig("../results/distribution_analysis/best_fit_comparison_EGLL_LBSF.pdf")
plt.show()
