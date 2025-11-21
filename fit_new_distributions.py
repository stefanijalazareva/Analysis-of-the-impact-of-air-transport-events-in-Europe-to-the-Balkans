"""
Fit Burr XII, Generalized Gamma, Lomax, Inverse Gaussian, Exponential
for all airports and save results in distribution_comparison.csv
so that ks_test_visualization.py can generate updated heatmap, boxplot and summary.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import burr12, gengamma, lomax, invgauss, expon
from tqdm import tqdm

# -------------------------------------------------------------
# 1. DEFINE NEW DISTRIBUTIONS
# -------------------------------------------------------------
NEW_DISTRIBUTIONS = {
    "burr12": burr12,
    "gengamma": gengamma,
    "lomax": lomax,
    "invgauss": invgauss,
    "expon": expon,
}


def fit_distribution(dist, data):
    """Fit distribution and compute params, AIC, KS."""
    try:
        params = dist.fit(data)
        frozen = dist(*params)

        # log-likelihood
        loglik = np.sum(frozen.logpdf(data))
        k = len(params)
        aic = 2 * k - 2 * loglik

        # KS test
        ks_stat, p_value = stats.kstest(data, frozen.cdf)

        return params, aic, ks_stat, p_value
    except Exception as e:
        print(f"Fit failed: {dist.name} -> {e}")
        return None, None, None, None


def analyze_airport(airport_code):
    """Loads delays for 1 airport and fits all 5 new distributions."""
    data_path = Path("data/RawData") / f"Delays_{airport_code}.npy"
    raw = np.load(data_path, allow_pickle=True)
    delays = raw[:, 3].astype(float)  # 4th column is delays

    results = []

    for name, dist in NEW_DISTRIBUTIONS.items():
        params, aic, ks_stat, p_value = fit_distribution(dist, delays)
        if params is None:
            continue

        results.append({
            "distribution": name,
            "aic": aic,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "parameters": str(params)
        })

    return pd.DataFrame(results)


def save_results(df, airport_code):
    """Appends new distribution results to existing distribution_comparison.csv."""
    airport_dir = Path("results/distribution_analysis") / airport_code
    airport_dir.mkdir(exist_ok=True)

    file = airport_dir / "distribution_comparison.csv"

    if file.exists():
        old = pd.read_csv(file)
        merged = pd.concat([old, df], ignore_index=True)
        merged.to_csv(file, index=False)
    else:
        df.to_csv(file, index=False)


def main():
    data_dir = Path("data/RawData")
    airports = [f.stem.replace("Delays_", "") for f in data_dir.glob("Delays_*.npy")]

    print(f"Fitting NEW distributions for {len(airports)} airports...\n")

    for airport in tqdm(airports):
        df = analyze_airport(airport)
        save_results(df, airport)

    print("\nDone! Now run:  python ks_test_visualization.py")


if __name__ == "__main__":
    main()
