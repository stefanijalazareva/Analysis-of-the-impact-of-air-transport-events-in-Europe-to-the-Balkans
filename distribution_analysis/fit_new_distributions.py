"""
Fit positive-support distributions on POSITIVE flight delays only:
Burr XII, Generalized Gamma, Lomax, Inverse Gaussian, Exponential

Results are saved to distribution_comparison.csv per airport
and are compatible with ks_test_visualization.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import burr12, gengamma, lomax, invgauss, expon
from tqdm import tqdm
import warnings
from pandas.errors import EmptyDataError
warnings.filterwarnings("ignore")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DISTRIBUTIONS = {
    "burr12": burr12,
    "gengamma": gengamma,
    "lomax": lomax,
    "invgauss": invgauss,
    "expon": expon,
}

MIN_SAMPLES = 100  # same logic as colleague


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def load_positive_delays(airport_code):
    """Load and return POSITIVE delays in minutes."""
    path = Path("data/RawData") / f"Delays_{airport_code}.npy"
    raw = np.load(path, allow_pickle=True)

    delays = raw[:, 3].astype(float)
    pos_delays = delays[delays > 0] / 60.0  # minutes

    return pos_delays


def fit_distribution(dist, data):
    """Fit distribution and compute LL, AIC, KS."""
    params = dist.fit(data)

    loglik = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * loglik

    ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)

    return params, loglik, aic, ks_stat, p_value


# -------------------------------------------------
# MAIN ANALYSIS
# -------------------------------------------------
def analyze_airport(airport_code):
    delays = load_positive_delays(airport_code)

    if len(delays) < MIN_SAMPLES:
        print(f"{airport_code}: not enough positive delays ({len(delays)})")
        return None

    results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            params, loglik, aic, ks_stat, p_value = fit_distribution(dist, delays)

            results.append({
                "airport": airport_code,
                "distribution": name,
                "n_samples": len(delays),
                "log_likelihood": loglik,
                "aic": aic,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "parameters": str(params)
            })

        except Exception as e:
            print(f"{airport_code} | {name} failed: {e}")

    return pd.DataFrame(results)

def save_results(df_new, airport_code):

    KEEP = {"normal", "nct", "lognorm", "gamma"}

    out_dir = Path("results/distribution_analysis") / airport_code
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / "distribution_comparison.csv"

    if file.exists():
        try:
            old = pd.read_csv(file)

            if "distribution" not in old.columns:
                raise EmptyDataError

            old_keep = old[old["distribution"].isin(KEEP)]
            merged = pd.concat([old_keep, df_new], ignore_index=True)

        except EmptyDataError:
            # file exists but is empty or broken
            merged = df_new
    else:
        merged = df_new

    merged.to_csv(file, index=False)

def main():
    data_dir = Path("data/RawData")
    airports = [f.stem.replace("Delays_", "") for f in data_dir.glob("Delays_*.npy")]

    print(f"Fitting distributions for {len(airports)} airports...\n")

    for airport in tqdm(airports):
        df = analyze_airport(airport)
        if df is not None:
            save_results(df, airport)

    print("\nDONE ")
    print("Next step: python ks_test_visualization.py")


if __name__ == "__main__":
    main()
