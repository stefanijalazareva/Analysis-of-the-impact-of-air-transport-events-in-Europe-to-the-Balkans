"""
Fit positive-support distributions on POSITIVE flight delays only:
Normal, NCT, Lognormal, Gamma,
Burr XII, Generalized Gamma, Lomax, Inverse Gaussian, Exponential

Results are saved to distribution_comparison.csv per airport
and are compatible with ks_test_visualization.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import burr12, gengamma, lomax, invgauss, expon,norm,nct,lognorm,gamma
from tqdm import tqdm
import warnings
from pandas.errors import EmptyDataError
warnings.filterwarnings("ignore")


# CONFIG
DISTRIBUTIONS = {
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

MIN_SAMPLES = 100

def load_all_delays_minutes(airport_code):
    path = Path("data/RawData") / f"Delays_{airport_code}.npy"
    raw = np.load(path, allow_pickle=True)

    delays_sec = raw[:, 3].astype(float)
    delays_min = delays_sec / 60.0

    return delays_min

def load_positive_delays_minutes(airport_code):
    path = Path("data/RawData") / f"Delays_{airport_code}.npy"
    raw = np.load(path, allow_pickle=True)

    delays_sec = raw[:, 3].astype(float)
    delays_min = delays_sec[delays_sec > 0] / 60.0

    return delays_min

SIGNED_DISTS = {"normal", "nct"}
POSITIVE_DISTS = {
    "lognorm", "gamma",
    "burr12", "gengamma", "lomax", "invgauss", "expon"
}



def fit_distribution(dist_name,dist, data):
    """Fit distribution and compute LL, AIC, KS."""
    params = dist.fit(data)

    loglik = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * loglik

    ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)

    return params, loglik, aic, ks_stat, p_value



# MAIN ANALYSIS
def analyze_airport(airport_code):
    delays_all = load_all_delays_minutes(airport_code)
    delays_pos = load_positive_delays_minutes(airport_code)

    results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            if name in SIGNED_DISTS:
                data = delays_all
            else:
                data = delays_pos

            if len(data) < MIN_SAMPLES:
                continue


            params = dist.fit(data)

            loglik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)

            results.append({
                "airport": airport_code,
                "distribution": name,
                "n_samples": len(data),
                "log_likelihood": loglik,
                "aic": aic,
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "parameters": str(params),
                "data_support": "signed" if name in SIGNED_DISTS else "positive",
                "units": "minutes"
            })

        except Exception as e:
            print(f"{airport_code} | {name} failed: {e}")

    return pd.DataFrame(results)

def save_results(df, airport_code):
    out_dir = Path("results/distribution_analysis") / airport_code
    out_dir.mkdir(parents=True, exist_ok=True)

    file = out_dir / "distribution_comparison_new.csv"
    df.to_csv(file, index=False)

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


