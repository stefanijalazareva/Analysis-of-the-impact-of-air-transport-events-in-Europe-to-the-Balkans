import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import nct
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Folder where the raw delay files (.npy) are stored
DATA_DIR = "data/RawData"

# Output directory for all bootstrap CI results (plots + table)
OUTPUT_DIR = Path("results/NCT_confidence_intervals_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Number of bootstrap repetitions (can increase later)
N_BOOT = 100

# Number of CPU cores to use in parallel (adjust if needed)
N_JOBS = 4

# Confidence interval width (95% CI)
CI_LEVEL = 95

AIRPORT_BATCH = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LEBL", "EDDM", "EGKK", "LIRF", "EIDW",
                 "LATI", "LQSA", "LBSF", "LBBG", "LDZA", "LDSP", "LDDU", "BKPR", "LYTV", "LWSK"]

def load_airport_data(icao):
    """
    Loads the delay data for one airport from .npy file,
    converts delays from seconds ➜ minutes,
    removes exact-zero delay values.
    """

    file_path = os.path.join(DATA_DIR, f"Delays_{icao}.npy")

    # If missing file → skip
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return None

    # Load numpy array: columns = Origin, Destination, Timestamp, Delay
    arr = np.load(file_path, allow_pickle=True)

    df = pd.DataFrame(arr, columns=["Origin", "Destination", "Timestamp", "Delay"])

    # Convert Delay from seconds → minutes
    x = df["Delay"].astype(float) / 60.0

    # Remove exact-on-time delays (= 0)
    x = x[x != 0].dropna()

    return x.to_numpy()

#               ONE WORKER TASK FOR MULTIPROCESSING
def single_bootstrap_fit(delays, seed):
    """
    One bootstrap sample + NCT fit.

    This is the function executed in PARALLEL by different CPU cores.
    """

    try:
        rng = np.random.default_rng(seed)
        n = len(delays)

        # Sample n values WITH replacement
        idx = rng.integers(0, n, size=n)
        sample = delays[idx]

        # Fit NCT distribution for this bootstrap sample
        params = nct.fit(sample)

        return params

    except Exception:
        return None    # In case of numerical failure


#              RUN PARALLEL BOOTSTRAP FOR ONE AIRPORT

def run_bootstrap_parallel(airport, delays, n_boot=N_BOOT, n_jobs=N_JOBS):
    """
    Does:
    1. Fit original NCT parameters
    2. Run bootstrap in parallel across CPU cores
    3. Collect results
    4. Compute CI
    """

    print(f" Starting bootstrap for airport: {airport}")

    #  FIT ORIGINAL DISTRIBUTION
    print(" STEP 1 — Fitting NCT to original data...")
    orig_params = nct.fit(delays)
    print(f"   ✔ Original fit OK:")
    print(f"     df={orig_params[0]:.3f}, nc={orig_params[1]:.3f}, loc={orig_params[2]:.3f}, scale={orig_params[3]:.3f}")

    #  PREPARE BOOTSTRAP SEEDS FOR REPRODUCIBILITY
    print("\n STEP 2 — Preparing random seeds...")
    rng = np.random.default_rng(12345)
    seeds = rng.integers(0, 2**32 - 1, size=n_boot)
    print(f"  Generated {len(seeds)} seeds")

    #  RUN BOOTSTRAP IN PARALLEL
    print("\n STEP 3 — Running multiprocess bootstrap...")

    bootstrap_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(single_bootstrap_fit, delays, s) for s in seeds]

        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                bootstrap_results.append(res)

            completed += 1
            if completed % 10 == 0:
                print(f"   → Progress: {completed}/{n_boot} bootstrap samples completed")

    bootstrap_results = np.array(bootstrap_results)

    print(f" Successful bootstrap fits: {len(bootstrap_results)}")


    #  COMPUTE CONFIDENCE INTERVALS
    print("\n STEP 4 — Computing confidence intervals...")

    lo = (100 - CI_LEVEL) / 2
    hi = 100 - lo

    ci_lower = np.percentile(bootstrap_results, lo, axis=0)
    ci_upper = np.percentile(bootstrap_results, hi, axis=0)

    param_names = ["df", "nc", "loc", "scale"]
    cis = {name: (ci_lower[i], ci_upper[i]) for i, name in enumerate(param_names)}

    print("  CI computed successfully")

    return orig_params, cis, bootstrap_results

#           SAVE OUTPUTS (TABLE + PLOT)
def save_results(airport, orig, cis, boot_samples):
    """
    Saves:
    1. CSV row with parameters + CI (append mode)
    2. Whisker + boxplot combined figure
    """

    #  SAVE TO CSV
    out_csv = OUTPUT_DIR / "bootstrap_CI_all_airports.csv"

    print(" STEP 5 — Saving results to table...")

    row = {
        "airport": airport,

        "df": orig[0],
        "df_low": cis["df"][0],
        "df_high": cis["df"][1],

        "nc": orig[1],
        "nc_low": cis["nc"][0],
        "nc_high": cis["nc"][1],

        "loc": orig[2],
        "loc_low": cis["loc"][0],
        "loc_high": cis["loc"][1],

        "scale": orig[3],
        "scale_low": cis["scale"][0],
        "scale_high": cis["scale"][1],
    }

    df_row = pd.DataFrame([row])

    if out_csv.exists():
        # Append new rows
        df_old = pd.read_csv(out_csv)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
        df_new.to_csv(out_csv, index=False)
        print(f"    Updated existing CSV file: {out_csv}")
    else:
        # Create new file
        df_row.to_csv(out_csv, index=False)
        print(f"    Created new CSV file: {out_csv}")


    #  SAVE PLOT (boxplot + whisker)
    print(" STEP 6 — Saving visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    param_names = ["df", "nc", "loc", "scale"]

    # --- Boxplot of bootstrap distributions ---
    sns.boxplot(data=boot_samples, ax=axes[0])
    axes[0].set_xticklabels(param_names)
    axes[0].set_title(f"{airport} — Bootstrap Boxplot")

    # --- Whisker Plot ---
    est = orig
    ci_low = np.array([cis[p][0] for p in param_names])
    ci_high = np.array([cis[p][1] for p in param_names])

    axes[1].errorbar(
        np.arange(4), est,
        yerr=[est - ci_low, ci_high - est],
        fmt='o', color='red', capsize=6
    )
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(param_names)
    axes[1].set_title(f"{airport} — Confidence Intervals")

    plt.tight_layout()
    filename = OUTPUT_DIR / f"{airport}_bootstrap_plot.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"   Saved plot: {filename}")


#                        MAIN DRIVER
if __name__ == "__main__":

    for code in AIRPORT_BATCH:

        print(f" PROCESSING AIRPORT: {code}")

        delays = load_airport_data(code)

        if delays is None or len(delays) < 50:
            print(f" Skipping {code}: insufficient data")
            continue

        # Run full parallel bootstrap
        orig, cis, boots = run_bootstrap_parallel(code, delays)

        # Save CSV row + plot
        save_results(code, orig, cis, boots)

    print("\n DONE ")
