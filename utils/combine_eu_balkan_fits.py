# combine_eu_balkan_fits.py
# Combine EU + Balkans delays on regional level, fit Normal / NCT / Burr XII,
# compute KS + bootstrap confidence intervals (parallel)
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, nct, kstest, burr
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


N_BOOT = 100
CI = 0.95             # 95% confidence interval
RANDOM_STATE = 42


def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)


def _bootstrap_one_fit(args):
    """
    One bootstrap iteration (one resample + one fit).

    Args:
        (data, dist_type, seed)

    Returns:
        list of params, or None if fit fails
    """
    data, dist_type, seed = args
    rng = np.random.default_rng(seed)

    data = np.asarray(data)
    n = len(data)

    sample = rng.choice(data, size=n, replace=True)

    try:
        if dist_type == "normal":
            mu, sigma = norm.fit(sample)
            return [mu, sigma]

        elif dist_type == "nct":
            df_val, nc, loc, scale = nct.fit(sample)
            return [df_val, nc, loc, scale]

        elif dist_type == "burr":
            c, d, loc, scale = burr.fit(sample)
            return [c, d, loc, scale]

    except Exception:
        return None


# Bootstrap CI computation (parallel or sequential)
def bootstrap_param_cis(
    data,
    dist_type,
    n_boot=N_BOOT,
    ci=CI,
    random_state=RANDOM_STATE,
    n_jobs=None,
    verbose=True,
    progress_label=""
):
    """
    Nonparametric percentile bootstrap CIs (parallel).
    Each iteration fits on full n (NO max sample cap).
    Progress logging is done ONLY in the main process:
        "iteration k/n_boot for {progress_label}"
    """
    alpha = 1 - ci
    low_q = 100 * (alpha / 2)
    high_q = 100 * (1 - alpha / 2)

    data = np.asarray(data)
    data = data[np.isfinite(data)]

    if dist_type == "burr":
        data = data[data > 0]

    n = len(data)
    if n < 200:
        return {}

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    if verbose:
        log(f"Bootstrap CI ({dist_type}) -> n={n:,}, n_boot={n_boot}, workers={n_jobs}")

    # --- sequential fallback (still logs only from main) ---
    if n_jobs == 1:
        rng = np.random.default_rng(random_state)
        boot_params = []
        failed = 0

        for i in range(1, n_boot + 1):
            seed = int(rng.integers(0, 2**32 - 1))
            res = _bootstrap_one_fit((data, dist_type, seed))
            if res is not None:
                boot_params.append(res)
            else:
                failed += 1

            if verbose:
                label = progress_label or dist_type
                log(f"iteration {i}/{n_boot} for {label}")

        if len(boot_params) < max(20, int(n_boot * 0.5)):
            log(f"  Too many failed fits for {dist_type}. Skipping CI.")
            return {}

        boot_params = np.array(boot_params)

    # --- parallel version (main process logs progress) ---
    else:
        rng = np.random.default_rng(random_state)
        seeds = rng.integers(0, 2**32 - 1, size=n_boot, dtype=np.uint32)

        tasks = [(data, dist_type, int(seeds[i])) for i in range(n_boot)]

        boot_params = []
        failed = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_bootstrap_one_fit, t) for t in tasks]

            for fut in as_completed(futures):
                res = fut.result()
                completed += 1

                if res is not None:
                    boot_params.append(res)
                else:
                    failed += 1

                if verbose:
                    label = progress_label or dist_type
                    log(f"iteration {completed}/{n_boot} for {label}")

        if len(boot_params) < max(20, int(n_boot * 0.5)):
            log(f"  Too many failed fits for {dist_type}. Skipping CI. "
                f"(success={len(boot_params)}, failed={failed})")
            return {}

        boot_params = np.array(boot_params)

    # Parameter name mapping
    if dist_type == "normal":
        names = ["Mean", "Std"]
    elif dist_type == "nct":
        names = ["df", "nc", "loc", "scale"]
    else:
        names = ["c", "d", "loc", "scale"]

    out = {}
    for j, name in enumerate(names):
        out[f"{name}_CI_low"] = float(np.percentile(boot_params[:, j], low_q))
        out[f"{name}_CI_high"] = float(np.percentile(boot_params[:, j], high_q))

    return out



# ---------------------------------------------------------
# Data loading + combining
# ---------------------------------------------------------
def load_airport_data(airport_code):
    filepath = os.path.join("data", "RawData", f"Delays_{airport_code}.npy")
    if not os.path.exists(filepath):
        log(f"File not found for {airport_code}: {filepath}")
        return None

    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=["Origin", "Destination", "Timestamp", "Delay"])

    # seconds -> minutes
    df["Delay"] = df["Delay"].astype(float) / 60.0

    # remove 0 delays
    df = df[df["Delay"] != 0]

    return df["Delay"].to_numpy()


def combine_all_delays():
    europe_airports = ["EGLL","LFPG","EHAM","EDDF","LEMD","LEBL","EDDM","EGKK","LIRF","EIDW"]
    balkans_airports = ["LATI","LQSA","LBSF","LBBG","LDZA","LDSP","LDDU","BKPR","LYTV","LWSK"]

    eu_delays, bk_delays = [], []

    log("Loading data for Europe...")
    for code in europe_airports:
        arr = load_airport_data(code)
        if arr is not None:
            eu_delays.append(arr)

    log("Loading data for Balkans...")
    for code in balkans_airports:
        arr = load_airport_data(code)
        if arr is not None:
            bk_delays.append(arr)

    eu = np.concatenate(eu_delays) if eu_delays else np.array([])
    bk = np.concatenate(bk_delays) if bk_delays else np.array([])

    return eu, bk


# ---------------------------------------------------------
# Row builders (point estimate + KS + CI)
# ---------------------------------------------------------
def fit_normal_row(region_name, data, n_boot, ci, n_jobs):
    data = data[np.isfinite(data)]

    # Point estimate fit
    mu, sigma = norm.fit(data)

    # KS test against fitted params
    ks_stat, _ = kstest(data, "norm", args=(mu, sigma))

    log(f"  Normal bootstrap for {region_name} ...")
    ci_dict = bootstrap_param_cis(
        data, "normal", n_boot=n_boot, ci=ci, n_jobs=n_jobs,
        verbose=True, progress_label=f"{region_name} Normal"
    )

    return {
        "Region": region_name,
        "Distribution": "Normal",
        "Mean": mu,
        "Mean_CI_low": ci_dict.get("Mean_CI_low"),
        "Mean_CI_high": ci_dict.get("Mean_CI_high"),
        "Std": sigma,
        "Std_CI_low": ci_dict.get("Std_CI_low"),
        "Std_CI_high": ci_dict.get("Std_CI_high"),
        "KS stat.": ks_stat,
    }


def fit_nct_row(region_name, data, n_boot, ci, n_jobs):
    data = data[np.isfinite(data)]

    # Point estimate fit
    params = nct.fit(data)
    df_val, nc_val, loc_val, scale_val = params

    # KS test against fitted params
    ks_stat, _ = kstest(data, "nct", args=params)

    log(f"  NCT bootstrap for {region_name} ...")
    ci_dict = bootstrap_param_cis(
        data, "nct", n_boot=n_boot, ci=ci, n_jobs=n_jobs,
        verbose=True, progress_label=f"{region_name} NCT"
    )

    return {
        "Region": region_name,
        "Distribution": "Noncentral t",
        "df": df_val,
        "df_CI_low": ci_dict.get("df_CI_low"),
        "df_CI_high": ci_dict.get("df_CI_high"),
        "nc": nc_val,
        "nc_CI_low": ci_dict.get("nc_CI_low"),
        "nc_CI_high": ci_dict.get("nc_CI_high"),
        "loc": loc_val,
        "loc_CI_low": ci_dict.get("loc_CI_low"),
        "loc_CI_high": ci_dict.get("loc_CI_high"),
        "scale": scale_val,
        "scale_CI_low": ci_dict.get("scale_CI_low"),
        "scale_CI_high": ci_dict.get("scale_CI_high"),
        "KS stat.": ks_stat,
    }


def fit_burr_row(region_name, data, n_boot, ci, n_jobs):
    data = data[np.isfinite(data)]
    pos = data[data > 0]
    if len(pos) < 200:
        raise RuntimeError(f"Not enough positive samples for Burr in {region_name}")

    # Point estimate fit
    params = burr.fit(pos)
    c, d, loc, scale = params

    # KS test against fitted params
    ks_stat, _ = kstest(pos, burr.cdf, args=params)

    log(f"  Burr bootstrap for {region_name} ...")
    ci_dict = bootstrap_param_cis(
        pos, "burr", n_boot=n_boot, ci=ci, n_jobs=n_jobs,
        verbose=True, progress_label=f"{region_name} Burr"
    )

    return {
        "Region": region_name,
        "Distribution": "Burr XII",
        "c": c,
        "c_CI_low": ci_dict.get("c_CI_low"),
        "c_CI_high": ci_dict.get("c_CI_high"),
        "d": d,
        "d_CI_low": ci_dict.get("d_CI_low"),
        "d_CI_high": ci_dict.get("d_CI_high"),
        "loc": loc,
        "loc_CI_low": ci_dict.get("loc_CI_low"),
        "loc_CI_high": ci_dict.get("loc_CI_high"),
        "scale": scale,
        "scale_CI_low": ci_dict.get("scale_CI_low"),
        "scale_CI_high": ci_dict.get("scale_CI_high"),
        "KS stat.": ks_stat,
    }

def enforce_order(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]


# Main
def analyze_and_compare():
    log("Starting EU vs Balkans combined fit analysis...")

    eu_delays, bk_delays = combine_all_delays()
    eu_delays = eu_delays[np.isfinite(eu_delays)]
    bk_delays = bk_delays[np.isfinite(bk_delays)]

    log(f"Loaded delays: Europe={len(eu_delays):,}, Balkans={len(bk_delays):,}")

    output_dir = os.path.join("results", "Combined_Comparison")
    os.makedirs(output_dir, exist_ok=True)
    log(f"Output directory: {output_dir}")

    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    log(f"Bootstrap settings: n_boot={N_BOOT}, CI={CI}, workers={n_jobs}")
    log("NOTE: max sample size cap is DISABLED (full n used per bootstrap iteration).")

    # ----- 1) NORMAL TABLE -----
    log("Fitting Normal distributions...")
    df_normal = pd.DataFrame([
        fit_normal_row("Europe", eu_delays, N_BOOT, CI, n_jobs),
        fit_normal_row("Balkans", bk_delays, N_BOOT, CI, n_jobs),
    ])

    normal_cols = [
        "Region", "Distribution",
        "Mean", "Mean_CI_low", "Mean_CI_high",
        "Std", "Std_CI_low", "Std_CI_high",
        "KS stat."
    ]
    df_normal = enforce_order(df_normal, normal_cols)

    normal_csv = os.path.join(output_dir, "table_normal.csv")
    df_normal.to_csv(normal_csv, index=False)
    log(f"Saved Normal table to: {normal_csv}")

    # ----- 2) NCT TABLE -----
    log("Fitting NCT distributions...")
    df_nct = pd.DataFrame([
        fit_nct_row("Europe", eu_delays, N_BOOT, CI, n_jobs),
        fit_nct_row("Balkans", bk_delays, N_BOOT, CI, n_jobs),
    ])

    nct_cols = [
        "Region", "Distribution",
        "df", "df_CI_low", "df_CI_high",
        "nc", "nc_CI_low", "nc_CI_high",
        "loc", "loc_CI_low", "loc_CI_high",
        "scale", "scale_CI_low", "scale_CI_high",
        "KS stat."
    ]
    df_nct = enforce_order(df_nct, nct_cols)

    nct_csv = os.path.join(output_dir, "table_nct.csv")
    df_nct.to_csv(nct_csv, index=False)
    log(f"Saved NCT table to: {nct_csv}")

    # ----- 3) BURR TABLE -----
    log("Fitting Burr XII distributions...")
    df_burr = pd.DataFrame([
        fit_burr_row("Europe", eu_delays, N_BOOT, CI, n_jobs),
        fit_burr_row("Balkans", bk_delays, N_BOOT, CI, n_jobs),
    ])

    burr_cols = [
        "Region", "Distribution",
        "c", "c_CI_low", "c_CI_high",
        "d", "d_CI_low", "d_CI_high",
        "loc", "loc_CI_low", "loc_CI_high",
        "scale", "scale_CI_low", "scale_CI_high",
        "KS stat."
    ]
    df_burr = enforce_order(df_burr, burr_cols)

    burr_csv = os.path.join(output_dir, "table_burr.csv")
    df_burr.to_csv(burr_csv, index=False)
    log(f"Saved Burr table to: {burr_csv}")

    # ----- Plot: Normal vs NCT PDFs -----
    log("Generating PDF comparison plot...")

    mu_eu = df_normal.loc[df_normal["Region"] == "Europe", "Mean"].values[0]
    sd_eu = df_normal.loc[df_normal["Region"] == "Europe", "Std"].values[0]
    mu_bk = df_normal.loc[df_normal["Region"] == "Balkans", "Mean"].values[0]
    sd_bk = df_normal.loc[df_normal["Region"] == "Balkans", "Std"].values[0]

    df_eu = df_nct.loc[df_nct["Region"] == "Europe", "df"].values[0]
    nc_eu = df_nct.loc[df_nct["Region"] == "Europe", "nc"].values[0]
    loc_eu = df_nct.loc[df_nct["Region"] == "Europe", "loc"].values[0]
    sc_eu = df_nct.loc[df_nct["Region"] == "Europe", "scale"].values[0]

    df_bk = df_nct.loc[df_nct["Region"] == "Balkans", "df"].values[0]
    nc_bk = df_nct.loc[df_nct["Region"] == "Balkans", "nc"].values[0]
    loc_bk = df_nct.loc[df_nct["Region"] == "Balkans", "loc"].values[0]
    sc_bk = df_nct.loc[df_nct["Region"] == "Balkans", "scale"].values[0]

    x = np.linspace(-60, 180, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, norm.pdf(x, mu_eu, sd_eu), lw=2, label="Europe - Normal")
    plt.plot(x, norm.pdf(x, mu_bk, sd_bk), lw=2, linestyle="--", label="Balkans - Normal")

    plt.plot(x, nct.pdf(x, df_eu, nc_eu, loc_eu, sc_eu), lw=2, label="Europe - NCT")
    plt.plot(x, nct.pdf(x, df_bk, nc_bk, loc_bk, sc_bk), lw=2, linestyle="--", label="Balkans - NCT")

    plt.title("Comparison of Normal vs Noncentral t Fits\nEurope vs Balkans (Combined Data)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Density")
    plt.axvline(x=0, color="gray", linestyle="--", label="On-time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "combined_normal_vs_nct.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log(f"Saved plot to: {plot_path}")

    log("Analysis complete!")


if __name__ == "__main__":
    analyze_and_compare()
