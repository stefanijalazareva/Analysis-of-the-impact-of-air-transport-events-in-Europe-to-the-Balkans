import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisk, burr, gengamma, weibull_min
import warnings
warnings.filterwarnings('ignore')

def load_airport_data(airport_code):
    """Load data for a specific airport and convert to DataFrame."""
    filepath = os.path.join('data', 'RawData', f'Delays_{airport_code}.npy')

    if not os.path.exists(filepath):
        print(f"Data file for {airport_code} not found.")
        return None

    data = np.load(filepath, allow_pickle=True)
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'ScheduledTimestamp', 'Delay'])

    df['ScheduledTimestamp'] = df['ScheduledTimestamp'].astype(float)
    df['Delay'] = df['Delay'].astype(float)
    df['PositiveDelay'] = df['Delay'].clip(lower=0)
    df['NegativeDelay'] = (-df['Delay']).clip(lower=0)

    return df

def fit_all_new_distributions(delays, airport_code, airport_name, output_dir, delay_type='positive'):
    """Fit all new distributions and compare them."""
    os.makedirs(output_dir, exist_ok=True)

    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    # Define all new distributions to test (using correct scipy names)
    new_distributions = [
        ('Log-Logistic (Fisk)', fisk),
        ('Burr XII', burr),
        ('Generalized Gamma', gengamma),
        ('Weibull Min', weibull_min),
        # ('Double Weibull', dweibull)  # Commented out as it may not be available in all scipy versions
    ]

    all_results = []
    successful_fits = []

    for dist_name, distribution in new_distributions:
        try:
            print(f"  Fitting {dist_name}...")

            # Fit distribution
            params = distribution.fit(delays_minutes)

            # Calculate goodness of fit metrics
            ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)

            try:
                log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))
                if not np.isfinite(log_likelihood):
                    raise ValueError(f"Invalid log-likelihood for {dist_name}")
            except:
                print(f"    Warning: Could not calculate log-likelihood for {dist_name}")
                continue

            n = len(delays_minutes)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Calculate Anderson-Darling test
            try:
                ad_stat, critical_vals, significance_level = stats.anderson(delays_minutes, dist='norm')
            except:
                ad_stat = np.nan

            # Calculate percentiles
            try:
                p90 = distribution.ppf(0.90, *params)
                p95 = distribution.ppf(0.95, *params)
                p99 = distribution.ppf(0.99, *params)
                median = distribution.ppf(0.50, *params)
            except:
                p90 = p95 = p99 = median = np.nan

            # Calculate moments if possible
            try:
                mean_est = distribution.mean(*params)
                var_est = distribution.var(*params)
            except:
                mean_est = var_est = np.nan

            result = {
                'Airport': airport_code,
                'Airport_Name': airport_name,
                'Delay_Type': delay_type,
                'Distribution': dist_name,
                'Parameters': params,
                'Num_Parameters': k,
                'KS_Statistic': ks_stat,
                'KS_P_value': p_value,
                'AD_Statistic': ad_stat,
                'Log_Likelihood': log_likelihood,
                'AIC': aic,
                'BIC': bic,
                'Mean': mean_est,
                'Variance': var_est,
                'Median': median,
                'P90': p90,
                'P95': p95,
                'P99': p99,
                'Sample_Size': n,
                'Data_Mean': np.mean(delays_minutes),
                'Data_Std': np.std(delays_minutes),
                'Data_Median': np.median(delays_minutes),
                'Data_P90': np.percentile(delays_minutes, 90),
                'Data_P95': np.percentile(delays_minutes, 95),
                'Data_P99': np.percentile(delays_minutes, 99)
            }

            all_results.append(result)
            successful_fits.append((dist_name, distribution, params))

        except Exception as e:
            print(f"    Error fitting {dist_name}: {e}")

    if not all_results:
        return None

    # Sort by AIC
    all_results.sort(key=lambda x: x['AIC'])

    # Create comprehensive comparison visualization
    create_comprehensive_comparison_plot(delays_minutes, all_results, successful_fits,
                                       airport_code, airport_name, delay_type, output_dir)

    return all_results

# ...existing code...
