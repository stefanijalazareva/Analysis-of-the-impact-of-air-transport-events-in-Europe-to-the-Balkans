import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (fisk, burr12, gengamma, weibull_min, gamma, lognorm,
                        expon, lomax, invgauss, beta, pareto, chi2,
                        f, t, norm, uniform, laplace, logistic)
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

def fit_comprehensive_distributions(delays, airport_code, airport_name, delay_type='positive'):
    """Fit comprehensive set of distributions and return results."""
    delays_nonzero = delays[delays > 0]
    if len(delays_nonzero) < 100:
        print(f"Not enough non-zero {delay_type} delay samples for {airport_code}")
        return None

    delays_minutes = delays_nonzero / 60

    # Define distributions to test
    distributions = [
        ('Log-Normal', lognorm),
        ('Gamma', gamma),
        ('Weibull', weibull_min),
        ('Log-Logistic', fisk),
        ('Burr XII', burr12),
        ('Generalized Gamma', gengamma),
        ('Exponential', expon),
        ('Lomax', lomax),
        ('Inverse Gaussian', invgauss),
        ('Beta', beta),
        ('Pareto', pareto),
        ('Chi-Square', chi2),
        ('F-Distribution', f),
        ('T-Distribution', t),
        ('Normal', norm),
        ('Uniform', uniform),
        ('Laplace', laplace),
        ('Logistic', logistic)
    ]

    results = []

    for dist_name, distribution in distributions:
        try:
            print(f"  Fitting {dist_name}...")

            # Special handling for different distributions
            if dist_name == 'Beta':
                # Beta distribution needs data in [0,1], so normalize
                norm_data = (delays_minutes - delays_minutes.min()) / (delays_minutes.max() - delays_minutes.min())
                norm_data = norm_data * 0.999 + 0.0005
                params = distribution.fit(norm_data)
                ks_stat, p_value = stats.kstest(norm_data, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(norm_data, *params))
                p90_norm = distribution.ppf(0.90, *params)
                p95_norm = distribution.ppf(0.95, *params)
                p99_norm = distribution.ppf(0.99, *params)
                p90 = p90_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()
                p95 = p95_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()
                p99 = p99_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()

            elif dist_name == 'Uniform':
                params = (delays_minutes.min(), delays_minutes.max() - delays_minutes.min())
                ks_stat, p_value = stats.kstest(delays_minutes, lambda x: uniform.cdf(x, *params))
                log_likelihood = np.sum(uniform.logpdf(delays_minutes, *params))
                p90 = uniform.ppf(0.90, *params)
                p95 = uniform.ppf(0.95, *params)
                p99 = uniform.ppf(0.99, *params)

            elif dist_name == 'Chi-Square':
                if delays_minutes.min() <= 0:
                    delays_minutes_pos = delays_minutes + 1e-6
                else:
                    delays_minutes_pos = delays_minutes
                params = distribution.fit(delays_minutes_pos)
                ks_stat, p_value = stats.kstest(delays_minutes_pos, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(delays_minutes_pos, *params))
                p90 = distribution.ppf(0.90, *params)
                p95 = distribution.ppf(0.95, *params)
                p99 = distribution.ppf(0.99, *params)

            elif dist_name == 'F-Distribution':
                params = distribution.fit(delays_minutes)
                if params[0] <= 0 or params[1] <= 0:
                    continue
                ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))
                p90 = distribution.ppf(0.90, *params)
                p95 = distribution.ppf(0.95, *params)
                p99 = distribution.ppf(0.99, *params)

            else:
                params = distribution.fit(delays_minutes)
                ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))
                p90 = distribution.ppf(0.90, *params)
                p95 = distribution.ppf(0.95, *params)
                p99 = distribution.ppf(0.99, *params)

            n = len(delays_minutes)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            results.append({
                'Airport': airport_code,
                'Airport_Name': airport_name,
                'Delay_Type': delay_type,
                'Distribution': dist_name,
                'KS_Statistic': ks_stat,
                'P_value': p_value,
                'AIC': aic,
                'BIC': bic,
                'P90': p90,
                'P95': p95,
                'P99': p99,
                'Sample_Size': n,
                'Data_P90': np.percentile(delays_minutes, 90),
                'Data_P95': np.percentile(delays_minutes, 95),
                'Data_P99': np.percentile(delays_minutes, 99),
                'Parameters': str(params)
            })

        except Exception as e:
            print(f"    Error fitting {dist_name}: {e}")

    return results

def create_detailed_statistical_report(airport_code, airport_name, results):
    """Create comprehensive statistical report with detailed analysis."""
    output_dir = os.path.join('results', 'detailed_statistical_reports')
    os.makedirs(output_dir, exist_ok=True)

    pos_results = [r for r in results if r['Delay_Type'] == 'positive']
    if not pos_results:
        return

    pos_df = pd.DataFrame(pos_results).sort_values('AIC')
    best_dist = pos_df.iloc[0]

    # Calculate additional statistics
    total_fits = len(pos_df)
    significant_fits = len(pos_df[pos_df['P_value'] > 0.05])

    # Evidence strength calculation
    if len(pos_df) > 1:
        aic_diff = pos_df.iloc[1]['AIC'] - best_dist['AIC']
        if aic_diff > 10: evidence = "Very Strong"
        elif aic_diff > 4: evidence = "Strong"
        elif aic_diff > 2: evidence = "Moderate"
        else: evidence = "Weak"
        evidence_score = min(aic_diff, 20) / 20 * 100
        aic_diff_str = f"{aic_diff:.2f}"
    else:
        evidence = "Single model"
        evidence_score = 100
        aic_diff_str = "N/A"

    # Model quality assessment
    if best_dist['P_value'] > 0.1:
        quality = "Excellent"
    elif best_dist['P_value'] > 0.05:
        quality = "Good"
    elif best_dist['P_value'] > 0.01:
        quality = "Fair"
    else:
        quality = "Poor"

    # Prediction accuracy analysis
    p95_error = abs(best_dist['P95'] - best_dist['Data_P95']) / best_dist['Data_P95'] * 100
    p90_error = abs(best_dist['P90'] - best_dist['Data_P90']) / best_dist['Data_P90'] * 100
    p99_error = abs(best_dist['P99'] - best_dist['Data_P99']) / best_dist['Data_P99'] * 100

    avg_error = (p90_error + p95_error + p99_error) / 3

    report_content = f"""# Statistical Distribution Analysis Report
## {airport_name} ({airport_code})

### Executive Summary
- **Analysis Date:** November 26, 2025
- **Recommended Distribution:** {best_dist['Distribution']}
- **Model Quality:** {quality}
- **Evidence Strength:** {evidence} (Score: {evidence_score:.1f}/100)
- **Sample Size:** {best_dist['Sample_Size']:,} positive delays
- **Statistical Significance:** {'PASS' if best_dist['P_value'] > 0.05 else 'FAIL'} (p = {best_dist['P_value']:.6f})

### Key Performance Indicators
| Metric | Value | Assessment |
|--------|--------|------------|
| AIC Score | {best_dist['AIC']:.2f} | {'Excellent' if best_dist['AIC'] < pos_df['AIC'].median() else 'Good'} |
| KS Statistic | {best_dist['KS_Statistic']:.4f} | {'Good' if best_dist['KS_Statistic'] < 0.05 else 'Moderate' if best_dist['KS_Statistic'] < 0.1 else 'Poor'} |
| p-value | {best_dist['P_value']:.6f} | {'Significant' if best_dist['P_value'] > 0.05 else 'Not Significant'} |
| Prediction Accuracy | {100-avg_error:.1f}% | {'Excellent' if avg_error < 5 else 'Good' if avg_error < 10 else 'Fair' if avg_error < 20 else 'Poor'} |

### Complete Distribution Analysis Results

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | Quality Score |
|------|--------------|-----|-----|---------|---------|-------------|---------------|
"""

    for i, (_, row) in enumerate(pos_df.iterrows()):
        rank = i + 1
        sig = 'Yes' if row['P_value'] > 0.05 else 'No'

        # Calculate quality score (0-100)
        aic_norm = max(0, 100 - (row['AIC'] - pos_df['AIC'].min()) / (pos_df['AIC'].max() - pos_df['AIC'].min()) * 50)
        p_norm = min(100, row['P_value'] * 1000)
        ks_norm = max(0, 100 - row['KS_Statistic'] * 1000)
        quality_score = (aic_norm + p_norm + ks_norm) / 3

        report_content += f"| {rank} | {row['Distribution']} | {row['AIC']:.0f} | {row['BIC']:.0f} | {row['KS_Statistic']:.4f} | {row['P_value']:.4f} | {sig} | {quality_score:.1f} |\n"

    report_content += f"""
### Best Model Detailed Analysis

#### {best_dist['Distribution']} Distribution Parameters
```
{best_dist['Parameters']}
```

#### Goodness of Fit Assessment
- **Kolmogorov-Smirnov Test:**
  - Statistic: {best_dist['KS_Statistic']:.4f}
  - p-value: {best_dist['P_value']:.6f}
  - Result: {'Model fits data well' if best_dist['P_value'] > 0.05 else 'Model does not fit data well'}

- **Information Criteria:**
  - AIC: {best_dist['AIC']:.2f}
  - BIC: {best_dist['BIC']:.2f}
  - Ranking: 1st out of {total_fits} distributions tested

#### Predictive Performance Analysis

| Percentile | Model Prediction | Actual Data | Absolute Error | Relative Error |
|------------|------------------|-------------|----------------|----------------|
| 90th | {best_dist['P90']:.2f} min | {best_dist['Data_P90']:.2f} min | {abs(best_dist['P90'] - best_dist['Data_P90']):.2f} min | {p90_error:.1f}% |
| 95th | {best_dist['P95']:.2f} min | {best_dist['Data_P95']:.2f} min | {abs(best_dist['P95'] - best_dist['Data_P95']):.2f} min | {p95_error:.1f}% |
| 99th | {best_dist['P99']:.2f} min | {best_dist['Data_P99']:.2f} min | {abs(best_dist['P99'] - best_dist['Data_P99']):.2f} min | {p99_error:.1f}% |

**Overall Prediction Accuracy:** {100-avg_error:.1f}%

### Statistical Comparison Analysis

#### Model Selection Evidence
- **Evidence Strength:** {evidence}
- **AIC Difference from 2nd Best:** {aic_diff_str}
- **Probability of Being Best Model:** {evidence_score:.1f}%

#### Distribution Family Analysis
"""

    # Group by distribution families
    families = {
        'Continuous Heavy-Tailed': ['Burr XII', 'Pareto', 'Lomax', 'Log-Logistic'],
        'Light-Tailed': ['Normal', 'Log-Normal', 'Exponential', 'Weibull'],
        'Flexible': ['Beta', 'Gamma', 'Generalized Gamma'],
        'Location-Scale': ['Logistic', 'Laplace', 'T-Distribution', 'Uniform'],
        'Special Cases': ['Chi-Square', 'F-Distribution', 'Inverse Gaussian']
    }

    for family, distributions in families.items():
        family_results = pos_df[pos_df['Distribution'].isin(distributions)]
        if not family_results.empty:
            best_in_family = family_results.iloc[0]
            report_content += f"\n- **{family} Family:** Best = {best_in_family['Distribution']} (AIC: {best_in_family['AIC']:.0f})"

    report_content += f"""

### Operational Recommendations

#### For Air Traffic Management
1. **Delay Planning Threshold (95th percentile):** {best_dist['P95']:.0f} minutes
2. **Extreme Event Threshold (99th percentile):** {best_dist['P99']:.0f} minutes
3. **Expected Daily Maximum Delay:** {best_dist['P99'] * 1.2:.0f} minutes

#### For Capacity Planning
- **Buffer Time Recommendation:** {best_dist['P95'] * 0.8:.0f} minutes
- **Schedule Padding:** {best_dist['P90']:.0f} minutes for 90% on-time performance
- **Crisis Management Trigger:** Delays exceeding {best_dist['P99']:.0f} minutes

#### Model Reliability Assessment
- **Confidence Level:** {'High' if quality == 'Excellent' else 'Medium' if quality == 'Good' else 'Low'}
- **Recommended Update Frequency:** {'Annually' if quality in ['Excellent', 'Good'] else 'Quarterly' if quality == 'Fair' else 'Monthly'}
- **Model Validation Status:** {'Approved for operational use' if best_dist['P_value'] > 0.05 else 'Requires validation before operational use'}

### Technical Appendix

#### Data Quality Assessment
- **Sample Size Adequacy:** {'Excellent' if best_dist['Sample_Size'] > 10000 else 'Good' if best_dist['Sample_Size'] > 1000 else 'Adequate' if best_dist['Sample_Size'] > 100 else 'Limited'} ({best_dist['Sample_Size']:,} observations)
- **Data Range:** {pos_df.iloc[0]['Data_P90']:.1f} - {pos_df.iloc[0]['Data_P99']:.1f} minutes (90th-99th percentile)
- **Distribution Characteristics:** {'Right-skewed heavy-tailed' if best_dist['Distribution'] in ['Burr XII', 'Pareto', 'Log-Normal'] else 'Moderate skewness' if best_dist['Distribution'] in ['Gamma', 'Weibull'] else 'Light-tailed'}

#### Alternative Models
Top 3 alternative models in case of model failure:
"""

    for i, (_, row) in enumerate(pos_df.iloc[1:4].iterrows()):
        rank = i + 2
        report_content += f"{rank}. **{row['Distribution']}** - AIC: {row['AIC']:.0f}, p-value: {row['P_value']:.4f}\n"

    report_content += f"""
#### Model Assumptions and Limitations
- **Independence:** Assumes delay observations are independent
- **Stationarity:** Assumes delay distribution is constant over time
- **Completeness:** Based on positive delays only
- **Seasonal Effects:** Not explicitly modeled
- **External Factors:** Weather, strikes, etc. not considered

---
*Report generated by Enhanced Statistical Distribution Analysis System v2.0*  
*Analysis Date: November 26, 2025*  
*Total Distributions Tested: {total_fits}*  
*Statistical Significance Rate: {significant_fits}/{total_fits} ({significant_fits/total_fits*100:.1f}%)*
"""

    filename = f"{airport_code}_{airport_name.replace(' ', '_')}_detailed_statistical_report.md"
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Generated detailed statistical report for {airport_name}")

def generate_reports_only():
    """Generate only the detailed statistical reports for all airports."""
    airport_codes = [
        'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW',  # Europe
        'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'   # Balkans
    ]

    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    print("Generating Detailed Statistical Reports...")
    print(f"Processing {len(airport_codes)} airports")

    successful_reports = 0

    for airport_code in airport_codes:
        try:
            airport_name = airport_names.get(airport_code, airport_code)
            print(f"\nGenerating report for {airport_name} ({airport_code})...")

            df = load_airport_data(airport_code)
            if df is None:
                continue

            # Analyze positive delays
            pos_results = fit_comprehensive_distributions(
                df['PositiveDelay'], airport_code, airport_name, 'positive'
            )

            if pos_results:
                create_detailed_statistical_report(airport_code, airport_name, pos_results)
                successful_reports += 1

        except Exception as e:
            print(f"Error generating report for {airport_code}: {e}")

    print(f"\nREPORT GENERATION COMPLETE!")
    print(f"Successfully generated: {successful_reports}/{len(airport_codes)} detailed statistical reports")
    print(f"Reports saved to: results/detailed_statistical_reports/")

if __name__ == "__main__":
    generate_reports_only()
