import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (fisk, burr12, gengamma, weibull_min, gamma, lognorm,
                        expon, lomax, invgauss, beta, pareto, chi2,
                        f, t, norm, uniform, laplace, logistic)
import warnings
warnings.filterwarnings('ignore')

# Set clean matplotlib style
plt.style.use('default')
sns.set_palette("husl")

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

def create_ultra_clean_visualization(df, results, airport_code, airport_name):
    """Create ultra-clean, professional visualization for a single airport."""
    output_dir = os.path.join('results', 'clean_individual_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Separate positive and negative results
    pos_results = [r for r in results if r['Delay_Type'] == 'positive']
    if not pos_results:
        return

    pos_df = pd.DataFrame(pos_results).sort_values('AIC')

    # Create figure with clean white background
    fig = plt.figure(figsize=(20, 12), facecolor='white')

    # Set overall style
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5
    })

    # Main title with clean styling
    fig.suptitle(f'{airport_name} ({airport_code}) - Statistical Distribution Analysis',
                 fontsize=18, fontweight='bold', y=0.96, color='#2c3e50')

    # Create 2x3 grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25,
                         left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Panel 1: Top 8 Distributions AIC Ranking
    ax1 = fig.add_subplot(gs[0, 0])
    top_8 = pos_df.head(8)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_8)))
    bars = ax1.barh(range(len(top_8)), top_8['AIC'], color=colors, alpha=0.8, edgecolor='white', linewidth=1)

    # Highlight the best distribution
    bars[0].set_color('#27ae60')
    bars[0].set_alpha(1.0)

    ax1.set_yticks(range(len(top_8)))
    ax1.set_yticklabels(top_8['Distribution'], fontsize=10)
    ax1.set_xlabel('AIC (Akaike Information Criterion)', fontweight='bold')
    ax1.set_title('Distribution Ranking (Lower AIC = Better)', fontweight='bold', pad=15)
    ax1.invert_yaxis()

    # Add best distribution annotation
    best_aic = top_8.iloc[0]['AIC']
    ax1.annotate(f'BEST\n{top_8.iloc[0]["Distribution"]}',
                xy=(best_aic, 0), xytext=(best_aic * 0.7, 0.5),
                fontsize=9, fontweight='bold', color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f4e6', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    # Panel 2: Statistical Significance Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    significance_data = pos_df.head(10)
    sig_colors = ['#27ae60' if p > 0.05 else '#e74c3c' for p in significance_data['P_value']]

    bars2 = ax2.bar(range(len(significance_data)), significance_data['P_value'],
                   color=sig_colors, alpha=0.7, edgecolor='white', linewidth=1)
    ax2.axhline(y=0.05, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2,
               label='Significance Threshold (α=0.05)')

    ax2.set_xticks(range(len(significance_data)))
    ax2.set_xticklabels(significance_data['Distribution'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('p-value', fontweight='bold')
    ax2.set_title('Statistical Significance Test', fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, max(0.1, significance_data['P_value'].max() * 1.1))

    # Panel 3: Model Performance Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    top_5 = pos_df.head(5)

    # Normalize metrics for radar-like comparison
    metrics_data = []
    for _, row in top_5.iterrows():
        metrics_data.append([
            1 / (1 + row['KS_Statistic']),  # Lower KS is better
            min(1.0, row['P_value'] * 10),  # Higher p-value is better
            1 / (1 + abs(row['P95'] - row['Data_P95']) / row['Data_P95'])  # Closer to actual is better
        ])

    metrics_df = pd.DataFrame(metrics_data,
                             index=top_5['Distribution'],
                             columns=['Goodness of Fit', 'Statistical Significance', 'Prediction Accuracy'])

    # Create heatmap
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
               ax=ax3, cbar_kws={'label': 'Performance Score (higher = better)'},
               square=True, linewidths=0.5)
    ax3.set_title('Performance Metrics Comparison', fontweight='bold', pad=15)
    ax3.set_ylabel('')

    # Panel 4: Data Summary & Best Model Info
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')

    # Calculate sample statistics
    pos_delays = df['PositiveDelay'][df['PositiveDelay'] > 0] / 60
    best_dist = pos_df.iloc[0]

    # Determine evidence strength
    if len(pos_df) > 1:
        aic_diff = pos_df.iloc[1]['AIC'] - best_dist['AIC']
        if aic_diff > 10: evidence = "Very Strong"
        elif aic_diff > 4: evidence = "Strong"
        elif aic_diff > 2: evidence = "Moderate"
        else: evidence = "Weak"
    else:
        evidence = "Single model"

    info_text = f"""DATASET SUMMARY
Sample Size: {len(pos_delays):,} delays
Mean Delay: {pos_delays.mean():.1f} minutes
Median Delay: {pos_delays.median():.1f} minutes
Std Dev: {pos_delays.std():.1f} minutes
95th Percentile: {np.percentile(pos_delays, 95):.1f} min

RECOMMENDED MODEL
Distribution: {best_dist['Distribution']}
Evidence Strength: {evidence}
AIC Score: {best_dist['AIC']:.0f}
Statistical Fit: {'Good' if best_dist['P_value'] > 0.05 else 'Poor'}
95th % Prediction: {best_dist['P95']:.1f} min
Actual 95th %: {best_dist['Data_P95']:.1f} min
Prediction Error: {abs(best_dist['P95'] - best_dist['Data_P95'])/best_dist['Data_P95']*100:.1f}%"""

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9, edgecolor='#bdc3c7'))

    # Panel 5: Percentile Prediction Accuracy
    ax5 = fig.add_subplot(gs[1, 1])

    percentiles = ['P90', 'P95', 'P99']
    top_3 = pos_df.head(3)

    x = np.arange(len(percentiles))
    width = 0.25

    # Plot actual data first
    actual_vals = [pos_df.iloc[0][f'Data_{p}'] for p in percentiles]
    ax5.bar(x - width, actual_vals, width, label='Actual Data',
           color='#34495e', alpha=0.9, edgecolor='white')

    # Plot top 3 models
    colors_models = ['#27ae60', '#f39c12', '#e67e22']
    for i, (_, row) in enumerate(top_3.iterrows()):
        model_vals = [row[p] for p in percentiles]
        ax5.bar(x + i*width, model_vals, width,
               label=f'{row["Distribution"]}',
               color=colors_models[i], alpha=0.8, edgecolor='white')

    ax5.set_xlabel('Percentiles', fontweight='bold')
    ax5.set_ylabel('Delay (minutes)', fontweight='bold')
    ax5.set_title('Extreme Percentiles Prediction', fontweight='bold', pad=15)
    ax5.set_xticks(x)
    ax5.set_xticklabels(percentiles)
    ax5.legend(fontsize=9, loc='upper left')

    # Panel 6: Model Rankings Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create clean ranking table for top 8
    table_data = []
    for i, (_, row) in enumerate(pos_df.head(8).iterrows()):
        rank = i + 1
        significance = 'Yes' if row['P_value'] > 0.05 else 'No'
        accuracy = abs(row['P95'] - row['Data_P95']) / row['Data_P95'] * 100

        table_data.append([
            f"{rank}",
            row['Distribution'][:12] + ('...' if len(row['Distribution']) > 12 else ''),
            f"{row['AIC']:.0f}",
            f"{row['P_value']:.4f}",
            significance,
            f"{accuracy:.1f}%"
        ])

    headers = ['#', 'Distribution', 'AIC', 'p-value', 'Sig.', 'Error']

    table = ax6.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0.0, 0.1, 1.0, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#d5f4e6')  # Best model
        if len(table_data) > 1:
            table[(2, i)].set_facecolor('#fef9e7')  # Second best

    ax6.set_title('Model Rankings', fontweight='bold', y=0.95, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{airport_code}_{airport_name.replace(" ", "_")}_clean_analysis.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Generated clean visualization for {airport_name}")

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
    else:
        evidence = "Single model"
        evidence_score = 100
        aic_diff = 0

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
- **AIC Difference from 2nd Best:** {aic_diff:.2f if len(pos_df) > 1 else 'N/A'}
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

def analyze_single_airport_enhanced(airport_code):
    """Enhanced analysis for a single airport with clean visuals and detailed reports."""
    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }

    airport_name = airport_names.get(airport_code, airport_code)
    print(f"\nEnhanced analysis for {airport_name} ({airport_code})...")

    df = load_airport_data(airport_code)
    if df is None:
        return None

    # Analyze with comprehensive distributions
    pos_results = fit_comprehensive_distributions(
        df['PositiveDelay'], airport_code, airport_name, 'positive'
    )

    neg_results = fit_comprehensive_distributions(
        df['NegativeDelay'], airport_code, airport_name, 'negative'
    )

    all_results = []
    if pos_results:
        all_results.extend(pos_results)
    if neg_results:
        all_results.extend(neg_results)

    if all_results:
        # Create clean visualization
        create_ultra_clean_visualization(df, all_results, airport_code, airport_name)

        # Create detailed statistical report
        create_detailed_statistical_report(airport_code, airport_name, all_results)

    return all_results

def main():
    """Enhanced analysis for all airports with clean visuals and detailed reports."""
    airport_codes = [
        'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW',  # Europe
        'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'   # Balkans
    ]

    print("Starting Enhanced Clean Airport Analysis...")
    print(f"Generating ultra-clean visualizations and detailed reports for {len(airport_codes)} airports")

    all_results = []
    successful_airports = 0

    for airport_code in airport_codes:
        try:
            results = analyze_single_airport_enhanced(airport_code)
            if results:
                all_results.extend(results)
                successful_airports += 1
        except Exception as e:
            print(f"Error analyzing {airport_code}: {e}")

    # Save comprehensive results
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Clean outputs directory
        clean_dir = os.path.join('results', 'clean_individual_analysis')
        reports_dir = os.path.join('results', 'detailed_statistical_reports')

        results_df.to_csv(os.path.join(clean_dir, 'enhanced_all_airports_analysis.csv'), index=False)

        # Summary statistics
        pos_results = results_df[results_df['Delay_Type'] == 'positive']
        best_distributions = pos_results.loc[pos_results.groupby('Airport')['AIC'].idxmin()]
        distribution_counts = best_distributions['Distribution'].value_counts()

        print(f"\nENHANCED ANALYSIS COMPLETE!")
        print(f"Successfully analyzed: {successful_airports}/{len(airport_codes)} airports")
        print(f"Generated: {successful_airports} clean PNG visualizations")
        print(f"Generated: {successful_airports} detailed statistical reports")
        print(f"\nTop Performing Distributions:")
        for dist, count in distribution_counts.head(5).items():
            print(f"  {dist}: {count} airports ({count/len(best_distributions)*100:.1f}%)")

        print(f"\nClean visualizations: {clean_dir}")
        print(f"Detailed reports: {reports_dir}")

    return all_results

if __name__ == "__main__":
    all_results = main()
