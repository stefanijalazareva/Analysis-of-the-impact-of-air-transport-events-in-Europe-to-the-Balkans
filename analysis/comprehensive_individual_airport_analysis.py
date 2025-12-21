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
    
    # Define ALL distributions to test including the ones you requested
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
                # Add small epsilon to avoid 0 and 1
                norm_data = norm_data * 0.999 + 0.0005
                params = distribution.fit(norm_data)
                # Calculate metrics on normalized data
                ks_stat, p_value = stats.kstest(norm_data, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(norm_data, *params))
                # Transform percentiles back to original scale
                p90_norm = distribution.ppf(0.90, *params)
                p95_norm = distribution.ppf(0.95, *params)
                p99_norm = distribution.ppf(0.99, *params)
                # Transform back to minutes scale
                p90 = p90_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()
                p95 = p95_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()
                p99 = p99_norm * (delays_minutes.max() - delays_minutes.min()) + delays_minutes.min()
                
            elif dist_name == 'Uniform':
                # Uniform distribution
                params = (delays_minutes.min(), delays_minutes.max() - delays_minutes.min())
                ks_stat, p_value = stats.kstest(delays_minutes, lambda x: uniform.cdf(x, *params))
                log_likelihood = np.sum(uniform.logpdf(delays_minutes, *params))
                p90 = uniform.ppf(0.90, *params)
                p95 = uniform.ppf(0.95, *params)
                p99 = uniform.ppf(0.99, *params)
                
            elif dist_name == 'Chi-Square':
                # Chi-square needs positive data, use degrees of freedom estimation
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
                # F distribution needs positive data and two parameters
                params = distribution.fit(delays_minutes)
                # Ensure parameters are reasonable
                if params[0] <= 0 or params[1] <= 0:
                    continue
                ks_stat, p_value = stats.kstest(delays_minutes, distribution.cdf, args=params)
                log_likelihood = np.sum(distribution.logpdf(delays_minutes, *params))
                p90 = distribution.ppf(0.90, *params)
                p95 = distribution.ppf(0.95, *params)
                p99 = distribution.ppf(0.99, *params)
                
            else:
                # Standard fitting for most distributions
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

def create_comprehensive_airport_visualization(df, results, airport_code, airport_name):
    """Create comprehensive visualization for a single airport with all distributions."""
    output_dir = os.path.join('results', 'comprehensive_individual_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate positive and negative results
    pos_results = [r for r in results if r['Delay_Type'] == 'positive']
    neg_results = [r for r in results if r['Delay_Type'] == 'negative']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    
    # Main title
    fig.suptitle(f'Comprehensive Distribution Analysis: {airport_name} ({airport_code})', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: AIC Comparison (Top 10 distributions)
    ax1 = fig.add_subplot(gs[0, :2])
    if pos_results:
        pos_df = pd.DataFrame(pos_results).sort_values('AIC').head(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(pos_df)))
        bars1 = ax1.bar(range(len(pos_df)), pos_df['AIC'], color=colors, alpha=0.8)
        ax1.set_xticks(range(len(pos_df)))
        ax1.set_xticklabels(pos_df['Distribution'], rotation=45, ha='right')
        ax1.set_title('Top 10 Distributions - AIC Comparison (Positive Delays)', fontweight='bold')
        ax1.set_ylabel('AIC (lower is better)')
        ax1.grid(True, alpha=0.3)
        
        # Highlight best
        bars1[0].set_color('gold')
        best_dist = pos_df.iloc[0]
        ax1.annotate(f'BEST: {best_dist["Distribution"]}\nAIC: {best_dist["AIC"]:.0f}',
                    xy=(0, best_dist['AIC']), xytext=(0.3, 0.9),
                    textcoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    # Plot 2: Statistical Significance
    ax2 = fig.add_subplot(gs[0, 2:])
    if pos_results:
        pos_df_full = pd.DataFrame(pos_results).sort_values('AIC')
        significance = pos_df_full['P_value'] > 0.05
        colors_sig = ['green' if sig else 'red' for sig in significance]
        bars2 = ax2.bar(range(len(pos_df_full)), pos_df_full['P_value'], 
                       color=colors_sig, alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                   label='Significance threshold (α=0.05)')
        ax2.set_xticks(range(len(pos_df_full)))
        ax2.set_xticklabels(pos_df_full['Distribution'], rotation=45, ha='right')
        ax2.set_title('Statistical Significance - All Distributions', fontweight='bold')
        ax2.set_ylabel('p-value (higher is better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Percentile Accuracy Comparison
    ax3 = fig.add_subplot(gs[1, :2])
    if pos_results:
        top_5_dists = pd.DataFrame(pos_results).sort_values('AIC').head(5)
        
        percentiles = ['P90', 'P95', 'P99']
        x = np.arange(len(percentiles))
        width = 0.15
        
        # Plot actual data percentiles first
        data_vals = [top_5_dists.iloc[0][f'Data_{p}'] for p in percentiles]
        ax3.bar(x - 2*width, data_vals, width, alpha=0.9, label='Actual Data', 
               color='black', edgecolor='white')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_5_dists)))
        for i, (_, row) in enumerate(top_5_dists.iterrows()):
            model_vals = [row[p] for p in percentiles]
            ax3.bar(x + (i-1)*width, model_vals, width, alpha=0.8,
                   label=f'{row["Distribution"]}', color=colors[i])
        
        ax3.set_xlabel('Percentiles')
        ax3.set_ylabel('Delay (minutes)')
        ax3.set_title('Extreme Percentiles - Top 5 Distributions vs Data', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(percentiles)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution Parameters Heatmap
    ax4 = fig.add_subplot(gs[1, 2:])
    if pos_results:
        # Create parameter comparison for top distributions
        top_dists = pd.DataFrame(pos_results).sort_values('AIC').head(8)
        
        # Create heatmap data
        heatmap_data = []
        for _, row in top_dists.iterrows():
            heatmap_data.append([
                row['AIC'],
                row['BIC'], 
                row['KS_Statistic'],
                row['P_value'],
                row['P95']
            ])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=top_dists['Distribution'],
                                 columns=['AIC', 'BIC', 'KS Stat', 'P-value', 'P95'])
        
        # Normalize for better visualization
        heatmap_normalized = heatmap_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        sns.heatmap(heatmap_normalized, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax4, cbar_kws={'label': 'Normalized Values'})
        ax4.set_title('Performance Metrics Heatmap (Top 8 Distributions)', fontweight='bold')
        ax4.set_ylabel('Distribution')
    
    # Plot 5: Detailed Performance Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    if pos_results:
        # Create detailed performance table
        top_10 = pd.DataFrame(pos_results).sort_values('AIC').head(10)
        
        table_data = []
        for i, (_, row) in enumerate(top_10.iterrows()):
            rank = i + 1
            good_fit = 'Yes' if row['P_value'] > 0.05 else 'No'
            evidence = 'Very Strong' if i == 0 and len(top_10) > 1 and (top_10.iloc[1]['AIC'] - row['AIC']) > 10 else \
                      'Strong' if i == 0 and len(top_10) > 1 and (top_10.iloc[1]['AIC'] - row['AIC']) > 4 else \
                      'Moderate' if i == 0 and len(top_10) > 1 and (top_10.iloc[1]['AIC'] - row['AIC']) > 2 else 'Weak'
            
            table_data.append([
                f"{rank}",
                row['Distribution'],
                f"{row['AIC']:.0f}",
                f"{row['BIC']:.0f}",
                f"{row['KS_Statistic']:.4f}",
                f"{row['P_value']:.4f}",
                good_fit,
                f"{row['P95']:.1f}",
                evidence if rank == 1 else '-'
            ])
        
        headers = ['Rank', 'Distribution', 'AIC', 'BIC', 'KS Stat', 'p-value', 
                  'Good Fit', 'P95 (min)', 'Evidence']
        
        table = ax5.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         bbox=[0.0, 0.0, 1.0, 1.0])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the best rows
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('lightgreen')  # Best distribution
            if len(table_data) > 1:
                table[(2, j)].set_facecolor('lightblue')  # Second best
        
        ax5.set_title('Detailed Performance Ranking', y=0.95, fontweight='bold', fontsize=14)
    
    # Plot 6: Sample Information and Recommendations
    ax6 = fig.add_subplot(gs[3, :2])
    ax6.axis('off')
    
    if pos_results:
        best_dist = pd.DataFrame(pos_results).sort_values('AIC').iloc[0]
        
        # Calculate sample statistics
        pos_delays = df['PositiveDelay'][df['PositiveDelay'] > 0] / 60
        
        info_text = f"""
SAMPLE INFORMATION:
Positive Delays: {len(pos_delays):,} observations
Mean Delay: {pos_delays.mean():.1f} minutes
Median Delay: {pos_delays.median():.1f} minutes
Standard Deviation: {pos_delays.std():.1f} minutes
Maximum Delay: {pos_delays.max():.1f} minutes

DATA PERCENTILES:
90th: {np.percentile(pos_delays, 90):.1f} min
95th: {np.percentile(pos_delays, 95):.1f} min
99th: {np.percentile(pos_delays, 99):.1f} min

RECOMMENDED MODEL:
Distribution: {best_dist['Distribution']}
Statistical Significance: {'Pass' if best_dist['P_value'] > 0.05 else 'Fail'}
Quality Rating: {'Excellent' if best_dist['P_value'] > 0.1 else 'Good' if best_dist['P_value'] > 0.05 else 'Poor'}
        """
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    # Plot 7: Prediction Accuracy
    ax7 = fig.add_subplot(gs[3, 2:])
    if pos_results:
        top_5 = pd.DataFrame(pos_results).sort_values('AIC').head(5)
        
        # Calculate prediction errors for P95
        actual_p95 = top_5.iloc[0]['Data_P95']
        prediction_errors = [(row['P95'] - actual_p95) / actual_p95 * 100 for _, row in top_5.iterrows()]
        
        colors = ['green' if abs(err) < 5 else 'orange' if abs(err) < 10 else 'red' for err in prediction_errors]
        bars = ax7.bar(range(len(top_5)), prediction_errors, color=colors, alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax7.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='±5% error')
        ax7.axhline(y=-5, color='green', linestyle='--', alpha=0.5)
        ax7.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='±10% error')
        ax7.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
        
        ax7.set_xticks(range(len(top_5)))
        ax7.set_xticklabels(top_5['Distribution'], rotation=45, ha='right')
        ax7.set_title('P95 Prediction Accuracy (Top 5 Distributions)', fontweight='bold')
        ax7.set_ylabel('Prediction Error (%)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{airport_code}_{airport_name.replace(" ", "_")}_comprehensive_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive visualization for {airport_name}")

def create_comprehensive_report(airport_code, airport_name, results):
    """Create comprehensive markdown report for individual airport."""
    output_dir = os.path.join('results', 'comprehensive_individual_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    pos_results = [r for r in results if r['Delay_Type'] == 'positive']
    if not pos_results:
        return
    
    pos_df = pd.DataFrame(pos_results).sort_values('AIC')
    best_dist = pos_df.iloc[0]
    
    # Calculate evidence strength
    if len(pos_df) > 1:
        aic_diff = pos_df.iloc[1]['AIC'] - best_dist['AIC']
        if aic_diff > 10: evidence = "Very Strong"
        elif aic_diff > 4: evidence = "Strong"
        elif aic_diff > 2: evidence = "Moderate"
        else: evidence = "Weak"
    else:
        evidence = "Single model"
    
    report_content = f"""# Comprehensive Distribution Analysis: {airport_name} ({airport_code})

## Executive Summary
**Recommended Distribution:** {best_dist['Distribution']}  
**Evidence Strength:** {evidence}  
**Statistical Significance:** {'Pass (p = {:.4f})'.format(best_dist['P_value']) if best_dist['P_value'] > 0.05 else 'Fail (p = {:.4f})'.format(best_dist['P_value'])}  
**Sample Size:** {best_dist['Sample_Size']:,} positive delays

## Complete Distribution Rankings

| Rank | Distribution | AIC | BIC | KS Stat | p-value | Significant | P95 (min) |
|------|--------------|-----|-----|---------|---------|-------------|-----------|
"""
    
    for i, (_, row) in enumerate(pos_df.iterrows()):
        rank = i + 1
        sig = 'Yes' if row['P_value'] > 0.05 else 'No'
        report_content += f"| {rank} | {row['Distribution']} | {row['AIC']:.0f} | {row['BIC']:.0f} | {row['KS_Statistic']:.4f} | {row['P_value']:.4f} | {sig} | {row['P95']:.1f} |\n"
    
    report_content += f"""
## Best Distribution Details
**{best_dist['Distribution']} Distribution**
- AIC: {best_dist['AIC']:.2f}
- BIC: {best_dist['BIC']:.2f}
- KS Statistic: {best_dist['KS_Statistic']:.4f}
- p-value: {best_dist['P_value']:.6f}
- Parameters: {best_dist['Parameters']}

## Predictions vs Actual Data
- **90th Percentile:** Model: {best_dist['P90']:.1f} min, Data: {best_dist['Data_P90']:.1f} min
- **95th Percentile:** Model: {best_dist['P95']:.1f} min, Data: {best_dist['Data_P95']:.1f} min  
- **99th Percentile:** Model: {best_dist['P99']:.1f} min, Data: {best_dist['Data_P99']:.1f} min

## Recommendations
- **Primary Model:** Use {best_dist['Distribution']} for delay modeling
- **Confidence Level:** {'High' if best_dist['P_value'] > 0.1 else 'Medium' if best_dist['P_value'] > 0.05 else 'Low'}
- **Operational Planning:** Plan for 95th percentile delays of {best_dist['P95']:.0f} minutes

*Analysis completed with {len(pos_df)} distribution models*
"""
    
    filename = f"{airport_code}_{airport_name.replace(' ', '_')}_comprehensive_report.md"
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        f.write(report_content)

def analyze_single_airport(airport_code):
    """Analyze comprehensive distributions for a single airport."""
    airport_names = {
        'EGLL': 'London Heathrow', 'LFPG': 'Paris Charles de Gaulle', 'EHAM': 'Amsterdam Schiphol',
        'EDDF': 'Frankfurt', 'LEMD': 'Madrid Barajas', 'LEBL': 'Barcelona', 'EDDM': 'Munich',
        'EGKK': 'London Gatwick', 'LIRF': 'Rome Fiumicino', 'EIDW': 'Dublin',
        'LATI': 'Tirana', 'LQSA': 'Sarajevo', 'LBSF': 'Sofia', 'LBBG': 'Burgas',
        'LDZA': 'Zagreb', 'LDSP': 'Split', 'LDDU': 'Dubrovnik', 'BKPR': 'Pristina',
        'LYTV': 'Tivat', 'LWSK': 'Skopje'
    }
    
    airport_name = airport_names.get(airport_code, airport_code)
    print(f"\nAnalyzing {airport_name} ({airport_code}) with comprehensive distributions...")
    
    df = load_airport_data(airport_code)
    if df is None:
        return None
    
    # Analyze positive delays with comprehensive distributions
    pos_results = fit_comprehensive_distributions(
        df['PositiveDelay'], airport_code, airport_name, 'positive'
    )
    
    # Analyze negative delays
    neg_results = fit_comprehensive_distributions(
        df['NegativeDelay'], airport_code, airport_name, 'negative'
    )
    
    all_results = []
    if pos_results:
        all_results.extend(pos_results)
    if neg_results:
        all_results.extend(neg_results)
    
    if all_results:
        # Create comprehensive visualization
        create_comprehensive_airport_visualization(df, all_results, airport_code, airport_name)
        
        # Create detailed report
        create_comprehensive_report(airport_code, airport_name, all_results)
    
    return all_results

def main():
    """Analyze all airports with comprehensive distributions."""
    airport_codes = [
        'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW',  # Europe
        'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA', 'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'   # Balkans
    ]
    
    print("Starting comprehensive distribution analysis for all airports...")
    print(f"Analyzing {len(airport_codes)} airports with 18 distribution models each")
    
    all_results = []
    successful_airports = 0
    
    for airport_code in airport_codes:
        try:
            results = analyze_single_airport(airport_code)
            if results:
                all_results.extend(results)
                successful_airports += 1
        except Exception as e:
            print(f"Error analyzing {airport_code}: {e}")
    
    # Save comprehensive results
    if all_results:
        output_dir = os.path.join('results', 'comprehensive_individual_analysis')
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_dir, 'comprehensive_all_airports_analysis.csv'), index=False)
        
        # Create summary statistics
        pos_results = results_df[results_df['Delay_Type'] == 'positive']
        best_distributions = pos_results.loc[pos_results.groupby('Airport')['AIC'].idxmin()]
        
        distribution_counts = best_distributions['Distribution'].value_counts()
        
        print(f"\nCOMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"Successfully analyzed: {successful_airports}/{len(airport_codes)} airports")
        print(f"Total distribution fits: {len(all_results)}")
        print(f"\nDistribution Winners:")
        for dist, count in distribution_counts.head(10).items():
            print(f"  {dist}: {count} airports ({count/len(best_distributions)*100:.1f}%)")
        
        print(f"\nResults saved to: {output_dir}")
        print("Check individual PNG files for detailed visualizations!")
    
    return all_results

if __name__ == "__main__":
    all_results = main()
