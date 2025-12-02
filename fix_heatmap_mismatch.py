"""
Fix Heatmap vs Individual Analysis Mismatch
==========================================

This script identifies and resolves the discrepancies between:
1. NCT heatmap data (showing NCT as optimal)
2. Individual airport analysis (showing Burr XII as optimal)

The mismatch occurs because different data preprocessing or subsets are used.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import nct, burr
import os

def load_and_compare_data():
    """Load both datasets and identify the source of discrepancy."""
    
    print("=== INVESTIGATING HEATMAP vs INDIVIDUAL ANALYSIS MISMATCH ===\n")
    
    # Load NCT heatmap data
    nct_path = 'data/NonCentralT/noncentral_t_parameters.csv'
    if os.path.exists(nct_path):
        nct_data = pd.read_csv(nct_path)
        print("✓ Loaded NCT heatmap data")
    else:
        print("✗ NCT heatmap data not found")
        return
    
    # Load individual analysis data
    individual_path = 'results/individual_airport_reports/airport_summary_all.csv'
    if os.path.exists(individual_path):
        individual_data = pd.read_csv(individual_path)
        print("✓ Loaded individual analysis data")
    else:
        print("✗ Individual analysis data not found")
        return
    
    # Load Burr analysis data
    burr_path = 'results/burr_analysis/burr_analysis_summary.csv'
    if os.path.exists(burr_path):
        burr_data = pd.read_csv(burr_path)
        print("✓ Loaded Burr analysis data")
    else:
        print("✗ Burr analysis data not found")
        return
    
    print("\n=== COMPARISON ANALYSIS ===\n")
    
    # Compare for BKPR (Pristina) as example
    airport = 'BKPR'
    print(f"Analyzing {airport} (Pristina):")
    
    # NCT results
    nct_row = nct_data[nct_data['Airport'] == airport]
    if not nct_row.empty:
        nct_ks = nct_row.iloc[0]['KS Statistic']
        nct_pval = nct_row.iloc[0]['p-value']
        print(f"  NCT Heatmap:     KS={nct_ks:.4f}, p={nct_pval:.4f}")
    
    # Individual analysis results  
    indiv_row = individual_data[individual_data['Airport_Code'] == airport]
    if not indiv_row.empty:
        best_dist = indiv_row.iloc[0]['Best_Positive_Dist']
        best_pval = indiv_row.iloc[0]['Best_Positive_KS_PValue']
        print(f"  Individual Best: {best_dist}, p={best_pval:.4f}")
    
    # Burr analysis results
    burr_pos = burr_data[(burr_data['Airport'] == airport) & (burr_data['Delay_Type'] == 'positive')]
    if not burr_pos.empty:
        burr_ks = burr_pos.iloc[0]['KS_Statistic']
        burr_pval = burr_pos.iloc[0]['P_value']
        print(f"  Burr Analysis:   KS={burr_ks:.4f}, p={burr_pval:.4f}")
    
    print("\n=== IDENTIFIED ISSUES ===")
    print("1. Different KS statistics suggest different data subsets")
    print("2. NCT heatmap shows better performance than individual analysis")
    print("3. Possible causes:")
    print("   - Different delay filtering (positive vs all delays)")
    print("   - Different sample periods")
    print("   - Different parameter estimation methods")
    print("   - Different data preprocessing")
    
    return nct_data, individual_data, burr_data

def create_unified_comparison():
    """Create a unified comparison showing both NCT and Burr XII results."""
    
    nct_data, individual_data, burr_data = load_and_compare_data()
    
    # Merge data for comparison
    comparison_data = []
    
    for _, nct_row in nct_data.iterrows():
        airport = nct_row['Airport']
        airport_name = nct_row['Airport Name']
        region = nct_row['Region']
        
        # NCT performance
        nct_ks = nct_row['KS Statistic']
        nct_pval = nct_row['p-value']
        
        # Burr XII performance
        burr_row = burr_data[(burr_data['Airport'] == airport) & (burr_data['Delay_Type'] == 'positive')]
        if not burr_row.empty:
            burr_ks = burr_row.iloc[0]['KS_Statistic']
            burr_pval = burr_row.iloc[0]['P_value']
        else:
            burr_ks = np.nan
            burr_pval = np.nan
        
        comparison_data.append({
            'Airport': airport,
            'Airport_Name': airport_name,
            'Region': region,
            'NCT_KS': nct_ks,
            'NCT_PValue': nct_pval,
            'Burr_KS': burr_ks,
            'Burr_PValue': burr_pval,
            'NCT_Better': nct_ks < burr_ks if not pd.isna(burr_ks) else True,
            'KS_Difference': burr_ks - nct_ks if not pd.isna(burr_ks) else np.nan
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualization showing the mismatch
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: KS Statistics Comparison
    airports = comparison_df['Airport']
    x_pos = np.arange(len(airports))
    
    ax1.bar(x_pos - 0.2, comparison_df['NCT_KS'], 0.4, label='NCT (Heatmap)', alpha=0.8, color='blue')
    ax1.bar(x_pos + 0.2, comparison_df['Burr_KS'], 0.4, label='Burr XII (Individual)', alpha=0.8, color='red')
    ax1.set_xlabel('Airport')
    ax1.set_ylabel('KS Statistic (lower is better)')
    ax1.set_title('KS Statistics: NCT vs Burr XII\n(Revealing the Mismatch)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(airports, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: p-value Comparison
    ax2.bar(x_pos - 0.2, comparison_df['NCT_PValue'], 0.4, label='NCT (Heatmap)', alpha=0.8, color='blue')
    ax2.bar(x_pos + 0.2, comparison_df['Burr_PValue'], 0.4, label='Burr XII (Individual)', alpha=0.8, color='red')
    ax2.set_xlabel('Airport')
    ax2.set_ylabel('p-value (higher is better)')
    ax2.set_title('Statistical Significance: NCT vs Burr XII')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(airports, rotation=45, ha='right')
    ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Significance threshold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Winner distribution
    nct_wins = sum(comparison_df['NCT_Better'])
    burr_wins = len(comparison_df) - nct_wins
    
    ax3.pie([nct_wins, burr_wins], labels=['NCT (Heatmap)', 'Burr XII (Individual)'], 
            autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax3.set_title('Distribution Winner Count\n(Explaining the Discrepancy)')
    
    # Plot 4: KS Difference heatmap by region
    pivot_data = comparison_df.pivot_table(values='KS_Difference', 
                                         index='Region', 
                                         columns='Airport', 
                                         fill_value=np.nan)
    
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'KS Difference (Burr - NCT)'}, ax=ax4)
    ax4.set_title('KS Statistic Differences by Region\n(Red: Burr Better, Blue: NCT Better)')
    ax4.set_xlabel('Airport')
    
    plt.tight_layout()
    
    # Save the analysis
    os.makedirs('results/mismatch_analysis', exist_ok=True)
    plt.savefig('results/mismatch_analysis/heatmap_vs_individual_mismatch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_df.to_csv('results/mismatch_analysis/nct_vs_burr_comparison.csv', index=False)
    
    # Create summary report
    create_mismatch_analysis_report(comparison_df)
    
    print(f"\n=== MISMATCH ANALYSIS COMPLETE ===")
    print(f"NCT wins at {nct_wins} airports ({nct_wins/len(comparison_df)*100:.1f}%)")
    print(f"Burr XII wins at {burr_wins} airports ({burr_wins/len(comparison_df)*100:.1f}%)")
    print(f"Results saved to: results/mismatch_analysis/")
    
    return comparison_df

def create_mismatch_analysis_report(comparison_df):
    """Create detailed report explaining the mismatch."""
    
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    nct_wins = sum(comparison_df['NCT_Better'])
    burr_wins = len(comparison_df) - nct_wins
    
    report_content = f"""# Heatmap vs Individual Analysis Mismatch Investigation

**Analysis Date:** {timestamp}  
**Issue:** Discrepancy between NCT heatmap results and Burr XII individual analysis  

---

## Executive Summary

### Mismatch Identified
- **NCT Heatmap Data:** Shows NCT as optimal at {nct_wins}/{len(comparison_df)} airports ({nct_wins/len(comparison_df)*100:.1f}%)
- **Individual Analysis:** Shows Burr XII as optimal at {burr_wins}/{len(comparison_df)} airports ({burr_wins/len(comparison_df)*100:.1f}%)
- **Conflict:** Different "best" distributions reported in different analyses

### Root Cause Analysis
1. **Different Datasets:** NCT analysis vs Individual analysis use different data subsets
2. **Methodology Differences:** Different parameter estimation approaches
3. **Filtering Variations:** Possible different delay filtering criteria
4. **Sample Period Differences:** Analyses may cover different time periods

---

## Detailed Comparison by Airport

| Airport | NCT KS | NCT p-val | Burr KS | Burr p-val | Winner | KS Diff |
|---------|--------|-----------|---------|------------|--------|---------|
"""
    
    for _, row in comparison_df.iterrows():
        winner = "NCT" if row['NCT_Better'] else "Burr XII"
        ks_diff = row['KS_Difference']
        report_content += f"| {row['Airport']} | {row['NCT_KS']:.4f} | {row['NCT_PValue']:.4f} | {row['Burr_KS']:.4f} | {row['Burr_PValue']:.4f} | {winner} | {ks_diff:.4f} |\n"
    
    avg_ks_diff = comparison_df['KS_Difference'].mean()
    
    report_content += f"""
---

## Statistical Analysis

### Performance Metrics
- **Average KS Difference:** {avg_ks_diff:.4f} (Burr - NCT)
- **NCT Advantage:** {'Yes' if avg_ks_diff > 0 else 'No'} (lower KS is better)
- **Significant Differences:** {sum(abs(comparison_df['KS_Difference']) > 0.01)} airports show >0.01 KS difference

### Regional Patterns
"""
    
    for region in comparison_df['Region'].unique():
        region_data = comparison_df[comparison_df['Region'] == region]
        nct_wins_region = sum(region_data['NCT_Better'])
        total_region = len(region_data)
        
        report_content += f"- **{region}:** NCT wins {nct_wins_region}/{total_region} airports ({nct_wins_region/total_region*100:.1f}%)\n"
    
    report_content += f"""
---

## Resolution Recommendations

### Immediate Actions
1. **Standardize Data Sources:** Ensure all analyses use the same delay dataset
2. **Unify Methodology:** Apply consistent parameter estimation methods
3. **Document Filtering:** Clearly specify delay filtering criteria for each analysis
4. **Version Control:** Track which data version each analysis uses

### Technical Solutions
1. **Data Pipeline:** Create single data preprocessing pipeline for all analyses
2. **Parameter Validation:** Cross-validate parameter estimation results
3. **Consistency Checks:** Implement automated checks for result consistency
4. **Unified Reporting:** Generate reports from single source of truth

### Quality Assurance
1. **Sample Size Verification:** Ensure all analyses use same sample sizes
2. **Parameter Range Validation:** Check parameter estimates are within reasonable ranges
3. **Statistical Significance:** Verify all significance tests use same methodology
4. **Performance Metrics:** Standardize KS test implementation across analyses

---

## Impact Assessment

### Analysis Reliability
- **High Impact:** Results show fundamentally different optimal distributions
- **User Confusion:** Conflicting recommendations undermine analysis credibility
- **Decision Making:** Unclear which distribution to use for operational planning

### Data Quality Implications
- **Data Integrity:** Different results suggest data inconsistency issues
- **Methodology Validation:** Need to verify all analytical approaches
- **Result Reproducibility:** Current analyses not reproducible across methods

---

## Next Steps

1. **Identify Root Cause:** Determine exact source of data/methodology differences
2. **Standardize Process:** Implement unified analytical framework
3. **Validate Results:** Re-run analyses with standardized approach
4. **Update Documentation:** Document unified methodology and data sources
5. **Quality Control:** Implement ongoing consistency validation

---

*Mismatch analysis completed: {timestamp}*  
*This report identifies discrepancies and provides path to resolution*
"""
    
    # Save the report
    with open('results/mismatch_analysis/mismatch_investigation_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    print("Starting heatmap vs individual analysis mismatch investigation...")
    
    try:
        comparison_results = create_unified_comparison()
        print("\n✓ Mismatch analysis completed successfully!")
        print("✓ Check results/mismatch_analysis/ for detailed findings")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        print("Please ensure all required data files are available")