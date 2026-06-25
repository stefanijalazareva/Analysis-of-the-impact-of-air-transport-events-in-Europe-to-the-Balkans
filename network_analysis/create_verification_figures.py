"""
Generates figures for temporal evolution and lag structure analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns
import sys
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

BASE_DIR = Path("results/full_monthly_networks")
GRANGER_DIR = Path("results/granger_europe_balkans")
OUTPUT_DIR = Path("results/document_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EU_AIRPORTS = {
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW'
}
BALKANS_AIRPORTS = {
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
}

COVID_START = datetime(2020, 3, 1)
COVID_RECOVERY = datetime(2021, 12, 31)


def load_temporal_metrics():
    metrics = {}
    for region in ['FULL', 'EUROPE', 'BALKANS']:
        csv_path = BASE_DIR / f"temporal_metrics_{region}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            metrics[region] = df
    return metrics


def load_or_create_directional_links():
    csv_path = BASE_DIR / "directionality" / "temporal_directional_links.csv"
    
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0 and 'year' in df.columns:
                df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                print(f"Loaded existing directional links: {len(df)} time periods")
                return df
        except Exception as e:
            print(f"Could not load existing file: {e}")
    
    print("Computing directional links from monthly networks...")
    records = []
    
    for folder in sorted(BASE_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in ['Aggregated', 'airport_dynamics', 'directionality', 'temporal_plots']:
            continue
        
        try:
            parts = folder.name.split("_")
            if len(parts) != 2:
                continue
            year, month = map(int, parts)
            
            possible_paths = [
                folder / "adjacency_matrix.csv",
                folder / "FULL" / "adjacency_matrix.csv",
            ]
            
            adj_file = None
            for path in possible_paths:
                if path.exists():
                    adj_file = path
                    break
            
            if adj_file is None:
                continue
            
            adj = pd.read_csv(adj_file, index_col=0)
            
            eu_to_balkans = adj.loc[
                adj.index.intersection(EU_AIRPORTS),
                adj.columns.intersection(BALKANS_AIRPORTS)
            ].values.sum()
            
            balkans_to_eu = adj.loc[
                adj.index.intersection(BALKANS_AIRPORTS),
                adj.columns.intersection(EU_AIRPORTS)
            ].values.sum()
            
            records.append({
                'year': year,
                'month': month,
                'date': datetime(year, month, 1),
                'EU_to_Balkans': int(eu_to_balkans),
                'Balkans_to_EU': int(balkans_to_eu),
                'asymmetry_ratio': eu_to_balkans / balkans_to_eu if balkans_to_eu > 0 else np.nan
            })
            print(f"  Processed {year}-{month:02d}: EU->Bal={eu_to_balkans}, Bal->EU={balkans_to_eu}")
            
        except Exception as e:
            continue
    
    if len(records) == 0:
        print("WARNING: No directional link data could be generated!")
        return pd.DataFrame(columns=['year', 'month', 'date', 'EU_to_Balkans', 'Balkans_to_EU', 'asymmetry_ratio'])
    
    df = pd.DataFrame(records)
    # Save for future use
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved directional links data: {len(df)} time periods")
    return df


def load_granger_results():
    """Load Granger causality results"""
    results = {}
    
    # Europe to Balkans
    eu_to_bal = GRANGER_DIR / "granger_europe_to_balkans.csv"
    if eu_to_bal.exists():
        results['EU_to_Balkans'] = pd.read_csv(eu_to_bal)
    
    # Balkans to Europe
    bal_to_eu = GRANGER_DIR / "granger_balkans_to_europe.csv"
    if bal_to_eu.exists():
        results['Balkans_to_EU'] = pd.read_csv(bal_to_eu)
    
    return results


def create_figure1_temporal_evolution(directional_df, metrics):
    """
    Figure 1: Temporal Evolution of Cross-Regional Connectivity
    Shows EU->Balkans and Balkans->EU links over time with COVID-19 highlighted
    """
    if len(directional_df) == 0:
        print("Skipping Figure 1: No directional link data available")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.2), sharex=True)
    
    # Panel A: Directional link counts
    ax = axes[0]
    ax.plot(directional_df['date'], directional_df['EU_to_Balkans'], 
            marker='o', linewidth=2, label=r'Europe $\rightarrow$ Balkans', color='#E69F00', markersize=5)
    ax.plot(directional_df['date'], directional_df['Balkans_to_EU'], 
            marker='s', linewidth=2, label=r'Balkans $\rightarrow$ Europe', color='#56B4E9', markersize=5)
    
    # Add COVID-19 shading
    ax.axvspan(COVID_START, COVID_RECOVERY, alpha=0.2, color='red', 
               label='COVID-19 Period')
    
    ax.set_ylabel('Number of Significant\nCausal Links', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel B: Asymmetry ratio
    ax = axes[1]
    ax.plot(directional_df['date'], directional_df['asymmetry_ratio'], 
            marker='D', linewidth=2, color='#009E73', markersize=5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Equal influence')
    ax.axvspan(COVID_START, COVID_RECOVERY, alpha=0.2, color='red')
    
    ax.set_ylabel('Asymmetry ratio', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel C: Network density for both regions
    if 'EUROPE' in metrics and 'BALKANS' in metrics:
        ax = axes[2]
        ax.plot(metrics['EUROPE']['date'], metrics['EUROPE']['density'], 
                marker='o', linewidth=2, label='Europe', color='#E69F00', markersize=5)
        ax.plot(metrics['BALKANS']['date'], metrics['BALKANS']['density'], 
                marker='s', linewidth=2, label='Balkans', color='#56B4E9', markersize=5)
        ax.axvspan(COVID_START, COVID_RECOVERY, alpha=0.2, color='red')
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Network Density', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig1_Temporal_Evolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig1_Temporal_Evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figure 1 created: Temporal Evolution")


def create_figure2_lag_structure(granger_results):
    """
    Figure 2: Lag Structure and Causality Strength
    Shows p-values and F-statistics for different lags
    """
    # Use cleaner styling
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 5.0))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Define colors
    color_eu = '#D55E00'  # vermillion
    color_bal = '#0072B2'  # blue
    
    if 'EU_to_Balkans' in granger_results:
        eu_bal = granger_results['EU_to_Balkans']
        
        # Panel A: Europe to Balkans p-values
        ax = axes[0, 0]
        ax.semilogy(eu_bal['lag_hours'], eu_bal['p_value'], 
                    marker='o', linewidth=2.5, markersize=6, color=color_eu, alpha=0.9,
                    label=r'Europe $\rightarrow$ Balkans')
        ax.set_xlabel('Lag (hours)', fontsize=14)
        ax.set_ylabel('p-value', fontsize=14)
        ax.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray', framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Panel B: Europe to Balkans F-statistics
        ax = axes[0, 1]
        ax.plot(eu_bal['lag_hours'], eu_bal['f_statistic'], 
                marker='o', linewidth=2.5, markersize=6, color=color_eu, alpha=0.9,
                label=r'Europe $\rightarrow$ Balkans')
        ax.set_xlabel('Lag (hours)', fontsize=14)
        ax.set_ylabel('F-statistic', fontsize=14)
        ax.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray', framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if 'Balkans_to_EU' in granger_results:
        bal_eu = granger_results['Balkans_to_EU']
        
        # Panel C: Balkans to Europe p-values
        ax = axes[1, 0]
        ax.semilogy(bal_eu['lag_hours'], bal_eu['p_value'], 
                    marker='s', linewidth=2.5, markersize=6, color=color_bal, alpha=0.9,
                    label=r'Balkans $\rightarrow$ Europe')
        ax.set_xlabel('Lag (hours)', fontsize=14)
        ax.set_ylabel('p-value', fontsize=14)
        ax.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray', framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Panel D: Balkans to Europe F-statistics
        ax = axes[1, 1]
        ax.plot(bal_eu['lag_hours'], bal_eu['f_statistic'], 
                marker='s', linewidth=2.5, markersize=6, color=color_bal, alpha=0.9,
                label=r'Balkans $\rightarrow$ Europe')
        ax.set_xlabel('Lag (hours)', fontsize=14)
        ax.set_ylabel('F-statistic', fontsize=14)
        ax.legend(fontsize=11, loc='best', frameon=True, edgecolor='gray', framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig2_Lag_Structure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig2_Lag_Structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figure 2 created: Lag Structure")


def create_verification_report(directional_df, granger_results, metrics):
    """Generate verification report"""
    report = []
    report.append("=" * 80)
    report.append("VERIFICATION REPORT: Document Claims vs. Actual Data")
    report.append("=" * 80)
    report.append("")
    
    if len(directional_df) > 0:
        # Cross-regional connectivity
        report.append("1. CROSS-REGIONAL CONNECTIVITY")
        report.append("-" * 40)
        
        eu_bal_stats = directional_df['EU_to_Balkans'].describe()
        bal_eu_stats = directional_df['Balkans_to_EU'].describe()
        
        report.append(f"Europe -> Balkans links:")
        report.append(f"  Document claims: 11-76 (mean=42.9, std=18.1)")
        report.append(f"  Actual data:     {eu_bal_stats['min']:.0f}-{eu_bal_stats['max']:.0f} (mean={eu_bal_stats['mean']:.1f}, std={eu_bal_stats['std']:.1f})")
        report.append("")
        
        report.append(f"Balkans -> Europe links:")
        report.append(f"  Document claims: 5-47 (mean=19.0, std=11.5)")
        report.append(f"  Actual data:     {bal_eu_stats['min']:.0f}-{bal_eu_stats['max']:.0f} (mean={bal_eu_stats['mean']:.1f}, std={bal_eu_stats['std']:.1f})")
        report.append("")
        
        # COVID-19 impact
        report.append("2. COVID-19 IMPACT")
        report.append("-" * 40)
        
        june_2019 = directional_df[(directional_df['year'] == 2019) & (directional_df['month'] == 6)]
        june_2020 = directional_df[(directional_df['year'] == 2020) & (directional_df['month'] == 6)]
        june_2021 = directional_df[(directional_df['year'] == 2021) & (directional_df['month'] == 6)]
        
        if not june_2019.empty and not june_2020.empty:
            eu_bal_2019 = june_2019['EU_to_Balkans'].values[0]
            eu_bal_2020 = june_2020['EU_to_Balkans'].values[0]
            reduction = (eu_bal_2019 - eu_bal_2020) / eu_bal_2019 * 100
            
            report.append(f"Europe -> Balkans (June 2019 to June 2020):")
            report.append(f"  Document claims: 58.8 -> 22.0 (63% reduction)")
            report.append(f"  Actual data:     {eu_bal_2019} -> {eu_bal_2020} ({reduction:.1f}% reduction)")
            
            if not june_2021.empty:
                eu_bal_2021 = june_2021['EU_to_Balkans'].values[0]
                report.append(f"  Recovery (June 2021): {eu_bal_2021} links")
            report.append("")
    
    if granger_results:
        report.append("3. GRANGER CAUSALITY STRENGTH")
        report.append("-" * 40)
        
        if 'EU_to_Balkans' in granger_results:
            eu_bal = granger_results['EU_to_Balkans']
            lag1 = eu_bal[eu_bal['lag_hours'] == 1].iloc[0]
            lag10 = eu_bal[eu_bal['lag_hours'] == 10].iloc[0]
            
            report.append(f"Europe -> Balkans:")
            report.append(f"  Lag 1h:  F-stat={lag1['f_statistic']:.1f}, p={lag1['p_value']:.2e}")
            report.append(f"  Lag 10h: F-stat={lag10['f_statistic']:.1f}, p={lag10['p_value']:.2e}")
            report.append("")
        
        if 'Balkans_to_EU' in granger_results:
            bal_eu = granger_results['Balkans_to_EU']
            lag1 = bal_eu[bal_eu['lag_hours'] == 1].iloc[0]
            
            report.append(f"Balkans -> Europe:")
            report.append(f"  Lag 1h:  F-stat={lag1['f_statistic']:.1f}, p={lag1['p_value']:.2e}")
            report.append("")
    
    report.append("=" * 80)
    
    # Save and print
    report_text = "\n".join(report)
    with open(OUTPUT_DIR / "verification_report.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)


def main():
    """Main execution"""
    print("=" * 80)
    print("Document Verification and Figure Generation")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    metrics = load_temporal_metrics()
    directional_df = load_or_create_directional_links()
    granger_results = load_granger_results()
    
    print(f"Loaded metrics for {len(metrics)} network types")
    print(f"Loaded {len(directional_df)} time periods of directional links")
    print(f"Loaded {len(granger_results)} Granger causality directions")
    print()
    
    # Create figures
    print("Creating figures...")
    create_figure1_temporal_evolution(directional_df, metrics)
    
    if granger_results:
        create_figure2_lag_structure(granger_results)
    
    print()
    create_verification_report(directional_df, granger_results, metrics)
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
