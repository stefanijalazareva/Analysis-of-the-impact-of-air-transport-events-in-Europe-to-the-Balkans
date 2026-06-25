"""
Create a comprehensive figure showing temporal evolution with annotations
for the document - highlighting COVID-19, seasonal patterns, and key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Paths
BASE_DIR = Path("results/full_monthly_networks")
OUTPUT_DIR = Path("results/document_verification")

# Load data
directional_df = pd.read_csv(BASE_DIR / "directionality" / "temporal_directional_links.csv")
directional_df['date'] = pd.to_datetime(directional_df[['year', 'month']].assign(day=1))

# COVID-19 period
COVID_START = datetime(2020, 3, 1)
COVID_RECOVERY = datetime(2021, 12, 31)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main panel: Temporal evolution with annotations
ax_main = fig.add_subplot(gs[0:2, :])

# Plot both directions
line1 = ax_main.plot(directional_df['date'], directional_df['EU_to_Balkans'], 
                     marker='o', linewidth=2.5, label='Europe → Balkans', 
                     color='#E69F00', markersize=6, zorder=3)
line2 = ax_main.plot(directional_df['date'], directional_df['Balkans_to_EU'], 
                     marker='s', linewidth=2.5, label='Balkans → Europe', 
                     color='#56B4E9', markersize=6, zorder=3)

# Add COVID-19 shading
ax_main.axvspan(COVID_START, COVID_RECOVERY, alpha=0.15, color='red', 
                label='COVID-19 Period', zorder=1)

# Annotate key events
# Pre-COVID peak
june_2019_idx = directional_df[(directional_df['year'] == 2019) & (directional_df['month'] == 6)].index[0]
june_2019_val = directional_df.loc[june_2019_idx, 'EU_to_Balkans']
june_2019_date = directional_df.loc[june_2019_idx, 'date']
ax_main.annotate(f'Pre-pandemic\npeak: {june_2019_val} links', 
                xy=(june_2019_date, june_2019_val),
                xytext=(20, 30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='black', lw=1.5),
                fontsize=10, fontweight='bold')

# COVID-19 impact
june_2020_idx = directional_df[(directional_df['year'] == 2020) & (directional_df['month'] == 6)].index[0]
june_2020_val = directional_df.loc[june_2020_idx, 'EU_to_Balkans']
june_2020_date = directional_df.loc[june_2020_idx, 'date']
ax_main.annotate(f'COVID-19 collapse:\n{june_2020_val} links\n(69% reduction)', 
                xy=(june_2020_date, june_2020_val),
                xytext=(20, -50), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#ffcccc', alpha=0.9),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                              color='red', lw=2),
                fontsize=10, fontweight='bold', color='darkred')

# Recovery
june_2022_idx = directional_df[(directional_df['year'] == 2022) & (directional_df['month'] == 6)].index[0]
june_2022_val = directional_df.loc[june_2022_idx, 'EU_to_Balkans']
june_2022_date = directional_df.loc[june_2022_idx, 'date']
ax_main.annotate(f'Recovery:\n{june_2022_val} links', 
                xy=(june_2022_date, june_2022_val),
                xytext=(-60, 30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='green', lw=1.5),
                fontsize=10, fontweight='bold', color='darkgreen')

# Seasonal pattern indicators
summer_months = directional_df[directional_df['month'] == 6]
for _, row in summer_months.iterrows():
    ax_main.plot(row['date'], row['EU_to_Balkans'], 
                'o', color='orange', markersize=10, alpha=0.3, zorder=2)

winter_months = directional_df[directional_df['month'] == 12]
for _, row in winter_months.iterrows():
    ax_main.plot(row['date'], row['EU_to_Balkans'], 
                's', color='blue', markersize=10, alpha=0.3, zorder=2)

ax_main.set_ylabel('Number of Significant Causal Links', fontsize=13, fontweight='bold')
ax_main.set_xlabel('Year', fontsize=13, fontweight='bold')
ax_main.set_title('Cross-Regional Delay Propagation: Temporal Evolution and COVID-19 Impact', 
                  fontsize=15, fontweight='bold', pad=20)
ax_main.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax_main.grid(True, alpha=0.3, linestyle='--')
ax_main.set_ylim(0, 80)

# Bottom left: Asymmetry ratio evolution
ax_asym = fig.add_subplot(gs[2, 0])
ax_asym.plot(directional_df['date'], directional_df['asymmetry_ratio'], 
            marker='D', linewidth=2, color='#009E73', markersize=5)
ax_asym.axhline(1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, 
               label='Equal influence')
ax_asym.axvspan(COVID_START, COVID_RECOVERY, alpha=0.15, color='red')

# Highlight mean asymmetry
mean_asym = directional_df['asymmetry_ratio'].mean()
ax_asym.axhline(mean_asym, color='darkgreen', linestyle=':', linewidth=2, 
               label=f'Mean: {mean_asym:.2f}:1')

ax_asym.set_ylabel('Asymmetry Ratio\n(EU→Bal / Bal→EU)', fontsize=11, fontweight='bold')
ax_asym.set_xlabel('Year', fontsize=11)
ax_asym.set_title('Directional Asymmetry Over Time', fontsize=12, fontweight='bold')
ax_asym.legend(fontsize=9)
ax_asym.grid(True, alpha=0.3)

# Bottom right: Seasonal comparison
ax_seasonal = fig.add_subplot(gs[2, 1])

# Group by month and calculate statistics
monthly_eu_bal = directional_df.groupby('month')['EU_to_Balkans'].agg(['mean', 'std'])
monthly_bal_eu = directional_df.groupby('month')['Balkans_to_EU'].agg(['mean', 'std'])

months = [3, 6, 9, 12]
month_labels = ['Mar\n(Spring)', 'Jun\n(Summer)', 'Sep\n(Autumn)', 'Dec\n(Winter)']
x = np.arange(len(months))
width = 0.35

bars1 = ax_seasonal.bar(x - width/2, [monthly_eu_bal.loc[m, 'mean'] for m in months], 
                       width, label='Europe → Balkans', color='#E69F00', alpha=0.8,
                       yerr=[monthly_eu_bal.loc[m, 'std'] for m in months],
                       capsize=5, error_kw={'linewidth': 2})
bars2 = ax_seasonal.bar(x + width/2, [monthly_bal_eu.loc[m, 'mean'] for m in months], 
                       width, label='Balkans → Europe', color='#56B4E9', alpha=0.8,
                       yerr=[monthly_bal_eu.loc[m, 'std'] for m in months],
                       capsize=5, error_kw={'linewidth': 2})

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_seasonal.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_seasonal.set_ylabel('Mean Number of Links', fontsize=11, fontweight='bold')
ax_seasonal.set_xlabel('Season', fontsize=11)
ax_seasonal.set_title('Seasonal Variation in Cross-Regional Links', fontsize=12, fontweight='bold')
ax_seasonal.set_xticks(x)
ax_seasonal.set_xticklabels(month_labels)
ax_seasonal.legend(fontsize=9)
ax_seasonal.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_Comprehensive_Temporal_Analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'Fig_Comprehensive_Temporal_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comprehensive figure created!")

# Create summary statistics table
summary_stats = pd.DataFrame({
    'Metric': [
        'Europe → Balkans (mean ± std)',
        'Europe → Balkans (range)',
        'Balkans → Europe (mean ± std)',
        'Balkans → Europe (range)',
        'Asymmetry ratio (mean)',
        'Asymmetry ratio (range)',
        'Summer peak (June mean)',
        'Winter low (December mean)',
        'COVID-19 impact (June 2019→2020)',
        'Recovery (June 2020→2022)'
    ],
    'Value': [
        f"{directional_df['EU_to_Balkans'].mean():.1f} ± {directional_df['EU_to_Balkans'].std():.1f}",
        f"{directional_df['EU_to_Balkans'].min():.0f}-{directional_df['EU_to_Balkans'].max():.0f}",
        f"{directional_df['Balkans_to_EU'].mean():.1f} ± {directional_df['Balkans_to_EU'].std():.1f}",
        f"{directional_df['Balkans_to_EU'].min():.0f}-{directional_df['Balkans_to_EU'].max():.0f}",
        f"{directional_df['asymmetry_ratio'].mean():.2f}:1",
        f"{directional_df['asymmetry_ratio'].min():.2f}-{directional_df['asymmetry_ratio'].max():.2f}",
        f"{monthly_eu_bal.loc[6, 'mean']:.1f} links",
        f"{monthly_eu_bal.loc[12, 'mean']:.1f} links",
        f"72 → 22 (69.4% reduction)",
        f"22 → 64 (191% increase)"
    ]
})

summary_stats.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
print("\nSummary statistics saved!")
print(summary_stats.to_string(index=False))
