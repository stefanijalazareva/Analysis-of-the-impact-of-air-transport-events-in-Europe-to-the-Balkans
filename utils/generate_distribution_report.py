"""
Generate publication-ready distribution analysis report
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import matplotlib.gridspec as gridspec

class DistributionReportGenerator:
    def __init__(self):
        self.results_dir = Path('results/distribution_analysis')
        self.report_dir = Path('reports/distribution_analysis')
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Define EU and Balkan airport groups
        self.eu_airports = ['EDDF', 'EDDM', 'EGKK', 'EGLL', 'EHAM', 'EIDW',
                          'LEBL', 'LEMD', 'LFPG', 'LIRF']
        self.balkan_airports = ['LATI', 'LBBG', 'LBSF', 'LDDU', 'LDSP', 'LDZA', 'BKPR']

    def generate_parameter_tables(self):
        """Generate LaTeX tables with distribution parameters and confidence intervals"""
        for region, airports in [('EU', self.eu_airports), ('Balkan', self.balkan_airports)]:
            tex_content = []
            tex_content.append(r"\begin{table}[htbp]")
            tex_content.append(r"\centering")
            tex_content.append(r"\caption{Distribution Parameters - " + region + " Airports}")
            tex_content.append(r"\begin{tabular}{lcccc}")
            tex_content.append(r"\toprule")
            tex_content.append(r"Airport & Distribution & Parameters & CI Lower & CI Upper \\")
            tex_content.append(r"\midrule")

            for airport in airports:
                if (self.results_dir / airport).exists():
                    # Load confidence intervals
                    with open(self.results_dir / airport / 'confidence_intervals.json') as f:
                        ci_data = json.load(f)

                    for dist_name, dist_results in ci_data.items():
                        params = dist_results['parameters']
                        means = dist_results['mean_params']
                        ci_lower = dist_results['ci_lower']
                        ci_upper = dist_results['ci_upper']

                        # Format parameters with confidence intervals
                        param_str = ""
                        for p, m, l, u in zip(params, means, ci_lower, ci_upper):
                            param_str += f"{p}: {m:.3f} ({l:.3f}, {u:.3f})\n"

                        tex_content.append(f"{airport} & {dist_name} & \\makecell{{{param_str}}} & \\\\")

            tex_content.append(r"\bottomrule")
            tex_content.append(r"\end{tabular}")
            tex_content.append(r"\end{table}")

            # Save table
            with open(self.report_dir / f'parameter_table_{region.lower()}.tex', 'w') as f:
                f.write('\n'.join(tex_content))

    def generate_ks_heatmap(self):
        """Generate heatmap of KS statistics for all airports and distributions"""
        all_data = []

        for airport in self.eu_airports + self.balkan_airports:
            if (self.results_dir / airport).exists():
                df = pd.read_csv(self.results_dir / airport / 'distribution_comparison.csv')
                df['airport'] = airport
                all_data.append(df)

        if all_data:
            combined_data = pd.concat(all_data)
            ks_pivot = combined_data.pivot(
                index='airport',
                columns='distribution',
                values='ks_statistic'
            )

            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(ks_pivot, annot=True, cmap='YlOrRd', fmt='.3f')
            plt.title('Kolmogorov-Smirnov Statistics by Airport and Distribution')
            plt.tight_layout()
            plt.savefig(self.report_dir / 'ks_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_fit_quality_comparison(self):
        """Generate comparative analysis of fit quality"""
        eu_data = []
        balkan_data = []

        # Collect data for both regions
        for airport in self.eu_airports + self.balkan_airports:
            if (self.results_dir / airport).exists():
                df = pd.read_csv(self.results_dir / airport / 'distribution_comparison.csv')
                df['airport'] = airport
                df['region'] = 'EU' if airport in self.eu_airports else 'Balkan'

                if airport in self.eu_airports:
                    eu_data.append(df)
                else:
                    balkan_data.append(df)

        if eu_data and balkan_data:
            eu_df = pd.concat(eu_data)
            balkan_df = pd.concat(balkan_data)

            # Create comparative plots
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2)

            # KS statistics comparison
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_boxplot_comparison(eu_df, balkan_df, 'ks_statistic',
                                       'KS Statistic', ax1)

            # P-values comparison
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_boxplot_comparison(eu_df, balkan_df, 'p_value',
                                       'P-value', ax2)

            # AIC comparison
            ax3 = fig.add_subplot(gs[1, :])
            self._plot_aic_comparison(eu_df, balkan_df, ax3)

            plt.tight_layout()
            plt.savefig(self.report_dir / 'fit_quality_comparison.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_boxplot_comparison(self, eu_df: pd.DataFrame, balkan_df: pd.DataFrame,
                               metric: str, ylabel: str, ax: plt.Axes):
        """Helper function to create boxplot comparisons"""
        data = []
        labels = []

        for dist in eu_df['distribution'].unique():
            eu_values = eu_df[eu_df['distribution'] == dist][metric]
            balkan_values = balkan_df[balkan_df['distribution'] == dist][metric]

            data.extend([eu_values, balkan_values])
            labels.extend([f'{dist}\nEU', f'{dist}\nBalkan'])

        ax.boxplot(data, labels=labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} Comparison')
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_aic_comparison(self, eu_df: pd.DataFrame, balkan_df: pd.DataFrame,
                           ax: plt.Axes):
        """Helper function to create AIC comparison plot"""
        eu_means = eu_df.groupby('distribution')['aic'].mean()
        balkan_means = balkan_df.groupby('distribution')['aic'].mean()

        x = np.arange(len(eu_means))
        width = 0.35

        ax.bar(x - width/2, eu_means, width, label='EU')
        ax.bar(x + width/2, balkan_means, width, label='Balkan')

        ax.set_ylabel('Mean AIC')
        ax.set_title('AIC Comparison by Region')
        ax.set_xticks(x)
        ax.set_xticklabels(eu_means.index)
        ax.legend()

    def generate_full_report(self):
        """Generate complete LaTeX report"""
        self.generate_parameter_tables()
        self.generate_ks_heatmap()
        self.generate_fit_quality_comparison()

        # Create main LaTeX document
        tex_content = [
            r"\documentclass{article}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage{makecell}",
            r"\title{Distribution Analysis Report}",
            r"\author{Statistical Analysis Team}",
            r"\begin{document}",
            r"\maketitle",

            r"\section{Distribution Parameters}",
            r"\input{parameter_table_eu}",
            r"\input{parameter_table_balkan}",

            r"\section{Fit Quality Analysis}",
            r"\subsection{Kolmogorov-Smirnov Statistics}",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{ks_heatmap}",
            r"\caption{KS Statistics Heatmap}",
            r"\end{figure}",

            r"\subsection{Comparative Analysis}",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{fit_quality_comparison}",
            r"\caption{Fit Quality Comparison between EU and Balkan Airports}",
            r"\end{figure}",

            r"\end{document}"
        ]

        # Save main LaTeX file
        with open(self.report_dir / 'distribution_report.tex', 'w') as f:
            f.write('\n'.join(tex_content))

if __name__ == "__main__":
    generator = DistributionReportGenerator()
    generator.generate_full_report()
