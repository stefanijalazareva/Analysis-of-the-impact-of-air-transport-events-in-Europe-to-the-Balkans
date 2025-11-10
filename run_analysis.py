"""
Run the distribution analysis with sampling for large datasets
"""
from statistical_distribution_analysis import DistributionAnalyzer
from data_loader import DataLoader
import numpy as np

def main():
    try:
        print("Starting distribution analysis...")
        analyzer = DistributionAnalyzer(DataLoader())

        # Run the analysis with sampling
        results = analyzer.run_analysis(max_samples=100000)  # Limit sample size for efficiency

        print("\nAnalysis complete!")
        print("Results saved in the results/distribution_analysis directory:")
        print("- Region comparison plots and statistics")
        print("- Distribution fit parameters and quality metrics")
        print("- QQ plots comparing different distributions")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
