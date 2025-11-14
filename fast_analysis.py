"""
Fast distribution analysis focusing on Normal and NCT fits
"""
from statistical_distribution_analysis import DistributionAnalyzer
from data_loader import DataLoader
import numpy as np

def main():
    try:
        print("Starting distribution analysis (optimized version)...")
        analyzer = DistributionAnalyzer(DataLoader())

        # Run the analysis with reduced bootstrap samples for speed
        results = analyzer.run_analysis(
            max_samples=50000,  # Reduced sample size
            bootstrap_samples=200  # Fewer bootstrap iterations
        )

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
