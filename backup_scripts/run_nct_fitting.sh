#!/bin/bash
# Run this script to perform the complete NCT fitting and traffic dependence modeling
# with sample data for faster results

# Step 1: Run global fitting with a small sample
echo "===== Step 1: Global NCT Fitting ====="
python fast_distribution_fitting.py --step global --sample 100000 --maxiter 300

# Step 2: Run daily fitting with limited days
echo ""
echo "===== Step 2: Daily NCT Fitting ====="
python fast_distribution_fitting.py --step daily --max-days 50

# Step 3: Run traffic dependence modeling with both methods
echo ""
echo "===== Step 3: Traffic Dependence Modeling ====="
python fast_distribution_fitting.py --step traffic --traffic-method both
