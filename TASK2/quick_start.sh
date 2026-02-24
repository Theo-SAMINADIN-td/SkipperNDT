#!/bin/bash
# Quick Start Script for MAP WIDTH REGRESSOR - TASK 2
# Usage: bash quick_start.sh

set -e

echo ""
echo "   MAP WIDTH REGRESSOR - QUICK START"
echo ""
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo " Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python version
echo " Checking Python version..."
python --version
echo ""

# Install requirements if needed
echo " Checking dependencies..."
pip install -q -r requirements.txt
echo " Dependencies installed"
echo ""

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo " Creating data directory..."
    mkdir data
    echo "     Place your .npz files in: data/"
fi

# Create outputs directory
if [ ! -d "outputs" ]; then
    mkdir outputs
fi

echo ""
echo ""
echo "   AVAILABLE COMMANDS"
echo ""
echo ""
echo "1⃣  Dataset Analysis (optional):"
echo "   python analyze_width_dataset.py"
echo ""
echo "2⃣  Training:"
echo "   python map_width_regressor.py"
echo ""
echo "3⃣  Evaluation:"
echo "   python evaluate_regression.py"
echo ""
echo "4⃣  Single Prediction:"
echo "   python predict_map_width.py data/sample.npz"
echo ""
echo "5⃣  Batch Predictions:"
echo "   python batch_predict_width.py data/ --output predictions.csv"
echo ""
echo ""
echo ""
echo " For detailed instructions, see: README_TASK2.md"
echo ""

# Optional: Ask if user wants to run dataset analysis
read -p "Do you want to run dataset analysis first? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python analyze_width_dataset.py
fi

echo ""
echo " Ready to train! Run: python map_width_regressor.py"
