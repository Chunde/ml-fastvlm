#!/bin/bash
# Bash script to run the FastVLM interactive prediction script

echo "Activating fastvlm environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fastvlm

if [ $? -ne 0 ]; then
    echo "Failed to activate fastvlm environment. Please make sure it exists."
    echo "You can create it with: conda create -n fastvlm python=3.10"
    exit 1
fi

echo "Running FastVLM interactive prediction script..."
python predict_interactive.py "$@"

if [ $? -ne 0 ]; then
    echo "Script execution failed."
    exit 1
fi

echo "Script completed successfully."