#!/bin/bash

# Script to automatically generate requirements.txt from neuro conda environment
# Usage: ./generate_requirements.sh

echo "Generating requirements.txt from neuro conda environment..."

# Activate conda environment and export pip packages
conda activate neuro && pip list --format=freeze | grep -E "(torch|torchvision|kagglehub|matplotlib|numpy|pillow|PIL|tqdm|pandas|scikit-learn|scikit-image|opencv)" > requirements.txt

echo "✅ requirements.txt generated successfully!"
echo "📦 Packages included:"
wc -l requirements.txt

