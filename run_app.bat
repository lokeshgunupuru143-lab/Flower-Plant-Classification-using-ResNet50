@echo off
title Flower Plant Classifier - Streamlit App
cd /d "%~dp0"
if exist "Flower Plant classification using DenseNet121\app.py" (
    cd "Flower Plant classification using DenseNet121"
)
echo Starting Flower Plant Classifier Web Application...
python -m streamlit run app.py
pause
