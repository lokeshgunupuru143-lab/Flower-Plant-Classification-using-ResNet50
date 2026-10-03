@echo off
title Flower Plant Classifier - Streamlit App
cd /d "%~dp0"
if exist "Flower Plant classification using ResNet50\app.py" (
    cd "Flower Plant classification using ResNet50"
)
echo Starting Flower Plant Classifier Web Application...
python -m streamlit run app.py
pause
