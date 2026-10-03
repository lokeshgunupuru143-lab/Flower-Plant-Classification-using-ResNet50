"""Root launcher for Streamlit app.py"""
import runpy
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "Flower Plant classification using DenseNet121" / "app.py"
if not target.is_file():
    raise FileNotFoundError(f"Cannot find main app at {target}")

# Execute the inner app.py
runpy.run_path(str(target), run_name="__main__")
