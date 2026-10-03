#!/usr/bin/env python3
"""Root forwarder for train.py"""
import subprocess
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "Flower Plant classification using ResNet50" / "train.py"
if not target.is_file():
    print(f"Target script not found: {target}")
    sys.exit(1)

result = subprocess.run([sys.executable, str(target)] + sys.argv[1:])
sys.exit(result.returncode)
