"""
URADES custom alloy screening.

Enter an alloy composition below and run the script.

Example:
    python examples/custom_alloy.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.core import run_URADES


# ============================================================
# ENTER YOUR ALLOY HERE
# ============================================================

composition = {
    "Nb": 75.4,
    "Hf": 15.0,
    "Ti": 5.5,
    "W": 4.1,
}

input_unit = "wt"


# ============================================================
# RUN URADES
# ============================================================

result = run_URADES(
    composition,
    input_unit=input_unit
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("=" * 70)
print("URADES — CUSTOM ALLOY SCREENING")
print("=" * 70)

print("\nInput composition:")
for element, value in composition.items():
    print(f"  {element:5s}: {value}")

print(f"\nInput unit: {input_unit}%")

print("\nURADES RESULT")
print("-" * 70)

for key, value in result.items():
    print(f"{key:25s}: {value}")

print("=" * 70)
