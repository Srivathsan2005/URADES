"""
URADES demonstration
====================

Runs representative alloys from the URADES datasets through the
complete screening workflow.

The demonstration shows:
    1. Case identification
    2. Boundary-condition checking
    3. Global Viability Index (GVI)
    4. Case-specific prediction/classification

Run from the repository root with:

    python examples/demo.py
"""

import os
import sys

# Allow the script to be run directly from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from urades.core import run_URADES
from urades.data import CASE1_DATA, CASE2_DATA, CASE3_DATA


def print_result(name, composition, unit):
    """Run URADES for one alloy and print the main results."""

    print("\n" + "=" * 70)
    print(f"ALLOY: {name}")
    print("=" * 70)

    print(f"Composition ({unit}%):")
    print(composition)

    result = run_URADES(composition, input_unit=unit)

    print("\nURADES RESULT")
    print("-" * 70)

    for key, value in result.items():
        print(f"{key:25s}: {value}")


def get_case1_example():
    """
    Select an actual Case 1 alloy from the validated dataset.

    The first dataset entry is used so that no demonstration
    composition is fabricated independently of the repository data.
    """
    name, comp, exp_dbtt = CASE1_DATA[0]
    return name, comp, "wt"


def get_case2_example():
    """
    Select an actual Case 2 alloy from the validated dataset.

    CASE2_DATA stores the composition as:
        name, W, Mo, Hf, Zr, Ti, experimental DBTT, LOOCV flag
    """
    name, W, Mo, Hf, Zr, Ti, exp_dbtt, in_loocv = CASE2_DATA[0]

    composition = {
        "W": W,
        "Mo": Mo,
        "Hf": Hf,
        "Zr": Zr,
        "Ti": Ti,
        "Nb": 100.0 - (W + Mo + Hf + Zr + Ti),
    }

    return name, composition, "at"


def get_case3_example():
    """
    Select an actual Case 3 alloy from the validated dataset.

    CASE3_DATA stores:
        name, composition dictionary, experimental classification
    """
    name, composition, exp_zone = CASE3_DATA[0]

    return name, composition, "at"


if __name__ == "__main__":

    print("\n" + "#" * 70)
    print("URADES — DEMONSTRATION")
    print("#" * 70)

    print(
        "\nThis demonstration runs one actual alloy from each "
        "URADES case through the complete screening framework."
    )

    # ------------------------------------------------------------------
    # CASE 1
    # ------------------------------------------------------------------
    name, composition, unit = get_case1_example()
    print_result(name, composition, unit)

    # ------------------------------------------------------------------
    # CASE 2
    # ------------------------------------------------------------------
    name, composition, unit = get_case2_example()
    print_result(name, composition, unit)

    # ------------------------------------------------------------------
    # CASE 3
    # ------------------------------------------------------------------
    name, composition, unit = get_case3_example()
    print_result(name, composition, unit)

    print("\n" + "#" * 70)
    print("DEMONSTRATION COMPLETE")
    print("#" * 70)
