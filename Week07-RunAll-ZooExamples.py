# -*- coding: utf-8 -*-
"""
Week 07 - Run All Zoo Examples
Runs every ZooData example script in sequence using a headless Matplotlib backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "Week07-Example1-ZooData-EDA.py",
    "Week07-Example2-ZooData-CorrelationAnalysis.py",
    "Week07-Example3-ZooData-PCA.py",
    "Week07-Example4-ZooData-Clustering.py",
    "Week07-Example5-ZooData-KMeansClustering.py",
    "Week07-Example6-ZooData-3DVisualization.py",
    "Week07-Example7-ZooData-Heatmap.py",
    "Week07-Example8-ZooData-Trisurface.py",
    "Week07-Example9-ZooData-CurseOfDimensionality.py",
]


def run_script(script_name: str) -> int:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print(f"[SKIP] {script_name} not found")
        return 0

    print(f"\n[RUN] {script_name}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        env=env,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print(f"[OK] {script_name}")
    else:
        print(f"[FAIL] {script_name} -> exit code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    print("=== Running All Zoo Examples ===")
    failures = []

    for script in SCRIPTS:
        exit_code = run_script(script)
        if exit_code != 0:
            failures.append((script, exit_code))

    print("\n=== Summary ===")
    if failures:
        for script, exit_code in failures:
            print(f"{script}: failed with exit code {exit_code}")
        raise SystemExit(1)

    print("All Zoo example scripts completed successfully.")
