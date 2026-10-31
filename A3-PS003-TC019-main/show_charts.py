"""
show_charts.py - Quick viewer for generated traffic & partition plots

Usage:
  python show_charts.py          # Opens all charts in Windows default photo viewer
  python show_charts.py --gui    # Displays charts interactively in a Matplotlib window
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def open_in_viewer(plots_dir: Path):
    """Opens generated plots in default Windows image viewer."""
    chart_files = [
        "iid_vs_non_iid_distributions.png",
        "traffic_flow_overview.png",
        "chronological_splits_iid.png",
        "chronological_splits_non_iid.png",
    ]
    print(f"Opening charts from: {plots_dir}\n")
    for f in chart_files:
        p = plots_dir / f
        if p.exists():
            print(f"-> Opening: {p.name}")
            try:
                os.startfile(str(p))
            except Exception as e:
                print(f"   Could not open {p.name}: {e}")
        else:
            print(f"   File not found: {p}")


def show_in_gui(plots_dir: Path):
    """Displays all 4 charts in an interactive matplotlib figure."""
    chart_files = [
        ("IID vs Non-IID Junction Distributions", plots_dir / "iid_vs_non_iid_distributions.png"),
        ("Traffic Flow Overview & Rolling Trend", plots_dir / "traffic_flow_overview.png"),
        ("Chronological Train/Test Splits (IID)", plots_dir / "chronological_splits_iid.png"),
        ("Chronological Train/Test Splits (Non-IID)", plots_dir / "chronological_splits_non_iid.png"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (title, p) in enumerate(chart_files):
        ax = axes[idx]
        if p.exists():
            img = mpimg.imread(str(p))
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"Not found:\n{p.name}", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    plt.suptitle("AIORI PS-003: BMD-45 Traffic Flow & Partition Visualizations", fontsize=15, fontweight="bold", y=1.01)
    print("Displaying interactive chart viewer (close the window to finish)...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Traffic & Partition Charts")
    parser.add_argument("--gui", action="store_true", help="Display all 4 charts inside an interactive Matplotlib window")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    plots_directory = project_dir / "outputs" / "splits" / "plots"

    if not plots_directory.exists():
        print(f"Plots directory not found: {plots_directory}")
        print("Please run 'python main_data_prep.py' first to generate plots.")
        exit(1)

    if args.gui:
        show_in_gui(plots_directory)
    else:
        open_in_viewer(plots_directory)
