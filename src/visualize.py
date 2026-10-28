"""
visualize.py - Traffic Flow & Partition Visualizations

Generates publication-quality charts for:
  1. Chronological traffic flow timeline and rolling averages.
  2. IID vs Non-IID client distribution comparisons (volume & vehicle class breakdown).
  3. Chronological train/test split timeline boundaries.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Clean visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]


def _safe_savefig(output_path: Path):
    """Safely saves a plot, handling potential Windows file locks from image viewers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    except OSError:
        # If open in Windows Photo viewer, save to temp and attempt replace
        temp_path = output_path.with_name(f"temp_{output_path.name}")
        plt.savefig(temp_path, dpi=200, bbox_inches="tight")
        try:
            import os
            os.replace(temp_path, output_path)
            print(f"Plot saved to: {output_path}")
        except Exception:
            print(f"Plot saved to: {temp_path} (original locked by external viewer)")


def plot_traffic_overview(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    sample_stride: int = 10,
):
    """
    Plots the full chronological traffic flow time series with rolling trend lines.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    df_sub = df.iloc[::sample_stride].reset_index(drop=True)

    # Top subplot: Total vehicles flow and rolling average
    ax1.plot(
        df_sub["time_step"],
        df_sub["total_vehicles"],
        color="#a0c4ff",
        alpha=0.5,
        linewidth=1,
        label="Raw Frame Vehicle Count",
    )
    ax1.plot(
        df_sub["time_step"],
        df_sub["flow_rolling_mean_15"],
        color="#003566",
        linewidth=2,
        label="15-Step Rolling Trend",
    )
    ax1.set_ylabel("Total Vehicles", fontsize=11, fontweight="bold")
    ax1.set_title(
        "BMD-45 Bengaluru Traffic Flow: Chronological Time-Series",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bottom subplot: Breakdown by vehicle types
    ax2.plot(df_sub["time_step"], df_sub["car"], color="#2b5c8f", label="Car", alpha=0.8)
    ax2.plot(df_sub["time_step"], df_sub["motorcycle"], color="#d95f02", label="Motorcycle / 2W", alpha=0.8)
    ax2.plot(df_sub["time_step"], df_sub["bus"], color="#1b9e77", label="Bus", alpha=0.8)
    ax2.plot(df_sub["time_step"], df_sub["truck"], color="#e7298a", label="Truck", alpha=0.8)
    ax2.set_xlabel("Chronological Time Step (Normalized CCTV Sequence)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Vehicle Count by Class", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if output_path:
        _safe_savefig(output_path)
    plt.close()


def plot_distribution_comparison(
    iid_dict: Dict[str, pd.DataFrame],
    non_iid_dict: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
):
    """
    Compares vehicle class proportions across junctions for IID vs Non-IID.
    """
    classes = ["car", "motorcycle", "bus", "truck"]
    class_labels = ["Car", "Motorcycle", "Bus", "Truck"]
    class_colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # 1. IID Proportions
    iid_names = []
    iid_data = {c: [] for c in classes}
    for j_id, df in iid_dict.items():
        total = max(df["total_vehicles"].sum(), 1)
        name = df["junction_name"].iloc[0] if "junction_name" in df else j_id
        iid_names.append(name.split()[0] + " " + name.split()[1] if len(name.split()) > 1 else name)
        for c in classes:
            iid_data[c].append((df[c].sum() / total) * 100)

    x = np.arange(len(iid_names))
    width = 0.18

    for idx, c in enumerate(classes):
        offset = (idx - 1.5) * width
        ax1.bar(
            x + offset,
            iid_data[c],
            width,
            label=class_labels[idx],
            color=class_colors[idx],
            alpha=0.9,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(iid_names, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Class Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("IID Client Distribution (Homogeneous)", fontsize=12, fontweight="bold", pad=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Non-IID Proportions
    non_iid_names = []
    non_iid_data = {c: [] for c in classes}
    for j_id, df in non_iid_dict.items():
        total = max(df["total_vehicles"].sum(), 1)
        name = df["junction_name"].iloc[0] if "junction_name" in df else j_id
        non_iid_names.append(name.split()[0] + " " + name.split()[1] if len(name.split()) > 1 else name)
        for c in classes:
            non_iid_data[c].append((df[c].sum() / total) * 100)

    for idx, c in enumerate(classes):
        offset = (idx - 1.5) * width
        ax2.bar(
            x + offset,
            non_iid_data[c],
            width,
            label=class_labels[idx],
            color=class_colors[idx],
            alpha=0.9,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(non_iid_names, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax2.set_title("Non-IID Client Distribution (Heterogeneous)", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Federated Learning Simulator: IID vs Non-IID Junction Profiles",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    if output_path:
        _safe_savefig(output_path)
    plt.close()


def plot_chronological_splits(
    client_dict: Dict[str, pd.DataFrame],
    train_ratio: float = 0.8,
    output_path: Optional[Path] = None,
):
    """
    Plots the chronological train/test split boundary for each junction.
    """
    num_junctions = len(client_dict)
    fig, axes = plt.subplots(num_junctions, 1, figsize=(14, 2.5 * num_junctions), sharex=False)
    if num_junctions == 1:
        axes = [axes]

    for idx, (j_id, df) in enumerate(client_dict.items()):
        ax = axes[idx]
        split_idx = int(len(df) * train_ratio)

        # Plot train portion
        ax.plot(
            np.arange(split_idx),
            df["total_vehicles"].iloc[:split_idx],
            color="#2ecc71",
            alpha=0.6,
            linewidth=1,
            label=f"Train Split (First {int(train_ratio*100)}%)",
        )
        # Plot test portion
        ax.plot(
            np.arange(split_idx, len(df)),
            df["total_vehicles"].iloc[split_idx:],
            color="#e67e22",
            alpha=0.7,
            linewidth=1,
            label=f"Test Split (Final {int((1-train_ratio)*100)}%)",
        )

        # Add vertical split boundary
        ax.axvline(x=split_idx, color="#c0392b", linestyle="--", linewidth=1.5, label="Temporal Cutoff")

        j_name = df["junction_name"].iloc[0] if "junction_name" in df else j_id
        ax.set_ylabel("Vehicles", fontsize=9, fontweight="bold")
        ax.set_title(
            f"{j_name} ({j_id}): Train={split_idx} frames, Test={len(df)-split_idx} frames",
            fontsize=10,
            fontweight="bold",
            loc="left",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Local Junction Chronological Time Step", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Strict Chronological Train/Test Split (Zero Lookahead Data Leakage)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    if output_path:
        _safe_savefig(output_path)
    plt.close()
