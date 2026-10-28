"""
partition.py - Federated Client Partitioning (IID & Non-IID)

Partitions the standardized BMD-45 traffic dataset across 3 to 5 simulated
virtual junctions (clients) for Federated Learning simulations.
Supports both:
  1. IID (Identical and Independently Distributed):
     Uniform distribution of traffic flow and vehicle classes across junctions.
  2. Non-IID (Heterogeneous Distributions):
     Realistic traffic profiles across distinct urban intersections
     (e.g., commuter car corridor, dense 2-wheeler market, freight ring road).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Predefined junction identities based on Bengaluru mobility zones
JUNCTION_PROFILES = [
    {
        "id": "junction_1",
        "name": "Silk Board Arterial",
        "description": "High-volume commuter corridor dominated by cars and cabs",
        "primary_class": "car",
        "volume_bias": 1.25,
    },
    {
        "id": "junction_2",
        "name": "Majestic Urban Core",
        "description": "Dense city center with high two-wheeler / motorcycle share",
        "primary_class": "motorcycle",
        "volume_bias": 1.05,
    },
    {
        "id": "junction_3",
        "name": "Hebbal Ring Road",
        "description": "Freight & logistics corridor with high truck & bus ratio",
        "primary_class": "truck",
        "volume_bias": 0.95,
    },
    {
        "id": "junction_4",
        "name": "Indiranagar Central",
        "description": "Mixed commercial & residential grid with moderate traffic",
        "primary_class": "balanced",
        "volume_bias": 0.85,
    },
    {
        "id": "junction_5",
        "name": "Whitefield Tech Park",
        "description": "Bottleneck corridor with high traffic variance and peak surges",
        "primary_class": "surge",
        "volume_bias": 1.15,
    },
]


def partition_iid(
    df: pd.DataFrame,
    num_junctions: int = 4,
    block_size: int = 16,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Partitions the dataset into `num_junctions` IID clients.

    Maintains local temporal continuity (in blocks of `block_size` consecutive frames)
    while uniformly interleaving blocks across clients. Each junction experiences
    statistically identical vehicle volume and class distribution across time.

    Args:
        df: Cleaned chronological traffic DataFrame.
        num_junctions: Number of virtual junctions (3 to 5).
        block_size: Number of consecutive frames per temporal chunk.
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of junction_id -> junction DataFrame.
    """
    if not (3 <= num_junctions <= 5):
        raise ValueError(f"num_junctions must be between 3 and 5, got {num_junctions}")

    df_sorted = df.sort_values("time_step").reset_index(drop=True).copy()
    total_rows = len(df_sorted)

    # Divide timeline into contiguous chronological blocks
    num_blocks = int(np.ceil(total_rows / block_size))
    block_assignments = np.zeros(total_rows, dtype=int)

    rng = np.random.RandomState(seed)

    # Interleaved round-robin assignment of blocks to clients
    for b in range(num_blocks):
        start_idx = b * block_size
        end_idx = min(start_idx + block_size, total_rows)
        client_id = b % num_junctions
        block_assignments[start_idx:end_idx] = client_id

    client_dfs = {}
    for k in range(num_junctions):
        j_profile = JUNCTION_PROFILES[k]
        j_id = j_profile["id"]
        j_name = j_profile["name"]

        client_df = df_sorted[block_assignments == k].copy()
        # Ensure chronological ordering within client
        client_df = client_df.sort_values("time_step").reset_index(drop=True)
        client_df["client_id"] = j_id
        client_df["junction_name"] = j_name
        client_df["distribution_type"] = "IID"
        client_df["client_time_step"] = np.arange(len(client_df))

        client_dfs[j_id] = client_df

    return client_dfs


def partition_non_iid(
    df: pd.DataFrame,
    num_junctions: int = 4,
    alpha: float = 0.5,
    mode: str = "profile",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Partitions dataset into `num_junctions` Non-IID clients with distinct traffic regimes.

    Modes:
      - 'profile': Assigns frames based on realistic junction characteristics
                   (e.g., commuter car corridor, 2-wheeler core, freight corridor).
      - 'dirichlet': Standard Dirichlet allocation Dir(alpha) on vehicle class composition.

    Args:
        df: Cleaned chronological traffic DataFrame.
        num_junctions: Number of virtual junctions (3 to 5).
        alpha: Dirichlet concentration parameter (lower = more heterogeneous).
        mode: 'profile' or 'dirichlet'.
        seed: Random seed.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of junction_id -> junction DataFrame.
    """
    if not (3 <= num_junctions <= 5):
        raise ValueError(f"num_junctions must be between 3 and 5, got {num_junctions}")

    df_sorted = df.sort_values("time_step").reset_index(drop=True).copy()
    rng = np.random.RandomState(seed)
    total_rows = len(df_sorted)

    client_dfs = {}

    if mode == "profile":
        # Realistic urban junction profiling
        # Assign probabilities to each frame based on its vehicle distribution
        weights = np.ones((total_rows, num_junctions), dtype=float)

        for k in range(num_junctions):
            profile = JUNCTION_PROFILES[k]
            p_class = profile["primary_class"]

            if p_class == "car":
                # Favor frames with high car ratio and high volume
                weights[:, k] = 1.0 + 3.0 * df_sorted["car_ratio"] + 0.1 * df_sorted["total_vehicles"]
            elif p_class == "motorcycle":
                # Favor frames with high motorcycle ratio
                weights[:, k] = 1.0 + 3.5 * df_sorted["motorcycle_ratio"]
            elif p_class == "truck":
                # Favor frames with high commercial/heavy vehicles (trucks + buses)
                heavy_ratio = df_sorted["truck_ratio"] + df_sorted["bus_ratio"]
                weights[:, k] = 1.0 + 4.0 * heavy_ratio
            elif p_class == "balanced":
                # Moderate balanced flow
                flow_diff = np.abs(df_sorted["total_vehicles"] - df_sorted["total_vehicles"].median())
                weights[:, k] = 1.0 + 2.0 / (1.0 + flow_diff)
            elif p_class == "surge":
                # High traffic peaks and variance
                weights[:, k] = 1.0 + 0.3 * (df_sorted["total_vehicles"] ** 1.3)

        # Normalize across clients per row
        probs = weights / weights.sum(axis=1, keepdims=True)

        # Assign each sample to a client according to probability weights
        client_indices = [[] for _ in range(num_junctions)]
        for i in range(total_rows):
            chosen_client = rng.choice(num_junctions, p=probs[i])
            client_indices[chosen_client].append(i)

    elif mode == "dirichlet":
        # Dirichlet distribution over vehicle classes
        classes = ["car", "motorcycle", "bus", "truck"]
        client_indices = [[] for _ in range(num_junctions)]

        # Class proportions matrix from Dirichlet
        class_proportions = rng.dirichlet(np.repeat(alpha, num_junctions), size=len(classes))

        for idx, row in df_sorted.iterrows():
            # Compute frame likelihood per client
            sample_weights = np.zeros(num_junctions)
            for c_idx, c_name in enumerate(classes):
                sample_weights += row[c_name] * class_proportions[c_idx]

            if sample_weights.sum() == 0:
                sample_weights = np.ones(num_junctions)

            prob = sample_weights / sample_weights.sum()
            chosen = rng.choice(num_junctions, p=prob)
            client_indices[chosen].append(idx)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'profile' or 'dirichlet'.")

    # Assemble client DataFrames
    for k in range(num_junctions):
        j_profile = JUNCTION_PROFILES[k]
        j_id = j_profile["id"]
        j_name = j_profile["name"]

        indices = client_indices[k]
        client_df = df_sorted.iloc[indices].copy()
        # Keep strict chronological order within client
        client_df = client_df.sort_values("time_step").reset_index(drop=True)
        client_df["client_id"] = j_id
        client_df["junction_name"] = j_name
        client_df["distribution_type"] = f"Non-IID ({mode})"
        client_df["client_time_step"] = np.arange(len(client_df))

        client_dfs[j_id] = client_df

    return client_dfs


def get_partition_stats(client_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Computes comparative statistics across all clients/junctions.

    Returns:
        pd.DataFrame: Table comparing count, mean volume, std, and vehicle % shares.
    """
    rows = []
    for j_id, df in client_dict.items():
        total_v = df["total_vehicles"].sum()
        safe_total = total_v if total_v > 0 else 1

        rows.append({
            "Junction ID": j_id,
            "Junction Name": df["junction_name"].iloc[0] if "junction_name" in df else j_id,
            "Distribution": df["distribution_type"].iloc[0] if "distribution_type" in df else "N/A",
            "Total Samples": len(df),
            "Mean Flow": round(df["total_vehicles"].mean(), 2),
            "Std Flow": round(df["total_vehicles"].std(), 2),
            "Max Flow": int(df["total_vehicles"].max()),
            "Car %": round((df["car"].sum() / safe_total) * 100, 1),
            "Motorcycle %": round((df["motorcycle"].sum() / safe_total) * 100, 1),
            "Bus %": round((df["bus"].sum() / safe_total) * 100, 1),
            "Truck %": round((df["truck"].sum() / safe_total) * 100, 1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    clean_csv = project_root / "outputs" / "cleaned_traffic_flow.csv"

    if not clean_csv.exists():
        from data_prep import clean_vehicle_counts
        clean_vehicle_counts(project_root / "outputs" / "vehicle_counts.csv", clean_csv)

    df_clean = pd.read_csv(clean_csv)

    print("\n========== IID PARTITION (4 JUNCTIONS) ==========")
    iid_splits = partition_iid(df_clean, num_junctions=4, seed=42)
    iid_stats = get_partition_stats(iid_splits)
    print(iid_stats.to_string(index=False))

    print("\n======== NON-IID PARTITION (4 JUNCTIONS) ========")
    non_iid_splits = partition_non_iid(df_clean, num_junctions=4, mode="profile", seed=42)
    non_iid_stats = get_partition_stats(non_iid_splits)
    print(non_iid_stats.to_string(index=False))
