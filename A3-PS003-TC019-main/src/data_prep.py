"""
data_prep.py - Data Preparation & Cleaning for BMD-45 Traffic Flow

Cleans YOLO annotation outputs (vehicle_counts.csv), fixes path inconsistencies,
sorts frames chronologically, computes temporal features, and generates the
master standardized traffic flow time-series.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def clean_vehicle_counts(
    raw_csv_path: Path,
    output_csv_path: Path = None,
) -> pd.DataFrame:
    """
    Cleans raw vehicle counts CSV:
    1. Normalizes image paths and removes duplicated entries.
    2. Extracts numerical frame ID from filename for chronological ordering.
    3. Sorts rows strictly by frame sequence.
    4. Computes temporal indices, vehicle class ratios, and rolling stats.

    Args:
        raw_csv_path: Path to outputs/vehicle_counts.csv
        output_csv_path: Optional path to save cleaned CSV.

    Returns:
        pd.DataFrame: Cleaned and chronologically ordered DataFrame.
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw vehicle counts file not found: {raw_csv_path}")

    print(f"Loading raw annotations from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    # 1. Standardize image_path and fix missing folder prefix for images_000
    def normalize_path(p: str) -> str:
        p_str = str(p).replace("\\", "/").strip()
        if "/" not in p_str:
            return f"images_000/{p_str}"
        return p_str

    df["clean_path"] = df["image_path"].apply(normalize_path)

    # Deduplicate based on standardized path
    initial_len = len(df)
    df = df.drop_duplicates(subset=["clean_path"]).copy()
    dedup_len = len(df)
    print(f"Deduplication: {initial_len} -> {dedup_len} records (removed {initial_len - dedup_len} duplicates)")

    # 2. Extract folder and frame_id
    df["folder"] = df["clean_path"].apply(lambda p: p.split("/")[0])
    df["frame_id"] = df["clean_path"].apply(
        lambda p: int(Path(p).stem)
    )

    # 3. Sort strictly by frame_id for strict chronological ordering
    df = df.sort_values("frame_id").reset_index(drop=True)

    # 4. Generate sequential timeline index (time_step: 0, 1, 2, ..., N-1)
    df["time_step"] = np.arange(len(df))

    # Calculate frame diffs (intervals between recorded frames)
    df["frame_interval"] = df["frame_id"].diff().fillna(0).astype(int)

    # Calculate vehicle breakdown percentages
    safe_total = df["total_vehicles"].replace(0, 1)
    df["car_ratio"] = (df["car"] / safe_total).round(4)
    df["motorcycle_ratio"] = (df["motorcycle"] / safe_total).round(4)
    df["bus_ratio"] = (df["bus"] / safe_total).round(4)
    df["truck_ratio"] = (df["truck"] / safe_total).round(4)

    # Calculate rolling short-term trends (5-step and 15-step rolling flow)
    df["flow_rolling_mean_5"] = (
        df["total_vehicles"].rolling(window=5, min_periods=1).mean().round(2)
    )
    df["flow_rolling_mean_15"] = (
        df["total_vehicles"].rolling(window=15, min_periods=1).mean().round(2)
    )
    df["flow_rolling_std_15"] = (
        df["total_vehicles"].rolling(window=15, min_periods=1).std().fillna(0).round(2)
    )

    # Select and order standard output columns
    columns_order = [
        "time_step",
        "frame_id",
        "folder",
        "clean_path",
        "total_vehicles",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "car_ratio",
        "motorcycle_ratio",
        "bus_ratio",
        "truck_ratio",
        "flow_rolling_mean_5",
        "flow_rolling_mean_15",
        "flow_rolling_std_15",
        "frame_interval",
    ]
    df = df[columns_order]

    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Cleaned dataset saved to: {output_csv_path} ({len(df)} rows)")

    return df


def get_traffic_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes overall summary statistics for the dataset.
    """
    return {
        "total_frames": int(len(df)),
        "min_frame_id": int(df["frame_id"].min()),
        "max_frame_id": int(df["frame_id"].max()),
        "mean_total_vehicles": float(df["total_vehicles"].mean()),
        "std_total_vehicles": float(df["total_vehicles"].std()),
        "max_total_vehicles": int(df["total_vehicles"].max()),
        "total_detected_vehicles": int(df["total_vehicles"].sum()),
        "total_cars": int(df["car"].sum()),
        "total_motorcycles": int(df["motorcycle"].sum()),
        "total_buses": int(df["bus"].sum()),
        "total_trucks": int(df["truck"].sum()),
        "car_pct": float((df["car"].sum() / df["total_vehicles"].sum()) * 100),
        "motorcycle_pct": float((df["motorcycle"].sum() / df["total_vehicles"].sum()) * 100),
        "bus_pct": float((df["bus"].sum() / df["total_vehicles"].sum()) * 100),
        "truck_pct": float((df["truck"].sum() / df["total_vehicles"].sum()) * 100),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "outputs" / "vehicle_counts.csv"
    clean_path = project_root / "outputs" / "cleaned_traffic_flow.csv"

    cleaned_df = clean_vehicle_counts(raw_path, clean_path)
    summary = get_traffic_summary(cleaned_df)

    print("\n================ DATA SUMMARY ================")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.2f}")
        else:
            print(f"{k:25s}: {v}")
    print("==============================================")
