"""
main_data_prep.py - Master Pipeline for Data Preparation & Junction Partitioning

Executes the complete data pipeline for AIORI Hackathon PS-003:
  1. Cleans and standardizes raw vehicle annotations from YOLO.
  2. Partitions the dataset across 3–5 virtual junctions (clients).
  3. Generates both IID and Non-IID client distributions.
  4. Applies strict chronological train/test splitting (zero lookahead leakage).
  5. Exports ready-to-use CSV splits, sliding-window tensors, and JSON metadata.
  6. Generates high-resolution visualization charts for presentations & reports.
"""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

# Add src to python path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir / "src"))

from data_prep import clean_vehicle_counts, get_traffic_summary
from partition import (
    partition_iid,
    partition_non_iid,
    get_partition_stats,
    JUNCTION_PROFILES,
)
from dataset import (
    chronological_train_test_split,
    prepare_client_dataloaders,
    TrafficScaler,
)
from visualize import (
    plot_traffic_overview,
    plot_distribution_comparison,
    plot_chronological_splits,
)


def run_pipeline(
    raw_csv: Path,
    output_dir: Path,
    num_junctions: int = 4,
    train_ratio: float = 0.8,
    window_size: int = 12,
    horizon: int = 1,
    batch_size: int = 32,
    seed: int = 42,
):
    print("=" * 65)
    print(" AIORI Hackathon PS-003: Data Preparation & Junction Partitioning")
    print("=" * 65)
    print(f"Project Directory : {project_dir}")
    print(f"Num Junctions     : {num_junctions}")
    print(f"Train/Test Split  : {int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test (Chronological)")
    print(f"Sliding Window    : {window_size} lookback steps -> {horizon} forecast step")
    print(f"Random Seed       : {seed}")
    print("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    plots_dir = splits_dir / "plots"
    splits_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Clean and standardize raw vehicle count data
    # -------------------------------------------------------------
    clean_csv_path = output_dir / "cleaned_traffic_flow.csv"
    print("\n[Step 1/5] Cleaning and chronologically ordering dataset...")
    df_clean = clean_vehicle_counts(raw_csv, clean_csv_path)
    summary = get_traffic_summary(df_clean)
    print(f"-> Successfully processed {len(df_clean)} frames.")
    print(f"-> Total detected vehicles: {summary['total_detected_vehicles']:,} across classes:")
    print(f"   Cars: {summary['car_pct']:.1f}% | 2W: {summary['motorcycle_pct']:.1f}% | Trucks: {summary['truck_pct']:.1f}% | Buses: {summary['bus_pct']:.1f}%")

    # -------------------------------------------------------------
    # Step 2: Generate IID and Non-IID Partitions
    # -------------------------------------------------------------
    print(f"\n[Step 2/5] Partitioning data across {num_junctions} virtual junctions...")
    print("  -> Creating IID distribution...")
    iid_dict = partition_iid(df_clean, num_junctions=num_junctions, seed=seed)
    iid_stats = get_partition_stats(iid_dict)

    print("  -> Creating Non-IID distribution (urban junction profiles)...")
    non_iid_dict = partition_non_iid(df_clean, num_junctions=num_junctions, mode="profile", seed=seed)
    non_iid_stats = get_partition_stats(non_iid_dict)

    print("\n--- IID Partition Statistics ---")
    print(iid_stats.to_string(index=False))

    print("\n--- Non-IID Partition Statistics ---")
    print(non_iid_stats.to_string(index=False))

    # Save stats summaries
    iid_stats.to_csv(splits_dir / "iid_partition_summary.csv", index=False)
    non_iid_stats.to_csv(splits_dir / "non_iid_partition_summary.csv", index=False)

    # -------------------------------------------------------------
    # Step 3: Chronological Train/Test Split & Export CSVs
    # -------------------------------------------------------------
    print(f"\n[Step 3/5] Applying strict chronological train/test split ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")

    # Save individual client CSVs for both distributions
    for dist_name, dist_dict in [("iid", iid_dict), ("non_iid", non_iid_dict)]:
        dist_dir = splits_dir / dist_name
        dist_dir.mkdir(parents=True, exist_ok=True)

        for j_id, j_df in dist_dict.items():
            train_df, test_df = chronological_train_test_split(j_df, train_ratio=train_ratio)
            train_path = dist_dir / f"{j_id}_train.csv"
            test_path = dist_dir / f"{j_id}_test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

    # Centralized baseline split (all data chronologically split)
    central_train_df, central_test_df = chronological_train_test_split(df_clean, train_ratio=train_ratio)
    central_dir = splits_dir / "centralized"
    central_dir.mkdir(parents=True, exist_ok=True)
    central_train_df.to_csv(central_dir / "central_train.csv", index=False)
    central_test_df.to_csv(central_dir / "central_test.csv", index=False)

    print(f"-> Centralized baseline: {len(central_train_df)} train frames, {len(central_test_df)} test frames.")

    # -------------------------------------------------------------
    # Step 4: Verify PyTorch DataLoaders & Window Sequences
    # -------------------------------------------------------------
    print(f"\n[Step 4/5] Building PyTorch sliding-window sequences (W={window_size}, H={horizon})...")
    client_splits, dataloaders, scalers = prepare_client_dataloaders(
        iid_dict,
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        batch_size=batch_size,
    )

    print("  Verification of DataLoader batch shapes:")
    sample_train_X, sample_train_y = next(iter(dataloaders["junction_1_train"]))
    print(f"  Junction 1 Train Batch -> X: {list(sample_train_X.shape)} (Batch, Window, Features), y: {list(sample_train_y.shape)}")

    sample_test_X, sample_test_y = next(iter(dataloaders["junction_1_test"]))
    print(f"  Junction 1 Test Batch  -> X: {list(sample_test_X.shape)}, y: {list(sample_test_y.shape)}")

    # -------------------------------------------------------------
    # Step 5: Generate Visualizations & Metadata
    # -------------------------------------------------------------
    print("\n[Step 5/5] Generating publication-quality charts and metadata...")
    plot_traffic_overview(df_clean, plots_dir / "traffic_flow_overview.png")
    plot_distribution_comparison(iid_dict, non_iid_dict, plots_dir / "iid_vs_non_iid_distributions.png")
    plot_chronological_splits(iid_dict, train_ratio, plots_dir / "chronological_splits_iid.png")
    plot_chronological_splits(non_iid_dict, train_ratio, plots_dir / "chronological_splits_non_iid.png")

    # Export JSON metadata
    metadata = {
        "dataset_name": "BMD-45 Bengaluru Mobility Dataset",
        "total_records": len(df_clean),
        "num_junctions": num_junctions,
        "train_ratio": train_ratio,
        "test_ratio": round(1.0 - train_ratio, 2),
        "window_size": window_size,
        "horizon": horizon,
        "feature_columns": ["total_vehicles", "car", "motorcycle", "bus", "truck"],
        "target_column": "total_vehicles",
        "junctions": [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "iid_samples": len(iid_dict[p["id"]]),
                "non_iid_samples": len(non_iid_dict[p["id"]]),
            }
            for p in JUNCTION_PROFILES[:num_junctions]
        ],
        "plots_generated": [
            "traffic_flow_overview.png",
            "iid_vs_non_iid_distributions.png",
            "chronological_splits_iid.png",
            "chronological_splits_non_iid.png",
        ],
    }

    with open(splits_dir / "partition_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata exported to: {splits_dir / 'partition_metadata.json'}")

    print("\n" + "=" * 65)
    print(" DATA PREPARATION & PARTITIONING COMPLETE!")
    print(f" All CSV splits & visual plots saved in: {splits_dir}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMD-45 Data Preparation & Federated Junction Partitioning")
    parser.add_argument("--num_junctions", type=int, default=4, choices=[3, 4, 5], help="Number of virtual junctions (3 to 5)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Chronological train split ratio (e.g. 0.8)")
    parser.add_argument("--window_size", type=int, default=12, help="Lookback window size in steps")
    parser.add_argument("--horizon", type=int, default=1, help="Forecasting horizon steps ahead")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    raw_csv = project_dir / "outputs" / "vehicle_counts.csv"
    output_dir = project_dir / "outputs"

    run_pipeline(
        raw_csv=raw_csv,
        output_dir=output_dir,
        num_junctions=args.num_junctions,
        train_ratio=args.train_ratio,
        window_size=args.window_size,
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
    )
