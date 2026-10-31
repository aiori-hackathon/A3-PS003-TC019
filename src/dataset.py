"""
dataset.py - Chronological Train/Test Split & Sliding-Window Sequence Dataset

Implements strict chronological train/test splitting to prevent lookahead data leakage.
Provides PyTorch Dataset and DataLoader generators for both local client models
(Federated Learning) and the global pooled model (Centralized Baseline).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficScaler:
    """
    MinMax or Standard Scaler fitted strictly on the training set
    to prevent temporal lookahead data leakage.
    """

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray):
        if self.method == "minmax":
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
            # Avoid division by zero
            self.max_ = np.where(self.max_ == self.min_, self.min_ + 1e-6, self.max_)
        elif self.method == "standard":
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0)
            self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.method == "minmax":
            return (data - self.min_) / (self.max_ - self.min_)
        elif self.method == "standard":
            return (data - self.mean_) / self.std_
        return data

    def inverse_transform_target(self, target_scaled: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """Inverse transforms target predictions back to raw vehicle count scale."""
        if self.method == "minmax":
            diff = self.max_[target_idx] - self.min_[target_idx]
            return target_scaled * diff + self.min_[target_idx]
        elif self.method == "standard":
            return target_scaled * self.std_[target_idx] + self.mean_[target_idx]
        return target_scaled

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }


class TrafficTimeSeriesDataset(Dataset):
    """
    Sliding-window Dataset for time-series traffic forecasting.

    Given lookback window W and forecast horizon H:
      X[i] = features[i : i + W]
      y[i] = target[i + W : i + W + H]
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 12,
        horizon: int = 1,
    ):
        assert len(features) == len(targets), "Features and targets must have same length"
        self.window_size = window_size
        self.horizon = horizon

        num_samples = len(features) - window_size - horizon + 1
        if num_samples <= 0:
            raise ValueError(
                f"Series length ({len(features)}) is too short for window_size={window_size} "
                f"and horizon={horizon}"
            )

        X_list = []
        y_list = []

        for i in range(num_samples):
            X_list.append(features[i : i + window_size])
            # Target is the forecast over the next `horizon` steps
            y_list.append(targets[i + window_size : i + window_size + horizon])

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame chronologically into train and test sets.

    Ensures zero temporal data leakage:
      - Train set: first (train_ratio * 100)% of timeline.
      - Test set: remaining (1 - train_ratio) * 100% of timeline.
    """
    if not (0.5 <= train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0.5 and 1.0, got {train_ratio}")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train_df, test_df


def prepare_client_dataloaders(
    client_dict: Dict[str, pd.DataFrame],
    window_size: int = 12,
    horizon: int = 1,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "total_vehicles",
    scaler_method: str = "minmax",
) -> Tuple[Dict[str, Dict], Dict[str, DataLoader], Dict]:
    """
    Prepares PyTorch DataLoaders for each client and the centralized baseline.

    Args:
        client_dict: Dict of junction_id -> junction DataFrame.
        window_size: Lookback history steps (default: 12).
        horizon: Prediction steps ahead (default: 1).
        train_ratio: Chronological train split fraction (default: 0.8).
        batch_size: Batch size for DataLoader.
        feature_cols: Features to use (defaults to total_vehicles + 4 vehicle types).
        target_col: Prediction target column ('total_vehicles').
        scaler_method: 'minmax' or 'standard'.

    Returns:
        client_splits: Metadata & raw split data per client.
        dataloaders: Dict containing 'train' and 'test' DataLoaders per client,
                     plus 'centralized_train' and 'centralized_test'.
        scalers: Dict of fitted scalers (per client and centralized).
    """
    if feature_cols is None:
        feature_cols = ["total_vehicles", "car", "motorcycle", "bus", "truck"]

    target_idx = feature_cols.index(target_col)

    client_splits = {}
    dataloaders = {}
    scalers = {}

    central_train_dfs = []
    central_test_dfs = []

    for client_id, df in client_dict.items():
        # 1. Chronological split
        train_df, test_df = chronological_train_test_split(df, train_ratio=train_ratio)
        central_train_dfs.append(train_df)
        central_test_dfs.append(test_df)

        # 2. Extract arrays
        train_feat = train_df[feature_cols].values.astype(np.float32)
        test_feat = test_df[feature_cols].values.astype(np.float32)

        # 3. Fit scaler STRICTLY on train features (no test lookahead!)
        scaler = TrafficScaler(method=scaler_method)
        scaler.fit(train_feat)
        scalers[client_id] = scaler

        train_scaled = scaler.transform(train_feat)
        test_scaled = scaler.transform(test_feat)

        train_targets = train_scaled[:, target_idx]
        test_targets = test_scaled[:, target_idx]

        # 4. Construct Datasets
        train_dataset = TrafficTimeSeriesDataset(
            features=train_scaled,
            targets=train_targets,
            window_size=window_size,
            horizon=horizon,
        )
        test_dataset = TrafficTimeSeriesDataset(
            features=test_scaled,
            targets=test_targets,
            window_size=window_size,
            horizon=horizon,
        )

        # 5. DataLoaders (train is shuffled, test is sequential)
        dataloaders[f"{client_id}_train"] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        dataloaders[f"{client_id}_test"] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        client_splits[client_id] = {
            "train_df": train_df,
            "test_df": test_df,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "scaler": scaler,
        }

    # ========================================================
    # Centralized Baseline Dataset (all data combined)
    # ========================================================
    central_train_df = pd.concat(central_train_dfs, ignore_index=True)
    central_test_df = pd.concat(central_test_dfs, ignore_index=True)

    central_train_feat = central_train_df[feature_cols].values.astype(np.float32)
    central_test_feat = central_test_df[feature_cols].values.astype(np.float32)

    central_scaler = TrafficScaler(method=scaler_method)
    central_scaler.fit(central_train_feat)
    scalers["centralized"] = central_scaler

    c_train_scaled = central_scaler.transform(central_train_feat)
    c_test_scaled = central_scaler.transform(central_test_feat)

    central_train_dataset = TrafficTimeSeriesDataset(
        features=c_train_scaled,
        targets=c_train_scaled[:, target_idx],
        window_size=window_size,
        horizon=horizon,
    )
    central_test_dataset = TrafficTimeSeriesDataset(
        features=c_test_scaled,
        targets=c_test_scaled[:, target_idx],
        window_size=window_size,
        horizon=horizon,
    )

    dataloaders["centralized_train"] = DataLoader(
        central_train_dataset, batch_size=batch_size, shuffle=True
    )
    dataloaders["centralized_test"] = DataLoader(
        central_test_dataset, batch_size=batch_size, shuffle=False
    )

    client_splits["centralized"] = {
        "train_df": central_train_df,
        "test_df": central_test_df,
        "train_samples": len(central_train_dataset),
        "test_samples": len(central_test_dataset),
        "scaler": central_scaler,
    }

    return client_splits, dataloaders, scalers


if __name__ == "__main__":
    from partition import partition_iid, partition_non_iid

    project_root = Path(__file__).resolve().parent.parent
    clean_csv = project_root / "outputs" / "cleaned_traffic_flow.csv"
    df = pd.read_csv(clean_csv)

    print("Partitioning 4 junctions (IID)...")
    iid_dict = partition_iid(df, num_junctions=4, seed=42)

    client_splits, dataloaders, scalers = prepare_client_dataloaders(
        iid_dict, window_size=12, horizon=1, train_ratio=0.8, batch_size=32
    )

    print("\n========= CLIENT SPLIT SUMMARY (IID) =========")
    for c_id, split_info in client_splits.items():
        print(
            f"{c_id:15s} | Train samples: {split_info['train_samples']:5d} | "
            f"Test samples: {split_info['test_samples']:5d}"
        )
    print("==============================================")
