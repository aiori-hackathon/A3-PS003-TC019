"""
test_pipeline.py - Unit Tests for Data Prep, Partitioning & Split Leakage Verification
Runs with standard unittest or pytest.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to python path
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "src"))

from data_prep import clean_vehicle_counts
from partition import partition_iid, partition_non_iid
from dataset import (
    chronological_train_test_split,
    TrafficScaler,
    TrafficTimeSeriesDataset,
    prepare_client_dataloaders,
)


class TestTrafficPipeline(unittest.TestCase):

    def setUp(self):
        """Generates synthetic sequential traffic flow data."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "time_step": np.arange(n),
            "frame_id": np.arange(100, 100 + n * 5, 5),
            "total_vehicles": np.random.poisson(lam=5, size=n),
            "car": np.random.poisson(lam=3, size=n),
            "motorcycle": np.random.poisson(lam=1, size=n),
            "bus": np.random.poisson(lam=0.5, size=n),
            "truck": np.random.poisson(lam=0.5, size=n),
        })
        safe_tot = df["total_vehicles"].replace(0, 1)
        df["car_ratio"] = df["car"] / safe_tot
        df["motorcycle_ratio"] = df["motorcycle"] / safe_tot
        df["bus_ratio"] = df["bus"] / safe_tot
        df["truck_ratio"] = df["truck"] / safe_tot
        self.sample_df = df

    def test_chronological_train_test_split_no_leakage(self):
        """Verifies that the train/test split has zero temporal leakage."""
        train_ratio = 0.8
        train_df, test_df = chronological_train_test_split(self.sample_df, train_ratio=train_ratio)

        # 1. Check sizes
        self.assertEqual(len(train_df), int(len(self.sample_df) * train_ratio))
        self.assertEqual(len(train_df) + len(test_df), len(self.sample_df))

        # 2. Strict chronological boundary (train max time < test min time)
        self.assertLess(
            train_df["time_step"].max(),
            test_df["time_step"].min(),
            "Data leakage detected! Train max time is not strictly less than Test min time",
        )

        # 3. Monotonic ordering in both splits
        self.assertTrue(train_df["time_step"].is_monotonic_increasing)
        self.assertTrue(test_df["time_step"].is_monotonic_increasing)

    def test_scaler_fit_on_train_only(self):
        """Tests that TrafficScaler fits strictly on train and transforms test properly."""
        train_data = np.array([[10.0], [20.0], [30.0]])
        test_data = np.array([[40.0], [50.0]])

        scaler = TrafficScaler(method="minmax")
        scaler.fit(train_data)

        self.assertEqual(scaler.min_[0], 10.0)
        self.assertEqual(scaler.max_[0], 30.0)

        # Test values must be scaled using train min/max
        scaled_test = scaler.transform(test_data)
        self.assertTrue(np.allclose(scaled_test, np.array([[1.5], [2.0]])))

        # Inverse transform recovers original scale
        inversed = scaler.inverse_transform_target(scaled_test[:, 0], target_idx=0)
        self.assertTrue(np.allclose(inversed, test_data[:, 0]))

    def test_sliding_window_dataset_shapes(self):
        """Tests that TrafficTimeSeriesDataset outputs correct (W, F) and (H,) shapes."""
        features = np.random.randn(50, 5).astype(np.float32)
        targets = features[:, 0]

        window_size = 12
        horizon = 1
        ds = TrafficTimeSeriesDataset(features, targets, window_size=window_size, horizon=horizon)

        expected_len = len(features) - window_size - horizon + 1
        self.assertEqual(len(ds), expected_len)

        X, y = ds[0]
        self.assertEqual(tuple(X.shape), (window_size, 5))
        self.assertEqual(tuple(y.shape), (horizon,))

        # Multi-step horizon test
        ds3 = TrafficTimeSeriesDataset(features, targets, window_size=10, horizon=3)
        X3, y3 = ds3[0]
        self.assertEqual(tuple(X3.shape), (10, 5))
        self.assertEqual(tuple(y3.shape), (3,))

    def test_iid_partitioning_balance(self):
        """Tests that IID partitioning distributes traffic counts evenly."""
        num_j = 4
        iid_dict = partition_iid(self.sample_df, num_junctions=num_j, block_size=5, seed=42)

        self.assertEqual(len(iid_dict), num_j)
        sample_sizes = [len(df) for df in iid_dict.values()]
        self.assertLessEqual(max(sample_sizes) - min(sample_sizes), 5)

        for j_id, df in iid_dict.items():
            self.assertTrue(df["time_step"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
