# A3-PS003-TC019
Federated Traffic-Flow Prediction with Communication-Cost Evaluation – AIORI Hackathon 3

## Team Introduction Video
**https://youtu.be/mxDhvBfXHCc** 

## Team Members
- **Student 1**: Anshika Sharon Sonwani
- **Student 2**: Freida B Rodrigues
- **Mentor**: Dr. Shamanth N

---

## Project Overview
Centralized traffic forecasting requires edge cameras and roadside sensors to continuously stream raw data to a central cloud, consuming high cellular bandwidth and concentrating private travel logs. Federated Learning (FL) addresses this by training local models on edge devices and sharing only mathematical weight updates with a central server.

In this project, we build a federated learning simulator using the **BMD-45 (Bengaluru Mobility Dataset)** partitioned across 3–5 virtual intersections. We evaluate how well a compact forecasting model using Federated Averaging (**FedAvg**) handles non-identically distributed (**non-IID**) traffic patterns, reporting prediction accuracy (MAE and RMSE) against network byte transmission costs per training round.

---

## Milestone 2: Data Preparation & Junction Partitioning (Completed)

### 1. Data Cleaning & Standardization
- Extracted vehicle detections from 25,551 annotated frames from the IISc AIM BMD-45 dataset.
- Cleaned and deduplicated records down to **20,560 strictly chronologically sorted frames** (spanning frame sequence 41 to 578,344).
- Extracted traffic flow volumes and multi-class breakdowns:
  - **Cars**: 48,096 (51.6%)
  - **Motorcycles / Two-Wheelers**: 21,074 (22.6%)
  - **Trucks / Heavy Vehicles**: 15,719 (16.9%)
  - **Buses / Public Transit**: 8,353 (9.0%)
  - **Overall Average Flow**: 4.54 vehicles/frame (max 24)

### 2. Virtual Junction Partitioning (3–5 Clients)
The dataset is partitioned into 4 simulated urban junctions in Bengaluru:
1. **Junction 1: Silk Board Arterial** – High commuter corridor, car/cab dominant (60.2% cars).
2. **Junction 2: Majestic Urban Core** – Dense city center with high two-wheeler density (36.1% motorcycles).
3. **Junction 3: Hebbal Ring Road** – Freight & logistics corridor with high commercial traffic (23.6% trucks, 13.2% buses).
4. **Junction 4: Indiranagar Central** – Mixed residential & commercial grid with balanced, moderate flow.

#### IID vs Non-IID Distribution Comparison
- **IID Distribution**: Every junction has identical vehicle volume and vehicle class proportions (~5,140 samples each).
- **Non-IID Distribution**: Skewed volume and vehicle type distributions reflecting real-world urban traffic diversity.

### 3. Chronological Train/Test Split (Zero Data Leakage)
- Strict chronological boundary: **80% Earlier Timeline for Training** and **20% Future Timeline for Testing**.
- Guarantees the model never looks ahead into future time steps.
- Feature scaling (`TrafficScaler`) is fitted strictly on the training partition.
- Centralized baseline dataset (16,448 train / 4,112 test frames) is generated for comparison against federated models.

---

## How to Run

### 1. Run Data Preparation & Partitioning Pipeline
```bash
python main_data_prep.py --num_junctions 4 --train_ratio 0.8 --window_size 12 --horizon 1
```

### 2. Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Generated Artifacts
- **Cleaned Data**: `outputs/cleaned_traffic_flow.csv`
- **Client Splits (IID)**: `outputs/splits/iid/` (`junction_1` to `junction_4`)
- **Client Splits (Non-IID)**: `outputs/splits/non_iid/` (`junction_1` to `junction_4`)
- **Centralized Baseline**: `outputs/splits/centralized/`
- **Metadata JSON**: `outputs/splits/partition_metadata.json`
- **Visualization Plots**: `outputs/splits/plots/`
  - `traffic_flow_overview.png`
  - `iid_vs_non_iid_distributions.png`
  - `chronological_splits_iid.png`
  - `chronological_splits_non_iid.png`
