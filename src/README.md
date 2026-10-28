# Source Code Modules (`src/`)

This directory contains the core implementation modules for **Federated Traffic-Flow Prediction with Communication-Cost Evaluation** (AIORI Hackathon 3, PS-003).

---

## Architecture Overview

```
src/
├── data_prep.py     # Cleans raw YOLO annotations, deduplicates, and sorts chronologically
├── partition.py     # Virtual junction partitioning across 3–5 clients (IID & Non-IID)
├── dataset.py       # Chronological train/test splitting (zero leakage) & PyTorch DataLoaders
└── visualize.py     # Publication-quality charts for traffic flow, distributions, and splits
```

---

## Module Details

### 1. `data_prep.py`
- Cleans and deduplicates the 25,551 YOLO detection records down to 20,560 unique frames.
- Standardizes frame sequence indices (`time_step` and `frame_id`).
- Computes vehicle class breakdowns (`car`, `motorcycle`, `bus`, `truck`) and rolling trend statistics.
- Outputs `outputs/cleaned_traffic_flow.csv`.

### 2. `partition.py`
- Simulates **3 to 5 virtual junctions** across Bengaluru mobility zones:
  - **Junction 1 (Silk Board Arterial)**: High commuter volume, car/cab dominant.
  - **Junction 2 (Majestic Urban Core)**: Dense urban core, high 2-wheeler / motorcycle share.
  - **Junction 3 (Hebbal Ring Road)**: Freight and logistics corridor, high truck & bus ratio.
  - **Junction 4 (Indiranagar Central)**: Mixed residential/commercial with balanced flow.
  - **Junction 5 (Whitefield Tech Park)**: Bottleneck corridor with high traffic variance.
- Implements:
  - **IID Partitioning**: Interleaved temporal blocks ensuring identical volume and vehicle class distributions across junctions.
  - **Non-IID Partitioning**: Realistic urban traffic profiles reflecting heterogeneous flow and vehicle class skew.

### 3. `dataset.py`
- **Zero-Leakage Chronological Split**: Strict cutoff (80% train / 20% test) where earlier timeline is used for training and future timeline is reserved for evaluation ("the model never sees the future").
- **TrafficScaler**: MinMax/Standard scaler fitted *strictly on training data* to prevent lookahead leakage.
- **TrafficTimeSeriesDataset**: PyTorch Dataset providing sliding lookback windows ($W=12$ steps) predicting future horizons ($H=1$ step ahead).
- Generates DataLoaders for each federated client as well as the **centralized baseline**.

### 4. `visualize.py`
Generates high-resolution visualization plots:
- Chronological traffic flow timeline and rolling trends.
- Side-by-side bar charts comparing IID vs Non-IID distributions across junctions.
- Temporal train/test split boundary visualizations.
