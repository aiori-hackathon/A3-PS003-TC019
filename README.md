# A3-PS003-TC019
# Federated Traffic-Flow Prediction with Communication-Cost Evaluation

**AIORI Hackathon 3 | Smart Cities**

## Project Videos

| Video | Link |
|---|---|
| **Team Introduction Video** | [Watch on YouTube](https://youtu.be/mxDhvBfXHCc) |
| **Final Model Demonstration** | [Watch on YouTube](https://youtu.be/Rxy1DE1UqmE) |

## Team Details

| Role | Name |
|---|---|
| Student 1 | Anshika Sharon Sonwani |
| Student 2 | Freida B Rodrigues |
| Mentor | Dr. Shamanth N |

---

## 1. Introduction

Urban traffic management systems increasingly rely on roadside cameras and sensors to monitor vehicle movement and predict traffic conditions. Traditional centralized traffic forecasting approaches require continuous transmission of raw traffic data to centralized cloud servers, resulting in increased communication overhead, bandwidth consumption, and potential privacy concerns.

**Federated Learning (FL)** offers an alternative by enabling multiple edge clients to train machine learning models locally while sharing model updates with a central aggregation server instead of transmitting their raw data.

This project presents a federated traffic-flow prediction framework using the **BMD-45 (Bengaluru Mobility Dataset)**. The system simulates multiple urban traffic intersections as independent federated clients and applies **Federated Averaging (FedAvg)** to collaboratively train a traffic forecasting model.

The framework investigates the impact of heterogeneous traffic distributions through IID and Non-IID data partitioning and evaluates prediction performance alongside communication costs.

The objective is to explore an efficient, decentralized approach to traffic forecasting that can support intelligent transportation systems and smart-city infrastructure.

---

## 2. Problem Statement

Conventional centralized traffic prediction systems depend on collecting and transmitting traffic observations from multiple intersections to a central server.

This approach introduces several challenges:

- High communication bandwidth requirements.
- Increased network transmission costs.
- Dependence on continuous connectivity.
- Privacy concerns associated with sharing raw mobility data.
- Difficulty handling differences in traffic patterns across geographical locations.

The project addresses these challenges by simulating a Federated Learning environment in which individual traffic intersections maintain their own local datasets and participate in collaborative model training.

The system evaluates whether federated learning can provide effective traffic-flow predictions while reducing the need for raw data transmission.

---

## 3. Project Objectives

The primary objectives are:

1. Develop a structured traffic-flow dataset from vehicle detection annotations.
2. Simulate multiple virtual traffic intersections using the BMD-45 dataset.
3. Implement IID and Non-IID data partitioning to represent different traffic characteristics.
4. Apply chronological train-test splitting to preserve temporal consistency and prevent data leakage.
5. Develop a traffic forecasting model using Federated Averaging (FedAvg).
6. Establish a centralized learning baseline for performance comparison.
7. Evaluate forecasting accuracy using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
8. Measure communication overhead in terms of bytes transmitted during federated training.

---

## 4. Dataset Description

The project uses the **BMD-45 (Bengaluru Mobility Dataset)** from the IISc AIM dataset collection.

The dataset consists of traffic images captured in urban road environments. Vehicle detection annotations are used to extract vehicle counts and traffic-flow characteristics.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Annotated frames extracted | 25,551 |
| Cleaned and deduplicated frames | 20,560 |
| First frame sequence | 41 |
| Last frame sequence | 578,344 |
| Average vehicles per frame | 4.54 |
| Maximum vehicles in a frame | 24 |

### Vehicle Category Distribution

| Vehicle Category | Count | Percentage |
|---|---:|---:|
| Cars | 48,096 | 51.6% |
| Motorcycles / Two-Wheelers | 21,074 | 22.6% |
| Trucks / Heavy Vehicles | 15,719 | 16.9% |
| Buses / Public Transit | 8,353 | 9.0% |
| **Total** | **93,242** | **100.1%*** |

*Percentages are rounded.*

The extracted vehicle counts are transformed into structured traffic-flow records for subsequent preprocessing, temporal feature generation, and model training.

---

## 5. System Architecture

The proposed system follows a decentralized Federated Learning architecture consisting of a central aggregation server and multiple virtual traffic intersection clients.

### Architecture Components

**1. Central Server**

- Coordinates the federated training process.
- Distributes the global model to participating clients.
- Receives locally trained model updates.
- Aggregates client updates using Federated Averaging.
- Maintains the updated global model.

**2. Virtual Junction Clients**

Each simulated junction represents an independent traffic monitoring location.

Every client:

- Maintains its own local traffic dataset.
- Receives the global model.
- Performs local model training.
- Sends model updates to the central server.

**3. Federated Aggregation**

The central server combines local model updates using the FedAvg algorithm to produce an updated global model.

**4. Evaluation Module**

The forecasting framework evaluates model performance using prediction error metrics and communication-cost measurements.

### Federated Learning Workflow

```text
       BMD-45 Dataset
              |
              v
    Data Preprocessing
              |
              v
     Junction Partitioning
              |
       +------+------+------+
       |      |      |      |
       v      v      v      v
      J1     J2     J3     J4
       |      |      |      |
       v      v      v      v
    Local Model Training
       |      |      |      |
       +------+------+------+
              |
              v
       Central Server
              |
              v
       FedAvg Aggregation
              |
              v
       Updated Global Model
              |
              v
       Next Training Round
              |
              v
      Model Evaluation
       /            \
      v              v
 Prediction      Communication
  Accuracy           Cost
```

---

## 6. Data Preprocessing and Junction Partitioning

### 6.1 Data Cleaning and Standardization

The initial vehicle detection records undergo preprocessing to generate a consistent traffic-flow dataset.

The preprocessing pipeline includes:

- Extraction of vehicle counts from annotated frames.
- Generation of multi-class vehicle distributions.
- Removal of duplicate records.
- Chronological sorting of observations.
- Standardization of traffic-flow features.
- Preparation of temporal input sequences for forecasting.

The resulting dataset contains 20,560 cleaned and chronologically sorted frames, spanning frame sequence 41 to 578,344.

### 6.2 Virtual Junction Partitioning

The dataset is distributed across four simulated Bengaluru traffic intersections.

| Client | Junction | Traffic Characteristics |
|---|---|---|
| Junction 1 | Silk Board Arterial | High commuter traffic with car/cab dominance (60.2% cars). |
| Junction 2 | Majestic Urban Core | Dense urban traffic with a high two-wheeler concentration (36.1% motorcycles). |
| Junction 3 | Hebbal Ring Road | Freight and logistics-oriented traffic (23.6% trucks and 13.2% buses). |
| Junction 4 | Indiranagar Central | Mixed residential and commercial traffic with balanced vehicle distribution. |

These virtual junctions represent different traffic characteristics within a simulated federated environment.

### 6.3 IID and Non-IID Data Distribution

Two different data distribution strategies are used to investigate the influence of client data heterogeneity.

**IID Distribution**

- Traffic observations are distributed across clients with approximately identical statistical characteristics.
- Vehicle-class proportions and traffic-volume distributions remain similar across junctions.
- Each client receives approximately 5,140 samples.

**Non-IID Distribution**

- Traffic observations are distributed with different statistical characteristics across clients.
- Each junction represents a different vehicle composition and traffic-volume pattern.
- The distribution reflects differences between commuter corridors, dense city centers, freight routes, and mixed urban environments.

Comparing these distributions helps evaluate the impact of heterogeneous client data on federated model training.

### 6.4 Chronological Train-Test Split

To maintain temporal consistency and avoid data leakage, the dataset is divided chronologically.

| Dataset | Percentage | Number of Frames |
|---|---:|---:|
| Training | 80% | 16,448 |
| Testing | 20% | 4,112 |
| Total | 100% | 20,560 |

The earlier 80% of observations are used for training, while the later 20% are reserved for testing.

Additional preprocessing considerations:

- Training observations always precede testing observations.
- Future observations are excluded from model training.
- Feature scaling using `TrafficScaler` is fitted exclusively on the training partition.
- The centralized baseline uses the same chronological evaluation boundary.

### 6.5 Temporal Feature Generation

Traffic forecasting uses historical observations to predict future traffic conditions.

The pipeline supports configurable temporal parameters:

- **Window Size:** Number of historical time steps provided as model input.
- **Prediction Horizon:** Number of future time steps to forecast.

For the configured experiment, a window size of 12 and a prediction horizon of 1 are used.

---

## 7. Federated Learning Methodology

### 7.1 Federated Averaging (FedAvg)

Federated Averaging is used to combine model updates from multiple participating junction clients.

Each client trains a local model using its own traffic observations. The resulting model parameters are communicated to the central server, where they are aggregated to update the global model.

The FedAvg aggregation rule is:

\[
w_{t+1} =
\sum_{k=1}^{K}
\frac{n_k}{N}w_{t+1}^{(k)}
\]

Where:

- \(w_{t+1}\) represents the updated global model parameters.
- \(K\) represents the number of participating clients.
- \(n_k\) represents the number of training samples at client \(k\).
- \(N\) represents the total number of training samples.
- \(w_{t+1}^{(k)}\) represents the locally trained model parameters of client \(k\).

The aggregated global model is distributed back to participating clients for subsequent training rounds.

### 7.2 Federated Training Process

The federated training workflow consists of the following operations:

1. Initialize the global forecasting model.
2. Distribute the global model to participating junction clients.
3. Train the model locally using each client's traffic observations.
4. Collect locally trained model updates.
5. Aggregate the updates using FedAvg.
6. Update the global model.
7. Repeat the process across communication rounds.
8. Evaluate the resulting model using the test dataset.

### 7.3 Centralized Learning Baseline

A centralized model is used as a reference for evaluating federated learning performance.

Both approaches use the prepared traffic-flow dataset and the same chronological train-test boundary.

This comparison provides insight into the differences between centralized training and decentralized collaborative learning.

---

## 8. Performance Evaluation

The framework evaluates prediction accuracy and communication efficiency.

### 8.1 Prediction Accuracy Metrics

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted traffic-flow values.

\[
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

**Root Mean Squared Error (RMSE)**

RMSE measures the square root of the average squared prediction error, giving greater importance to larger errors.

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

Where:

- \(y_i\) represents the actual traffic-flow value.
- \(\hat{y}_i\) represents the predicted traffic-flow value.
- \(n\) represents the number of observations.

Lower MAE and RMSE values indicate better prediction accuracy.

### 8.2 Communication-Cost Evaluation

An important aspect of the project is evaluating the communication overhead associated with Federated Learning.

Instead of transmitting raw traffic observations, participating clients communicate model updates to the central server.

Communication cost is evaluated through:

- Model update size.
- Bytes transmitted during each communication round.
- Total communication volume across training rounds.
- Communication overhead across participating clients.

For a communication round, the total transmission cost can be represented as:

\[
C_t =
\sum_{k=1}^{K}
(B_{k,\text{upload}}+
B_{k,\text{download}})
\]

Where:

- \(C_t\) represents total communication bytes during round \(t\).
- \(B_{k,\text{upload}}\) represents bytes uploaded by client \(k\).
- \(B_{k,\text{download}}\) represents bytes downloaded by client \(k\).

The cumulative communication cost is:

\[
C_{\text{total}} =
\sum_{t=1}^{R} C_t
\]

Where \(R\) represents the total number of communication rounds.

### 8.3 Experimental Comparisons

The evaluation considers the following comparisons:

| Experiment | Evaluation Focus |
|---|---|
| Centralized vs Federated | Prediction performance |
| IID vs Non-IID | Impact of client data heterogeneity |
| Multiple Junction Clients | Performance across different traffic profiles |
| Communication Rounds | Network transmission overhead |
| Model Updates | Communication efficiency |

---

## 9. Technologies and Tools

| Technology | Application |
|---|---|
| Python | Core development and implementation |
| YOLO | Vehicle detection and annotation |
| OpenCV | Image processing |
| Pandas | Dataset cleaning and manipulation |
| NumPy | Numerical computation |
| PyTorch | Deep learning and model training |
| Scikit-learn | Data preprocessing and evaluation utilities |
| Matplotlib | Data visualization |
| Federated Averaging | Global model aggregation |

---

## 10. Project Directory Structure

```text
Traffix/
│
├── BMD45/
│   ├── images_000/
│   ├── images_001/
│   ├── images_002/
│   ├── images_003/
│   └── images_004/
│
├── outputs/
│   ├── labels/
│   ├── annotated_images/
│   ├── vehicle_counts.csv
│   ├── annotation_summary.csv
│   │
│   ├── cleaned_traffic_flow.csv
│   │
│   └── splits/
│       ├── iid/
│       │   ├── junction_1/
│       │   ├── junction_2/
│       │   ├── junction_3/
│       │   └── junction_4/
│       │
│       ├── non_iid/
│       │   ├── junction_1/
│       │   ├── junction_2/
│       │   ├── junction_3/
│       │   └── junction_4/
│       │
│       ├── centralized/
│       ├── plots/
│       └── partition_metadata.json
│
├── tests/
│   └── test_pipeline.py
│
├── annotations.py
├── main_data_prep.py
├── requirements.txt
├── yolo26n.pt
└── README.md
```

*The directory structure illustrates the organization of the dataset, preprocessing outputs, and generated artifacts.*

---

## 11. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- pip package manager.
- Visual Studio Code or another Python development environment.
- Required project dependencies.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Traffix
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Vehicle Annotation

```bash
python annotations.py
```

This executes the vehicle detection and annotation pipeline, generating bounding-box labels, annotated images, and vehicle-count records.

### Step 5: Run Data Preparation and Junction Partitioning

```bash
python main_data_prep.py --num_junctions 4 --train_ratio 0.8 --window_size 12 --horizon 1
```

**Parameter Description**

| Parameter | Description |
|---|---|
| `num_junctions` | Number of simulated traffic clients |
| `train_ratio` | Proportion of observations allocated for training |
| `window_size` | Number of historical time steps used as input |
| `horizon` | Number of future time steps to predict |

### Step 6: Run Unit Tests

```bash
python -m unittest tests/test_pipeline.py
```

---

## 12. Generated Outputs

The data preparation pipeline produces structured datasets and visualizations.

| Output | Description |
|---|---|
| `cleaned_traffic_flow.csv` | Cleaned and standardized traffic-flow records |
| `splits/iid/` | IID client datasets |
| `splits/non_iid/` | Non-IID client datasets |
| `splits/centralized/` | Centralized baseline datasets |
| `partition_metadata.json` | Junction partitioning information |
| `splits/plots/` | Data distribution and temporal visualizations |

### Data Visualization

The pipeline generates visualizations to analyze traffic patterns and data distribution:

- **traffic_flow_overview.png** – Overview of traffic-flow characteristics.
- **iid_vs_non_iid_distributions.png** – Comparison of traffic distributions across IID and Non-IID clients.
- **chronological_splits_iid.png** – Chronological training and testing distribution for IID data.
- **chronological_splits_non_iid.png** – Chronological training and testing distribution for Non-IID data.

---

## 13. Conclusion

This project explores the application of Federated Learning to traffic-flow prediction in smart-city environments.

Using the BMD-45 Bengaluru Mobility Dataset, the framework simulates multiple virtual traffic intersections with distinct traffic characteristics and enables collaborative model training through Federated Averaging.

The IID and Non-IID partitioning strategies provide a way to investigate the influence of heterogeneous traffic patterns on federated model learning. The centralized baseline establishes a reference for comparing prediction performance, while communication-cost evaluation provides insight into the network overhead associated with federated training.

By combining traffic-flow forecasting, decentralized learning, and communication-efficiency analysis, the project demonstrates an approach toward collaborative and privacy-conscious intelligent transportation systems.

---

## 14. Dataset Source and Acknowledgements

The project uses the **BMD-45 (Bengaluru Mobility Dataset)**, sourced from the IISc AIM dataset collection. We acknowledge the researchers and contributors responsible for making the dataset available for research and development purposes.

We also sincerely thank our mentor, **Dr. Shamanth N**, for his valuable guidance and support throughout the project.
---

**Project ID:** A3-PS003-TC019  
**Hackathon:** AIORI Hackathon 3  
**Domain:** Smart Cities  
**Project Title:** Federated Traffic-Flow Prediction with Communication-Cost Evaluation
