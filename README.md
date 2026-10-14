
# A3-PS003-TC019
Federated Traffic-Flow Prediction with Communication-Cost Evaluation – AIORI Hackathon 3
## Team Introduction Video
**https://youtu.be/mxDhvBfXHCc** 
## Team Members
- Student 1: Anshika Sharon Sonwani
- Student 2: Freida B Rodrigues
- Mentor: Shamanth N

## Project Overview
This project aims to develop a federated learning simulator for traffic-flow prediction. It allows multiple virtual traffic junctions to train local models while keeping their raw data locally. The project evaluates prediction accuracy and communication costs by comparing federated learning with a centralized model.

## Proposed Solution
- Split the IISc AIM BDE dataset across 3–5 virtual traffic junctions.
- Train a local prediction model at each junction.
- Use Federated Averaging (FedAvg) to combine local model updates.
- Compare the federated approach with a centralized model trained on all the data.
- Measure prediction accuracy and communication cost across training rounds.
