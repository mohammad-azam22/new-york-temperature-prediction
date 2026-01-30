# 📊 New York Temperature Prediction & Monitoring

This repository contains an end-to-end **ML system** that predicts the **daily temperature of New York City** using historical weather data and monitors model performance and drift in production-like conditions. The project demonstrates experiment tracking, model monitoring, alerting, and visualization using modern MLOps tools.

---

## 🧠 Project Overview

The goal of this project is to build a predictive model for **New York daily temperature**, track experiments, and monitor model health over time using **Evidently** and **Grafana**.

Key components include:

- Data collection using the **Open-Meteo API**
- Model training with multiple regression algorithms and tracking with **MLflow**
- Model inference on daily data
- Monitoring of prediction drift, data drift, and performance metrics
- Dashboarding and alerting with **Grafana**
- Metrics storage in **PostgreSQL**

---

## 🚀 Features

✔ Real-world weather data from the free Open-Meteo API  
✔ Multiple models compared with **MLflow**  
✔ Monitoring using **Evidently**  
✔ Metrics stored in **PostgreSQL**  
✔ Interactive dashboards in **Grafana**  
✔ Alert rules for drift detection

---

## 🗂️ Repository Structure

├── `config/`  
├── `data/` # Reference and raw data  
├── `models/` # Trained models  
├── `notebook.ipynb` # EDA & experiment walkthrough  
├── `metrics_calculation.py` # Script for daily inference + monitoring  
├── `grafana_dashboard.json`  
├── `docker-compose.yaml` # Compose setup  
└── `README.md`


---

## 🧪 Model Training & MLflow

The training pipeline:

1. Downloaded **New York historical weather data (2022)** using the Open-Meteo API
2. Selected target variable: `temperature_2m`
3. Trained multiple models:  
   | Model | Train MAE | Train RMSE | Train R² Score | Validation MAE | Validation RMSE | Validation R² Score |
   |-------|-----------|------------|----------------|----------|-----------|----------------|
   | Linear Regression | 0.3664 | 0.5766 | 0.9956 | 0.3722 | 0.5435 | 0.9961 |
   | Lasso | 1.7666 | 2.1240 | 0.9416 | 1.7552 | 2.1333 | 0.9414 |
   | Ridge | 0.3626 | 0.5699 | 0.9957 | 0.3682 | 0.5373 | 0.9962 |
   | Elastic Net | 2.8368 | 3.3264 | 0.8568 | 2.8047 | 3.2952 | 0.8602 |
   | Decision Tree | 0.0 | 0.0 | 1.0 | 0.2613 | 0.3770 | 0.9981 |
   | **Random Forest** | 0.0422 | 0.0696 | 0.9999 | 0.1121 | 0.1935 | 0.9995 |
   | Support Vector Regressor (SVR) | 0.4025 | 0.9685 | 0.9878 | 0.4965 | 1.2728 | 0.9791 |
   | Multi-Layer Perceptron | 1.4410 | 2.0596 | 0.9451 | 0.6264 | 0.9363 | 0.9887 |
4. Logged experiments, parameters, and metrics to **MLflow**
5. Selected **Random Forest** as the best model based on validation metrics
6. Retrained the best model on the full dataset  
7. Saved the model for inference

📊 Example validation results (2022 → 2023 split):

| Metric | Train | Test |
|--------|-------|------|
| MSE | 0.042 | 0.114 |
| RMSE | 0.068 | 0.196 |
| R² | 0.99994 | 0.99951 |

---

## 🧠 Monitoring & Drift Detection

After training, the model was evaluated on daily 2024 data:

1. Predictions made using the saved model
2. Metrics computed:
   - MAE
   - RMSE
   - R² score
3. Drift metrics computed using **Evidently**:
   - Prediction drift
   - Number of drifted columns
   - Share of missing values
4. Metrics were inserted into **PostgreSQL**

This allows tracking model behavior over time and understanding whether the data distribution has changed significantly from the reference (2023) dataset.

---

## 📊 Dashboards & Alerting

![Grafana Dashboard](./Grafana%20Dashboard.pn)

Using **Grafana**:

- Visualized daily metrics from PostgreSQL
- Built interactive dashboards for MAE, RMSE, R², prediction drift, etc.
- Set up alert rules, for example:
  - Trigger alert when **prediction drift > 1.0**
  - Notification can be configured via email or webhook

This demonstrates an MLOps-style **monitoring and alerting workflow** for deployed models.

---

## 🐳 Running the Project (Docker)

Start the services:

```bash
docker-compose up --build
```

This will spin up:
- PostgreSQL
- Grafana
-  Adminer (for database exploration)

📌 Make sure to adjust environment variables in the docker-compose.yaml if needed.

## 🛠️ How to Use

1. Once the docker container is running, execute the inference & data loading script (`metrics_calculation.py`)

2. Login to Grafana:

3. Configure PostgreSQL as a datasource

4. Import grafana_dashboard.json to view dashboards

5. Set up alert channels in Grafana for drift monitoring

📌 Retraining of the model is manual — Grafana alerts indicate when retraining should be considered.

📌 This project illustrates best practices in model monitoring and observability, not automated model retraining workflows.

## 🧾 Dependencies

| Component | Tech |
|-----------|------|
| Code and model training | Python & Scikit-Learn |
| Experiment tracking | MLflow |
| Metcis calculaiton | Evidently |
| Data storage | PostgreSQL |
| Dashboard and alerting | Grafana |
| Containerization | Docker & Docker-Compose |

## 📄 License

This repository is licensed under the MIT License. See the license tab for more details.
