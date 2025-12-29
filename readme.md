# 📡 End-to-End MLOps Pipeline: Telco Customer Churn

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## 📋 Overview
This project is not just a machine learning model; it is a **production-ready MLOps pipeline** designed to predict customer churn in the Telecommunications industry.

The goal was to move beyond ad-hoc notebooks and build a system that ensures **reproducibility**, **automation**, and **observability**. It leverages **MLflow** for experiment tracking and model registry, and **GitHub Actions** for Continuous Integration (CI).

## 🏗️ Architecture & Workflow
The pipeline is designed to automate the lifecycle of the ML model:

1.  **Data Ingestion:** Automated handling of the Telco Customer Churn dataset.
2.  **Preprocessing:** Feature engineering and cleaning pipelines using Scikit-Learn `ColumnTransformer`.
3.  **Experimentation:** Training multiple models (Random Forest, XGBoost, SVM) with **MLflow** logging parameters, metrics, and artifacts.
4.  **CI/CD:** A GitHub Actions workflow triggers on push to:
    * Lint the code (Flake8).
    * Test data integrity (Pytest).
    * Dry-run the training pipeline.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Orchestration:** MLflow (Tracking & Registry)
* **CI/CD:** GitHub Actions
* **Modeling:** Scikit-Learn, Pandas, NumPy
* **Containerization:** Docker (Optional/In-progress)

## 🚀 Key Features
* **Experiment Tracking:** All runs are logged. No more "which parameters did I use for that result?"
* **Reproducibility:** Rigid dependency management ensures the environment is identical across machines.
* **Automated Testing:** Unit tests ensure that data leaks and schema mismatches are caught before training begins.
* **Modular Codebase:** Code is structured as a package (`src/`), not a monolithic notebook.

## 💻 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/telco-churn-mlops.git](https://github.com/yourusername/telco-churn-mlops.git)
cd telco-churn-mlops

```

### 2. Set up the Environment

```bash
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Run the Pipeline

To execute the training pipeline and log to MLflow:

```bash
python main.py
# OR if using DVC/Make
make train

```

### 4. View MLflow Dashboard

```bash
mlflow ui
# Navigate to http://localhost:5000 to compare experiments.

```

## 📊 Results & Metrics
The pipeline was evaluated using K-Fold Cross-Validation to ensure stability.

* **Best Model Performance:** ~79.8% (Mean CV Score)
* **Stability:** ±0.3% Standard Deviation (Indicates highly consistent performance across data folds)
* **Efficiency:** ~0.11s training time (Low computational cost, suitable for frequent retraining)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Score** | **0.7976** | The model correctly predicts the outcome ~80% of the time. |
| **Std Dev** | **0.0030** | The model's performance varies by less than 1%, proving reliability. |
| **Fit Time** | **0.107s** | fast training enables rapid iteration and low-latency CI/CD pipelines. |

## 📂 Project Structure

```text
├── .github/workflows/   # CI/CD (GitHub Actions)
├── src/                 # Source code for data, training, and evaluation
├── tests/               # Unit tests
├── notebooks/           # EDA and prototyping notebooks
├── main.py              # Entry point for the pipeline
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

```

---

*Built by Vadhh - [[Link to LinkedIn]](https://www.linkedin.com/in/afr1dho/)*
