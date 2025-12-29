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

## 📊 Results & Performance
The model was evaluated using strict Cross-Validation (CV) to ensure stability and prevent overfitting.

<table>
  <tr>
    <td width="50%">
      <h3>Key Metrics</h3>
      <ul>
        <li><b>ROC AUC:</b> 0.856</li>
        <li><b>Recall:</b> 80.5%</li>
        <li><b>F1-Score:</b> 0.788</li>
        <li><b>CV Score:</b> 0.798</li>
      </ul>
    </td>
    <td width="50%">
      <h3>Confusion Matrix</h3>
      <img width="600" height="400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d8c84fa1-6991-47bf-a420-88a9bef11b57" />
    </td>
  </tr>
</table>

**Key Insight:**
With a Recall of ~80%, this model enables the retention team to proactively target the majority of at-risk users, potentially saving significant revenue vs. a random outreach strategy.



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
