import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure Matplotlib doesn't try to open a GUI window (critical for servers/CI)
plt.switch_backend('Agg')

def load_data(data_path):
    """Loads CSV and separates features (X) from target (y)."""
    print(f"[INFO] Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        # Assuming 'Churn' is the target column name from Phase 1
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        return X, y
    except FileNotFoundError:
        print(f"[ERROR] File not found: {data_path}")
        exit(1)

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generates and saves a Confusion Matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Best Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {output_path}")

def plot_feature_importance(model, feature_names, output_path):
    """Generates and saves a Feature Importance bar plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features to keep it readable
    top_n = 20
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices][:top_n], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Feature importance saved to {output_path}")

def main(train_path, test_path):
    # 1. Initialize DagsHub (Connects MLflow to Remote)
    # REPLACE THESE WITH YOUR EXACT DETAILS
    dagshub.init(repo_owner='vadhh', repo_name='SMSML_Afridho_Tavadhu', mlflow=True)
    
    # 2. Load Data
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # 3. Define Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    # 4. Start MLflow Run
    experiment_name = "Telco_Churn_RandomForest_Optimization"
    mlflow.set_experiment(experiment_name)

    print(f"[INFO] Starting MLflow run in experiment: {experiment_name}")
    
    with mlflow.start_run() as run:
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")
        
        # A. Enable Autologging (Captures params, metrics, model automatically)
        mlflow.sklearn.autolog()

        # B. Model Tuning (GridSearch)
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid, 
            cv=3, 
            scoring='accuracy', 
            verbose=2,
            n_jobs=-1
        )
        
        print("[INFO] Starting GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"[INFO] Best Parameters: {grid_search.best_params_}")

        # C. Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Test Accuracy: {acc:.4f}")

        # D. Custom Artifact Generation (CRITICAL FOR RUBRIC)
        # Create a temp folder for artifacts
        os.makedirs("artifacts", exist_ok=True)
        
        # 1. Confusion Matrix
        cm_path = "artifacts/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        
        # 2. Feature Importance
        fi_path = "artifacts/feature_importance.png"
        plot_feature_importance(best_model, X_train.columns, fi_path)

        # E. Log Custom Artifacts to MLflow
        print("[INFO] Uploading artifacts to DagsHub/MLflow...")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        
        # Log the classification report as text
        report = classification_report(y_test, y_pred)
        with open("artifacts/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("artifacts/classification_report.txt")

        print("[SUCCESS] Run Complete. Check DagsHub for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Tune Random Forest for Telco Churn")
    # Default paths assume script is run from project root
    parser.add_argument("--train", type=str, default="data_processed/train.csv")
    parser.add_argument("--test", type=str, default="data_processed/test.csv")
    
    args = parser.parse_args()
    
    main(args.train, args.test)
