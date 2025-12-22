import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_folder):
    print(f"[INFO] Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found at {input_path}")
        return

    # 1. Drop customerID (irrelevant for training)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("[INFO] Dropped 'customerID' column.")

    # 2. Handle TotalCharges (Force numeric conversion)
    # This column often contains " " (spaces) which pandas reads as object.
    print("[INFO] Processing 'TotalCharges'...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN values (caused by spaces) with 0 or Mean. 
    # Using 0 is safer for Churn analysis if tenure is 0.
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Encoding Categorical Variables
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Binary encoding for target 'Churn' and others
    le = LabelEncoder()
    
    # Specific handling for Target variable to ensure 1/0
    if 'Churn' in df.columns:
        df['Churn'] = le.fit_transform(df['Churn'])
        print("[INFO] Encoded target 'Churn'.")
    
    # Simple Label Encoding for binary/ordinal, OneHot for others.
    # For this rubric, we will keep it simple with pd.get_dummies to ensure compatibility.
    print("[INFO] Encoding categorical features...")
    df = pd.get_dummies(df, drop_first=True)

    # 4. Split Data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    print(f"[INFO] Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Recombine for saving (easier for downstream MLflow steps)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # 5. Save Output
    os.makedirs(output_folder, exist_ok=True)
    
    train_path = os.path.join(output_folder, 'train.csv')
    test_path = os.path.join(output_folder, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"[SUCCESS] Data saved to:\n  - {train_path}\n  - {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Preprocessing for Telco Churn")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data csv")
    parser.add_argument("--output", type=str, default="data_processed", help="Folder to save processed data")
    
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)
