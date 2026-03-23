import io
import sys
import time
import os

import boto3
import botocore
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error

from src.model.model_factory import ModelFactory
from src.utils.config import load_config

# ==========================================
# DYNAMIC CONFIGURATION
# ==========================================
try:
    config = load_config()
except Exception as e:
    print(f" [CRITICAL] Error during Auto-Discovery: {e}")
    sys.exit(1)

AWS_REGION = config.get("aws_region")
TARGET_BUCKET = config.get("s3_bucket")

# =====================================================================
# BASELINE GRID CONFIGURATION (Modify before running)
# =====================================================================
TARGET_DATASET = "airlines"  # Change to "taxi" for regression
TREES_GRID = [50, 100, 200, 300]
# =====================================================================

s3_client = boto3.client('s3', region_name=AWS_REGION)

# Appends baseline metrics to a persistent S3 CSV file for comparison.
def save_baseline_metrics(dataset, n_trees, train_time, inf_time, metrics_dict, config):
    
    s3_key = f"results/{dataset}/{dataset}_baseline_results.csv"
    
    new_row_df = pd.DataFrame([{
        'Dataset': dataset, 
        'Trees': n_trees, 
        'Train_Time': round(train_time, 2), 
        'Infer_Time': round(inf_time, 2), 
        'Metrics': str(metrics_dict)
    }])

    try:
        obj = s3_client.get_object(Bucket=TARGET_BUCKET, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_final = new_row_df
        else:
            print(f" [METRICS ERROR] Unexpected S3 error: {e}")
            return
            
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=TARGET_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICS] Baseline results securely appended to: s3://{TARGET_BUCKET}/{s3_key}")

# Downloads an entire CSV dataset from S3 directly into a Pandas DataFrame in RAM.
# WARNING: Highly susceptible to Out-Of-Memory (OOM) errors on large datasets.
def load_dataset_from_s3(bucket, key):
    
    print(f" [DOWNLOAD] Fetching s3://{bucket}/{key} into RAM...")
    start_dl = time.time()
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    
    print(f" [DOWNLOAD] Completed in {time.time() - start_dl:.2f}s. Loaded {len(df)} rows.")
    return df


def main():
    print("\n" + "=" * 60)
    print(" MONOLITHIC BASELINE BENCHMARK - RANDOM FOREST")
    print("=" * 60)
    print(f" Target Dataset : {TARGET_DATASET.upper()}")
    print(f" Trees Grid     : {TREES_GRID}")
    print("=" * 60 + "\n")

    # 1. Initialize ML Handler
    try:
        ml_handler = ModelFactory.get_model(TARGET_DATASET)
        task_type = getattr(ml_handler, 'task_type', 'classification')
    except ValueError as e:
        print(f" [CRITICAL] {e}")
        sys.exit(1)

    train_s3_key = config['datasets_metadata'][TARGET_DATASET]['train_path']
    test_s3_key = config['datasets_metadata'][TARGET_DATASET]['test_path']

    # 2. Massive RAM Allocation (Where the local approach usually fails on Big Data)
    try:
        print(" [RAM ALLOCATION] WARNING: Loading entire Train and Test sets into memory...")
        df_train = load_dataset_from_s3(TARGET_BUCKET, train_s3_key)
        df_test = load_dataset_from_s3(TARGET_BUCKET, test_s3_key)
    except Exception as e:
        print(f" [CRITICAL] Memory or S3 Error during dataset loading: {e}")
        sys.exit(1)

    target_col = ml_handler.target_column
    y_true = df_test[target_col].values

    # 3. Iterate over the Hyperparameter Grid
    for trees in TREES_GRID:
        print(f"\n --- STARTING BENCHMARK FOR {trees} TREES ---")
        
        # Standard monolithic configuration
        params = {
            "trees": trees,
            "max_depth": None, 
            "max_features": "sqrt", 
            "criterion": "gini" if task_type == 'classification' else "squared_error",
            "seed": 42
        }

        # Training phase
        print(f" [BASELINE TRAIN] Training model...")
        train_start = time.time()
        rf_model = ml_handler.process_and_train(df_train, params)
        train_time = time.time() - train_start
        print(f" [BASELINE TRAIN] Completed in {train_time:.2f}s")

        # Inference phase
        print(f" [BASELINE INFER] Executing prediction on {len(df_test)} rows...")
        infer_start = time.time()
        predictions = ml_handler.process_and_predict(rf_model, df_test)
        infer_time = time.time() - infer_start
        print(f" [BASELINE INFER] Completed in {infer_time:.2f}s")

        # Evaluation phase
        if task_type == 'classification':
            # Reusing the same matrix extraction logic used by the distributed Master
            votes_0 = predictions[:, 0]
            votes_1 = predictions[:, 1]
            y_prob = votes_1 / (votes_0 + votes_1)
            final_prediction = np.argmax(predictions, axis=1)

            auc = roc_auc_score(y_true, y_prob)
            acc = accuracy_score(y_true, final_prediction)

            print(f" [EVALUATION] ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f}")
            metrics_dict = {'ROC-AUC': round(auc, 4), 'Accuracy': round(acc, 4)}

        else:
            # Regression simply uses the returned mean array
            mse = mean_squared_error(y_true, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)

            print(f" [EVALUATION] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2 Score: {r2:.4f}")
            metrics_dict = {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2 Score': round(r2, 4)}

        # Save result
        save_baseline_metrics(TARGET_DATASET, trees, train_time, infer_time, metrics_dict, config)

    print("\n [SUCCESS] Baseline execution complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Baseline terminated by user.")
        sys.exit(0)
