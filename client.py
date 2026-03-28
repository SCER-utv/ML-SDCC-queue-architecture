import sys
import json
from datetime import datetime

import boto3

from src.utils.config import load_config

# DYNAMIC CONFIGURATION & AUTO-DISCOVERY 
try:
    config = load_config()
except Exception as e:
    print(f" [CRITICAL] Error during Auto-Discovery from S3: {e}")
    sys.exit(1)

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
AWS_REGION = config.get("aws_region")
S3_BUCKET = config.get("s3_bucket")
DATASETS_METADATA = config.get("datasets_metadata", {})


# Scans the S3 bucket to retrieve all trained model directories available for the specified dataset.
def list_available_models(s3_client, bucket, dataset):
    prefix = f"models/{dataset}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    models = []
    if 'CommonPrefixes' in resp:
        for obj in resp['CommonPrefixes']:
            folder_name = obj['Prefix'].replace(prefix, '').strip('/')
            models.append(folder_name)
    return models


# Clears the terminal screen for better UI readability
def clear_screen():
    print("\n" * 2)

def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    clear_screen()
    print("=" * 60)
    print("  DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
    print("=" * 60)

    print("\nSelect Operation Mode:")
    print("  1)  Distributed Training (+ Bulk Inference Test)")
    print("  2)  Real-Time Inference (Single Prediction)")
    
    while True:
        mode_choice = input("\n Enter 1 or 2: ").strip()
        if mode_choice in ['1', '2']:
            mode = 'train' if mode_choice == '1' else 'infer'
            break
        print(" Invalid choice. Please try again.")

    # DYNAMIC DATASET MENU
    print("\n" + "-" * 40)
    print(" Select Target Dataset:")
    
    available_datasets = list(DATASETS_METADATA.keys())
    if not available_datasets:
        print(" [ERROR] No datasets found in config metadata!")
        sys.exit(1)

    dataset_map = {}

    # Dynamically list all datasets registered in config.json
    for i, ds_name in enumerate(available_datasets, start=1):
        ds_type = DATASETS_METADATA[ds_name]["type"]
        print(f" {i}) {ds_name.capitalize()} ({ds_type.capitalize()})")
        dataset_map[str(i)] = ds_name
        
    while True:
        ds_choice = input(f"\n Enter a number [1-{len(available_datasets)}]: ").strip()
        if ds_choice in dataset_map:
            dataset = dataset_map[ds_choice]
            break
        print(" Invalid dataset selection.")

    if mode == 'train':
        print("\n" + "-" * 40)
        print(f"  Hyperparameter Configuration for: {dataset.upper()}")
        
        while True:
            try:
                workers = int(input(" Enter number of Workers (e.g., 4): "))
                trees = int(input(" Enter TOTAL number of Trees (e.g., 100): "))
                if workers > 0 and trees > 0:
                    break
                print(" Values must be greater than zero.")
            except ValueError:
                print(" Invalid input. Please enter integers only.")

        # AGGIUNTA: Scelta della Strategia (Homogeneous vs Heterogeneous)
        print("\n Select Training Strategy:")
        print("  1) Homogeneous  [Same parameters for all workers]")
        print("  2) Heterogeneous [Different parameters per worker, variance boosting]")

        while True:
            strat_choice = input(" Enter 1 or 2: ").strip()
            if strat_choice in ['1', '2']:
                strategy_type = "homogeneous" if strat_choice == '1' else "heterogeneous"
                break
            print(" Invalid choice. Please enter 1 or 2.")

        # Generate a unique and descriptive Job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ex: job_taxi_100trees_4workers_homogeneous_20260328_180000
        job_id = f"job_{dataset}_{trees}trees_{workers}workers_{strategy_type}_{timestamp}"
        
        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": dataset,
            "num_workers": workers,
            "num_trees": trees,
            "strategy": strategy_type
        }

    elif mode == 'infer':
        print("\n" + "-" * 40)
        print(f" [SEARCH] Scanning S3 for saved '{dataset}' models...")
        
        models = list_available_models(s3_client, S3_BUCKET, dataset)
        
        if not models:
            print(f"\n [ERROR] No trained models found for '{dataset}'. Run a training job first!")
            sys.exit(0)
            
        print("\n=== AVAILABLE MODELS ===")
        for i, m in enumerate(models):
            # Dividiamo la stringa usando l'underscore per estrarre i metadati
            parts = m.split('_')
            try:
                # 1. NUOVO FORMATO (Con Strategia)
                # Es: job_taxi_100trees_4workers_homogeneous_20260328_180000
                if "homogeneous" in m or "heterogeneous" in m:
                    trees_count = parts[2].replace('trees', '')
                    workers_count = parts[3].replace('workers', '')
                    strat_label = parts[4][:4].upper() # Prende le prime 4 lettere: "HOMO" o "HETE"
                    raw_date = parts[5]
                    raw_time = parts[6]
                    
                # 2. VECCHIO FORMATO (Senza Strategia, ma con i worker)
                # Es: job_taxi_100trees_4workers_20260328_180000
                elif "workers" in m:
                    trees_count = parts[2].replace('trees', '')
                    workers_count = parts[3].replace('workers', '')
                    strat_label = "N/A " # Forziamo "N/A " per allineamento
                    raw_date = parts[4]
                    raw_time = parts[5]
                    
                # 3. LEGACY FORMAT (Vecchissimo formato senza worker)
                # Es: job_taxi_100trees_20260328_180000
                else:
                    trees_count = parts[2].replace('trees', '')
                    workers_count = "? " 
                    strat_label = "N/A "
                    raw_date = parts[3]
                    raw_time = parts[4]
                    
                # Formattazione per la stampa a schermo
                date_formatted = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
                time_formatted = f"{raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
                
                print(f"  [{i}]  Trees: {trees_count:<4} | Workers: {workers_count:<2} | Strat: {strat_label} | Date: {date_formatted} {time_formatted}  (ID: {m})")
                
            except Exception:
                # Se la cartella ha un nome manuale o un formato illeggibile, la stampiamo "grezza" senza far crashare il Client
                print(f"  [{i}] {m}")
        
        while True:
            try:
                model_choice = int(input(f"\n Select Model ID [0-{len(models)-1}]: "))
                if 0 <= model_choice < len(models):
                    target_model = models[model_choice]
                    break
                print(" Invalid ID selected.")
            except ValueError:
                print(" Please enter a valid number.")

        # Auto-detect required features based on config.json metadata
        required_features = DATASETS_METADATA[dataset]["features"]

        print("\n" + "-" * 40)
        print(" Real-Time Prediction Input")
        print(f" WARNING: The '{dataset.upper()}' dataset requires EXACTLY {required_features} features!")
        
        while True:
            raw_tuple = input(f" Enter {required_features} comma-separated values: ").strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                
                if len(tuple_data) == required_features:
                    break
                else:
                    print(f" [ERROR] You provided {len(tuple_data)} values, but the model expects {required_features}.")
            except ValueError:
                print(" [ERROR] Formatting error. Use numbers only (e.g., 10.5, 3).")

        req_id = f"req_{dataset}_{int(datetime.now().timestamp())}"
        
        payload = {
            "mode": "infer",
            "job_id": req_id,
            "dataset": dataset,
            "target_model": target_model,
            "tuple_data": tuple_data
        }

    # SQS DISPATCH 
    print("\n" + "=" * 60)
    print(" Dispatching request to Master Node...")
    
    try:
        # Enqueue the JSON payload into the FIFO queue
        sqs_client.send_message(
            QueueUrl=CLIENT_QUEUE_URL,
            MessageBody=json.dumps(payload),
            MessageGroupId="ML_Jobs",
            MessageDeduplicationId=payload['job_id']
        )
        print(f" [SUCCESS] Message enqueued successfully.")
        print(f" [INFO] Generated Job ID: {payload['job_id']}")
        print("=" * 60 + "\n")
        
        if payload['mode'] == 'infer':
            print(" -> Check the Master Node logs to see the cluster's real-time prediction!")
            
    except Exception as e:
        print(f"\n [CRITICAL ERROR] Failed to dispatch SQS message: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Client terminated by user.")
        sys.exit(0)
