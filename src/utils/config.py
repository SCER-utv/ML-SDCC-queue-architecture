import json
import os

import boto3

_cached_config = None

# Scans the S3 interim data bucket to discover optimized datasets, extracts their headers via S3 Select to count features, and maps future processed S3 paths
def discover_datasets(s3_bucket, region, dataset_registry):
    s3 = boto3.client('s3', region_name=region)
    datasets = {}
    
    # 1. Scan interim data (Single Source of Truth)
    prefix = "data/interim/"
    
    try:
        resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        
        if 'Contents' not in resp:
            print(f" [DISCOVERY WARN] No objects found in bucket under prefix '{prefix}'.")
            return datasets

        for obj in resp['Contents']:
            key = obj['Key']
            
            # 2. Identify datasets from optimized files
            if key.endswith('_optimized.csv'):
                # Extract filename (e.g., "data/interim/airlines/airlines_optimized.csv" -> "airlines_optimized.csv")
                filename = key.split('/')[-1]
                
                # Extract dataset raw name (e.g., "airlines_optimized.csv" -> "airlines")
                dataset_name = filename.replace('_optimized.csv', '')
                
                if dataset_name not in dataset_registry:
                    continue
                    
                target_col = dataset_registry[dataset_name]["target"]
                task_type = dataset_registry[dataset_name]["type"]
                
                # 3. Define future paths for processed data splits
                train_key = f"data/processed/{dataset_name}/{dataset_name}_train.csv"
                test_key = f"data/processed/{dataset_name}/{dataset_name}_test.csv"
                
                # 4. S3 Select: Read only the header row from the interim file to map features
                select_resp = s3.select_object_content(
                    Bucket=s3_bucket,
                    Key=key,
                    ExpressionType='SQL',
                    Expression='SELECT * FROM S3Object LIMIT 1',
                    InputSerialization={'CSV': {'FileHeaderInfo': 'NONE'}},
                    OutputSerialization={'CSV': {}}
                )
                
                header = ""
                for event in select_resp['Payload']:
                    if 'Records' in event:
                        header += event['Records']['Payload'].decode('utf-8')
                        
                columns = [col.strip() for col in header.split(',') if col.strip()]
                if not columns or target_col not in columns:
                    print(f" [DISCOVERY WARN] Target column '{target_col}' not found in CSV header for '{dataset_name}'. Skipping.")
                    continue
                    
                features_count = len(columns) - 1
                
                # 5. Populate registry with validated metadata
                datasets[dataset_name] = {
                    "type": task_type,
                    "target": target_col,
                    "features": features_count,
                    "train_path": train_key,
                    "test_path": test_key
                }
                
    except Exception as e:
        print(f" [DISCOVERY ERROR] S3 error during scan of prefix {prefix}: {e}")
        
    return datasets

# Loads configuration from config.json, executes S3 auto-discovery, and caches the result to prevent redundant I/O operations.
def load_config():
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f" [SYSTEM ERROR] Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset_registry = config.get("dataset_registry", {})
    print("\n [AUTO-DISCOVERY] Scanning S3 'data/interim/' prefix for valid datasets...")

    # Inject dynamically discovered datasets into the config dictionary
    config['datasets_metadata'] = discover_datasets(config['s3_bucket'], config['aws_region'], dataset_registry)

    config['_root_dir'] = root_dir
    _cached_config = config
    return config
