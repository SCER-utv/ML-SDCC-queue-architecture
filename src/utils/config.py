import json
import os
import boto3

_cached_config = None

def discover_datasets(s3_bucket, region, target_registry):
    s3 = boto3.client('s3', region_name=region)
    datasets = {}
    
    # Esploriamo entrambe le cartelle per trovare i dataset su S3
    for task_type in ['classification', 'regression']:
        prefix = f"data/{task_type}/"
        try:
            resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix, Delimiter='/')
            
            for obj in resp.get('CommonPrefixes', []):
                dataset_name = obj['Prefix'].split('/')[-2]
                
                train_key = f"{obj['Prefix']}{dataset_name}_train.csv"
                test_key = f"{obj['Prefix']}{dataset_name}_test.csv"
                
                # S3 SELECT: Leggiamo solo l'intestazione senza scaricare il file!
                select_resp = s3.select_object_content(
                    Bucket=s3_bucket,
                    Key=train_key,
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
                if not columns:
                    continue
                    
                # <-- CERCHIAMO IL TARGET NEL REGISTRO -->
                if dataset_name not in target_registry:
                    print(f"[WARNING] Il dataset '{dataset_name}' trovato su S3 non ha un target nel config.json! Verrà ignorato.")
                    continue
                    
                target_col = target_registry[dataset_name]
                
                if target_col not in columns:
                    print(f"[WARNING] La colonna target '{target_col}' non esiste nel CSV di '{dataset_name}'! Verrà ignorato.")
                    continue
                    
                # Le feature sono il numero totale di colonne MENO la colonna target (indipendentemente da dove si trova!)
                features_count = len(columns) - 1
                
                datasets[dataset_name] = {
                    "type": task_type,
                    "target": target_col,
                    "features": features_count,
                    "train_path": train_key,
                    "test_path": test_key
                }
        except Exception as e:
            print(f"[DISCOVERY WARNING] Errore S3 per la cartella {task_type}: {e}")
            
    return datasets

def load_config():
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trovato in: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    #  AUTO-DISCOVERY CON REGISTRO TARGET 
    target_registry = config.get("target_registry", {})
    print("\n[Auto-Discovery] Scansione intelligente dei dataset su S3 in corso...")
    config['datasets_metadata'] = discover_datasets(config['s3_bucket'], config['aws_region'], target_registry)

    config['_root_dir'] = root_dir
    _cached_config = config
    return config
