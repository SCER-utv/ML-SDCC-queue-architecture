def discover_datasets(s3_bucket, region, dataset_registry):
    s3 = boto3.client('s3', region_name=region)
    datasets = {}
    
    # 1. IL SISTEMA CERCA NEI DATI OTTIMIZZATI (La Fonte Unica di Verità)
    prefix = "data/interim/"
    
    try:
        resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        
        if 'Contents' not in resp:
            print(f" [DEBUG] Attenzione: Nessun file trovato nel bucket sotto '{prefix}'")
            return datasets
            
        for obj in resp['Contents']:
            key = obj['Key']
            
            # 2. Identifichiamo il dataset dal file ottimizzato
            if key.endswith('_optimized.csv'):
                # Esempio: da "data/interim/airlines/airlines_optimized.csv" estrae "airlines_optimized.csv"
                filename = key.split('/')[-1]
                
                # Esempio: da "airlines_optimized.csv" estrae "airlines"
                dataset_name = filename.replace('_optimized.csv', '')
                
                if dataset_name not in dataset_registry:
                    continue
                    
                target_col = dataset_registry[dataset_name]["target"]
                task_type = dataset_registry[dataset_name]["type"]
                
                # 3. Definiamo i percorsi FUTURI (dove il Master salverà lo split)
                # Il Client sa che il Master metterà i file splittati in queste cartelle!
                train_key = f"data/processed/{dataset_name}/{dataset_name}_train.csv"
                test_key = f"data/processed/{dataset_name}/{dataset_name}_test.csv"
                
                # 4. S3 SELECT: Leggiamo l'intestazione dal file INTERIM originale!
                select_resp = s3.select_object_content(
                    Bucket=s3_bucket,
                    Key=key, # <-- Usa la key del file _optimized.csv
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
                    print(f"[WARNING] La colonna target '{target_col}' non trovata nel CSV di '{dataset_name}'. Ignorato.")
                    continue
                    
                features_count = len(columns) - 1
                
                # 5. Salviamo nel dizionario i percorsi di PROCESSED
                datasets[dataset_name] = {
                    "type": task_type,
                    "target": target_col,
                    "features": features_count,
                    "train_path": train_key, # Il Master e i Worker li cercheranno qui!
                    "test_path": test_key
                }
                
    except Exception as e:
        print(f"[DISCOVERY WARNING] Errore S3 durante la scansione di {prefix}: {e}")
        
    return datasets
