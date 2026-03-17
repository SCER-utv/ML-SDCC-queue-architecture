import boto3
import json
import sys
from datetime import datetime

# --- CARICAMENTO DINAMICO CONFIGURAZIONE E AUTO-DISCOVERY ---
from src.utils.config import load_config

try:
    config = load_config()
except Exception as e:
    print(f" Errore critico durante l'Auto-Discovery da S3: {e}")
    sys.exit(1)

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
AWS_REGION = config.get("aws_region")
S3_BUCKET = config.get("s3_bucket")
DATASETS_METADATA = config.get("datasets_metadata", {})

def list_available_models(s3_client, bucket, dataset):
    prefix = f"models/{dataset}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    models = []
    if 'CommonPrefixes' in resp:
        for obj in resp['CommonPrefixes']:
            folder_name = obj['Prefix'].replace(prefix, '').strip('/')
            models.append(folder_name)
    return models

def clear_screen():
    print("\n" * 2)

def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    clear_screen()
    print("="*60)
    print("  DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
    print("="*60)

    print("\nCosa desideri fare?")
    print("  1)  Addestramento Distribuito (+ Test Bulk)")
    print("  2)  Inferenza Real-Time (Singola Predizione)")
    
    while True:
        scelta_mode = input("\n Inserisci 1 o 2: ").strip()
        if scelta_mode in ['1', '2']:
            mode = 'train' if scelta_mode == '1' else 'infer'
            break
        print(" Scelta non valida.")

    # --- MENU DATASET GENERATO DINAMICAMENTE ---
    print("\n" + "-"*40)
    print(" Seleziona il Dataset di riferimento:")
    
    available_datasets = list(DATASETS_METADATA.keys())
    if not available_datasets:
        print(" ERRORE: Nessun dataset trovato in 'data/processed/' su S3!")
        sys.exit(1)

    dataset_map = {}
    
    for i, ds_name in enumerate(available_datasets, start=1):
        ds_type = DATASETS_METADATA[ds_name]["type"]
        print(f"  {i}) {ds_name.capitalize()} ({ds_type.capitalize()})")
        dataset_map[str(i)] = ds_name
        
    while True:
        scelta_ds = input(f"\n Inserisci un numero da 1 a {len(available_datasets)}: ").strip()
        if scelta_ds in dataset_map:
            dataset = dataset_map[scelta_ds]
            break
        print(" Scelta non valida.")

    payload = {}

    if mode == 'train':
        print("\n" + "-"*40)
        print(f"  Configurazione Parametri per: {dataset.upper()}")
        
        while True:
            try:
                workers = int(input(" Inserisci il numero di Worker (es. 4): "))
                trees = int(input(" Inserisci il numero TOTALE di alberi (es. 100): "))
                if workers > 0 and trees > 0:
                    break
                print(" Inserisci numeri maggiori di zero.")
            except ValueError:
                print(" Inserisci dei numeri validi.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{dataset}_{trees}trees_{workers}workers_{timestamp}"
        
        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": dataset,
            "num_workers": workers,
            "num_trees": trees
        }

    elif mode == 'infer':
        print("\n" + "-"*40)
        print(f"🔍 Ricerca dei modelli '{dataset}' salvati su S3...")
        
        models = list_available_models(s3_client, S3_BUCKET, dataset)
        
        if not models:
            print(f"\n Nessun modello trovato per il dataset '{dataset}'. Esegui prima un addestramento!")
            sys.exit(0)
            
        print("\n=== MODELLI DISPONIBILI ===")
        for i, m in enumerate(models):
            parts = m.split('_')
            try:
                if "workers" in m:
                    alberi = parts[2].replace('trees', '')
                    workers_count = parts[3].replace('workers', '')
                    data_raw = parts[4]
                    ora_raw = parts[5]
                else:
                    alberi = parts[2].replace('trees', '')
                    workers_count = "?" 
                    data_raw = parts[3]
                    ora_raw = parts[4]
                    
                data_formattata = f"{data_raw[6:8]}/{data_raw[4:6]}/{data_raw[0:4]}"
                ora_formattata = f"{ora_raw[0:2]}:{ora_raw[2:4]}:{ora_raw[4:6]}"
                
                print(f"  [{i}]  Alberi: {alberi:<4} |  Worker: {workers_count:<2} |  Data: {data_formattata} {ora_formattata}  (ID: {m})")
            except Exception:
                print(f"  [{i}] {m}")
        
        while True:
            try:
                scelta_modello = int(input(f"\n Scegli l'ID del modello [0-{len(models)-1}]: "))
                if 0 <= scelta_modello < len(models):
                    target_model = models[scelta_modello]
                    break
                print(" ID non valido.")
            except ValueError:
                print(" Inserisci un numero.")

        num_richiesto = DATASETS_METADATA[dataset]["features"]

        print("\n" + "-"*40)
        print(" Inserimento Dati per Predizione")
        print(f"ATTENZIONE: Il dataset '{dataset.upper()}' richiede ESATTAMENTE {num_richiesto} parametri!")
        
        while True:
            raw_tuple = input(f" Inserisci i {num_richiesto} valori (separati da virgola): ").strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                
                if len(tuple_data) == num_richiesto:
                    break
                else:
                    print(f" ERRORE: Hai inserito {len(tuple_data)} valori, ma il modello ne aspetta {num_richiesto}!")
            except ValueError:
                print(" ERRORE di formattazione. Usa solo numeri (es. 10.5).")

        req_id = f"req_{dataset}_{int(datetime.now().timestamp())}"
        
        payload = {
            "mode": "infer",
            "job_id": req_id,
            "dataset": dataset,
            "target_model": target_model,
            "tuple_data": tuple_data
        }

    print("\n" + "="*60)
    print(" Invio richiesta al Master Node in corso...")
    
    try:
        sqs_client.send_message(
            QueueUrl=CLIENT_QUEUE_URL,
            MessageBody=json.dumps(payload),
            MessageGroupId="ML_Jobs",
            MessageDeduplicationId=payload['job_id']
        )
        print(f" SUCCESS! Il messaggio è stato accodato con successo.")
        print(f" Job ID generato: {payload['job_id']}")
        print("="*60 + "\n")
        
        if payload['mode'] == 'infer':
            print("Guarda i log del Master Node per leggere la previsione calcolata dal cluster!")
            
    except Exception as e:
        print(f"\n ERRORE IMPREVISTO DURANTE L'INVIO: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Chiusura forzata del client. Arrivederci!")
        sys.exit(0)
