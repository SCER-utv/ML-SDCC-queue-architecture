import boto3
import json
import sys
from datetime import datetime

# ==========================================
# CONFIGURAZIONI
# ==========================================
CLIENT_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/248593862537/JobRequestQueue.fifo'
AWS_REGION = 'us-east-1'
S3_BUCKET = "distributed-random-forest-bkt"

def list_available_models(s3_client, bucket, dataset):
    """Cerca su S3 i modelli addestrati per il dataset specificato"""
    prefix = f"models/{dataset}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    models = []
    if 'CommonPrefixes' in resp:
        for obj in resp['CommonPrefixes']:
            folder_name = obj['Prefix'].replace(prefix, '').strip('/')
            models.append(folder_name)
    return models

def clear_screen():
    # Stampa un po' di righe vuote per pulire il terminale
    print("\n" * 2)

def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    clear_screen()
    print("="*60)
    print(" DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
    print("="*60)

    # --- STEP 1: SCELTA MODALITÀ ---
    print("\nCosa desideri fare?")
    print("  1) Addestramento Distribuito (+ Test Bulk)")
    print("  2) Inferenza Real-Time (Singola Predizione)")
    
    while True:
        scelta_mode = input("\n👉 Inserisci 1 o 2: ").strip()
        if scelta_mode in ['1', '2']:
            mode = 'train' if scelta_mode == '1' else 'infer'
            break
        print(" Scelta non valida.")

    # --- STEP 2: SCELTA DATASET ---
    print("\n" + "-"*40)
    print(" Seleziona il Dataset di riferimento:")
    print("  1) Taxi (Regressione)")
    print("  2) Higgs (Classificazione)")
    print("  3) Airlines (Classificazione)")
    
    dataset_map = {'1': 'taxi', '2': 'higgs', '3': 'airlines'}
    while True:
        scelta_ds = input("\n👉 Inserisci 1, 2 o 3: ").strip()
        if scelta_ds in dataset_map:
            dataset = dataset_map[scelta_ds]
            break
        print(" Scelta non valida.")

    payload = {}

    # ==========================================
    # RAMO A: TRAINING
    # ==========================================
    if mode == 'train':
        print("\n" + "-"*40)
        print(f" Configurazione Parametri per: {dataset.upper()}")
        
        while True:
            try:
                workers = int(input("👉 Inserisci il numero di Worker (es. 4): "))
                trees = int(input("👉 Inserisci il numero TOTALE di alberi (es. 100): "))
                if workers > 0 and trees > 0:
                    break
                print(" Inserisci numeri maggiori di zero.")
            except ValueError:
                print(" Inserisci dei numeri validi.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{dataset}_{trees}trees_{timestamp}"

        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": dataset,
            "num_workers": workers,
            "num_trees": trees
        }

    # ==========================================
    # RAMO B: INFERENZA
    # ==========================================
    elif mode == 'infer':
        print("\n" + "-"*40)
        print(f" Ricerca dei modelli '{dataset}' salvati su S3...")
        
        models = list_available_models(s3_client, S3_BUCKET, dataset)
        
        if not models:
            print(f"\n Nessun modello trovato per il dataset '{dataset}'. Esegui prima un addestramento!")
            sys.exit(0)
            
        print("\n=== Modelli Disponibili ===")
        for i, m in enumerate(models):
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

        print("\n" + "-"*40)
        print(" Inserimento Dati")
        print("Inserisci i valori della riga separati da virgola (es: 1.0, 2.5, 3.1)")
        
        while True:
            raw_tuple = input(" Valori: ").strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                if len(tuple_data) > 0:
                    break
                print(" Inserisci almeno un valore.")
            except ValueError:
                print(" Errore di formattazione. Assicurati di usare solo numeri e virgole.")

        req_id = f"req_{dataset}_{int(datetime.now().timestamp())}"
        
        payload = {
            "mode": "infer",
            "job_id": req_id,
            "dataset": dataset,
            "target_model": target_model,
            "tuple_data": tuple_data
        }

    # ==========================================
    # INVIO MESSAGGIO ALLA CODA
    # ==========================================
    print("\n" + "="*60)
    print(" Invio richiesta al Master Node in corso...")
    
    try:
        response = sqs_client.send_message(
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
