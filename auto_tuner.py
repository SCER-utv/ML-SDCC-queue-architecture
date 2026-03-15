import boto3
import json
import sys
import itertools
from datetime import datetime

# --- CARICAMENTO DINAMICO CONFIGURAZIONE ---
from src.utils.config import load_config

try:
    config = load_config()
except Exception as e:
    print(f" Errore critico durante l'Auto-Discovery: {e}")
    sys.exit(1)

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
AWS_REGION = config.get("aws_region")

# GRIGLIA DEGLI IPERPARAMETRI (Modifica questi array prima di avviare)
Hai detto
DATASET_TO_TEST = "airlines"          
WORKERS_TO_TEST = [1, 2, 3, 4, 5, 6, 7, 8]  # Raddoppio dei nodi per misurare lo Speedup
TREES_TO_TEST   = [5, 10, 25, 50, 100, 200, 400]  # Raddoppio del cari
# =====================================================================

def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    
    # Crea in automatico il prodotto cartesiano (tutte le combinazioni possibili)
    # Es: (2, 20), (2, 50), ..., (6, 200)
    combinazioni = list(itertools.product(WORKERS_TO_TEST, TREES_TO_TEST))

    print("\n" + "="*60)
    print(" DISTRIBUTED RANDOM FOREST - FULLY AUTOMATED GRID SEARCH")
    print("="*60)
    print(f" Dataset target : {DATASET_TO_TEST.upper()}")
    print(f" Worker da testare : {WORKERS_TO_TEST}")
    print(f" Alberi da testare : {TREES_TO_TEST}")
    print(f" Totale Job Generati: {len(combinazioni)}")
    print("="*60 + "\n")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Ciclo di invio automatico a SQS
    for w, t in combinazioni:
        job_id = f"job_{DATASET_TO_TEST}_W{w}_T{t}_TUNING_{batch_timestamp}"
        
        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": DATASET_TO_TEST,
            "num_workers": w,
            "num_trees": t,
            # FONDAMENTALE: Fissiamo i dati per fare un confronto scientifico!
            # Il Master userà i file già presenti (train e validation) senza rigenerarli
            "dynamic_split": False 
        }
        
        try:
            sqs_client.send_message(
                QueueUrl=CLIENT_QUEUE_URL,
                MessageBody=json.dumps(payload),
                MessageGroupId="ML_Jobs",
                MessageDeduplicationId=job_id
            )
            print(f" [ACCODATO] Worker: {w:<2} | Alberi: {t:<4} | Job ID: {job_id}")
        except Exception as e:
            print(f" [ERRORE] Invio fallito per W{w}-T{t}: {e}")

    print("\n Tutte le configurazioni sono state inviate alla coda SQS con successo!")
    print(" Il Master Node le smaltirà una ad una in sequenza in modo completamente autonomo.")
    print(f" Controlla il file s3://{config.get('s3_bucket')}/results/{DATASET_TO_TEST}/{DATASET_TO_TEST}_results.csv per i risultati finali.")

if __name__ == "__main__":
    main()
