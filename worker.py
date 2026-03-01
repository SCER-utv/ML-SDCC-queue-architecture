import boto3
import json
import time
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# CONFIGURAZIONE INIZIALE
# ==========================================
AWS_REGION = 'us-east-1'  # La tua region
WORKER_TRAINING_QUEUE_URL = ''
WORKER_RESPONSE_QUEUE_URL = ''
S3_MODELS_BUCKET = 'distributed-random-forest-bkt'  # Bucket dove salvare i modelli

# Inizializzazione client
sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)


# ==========================================
# FASE 1: ADDESTRAMENTO E SALVATAGGIO
# ==========================================
def process_task(task_data):
    """
    Legge i dati, addestra la foresta parziale e salva il modello su S3.
    """
    job_id = task_data['job_id']
    task_id = task_data['task_id']
    dataset_uri = task_data['dataset_uri']
    trees_to_train = task_data['trees_to_train']

    print(f"🌲 [Task: {task_id}] Inizio addestramento di {trees_to_train} alberi...")

    # 1. LETTURA DATI (Gestione intelligente della memoria)
    # Per non far esplodere la RAM, il Worker dovrebbe leggere solo la sua porzione di dati.
    # NOTA: Per un dataset da 40M di righe, l'ideale è che il Master invii nel JSON
    # anche 'skiprows' e 'nrows' in base allo shard_index.
    # Qui simuliamo la lettura diretta da S3 tramite Pandas.
    print(f"   -> Download dati da {dataset_uri}...")
    df = pd.read_csv(dataset_uri)  # Sostituisci con logica a chunk (nrows) se necessario

    X = df.drop(columns=['TARGET_DELAY']).values
    y = df['TARGET_DELAY'].values

    # 2. ADDESTRAMENTO DELLA FORESTA PARZIALE
    rf = RandomForestClassifier(
        n_estimators=trees_to_train,
        max_depth=20,
        n_jobs=-1,  # Usa tutti i core della macchina EC2
        class_weight='balanced',
        random_state=42  # Opzionale: variare il seed per ogni worker
    )
    rf.fit(X, y)

    # 3. SALVATAGGIO DEL MODELLO (Localmente e poi su S3)
    local_model_path = f"/tmp/{job_id}_{task_id}.joblib"
    s3_model_key = f"modelli/{job_id}/{task_id}.joblib"

    print(f"   -> Salvataggio modello in {local_model_path}...")
    joblib.dump(rf, local_model_path)

    print(f"   -> Upload su S3: s3://{S3_MODELS_BUCKET}/{s3_model_key}")
    s3_client.upload_file(local_model_path, S3_MODELS_BUCKET, s3_model_key)

    # Pulizia del file locale per liberare spazio sull'EC2
    os.remove(local_model_path)

    return f"s3://{S3_MODELS_BUCKET}/{s3_model_key}"


# ==========================================
# CICLO PRINCIPALE DEL WORKER
# ==========================================
def main():
    print("👷 Worker Node Avviato. In attesa di Task sulla coda...")

    while True:
        # 1. Polling: Peschiamo UN SOLO messaggio alla volta (MaxNumberOfMessages=1)
        response = sqs_client.receive_message(
            QueueUrl=WORKER_TRAINING_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20  # Risparmia chiamate API a vuoto
        )

        if 'Messages' in response:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            task_data = json.loads(message['Body'])

            print(f"\n📥 Preso in carico Task: {task_data['task_id']} per il Job: {task_data['job_id']}")

            try:
                # 2. Addestra e carica su S3 (Il "lavoro sporco")
                s3_model_uri = process_task(task_data)

                # 3. Invia la risposta al Master sulla coda di ritorno
                risposta_master = {
                    "job_id": task_data['job_id'],
                    "task_id": task_data['task_id'],
                    "s3_model_uri": s3_model_uri,
                    "status": "COMPLETED"
                }

                sqs_client.send_message(
                    QueueUrl=WORKER_RESPONSE_QUEUE_URL,
                    MessageBody=json.dumps(risposta_master)
                )
                print(f"📤 Risposta inviata al Master con URI: {s3_model_uri}")

                # 4. Elimina il messaggio originale dalla coda di Training
                # (Se non lo eliminiamo, alla fine del Visibility Timeout un altro Worker lo rifarà!)
                sqs_client.delete_message(
                    QueueUrl=WORKER_TRAINING_QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )
                print("✅ Task completato e rimosso dalla coda. Pronto per il prossimo!\n")

            except Exception as e:
                print(f"❌ Errore durante l'esecuzione del Task: {str(e)}")
                # In caso di eccezione, NON eliminiamo il messaggio.
                # Scadrà il Visibility Timeout e verrà riprocessato.


if __name__ == "__main__":
    main()