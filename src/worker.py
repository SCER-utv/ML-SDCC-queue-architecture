import boto3
import json
import time
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.model.model_factory import ModelFactory

# ==========================================
# CONFIGURAZIONE CODE SQS (Le stesse del Master)
# ==========================================
AWS_REGION = 'us-east-1'
CLIENT_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/123/client-queue'

TRAIN_TASK_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/train-task-queue'
TRAIN_RESPONSE_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/train-response-queue'

INFER_TASK_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/infer-task-queue'
INFER_RESPONSE_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/infer-response-queue'

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)


# Funzione helper per estrarre bucket e key da un URI s3://
def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


# ==========================================
# CORE LOGIC: TRAINING
# ==========================================
def train(train_task_data):
    job_id = train_task_data['job_id']
    task_id = train_task_data['task_id']
    dataset_uri = train_task_data['dataset_s3_path']


    print(f"🌲 [TRAIN] Avvio {task_id}. Lettura di {train_task_data['num_rows']} righe da S3...")

    # 1. LETTURA PARZIALE DA S3 (Zero-Waste RAM)
    # Calcoliamo le righe da saltare preservando l'header (riga 0 del CSV)
    skip_rows = train_task_data['skip_rows']
    if skip_rows > 0:
        # Salta le righe dalla 1 fino a skip_rows (mantiene la 0 che è l'intestazione)
        rows_to_skip = range(1, skip_rows + 1)
    else:
        rows_to_skip = None

    df = pd.read_csv(
        dataset_uri,
        skiprows=rows_to_skip,
        nrows=train_task_data['num_rows']
    )

    ml_handler = ModelFactory.get_model(dataset_name=train_task_data['dataset'])

    print("avvio timer")
    start_time = time.time()
    print("timer avviato")
    rf = ml_handler.process_and_train(df, train_task_data)
    print(f"   -> [job id:{job_id}, task id: {task_id}] Training completato in {time.time() - start_time:.2f}s")


    # 3. SALVATAGGIO E UPLOAD
    local_model_path = f"/tmp/{task_id}_{job_id}.joblib"
    joblib.dump(rf, local_model_path)

    bucket, _ = parse_s3_uri(dataset_uri)
    s3_key = f"models/{job_id}/task_{task_id}.joblib"

    print("   -> Upload del modello su S3 in corso...")
    s3_client.upload_file(local_model_path, bucket, s3_key)

    os.remove(local_model_path)  # Pulizia disco locale
    return f"s3://{bucket}/{s3_key}"


# ==========================================
# CORE LOGIC: INFERENZA
# ==========================================
def esegui_inferenza(infer_task_data):
    job_id = infer_task_data['job_id']
    task_id = infer_task_data['task_id']
    model_s3_uri = infer_task_data['model_s3_uri']
    test_dataset_uri = infer_task_data['test_dataset_uri']

    print(f"🔮 [INFER] Avvio inferenza {task_id}. Scaricamento modello e test set...")
    bucket, model_key = parse_s3_uri(model_s3_uri)

    # 1. SCARICA IL MODELLO DA S3
    local_model_path = f"/tmp/model_{job_id}_{task_id}.joblib"
    s3_client.download_file(bucket, model_key, local_model_path)
    rf = joblib.load(local_model_path)

    ml_handler = ModelFactory.get_model(dataset_name=infer_task_data['dataset'])

    # 2. SCARICA L'INTERO TEST SET
    df_test = pd.read_csv(test_dataset_uri)
    print(f"   -> Calcolo previsioni in corso (Delegate alla Factory)...")
    start_time = time.time()

    # Ritornerà una matrice a 2 colonne (Classificazione) o un array 1D (Regressione)
    risultati_numpy = ml_handler.process_and_predict(rf, df_test)

    print(f"   -> Previsioni completate in {time.time() - start_time:.2f} secondi.")

    # 4. SALVATAGGIO IN .NPY COMPRESSO E UPLOAD (Meglio del CSV!)
    local_npy_path = f"/tmp/results_{job_id}_{task_id}.npy"
    np.save(local_npy_path, risultati_numpy)

    s3_voti_key = f"results/{job_id}/{task_id}.npy"
    s3_client.upload_file(local_npy_path, bucket, s3_voti_key)

    os.remove(local_model_path)
    os.remove(local_npy_path)
    return f"s3://{bucket}/{s3_voti_key}"


# ==========================================
# EVENT LOOP: PRIORITY POLLING
# ==========================================
def main():
    print("🤖 Worker Node Avviato e pronto a ricevere ordini...")

    while True:
        try:
            # ==========================================
            # PRIORITÀ 1: TRAINING
            # ==========================================
            res_train = sqs_client.receive_message(
                QueueUrl=TRAIN_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_train:
                msg = res_train['Messages'][0]
                train_task_data = json.loads(msg['Body'])

                # Esegue il lavoro pesante
                s3_model_uri = train(train_task_data)

                # Risponde al Master
                train_resp = {
                    "job_id": train_task_data['job_id'],
                    "task_id": train_task_data['task_id'],
                    "s3_model_uri": s3_model_uri
                }
                sqs_client.send_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MessageBody=json.dumps(train_resp))

                # CANCELLA IL MESSAGGIO SOLO DOPO IL SUCCESSO (Fault Tolerance)
                sqs_client.delete_message(QueueUrl=TRAIN_TASK_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                print(f"✅ Training {train_task_data['task_id']} completato con successo!\n")

                continue  # 🔥 FONDAMENTALE: Torna su e ricontrolla la coda di training!

            # ==========================================
            # PRIORITÀ 2: INFERENZA
            # ==========================================
            # Ci arriva SOLO se la coda di training non ha restituito messaggi.
            res_infer = sqs_client.receive_message(
                QueueUrl=INFER_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_infer:
                msg = res_infer['Messages'][0]
                infer_task_data = json.loads(msg['Body'])

                # Esegue il lavoro pesante
                s3_voti_uri = esegui_inferenza(infer_task_data)

                # Risponde al Master
                risposta = {
                    "job_id": infer_task_data['job_id'],
                    "task_id": infer_task_data['task_id'],
                    "s3_voti_uri": s3_voti_uri
                }
                sqs_client.send_message(QueueUrl=INFER_RESPONSE_QUEUE, MessageBody=json.dumps(risposta))

                # CANCELLA IL MESSAGGIO SOLO DOPO IL SUCCESSO
                sqs_client.delete_message(QueueUrl=INFER_TASK_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                print(f"✅ Inferenza {train_task_data['task_id']} completata con successo!\n")

                continue  # Ricomincia il ciclo

            # ==========================================
            # RIPOSO
            # ==========================================
            # Se entrambe le code sono vuote, respira un secondo per non intasare le API AWS
            time.sleep(2)

        except Exception as e:
            # Se esplode per RAM o errori vari, catturiamo l'eccezione in modo che il
            # demone Worker non muoia, ma NON cancelliamo il messaggio dalla coda!
            # Così il Visibility Timeout scadrà e il messaggio tornerà disponibile.
            print(f"❌ Errore durante l'elaborazione del task: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()