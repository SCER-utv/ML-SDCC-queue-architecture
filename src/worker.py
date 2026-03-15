import boto3
import json
import time
import os
import threading
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.model.model_factory import ModelFactory
from src.utils.config import load_config

# ==========================================
# CONFIGURAZIONE DINAMICA DA JSON
# ==========================================
config = load_config()

AWS_REGION = config.get("aws_region")
CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
TRAIN_TASK_QUEUE = config["sqs_queues"]["train_task"]
TRAIN_RESPONSE_QUEUE = config["sqs_queues"]["train_response"]
INFER_TASK_QUEUE = config["sqs_queues"]["infer_task"]
INFER_RESPONSE_QUEUE = config["sqs_queues"]["infer_response"]

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Thread in background che allunga la vita del messaggio ogni 2 minuti
def extend_sqs_visibility(queue_url, receipt_handle, stop_event):
    
    while not stop_event.is_set():
        # Dorme 2 minuti. Se nel frattempo il training finisce (stop_event viene settato),
        # il ciclo si interrompe prima di fare un'altra chiamata ad AWS.

        stop_event.wait(120) 
        if not stop_event.is_set():
            try:
                # Estende il timeout di altri 5 minuti (300 secondi)
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=300 
                )
                print(" [HEARTBEAT] Timeout del messaggio SQS esteso di 5 minuti.")
            except Exception as e:
                print(f" [HEARTBEAT] Errore estensione SQS (forse già cancellato): {e}")

# Funzione helper per estrarre bucket e key da un URI s3://
def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


# ==========================================
# CORE LOGIC: TRAINING
# ==========================================
def train(train_task_data, receipt_handle):
    job_id = train_task_data['job_id']
    task_id = train_task_data['task_id']
    dataset_uri = train_task_data['dataset_s3_path']

    # --- INIZIO HEARTBEAT ---
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=extend_sqs_visibility, 
        args=(TRAIN_TASK_QUEUE, receipt_handle, stop_event)
    )
    heartbeat_thread.start()
    # ------------------------

    print(f" [TRAIN] Avvio {task_id}. Lettura di {train_task_data['num_rows']} righe da S3...")

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

    try: 
        # 3. SALVATAGGIO E UPLOAD
        local_model_path = f"/tmp/{task_id}_{job_id}.joblib"
        joblib.dump(rf, local_model_path)

        bucket, _ = parse_s3_uri(dataset_uri)
        dataset_name = train_task_data['dataset']
        s3_key = f"models/{dataset_name}/{job_id}/task_{task_id}.joblib"

        print("   -> Upload del modello su S3 in corso...")
        s3_client.upload_file(local_model_path, bucket, s3_key)

        os.remove(local_model_path)  # Pulizia disco locale
        return f"s3://{bucket}/{s3_key}"

    finally:
        # --- FINE HEARTBEAT ---
        # Qualsiasi cosa succeda (successo o errore), fermiamo il thread dell'Heartbeat
        stop_event.set()
        heartbeat_thread.join()


# ==========================================
# CORE LOGIC: INFERENZA
# ==========================================
def esegui_inferenza(infer_task_data, receipt_handle): 
    job_id = infer_task_data['job_id']
    task_id = infer_task_data['task_id']
    model_s3_uri = infer_task_data['model_s3_uri']

    # --- INIZIO HEARTBEAT ---
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=extend_sqs_visibility, 
        args=(INFER_TASK_QUEUE, receipt_handle, stop_event) # <--- Usa INFER_TASK_QUEUE
    )
    heartbeat_thread.start()
    # ------------------------

    try:
        print(f" [INFER] Avvio inferenza {task_id}. Scaricamento modello...")
        bucket, model_key = parse_s3_uri(model_s3_uri)

        # 1. SCARICA IL MODELLO DA S3
        local_model_path = f"/tmp/model_{job_id}_{task_id}.joblib"
        s3_client.download_file(bucket, model_key, local_model_path)
        rf = joblib.load(local_model_path)

        # CASO 1: INFERENZA SU SINGOLA TUPLA (Real-time)
        if 'tuple_data' in infer_task_data:
            print(f" [INFER] Predizione su singola tupla in corso...")
            # Formattiamo i dati per sklearn (1 riga, N colonne)
            dati = np.array(infer_task_data['tuple_data']).reshape(1, -1)
            
            # Usiamo direttamente la predizione del modello scikit-learn
            all_pred = [float(tree.predict(dati)[0]) for tree in rf.estimators_]
            os.remove(local_model_path)
            # Non salviamo nulla su S3, ritorniamo il dizionario direttamente al Master!
            return {"tipo": "singolo", "valore": all_pred}

        # CASO 2: INFERENZA BULK DA S3 (Test set per metriche classiche)
        else:
            print(f" [INFER] Inferenza su Dataset Intero in corso...")
            test_dataset_uri = infer_task_data['test_dataset_uri']
            df_test = pd.read_csv(test_dataset_uri)
            
            ml_handler = ModelFactory.get_model(dataset_name=infer_task_data['dataset'])
            
            print(f"   -> Calcolo previsioni in corso...")
            start_time = time.time()
            risultati_numpy = ml_handler.process_and_predict(rf, df_test)
            print(f"   -> Previsioni completate in {time.time() - start_time:.2f} secondi.")

            # SALVATAGGIO IN .NPY COMPRESSO E UPLOAD
            local_npy_path = f"/tmp/results_{job_id}_{task_id}.npy"
            np.save(local_npy_path, risultati_numpy)

            dataset_name = infer_task_data['dataset']
            s3_voti_key = f"results/{dataset_name}/{job_id}/task_{task_id}.npy"
            s3_client.upload_file(local_npy_path, bucket, s3_voti_key)

            os.remove(local_model_path)
            os.remove(local_npy_path)
            return {"tipo": "bulk", "valore": f"s3://{bucket}/{s3_voti_key}"}

    finally:
        stop_event.set()
        heartbeat_thread.join()

# ==========================================
# EVENT LOOP: PRIORITY POLLING
# ==========================================
def main():
    print(" Worker Node Avviato e pronto a ricevere ordini...")

    while True:
        # Teniamo traccia del messaggio corrente per la gestione degli errori
        current_queue = None
        current_receipt = None
        
        try:
            # PRIORITÀ 1: TRAINING
            res_train = sqs_client.receive_message(
                QueueUrl=TRAIN_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_train:
                msg = res_train['Messages'][0]
                current_queue = TRAIN_TASK_QUEUE
                current_receipt = msg['ReceiptHandle']
                
                train_task_data = json.loads(msg['Body'])

                # Esegue il lavoro pesante
                s3_model_uri = train(train_task_data, current_receipt)

                # Risponde al Master
                train_resp = {
                    "job_id": train_task_data['job_id'],
                    "task_id": train_task_data['task_id'],
                    "s3_model_uri": s3_model_uri
                }
                sqs_client.send_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MessageBody=json.dumps(train_resp))

                # CANCELLA IL MESSAGGIO SOLO DOPO IL SUCCESSO (Fault Tolerance)
                sqs_client.delete_message(QueueUrl=TRAIN_TASK_QUEUE, ReceiptHandle=current_receipt)
                print(f" Training {train_task_data['task_id']} completato con successo!\n")

                # FONDAMENTALE: Torna su e ricontrolla la coda di training!
                continue 

            # PRIORITÀ 2: INFERENZA
            # Ci arriva SOLO se la coda di training non ha restituito messaggi.
            res_infer = sqs_client.receive_message(
                QueueUrl=INFER_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_infer:
                msg = res_infer['Messages'][0]
                current_queue = INFER_TASK_QUEUE
                current_receipt = msg['ReceiptHandle']
                
                infer_task_data = json.loads(msg['Body'])

                # Esegue il lavoro pesante
                s3_voti_uri = esegui_inferenza(infer_task_data, current_receipt)

                # Risponde al Master
                risposta = {
                    "job_id": infer_task_data['job_id'],
                    "task_id": infer_task_data['task_id'],
                    "s3_voti_uri": s3_voti_uri
                }
                sqs_client.send_message(QueueUrl=INFER_RESPONSE_QUEUE, MessageBody=json.dumps(risposta))

                # CANCELLA IL MESSAGGIO SOLO DOPO IL SUCCESSO
                sqs_client.delete_message(QueueUrl=INFER_TASK_QUEUE, ReceiptHandle=current_receipt)
                print(f" Inferenza {infer_task_data['task_id']} completata con successo!\n")
                
                continue  # Ricomincia il ciclo

            # Se entrambe le code sono vuote, respira 2 secondi per non intasare le API AWS
            time.sleep(2)

        except Exception as e:
            
            # GESTIONE MORTE LENTA / OOM
            print(f" \n[FAULT TOLERANCE] Rilevato errore critico nel Worker: {e}")
            
            # Se abbiamo un messaggio "in mano", lo rigettiamo istantaneamente nella coda
            if current_queue and current_receipt:
                try:
                    print(" [FAULT TOLERANCE] Eseguo Rilascio Immediato (VisibilityTimeout=0) per riassegnazione rapida...")
                    sqs_client.change_message_visibility(
                        QueueUrl=current_queue,
                        ReceiptHandle=current_receipt,
                        VisibilityTimeout=0
                    )
                except Exception as inner_e:
                    print(f" [FAULT TOLERANCE] Impossibile rilasciare il messaggio (forse già scaduto): {inner_e}")
            
            # Riposo per evitare loop di riavvio se l'errore è infrastrutturale
            time.sleep(10)


if __name__ == "__main__":
    main()
