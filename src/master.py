import boto3
import json
import math
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

from src.model.model_factory import ModelFactory
from src.utils.config import load_config

# ==========================================
# CONFIGURAZIONE CODE SQS (Le 5 Code)
# ==========================================
AWS_REGION = 'us-east-1'
CLIENT_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/248593862537/JobRequestQueue.fifo'

TRAIN_TASK_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/train-task-queue'
TRAIN_RESPONSE_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/train-response-queue'

INFER_TASK_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/infer-task-queue'
INFER_RESPONSE_QUEUE = 'https://sqs.us-east-1.amazonaws.com/248593862537/infer-response-queue'

ASG_NAME = 'DRF-Worker-ASG-new'

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
asg_client = boto3.client('autoscaling', region_name=AWS_REGION)


# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def scale_worker_infrastructure(num_workers):
    print(f"📈 [ASG] Imposto la capacità desiderata a {num_workers} Worker...")
    asg_client.update_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
    )


# [NUOVO METODO ZERO-COPY] Interroga S3 senza scaricare il file
def _get_total_rows_s3_select(bucket, key):
    print(f" [S3 Select] Lancio query SQL 'SELECT count(*)' su s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    try:
        resp = s3.select_object_content(
            Bucket=bucket, Key=key,
            ExpressionType='SQL', Expression='SELECT count(*) FROM S3Object',
            InputSerialization={'CSV': {'FileHeaderInfo': 'USE', 'AllowQuotedRecordDelimiter': False}},
            OutputSerialization={'CSV': {}}
        )
        for event in resp['Payload']:
            if 'Records' in event:
                total_rows = int(event['Records']['Payload'].decode('utf-8').strip())
                print(f" [S3 Select] Trovate {total_rows} righe!")
                return total_rows
        return 0
    except Exception as e:
        print(f"[ERRORE S3 Select]: {e}")
        raise e


def generate_initial_training_tasks(job_data):
    config = load_config()
    num_workers = job_data['num_workers']
    num_trees_total = job_data['num_trees']
    dataset = job_data['dataset']
    job_id = job_data['job_id']

    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    dataset_paths = config['paths'][dataset]

    train_s3_key = dataset_paths['train']
    train_s3_uri = f"s3://{target_bucket}/{train_s3_key}"


    # 1. Conta il numero di righe all'interno del dataset (fatto per garantire generalità per ogni dataset)
    try:
        total_rows = _get_total_rows_s3_select(target_bucket, train_s3_key)
    except Exception:
        print("Fallimento critico S3 Select. Esco.")
        return

    rows_per_worker = total_rows // num_workers
    remainder_rows = total_rows % num_workers

    trees_per_worker = math.floor(num_trees_total / num_workers)
    trees_remainder = num_trees_total % num_workers

    percorso_strategie = "config/worker_strategies.json"
    try:
        with open(percorso_strategie, 'r') as f:
            all_strategies = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE CRITICO: Impossibile trovare il file {percorso_strategie}!")
        all_strategies = {}

        # 2. Capiamo se il dataset è classificazione o regressione
    ml_handler = ModelFactory.get_model(dataset)
    task_type = getattr(ml_handler, 'task_type', 'classification')

    # 3. Estrazione sicura (zero crash)
    str_num = str(num_workers)
    blocco_task = all_strategies.get(task_type, {})
    lista_strategie_corretta = blocco_task.get(str_num, [])

    # 4. Rete di sicurezza (Fallback)
    if not lista_strategie_corretta:
        print(f"⚠️ ATTENZIONE: Nessuna configurazione esatta per {task_type} con {num_workers} worker. Uso il default.")
        lista_strategie_corretta = [{"max_depth": "None", "max_features": "sqrt", "criterion": "gini"}]
    current_skip = 0

    print(f"🔀 [FAN-OUT] Suddivisione {num_trees_total} alberi in {num_workers} task di training...")
    for i in range(num_workers):
        trees = trees_per_worker + (1 if i < trees_remainder else 0)
        n_rows = rows_per_worker + (remainder_rows if i == num_workers - 1 else 0)
        conf = lista_strategie_corretta[i % len(lista_strategie_corretta)]

        raw_depth = conf['max_depth']

        if(raw_depth == "None"):
            max_depth = None
        else:
            try:
                max_depth = int(raw_depth)
            except (ValueError, TypeError):
                max_depth = None


        raw_features = conf['max_features']
        if raw_features in ["sqrt", "log2"]:
            max_features = raw_features  # Va bene così, è una stringa valida per sklearn
        elif raw_features in ["None", "null", None]:
            max_features = None
        else:
            try:
                # Prova a convertirlo in numero (es. la stringa "0.2" diventa il float 0.2)
                val_float = float(raw_features)

                # Se è un numero intero tondo (es. 10.0), scikit-learn preferisce l'intero (10)
                if val_float.is_integer():
                    max_features = int(val_float)
                else:
                    max_features = val_float  # È una frazione, es. 0.2 o 0.3
            except (ValueError, TypeError):
                # Fallback di sicurezza se arriva roba incomprensibile
                max_features = "sqrt"


        task_payload = {
            "job_id": job_id,
            "task_id": f"task_{i + 1}",
            "seed": i * 1000,
            "dataset": dataset,
            "dataset_s3_path": train_s3_uri,
            "trees": trees,
            "max_depth": max_depth,
            "max_features": max_features,
            "criterion": conf['criterion'],
            "skip_rows": current_skip,
            "num_rows": n_rows
        }

        current_skip += n_rows
        sqs_client.send_message(QueueUrl=TRAIN_TASK_QUEUE, MessageBody=json.dumps(task_payload))
        print(f"   -> Inviato {task_payload['task_id']} ({trees} alberi) in coda di addestramento.")


def generate_inference_tasks(job_id, train_resp, dataset):
    task_id = train_resp['task_id']
    model_s3_uri = train_resp['s3_model_uri']

    config = load_config()
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    dataset_paths = config['paths'][dataset]

    test_s3_key = dataset_paths['test']
    test_s3_uri = f"s3://{target_bucket}/{test_s3_key}"


    infer_task = {
        "job_id": job_id,
        "task_id": task_id,
        "dataset": dataset,
        "test_dataset_uri": test_s3_uri,
        "model_s3_uri": model_s3_uri  # Il modello appena creato!
    }
    sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
    print(f"   ⚡ [INFERENZA DISPACCIATA] Task {task_id} inviato alla coda di inferenza!")


def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


def aggrega_e_valuta(job_id, dataset_name, risultati_inferenza_s3):
    print("\n" + "=" * 50)
    print("📊 FASE DI AGGREGAZIONE E VALUTAZIONE FINALE")
    print("=" * 50)

    s3 = boto3.client('s3')
    config = load_config()

    # 1. SCARICA TUTTI I FILE .NPY DAI WORKER
    voti_list = []
    print(f"📥 Scaricamento di {len(risultati_inferenza_s3)} file di risultati da S3...")

    for task_id, s3_uri in risultati_inferenza_s3.items():
        bucket, key = parse_s3_uri(s3_uri)
        local_path = f"/tmp/res_{task_id}.npy"
        s3.download_file(bucket, key, local_path)

        # Carica in memoria e aggiungi alla lista
        array_risultato = np.load(local_path)
        voti_list.append(array_risultato)
        os.remove(local_path)

    # 2. SCARICA LA GROUND TRUTH (I valori reali dal test set)
    # Usiamo la Factory per sapere qual è la colonna target!
    ml_handler = ModelFactory.get_model(dataset_name)
    target_col = ml_handler.target_column

    test_s3_key = config['paths'][dataset_name]['test']
    test_s3_uri = f"s3://{config.get('s3_bucket')}/{test_s3_key}"

    print(f"🎯 Lettura dei valori reali (Ground Truth) dalla colonna '{target_col}'...")
    # Leggiamo SOLO la colonna target per non saturare la RAM del Master
    df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
    y_true = df_test[target_col].values

    # 3. LA MAGIA DELL'AGGREGAZIONE (Riconoscimento automatico del task)
    forma_dati = voti_list[0].shape

    if len(forma_dati) == 2:
        # ---------------------------------------------------------
        # CLASSIFICAZIONE (La shape è N_righe x 2_colonne)
        # ---------------------------------------------------------
        print("🧮 Rilevato task di Classificazione. Eseguo il conteggio dei voti...")

        # Sommiamo tutte le matrici. Se abbiamo 4 worker, sommiamo i voti di tutti!
        # Risultato: matrice N x 2 con il totale dei voti globali per riga.
        totale_voti = np.sum(voti_list, axis=0)

        # La probabilità della classe 1 è data dalla formula: $Prob(1) = \frac{Voti_1}{Voti_0 + Voti_1}$
        voti_0 = totale_voti[:, 0]
        voti_1 = totale_voti[:, 1]
        y_prob = voti_1 / (voti_0 + voti_1)

        # La classe secca è l'indice della colonna col valore massimo (0 o 1)
        y_pred_class = np.argmax(totale_voti, axis=1)

        # Metriche
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred_class)

        print(f"\n🏆 RISULTATI GLOBALI (Random Forest Distribuita):")
        print(f"   -> ROC-AUC:   {auc:.4f}")
        print(f"   -> Accuracy:  {acc:.4f}")

    else:
        # ---------------------------------------------------------
        # REGRESSIONE (La shape è N_righe array 1D)
        # ---------------------------------------------------------
        print("📈 Rilevato task di Regressione. Calcolo la media globale...")

        # Per la regressione, la previsione finale della foresta è la media
        # delle previsioni di tutti gli alberi (e quindi la media delle medie dei worker).
        y_pred = np.mean(voti_list, axis=0)

        # Metriche
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"\n🏆 RISULTATI GLOBALI (Random Forest Distribuita):")
        print(f"   -> RMSE:      {rmse:.4f}")
        print(f"   -> R2 Score:  {r2:.4f}")

    print("=" * 50 + "\n")


# ==========================================
# CICLO PRINCIPALE DEL MASTER (EVENT LOOP REATTIVO)
# ==========================================
def main():
    print("🚀 Master Node Avviato. In attesa del Client...")

    while True:
        # 1. Attesa del comando dal Client
        response = sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]
            job_data = json.loads(client_msg['Body'])

            # Usiamo l'ID del messaggio SQS come Job ID!
            job_id = client_msg['MessageId']
            job_data['job_id'] = job_id
            dataset = job_data['dataset']

            num_workers = job_data['num_workers']

            print(f"\n" + "=" * 50)
            print(f"🎬 INIZIO PIPELINE ASINCRONA PER JOB: {job_id}")
            print("=" * 50)

            # 2. Accendi le macchine
            scale_worker_infrastructure(num_workers)

            # 3. Genera tutti i task di Training
            generate_initial_training_tasks(job_data)

            # --- STATO DELLA PIPELINE ---
            train_completati = set()  # Tiene traccia di chi ha finito il training
            risultati_inferenza_s3 = {}  # task_id -> s3_uri del file .npy

            print("\n🔄 [EVENT LOOP] Master in ascolto attivo delle risposte...\n")

            # 4. IL CUORE DELL'ASINCRONIA: Ciclo finché non finiscono tutte le INFERENZE
            while len(risultati_inferenza_s3) < num_workers:

                # --- ASCOLTO RISPOSTE TRAINING ---
                # Usiamo un WaitTime breve (es. 2 secondi) per non bloccare troppo il loop
                res_train = sqs_client.receive_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MaxNumberOfMessages=10,
                                                       WaitTimeSeconds=2)
                if 'Messages' in res_train:
                    for msg in res_train['Messages']:
                        train_resp = json.loads(msg['Body'])
                        task_id = train_resp['task_id']


                        if task_id not in train_completati:
                            train_completati.add(task_id)
                            print(f"   ✅ [TRAIN FATTO] {task_id} ha finito l'addestramento.")
                            generate_inference_tasks(job_id, train_resp, dataset)

                        ##salva stato e poi cancella, DA IMPLEMENTARE
                        ########################################################################
                        # Cancello il messaggio di risposta training
                        sqs_client.delete_message(QueueUrl=TRAIN_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

                # --- ASCOLTO RISPOSTE INFERENZA ---
                res_infer = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, MaxNumberOfMessages=10,
                                                       WaitTimeSeconds=2)
                if 'Messages' in res_infer:
                    for msg in res_infer['Messages']:
                        body = json.loads(msg['Body'])
                        task_id = body['task_id']
                        voti_s3_uri = body['s3_voti_uri']

                        if task_id not in risultati_inferenza_s3:
                            risultati_inferenza_s3[task_id] = voti_s3_uri
                            print(
                                f"   🔮 [INFERENZA FATTA] {task_id} ha completato le previsioni! ({len(risultati_inferenza_s3)}/{num_workers})")

                        # Cancello il messaggio di risposta inferenza
                        sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

            # 5. USCITI DAL LOOP: Tutti hanno finito sia Training che Inferenza!
            print("\n🎉 Tutti i Worker hanno completato la pipeline end-to-end!")





            # 6. SCALE-TO-ZERO IMMEDIATO (per smettere di pagare mentre calcoliamo la metrica)
            scale_worker_infrastructure(0)

            # 7. AGGREGAZIONE FINALE E CALCOLO METRICHE
            print("📊 Calcolo delle metriche finali in corso (ROC-AUC e Accuracy)...")

            try:
                aggrega_e_valuta(job_id, dataset, risultati_inferenza_s3)
            except Exception as e:
                print(f"❌ Errore durante l'aggregazione finale: {e}")


            # 8. PULIZIA
            sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=client_msg['ReceiptHandle'])
            print(f"🏁 JOB {job_id} COMPLETATO E CHIUSO.\n")


if __name__ == "__main__":
    main()