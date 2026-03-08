import boto3
import json
import math
import os
import time
import io
import threading
import botocore
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error

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

def conta_parti_modello(bucket, dataset, target_model):
    s3 = boto3.client('s3', region_name=AWS_REGION)
    prefix = f"models/{dataset}/{target_model}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # Ritorna la lista degli URI completi dei file .joblib
    return [f"s3://{bucket}/{obj['Key']}" for obj in resp.get('Contents', []) if obj['Key'].endswith('.joblib')]
    
# Thread in background che allunga la vita del messaggio del Client
def extend_client_sqs_visibility(queue_url, receipt_handle, stop_event):
    
    while not stop_event.is_set():
        # Dorme 2 minuti.
        stop_event.wait(120) 
        if not stop_event.is_set():
            try:
                # Estende il timeout di altri 5 minuti (300 secondi)
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=300 
                )
                print(" [MASTER HEARTBEAT] Timeout del Job esteso di 5 minuti.")
            except Exception as e:
                pass # Se dà errore, probabilmente è già stato cancellato
                
def scale_worker_infrastructure(num_workers):
    print(f" [ASG] Imposto la capacità desiderata a {num_workers} Worker...")
    asg_client.update_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
    )
    
    # Se stiamo spegnendo le macchine (num_workers == 0), usciamo.
    if num_workers == 0:
        return
        
    print(f" [ASG] Attendo l'avvio delle istanze per rinominarle (DRF-worker1, DRF-worker2...)...")
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    
    # Aspettiamo finché AWS non ha effettivamente creato le istanze (Polling)
    max_attesa = 24 # 24 * 5 sec = 2 minuti massimi di attesa
    for _ in range(max_attesa):
        time.sleep(5)
        # Cerchiamo le istanze che appartengono al nostro Auto Scaling Group
        # e che non siano nello stato 'terminated' o 'shutting-down'
        risposta = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:aws:autoscaling:groupName', 'Values': [ASG_NAME]},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
            ]
        )
        
        istanze_trovate = []
        for reservation in risposta.get('Reservations', []):
            for inst in reservation.get('Instances', []):
                istanze_trovate.append(inst['InstanceId'])
                
        # Se l'ASG ha partorito tutte le macchine richieste, procediamo col rinominarle
        if len(istanze_trovate) == num_workers:
            print(f" Trovate {len(istanze_trovate)} istanze. Applicazione dei nomi in corso...")
            
            # Rinominiamo le istanze con un ciclo for
            for i, instance_id in enumerate(istanze_trovate):
                nome_worker = f"DRF-worker{i+1}"
                ec2_client.create_tags(
                    Resources=[instance_id],
                    Tags=[{'Key': 'Name', 'Value': nome_worker}]
                )
            print(" Nomi applicati con successo su EC2!")
            break


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
        print(f" ERRORE CRITICO: Impossibile trovare il file {percorso_strategie}!")
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
        print(f" ATTENZIONE: Nessuna configurazione esatta per {task_type} con {num_workers} worker. Uso il default.")
        lista_strategie_corretta = [{"max_depth": "None", "max_features": "sqrt", "criterion": "gini"}]
    current_skip = 0

    print(f" [FAN-OUT] Suddivisione {num_trees_total} alberi in {num_workers} task di training...")
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
        "model_s3_uri": model_s3_uri  
    }
    sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
    print(f" [INFERENZA DISPACCIATA] Task {task_id} inviato alla coda di inferenza!")


def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


# SALVATAGGIO METRICHE SU S3
def save_metrics(dataset, n_workers, n_trees, strategy_name, train_time, inf_time, metrics_dict, config):
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    
    # Percorso dinamico del file CSV dei risultati su S3
    s3_key = f"results/{dataset}/{dataset}_results.csv"
    
    # Creiamo il dataframe con la nuova riga di risultati
    new_row_df = pd.DataFrame([{
        'Dataset': dataset, 
        'Workers': n_workers, 
        'Trees': n_trees, 
        'Strategy': strategy_name, 
        'Train_Time': round(train_time, 2), 
        'Infer_Time': round(inf_time, 2), 
        'Metrics': str(metrics_dict)
    }])

    try:
        # Tenta di scaricare il CSV esistente da S3
        obj = s3_client.get_object(Bucket=target_bucket, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
        # Se esiste, accoda la nuova riga (APPEND continuo)
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
        
    except botocore.exceptions.ClientError as e:
        # Se il file non esiste (errore 404 NoSuchKey), il file finale sarà solo la nuova riga
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_final = new_row_df
        else:
            print(f"!! Errore imprevisto di S3 durante il salvataggio: {e}")
            return
            
    # Salva il file aggiornato sovrascrivendolo su S3
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICHE] Risultati accodati permanentemente in: s3://{target_bucket}/{s3_key}")

def aggrega_e_valuta(job_id, dataset_name, risultati_inferenza_s3, num_workers, trees, train_time, infer_time):
    print("\n" + "=" * 50)
    print(" FASE DI AGGREGAZIONE E VALUTAZIONE FINALE")
    print("=" * 50)

    ml_handler = ModelFactory.get_model(dataset_name)
    task_type = getattr(ml_handler, 'task_type', 'classification')

    s3 = boto3.client('s3')
    config = load_config()

    # 1. SCARICA TUTTI I FILE .NPY DAI WORKER
    voti_list = []
    print(f" Scaricamento di {len(risultati_inferenza_s3)} file di risultati da S3...")

    for task_id, s3_uri in risultati_inferenza_s3.items():
        bucket, key = parse_s3_uri(s3_uri)
        local_path = f"/tmp/res_{task_id}.npy"
        s3.download_file(bucket, key, local_path)

        # Carica in memoria e aggiungi alla lista
        array_risultato = np.load(local_path)
        voti_list.append(array_risultato)
        os.remove(local_path)

    # 2. SCARICA LA GROUND TRUTH (I valori reali dal test set)
    target_col = ml_handler.target_column
    test_s3_key = config['paths'][dataset_name]['test']
    test_s3_uri = f"s3://{config.get('s3_bucket')}/{test_s3_key}"

    print(f" Lettura dei valori reali (Ground Truth) dalla colonna '{target_col}'...")
    df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
    y_true = df_test[target_col].values

    # 3. LA MAGIA DELL'AGGREGAZIONE
    forma_dati = voti_list[0].shape

    if len(forma_dati) == 2:
        # ---------------------------------------------------------
        # CLASSIFICAZIONE (La shape è N_righe x 2_colonne)
        # ---------------------------------------------------------
        print(" Rilevato task di Classificazione. Eseguo il conteggio dei voti...")

        totale_voti = np.sum(voti_list, axis=0)
        voti_0 = totale_voti[:, 0]
        voti_1 = totale_voti[:, 1]
        y_prob = voti_1 / (voti_0 + voti_1)
        y_pred_class = np.argmax(totale_voti, axis=1)

        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred_class)

        print(f"\n RISULTATI GLOBALI (Random Forest Distribuita):")
        print(f"   -> ROC-AUC:   {auc:.4f}")
        print(f"   -> Accuracy:  {acc:.4f}")
        
        # Dizionario metriche da salvare
        metrics_dict = {'ROC-AUC': round(auc, 4), 'Accuracy': round(acc, 4)}

    else:
        # ---------------------------------------------------------
        # REGRESSIONE (La shape è N_righe array 1D)
        # ---------------------------------------------------------
        print(" Rilevato task di Regressione. Calcolo la media globale...")

        y_pred = np.mean(voti_list, axis=0)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred) # <--- NUOVO CALCOLO MAE

        print(f"\n RISULTATI GLOBALI (Random Forest Distribuita):")
        print(f" -> RMSE:      {rmse:.4f}")
        print(f" -> MAE:       {mae:.4f}") # <--- NUOVA STAMPA MAE
        print(f" -> R2 Score:  {r2:.4f}")
        
        # Dizionario metriche da salvare con MAE aggiunto
        metrics_dict = {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2 Score': round(r2, 4)}

    print("=" * 50 + "\n")

    # 4. SALVATAGGIO FINALE SU S3 (Spostato qui in fondo, quando le metriche esistono!)
    save_metrics(
        dataset=dataset_name, 
        n_workers=num_workers, 
        n_trees=trees, 
        strategy_name="SQS_Async", 
        train_time=train_time, 
        inf_time=infer_time, 
        metrics_dict=metrics_dict, 
        config=config
    )


def get_job_state(job_id):
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table('JobStatus')
    try:
        response = table.get_item(Key={'job_id': job_id})
        if 'Item' in response:
            return set(response['Item'].get('completed_train', [])), response['Item'].get('completed_infer', {})
    except Exception:
        pass
    return set(), {} # Se non esiste, il job è nuovo

def update_job_state(job_id, completed_train_set, completed_infer_dict):
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table('JobStatus')
    table.put_item(Item={
        'job_id': job_id,
        'completed_train': list(completed_train_set), # Dynamo non accetta i Set di Python direttamente
        'completed_infer': completed_infer_dict
    })

# ==========================================
# CICLO PRINCIPALE DEL MASTER (EVENT LOOP REATTIVO)
# ==========================================
def main():
    print(" Master Node Avviato. In attesa del Client...")

    while True:
        # 1. Attesa del comando dal Client
        response = sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]

            receipt_handle = client_msg['ReceiptHandle']
            
            job_data = json.loads(client_msg['Body'])

            # Estraiamo il Job ID parlante creato dal Client (es. job_taxi_200trees_...)
            # Usiamo l'ID di SQS solo come "piano B" se per qualche motivo manca nel JSON
            job_id = job_data.get('job_id', client_msg['MessageId'])
            job_data['job_id'] = job_id
            dataset = job_data['dataset']

            mode = job_data.get('mode', 'train')

            print(f"\n" + "=" * 50)
            print(f" INIZIO PIPELINE ASINCRONA PER JOB: {job_id}")
            print("=" * 50)

            # --- INIZIO HEARTBEAT DEL MASTER ---
            stop_event_master = threading.Event()
            heartbeat_thread_master = threading.Thread(
                target=extend_client_sqs_visibility, 
                args=(CLIENT_QUEUE_URL, receipt_handle, stop_event_master)
            )
            heartbeat_thread_master.start()
            # -----------------------------------

            try:
                # RAMO A: TRAINING + TEST BULK
                if mode == 'train':
                    num_workers = job_data['num_workers']
                    start_totale = time.time() 
                    start_train = time.time()
                    tempo_training = 0
                    tempo_inferenza = 0
        
                    scale_worker_infrastructure(num_workers)
        
                    # STATO DELLA PIPELINE SU DYNAMODB
                    train_completati, risultati_inferenza_s3 = get_job_state(job_id)
                    
                    if len(train_completati) == 0:
                        generate_initial_training_tasks(job_data)
                    else:
                        print(f" [RECOVERY] Trovati {len(train_completati)} task di training già completati su DynamoDB!")
        
                    print("\n [EVENT LOOP] Master in ascolto attivo delle risposte...\n")
        
                    while len(risultati_inferenza_s3) < num_workers:
                        # --- ASCOLTO TRAINING ---
                        res_train = sqs_client.receive_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
                        if 'Messages' in res_train:
                            for msg in res_train['Messages']:
                                train_resp = json.loads(msg['Body'])
                                task_id = train_resp['task_id']
        
                                if task_id not in train_completati:
                                    train_completati.add(task_id)
                                    print(f" [TRAIN FATTO] {task_id} ha finito l'addestramento.")
                                    update_job_state(job_id, train_completati, risultati_inferenza_s3)
                                    generate_inference_tasks(job_id, train_resp, dataset)
        
                                sqs_client.delete_message(QueueUrl=TRAIN_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                        
                        if len(train_completati) == num_workers and tempo_training == 0:
                            tempo_training = time.time() - start_train
                            start_infer = time.time() 
        
                        # --- ASCOLTO INFERENZA ---
                        res_infer = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
                        if 'Messages' in res_infer:
                            for msg in res_infer['Messages']:
                                body = json.loads(msg['Body'])
                                task_id = body['task_id']
                                voti_s3_uri = body['s3_voti_uri'] # Nel bulk è la stringa URI
        
                                if task_id not in risultati_inferenza_s3:
                                    risultati_inferenza_s3[task_id] = voti_s3_uri
                                    print(f" [INFERENZA FATTA] {task_id} completato! ({len(risultati_inferenza_s3)}/{num_workers})")
                                    update_job_state(job_id, train_completati, risultati_inferenza_s3)
        
                                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
        
                    if tempo_inferenza == 0:
                        tempo_inferenza = time.time() - start_infer
        
                    print("\n Tutti i Worker hanno completato la pipeline end-to-end!")
                    scale_worker_infrastructure(0)
                    print(" Calcolo delle metriche finali in corso...")
                    tempo_totale = time.time() - start_totale
        
                    try:
                        aggrega_e_valuta(job_id, dataset, risultati_inferenza_s3, num_workers, job_data['num_trees'], tempo_training, tempo_inferenza)
                    except Exception as e:
                        print(f" Errore durante l'aggregazione finale: {e}")
                        
                    print(f" TEMPI -> Train: {tempo_training:.2f}s | Infer: {tempo_inferenza:.2f}s | Totale Sistema: {tempo_totale:.2f}s")


                # RAMO B: INFERENZA REAL-TIME SINGOLA
                elif mode == 'infer':
                    start_totale = time.time()
                    target_model = job_data['target_model']
                    tuple_data = job_data['tuple_data']
                    
                    bucket = load_config().get("s3_bucket")
                    modelli_s3_uris = conta_parti_modello(bucket, dataset, target_model)
                    num_workers = len(modelli_s3_uris)
                    
                    print(f"📦 Modello '{target_model}' diviso in {num_workers} parti. Avvio Worker...")
                    scale_worker_infrastructure(num_workers)
                    
                    for i, uri in enumerate(modelli_s3_uris):
                        task_id = f"task_infer_rt_{i+1}"
                        infer_task = {
                            "job_id": job_id, "task_id": task_id, "dataset": dataset,
                            "model_s3_uri": uri, "tuple_data": tuple_data
                        }
                        sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
                        
                    voti_ricevuti = []
                    while len(voti_ricevuti) < num_workers:
                        res = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, WaitTimeSeconds=2)
                        if 'Messages' in res:
                            for msg in res['Messages']:
                                body = json.loads(msg['Body'])
                                res_dati = body['s3_voti_uri'] 
                                
                                # Verifica che sia la risposta strutturata real-time (dict) e non il bulk (stringa)
                                if isinstance(res_dati, dict) and res_dati.get("tipo") == "singolo":
                                    voti_ricevuti.append(res_dati['valore'])
                                    print(f"   -> Ricevuta predizione locale da un worker: {res_dati['valore']}")
                                    
                                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                                
                    scale_worker_infrastructure(0)
                    
                    # Consenso distribuito (Aggregation)
                    ml_handler = ModelFactory.get_model(dataset)
                    if ml_handler.task_type == 'classification':
                        predizione_finale = max(set(voti_ricevuti), key=voti_ricevuti.count) # Moda / Maggioranza
                        tipo_task = "Classificazione (Voto Maggioranza)"
                    else:
                        predizione_finale = sum(voti_ricevuti) / len(voti_ricevuti) # Media
                        tipo_task = "Regressione (Media)"
                        
                    tempo_totale = time.time() - start_totale
                    print("\n" + "=" * 60)
                    print(f" RISULTATO FINALE INFERENZA ({tipo_task}): {predizione_finale}")
                    print(f" Tempo di latenza sistema: {tempo_totale:.2f}s")
                    print("=" * 60 + "\n")

            finally:
                # --- FINE HEARTBEAT DEL MASTER ---
                stop_event_master.set()
                heartbeat_thread_master.join()
                
            sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=receipt_handle)
            print(f" JOB {job_id} COMPLETATO E CHIUSO.\n")

if __name__ == "__main__":
    main()
