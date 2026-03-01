import boto3
import json
import time
import math

# ==========================================
# CONFIGURAZIONE INIZIALE
# ==========================================
AWS_REGION = 'us-east-1'  # Inserisci la tua region
CLIENT_QUEUE_URL = ''
WORKER_TRAINING_QUEUE_URL = ''
WORKER_RESPONSE_QUEUE_URL = ''
ASG_NAME = 'RandomForest-Worker-ASG'

# Inizializzazione dei client Boto3
sqs_client = boto3.client('sqs', region_name=AWS_REGION)
asg_client = boto3.client('autoscaling', region_name=AWS_REGION)


# ==========================================
# FASE 1: ASCOLTO DEL CLIENT
# ==========================================
def poll_client_queue():
    """
    Fa polling sulla coda del Client.
    Ritorna il payload del Job e il ReceiptHandle per cancellare il messaggio alla fine.
    """
    print("🎧 Master in ascolto di nuovi Job sulla coda Client...")
    response = sqs_client.receive_message(
        QueueUrl=CLIENT_QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20  # Long Polling (risparmia soldi ed è più reattivo)
    )

    if 'Messages' in response:
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']

        # Il client ci manda un JSON. Lo trasformiamo in dizionario Python
        job_data = json.loads(message['Body'])
        print(f"📥 Ricevuto nuovo Job di addestramento: {job_data['job_id']}")
        return job_data, receipt_handle

    return None, None


# ==========================================
# FASE 2: SCALING DELL'INFRASTRUTTURA
# ==========================================
def scale_worker_infrastructure(num_workers_richiesti):
    """
    Comunica con l'Auto Scaling Group per accendere il numero esatto di macchine.
    """
    print(f"📈 Richiesta Auto Scaling: porto la flotta a {num_workers_richiesti} Worker...")

    asg_client.update_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME,
        MinSize=0,
        DesiredCapacity=num_workers_richiesti,
        MaxSize=10  # Limite di sicurezza per il budget!
    )
    print("✅ Comando inviato ad AWS. I Worker si stanno avviando in background.")


# ==========================================
# FASE 3: GENERAZIONE DEI TASK (FAN-OUT)
# ==========================================
def generate_worker_tasks(job_data):
    """
    Divide il lavoro e inserisce i Task individuali nella coda dei Worker.
    """
    num_workers = job_data['num_workers']
    num_trees_total = job_data['num_trees']
    dataset_uri = job_data['dataset_uri']
    job_id = job_data['job_id']

    # Quanti alberi deve fare ogni Worker? (Gestiamo anche il resto della divisione)
    trees_per_worker = math.floor(num_trees_total / num_workers)
    trees_remainder = num_trees_total % num_workers

    print(f"🔀 Suddivisione di {num_trees_total} alberi in {num_workers} task...")

    for i in range(num_workers):
        # Assegniamo gli alberi extra al primo worker se la divisione non è perfetta
        trees_for_this_task = trees_per_worker + (1 if i < trees_remainder else 0)

        task_payload = {
            "job_id": job_id,
            "task_id": f"task_{i + 1}",
            "dataset_uri": dataset_uri,
            "shard_index": i,  # Utile se i dati su S3 sono già divisi in partizioni
            "trees_to_train": trees_for_this_task
        }

        # Invio del singolo task alla coda di Training
        sqs_client.send_message(
            QueueUrl=WORKER_TRAINING_QUEUE_URL,
            MessageBody=json.dumps(task_payload)
        )
        print(f"   -> Inviato {task_payload['task_id']} ({trees_for_this_task} alberi) sulla coda Worker.")


# ==========================================
# FASE 4: ATTESA DELLE RISPOSTE (FAN-IN)
# ==========================================
def wait_for_worker_responses(job_id, num_tasks_expected):
    """
    Fa polling sulla coda di ritorno finché non raccoglie i risultati di tutti i task.
    """
    print(f"⏳ In attesa di {num_tasks_expected} risposte sulla coda per il Job {job_id}...")

    # Usiamo un dizionario per evitare i doppioni (SQS potrebbe inviare messaggi doppi)
    risposte_ricevute = {}

    while len(risposte_ricevute) < num_tasks_expected:
        response = sqs_client.receive_message(
            QueueUrl=WORKER_RESPONSE_QUEUE_URL,
            MaxNumberOfMessages=10,  # Leggiamo fino a 10 messaggi alla volta per fare prima
            WaitTimeSeconds=20  # Long Polling
        )

        if 'Messages' in response:
            for msg in response['Messages']:
                body = json.loads(msg['Body'])

                # Verifichiamo che il messaggio appartenga al Job corrente
                if body.get('job_id') == job_id:
                    task_id = body['task_id']
                    s3_model_uri = body['s3_model_uri']  # Dove il worker ha salvato i suoi alberi

                    if task_id not in risposte_ricevute:
                        risposte_ricevute[task_id] = s3_model_uri
                        print(
                            f"   ✅ Ricevuto completamento per {task_id}! ({len(risposte_ricevute)}/{num_tasks_expected})")

                    # Il Master cancella il messaggio: il Worker ha fatto il suo dovere!
                    sqs_client.delete_message(
                        QueueUrl=WORKER_RESPONSE_QUEUE_URL,
                        ReceiptHandle=msg['ReceiptHandle']
                    )

        # Una piccola pausa per non inondare le API di AWS di richieste
        time.sleep(1)

    print("🎉 Fan-In completato! Tutti i Worker hanno terminato il lavoro.")
    return list(risposte_ricevute.values())  # Ritorna la lista di tutti gli URI su S3


# ==========================================
# CICLO PRINCIPALE DEL MASTER
# ==========================================
def main():
    print("🚀 Master Node Avviato e pronto.")

    while True:
        # 1. Prende il messaggio dal Client
        job_data, receipt_handle = poll_client_queue()

        if job_data:
            job_id = job_data['job_id']
            num_workers = job_data['num_workers']

            # 2. Scala l'infrastruttura accendendo le macchine
            scale_worker_infrastructure(num_workers)

            # 3. Distribuisce i Task (Fan-Out)
            generate_worker_tasks(job_data)

            # 4. Cancella il messaggio del Client
            sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=receipt_handle)
            print("🏁 Job distribuito. Ora il Master si mette in attesa...\n")

            # 5. Aspetta i risultati (Fan-In)
            lista_modelli_s3 = wait_for_worker_responses(job_id, num_workers)

            # 6. (Qui andrà il codice in cui il Master scarica i file da S3 e fa il merge degli alberi)
            print(f"🌳 Unione dei modelli da queste posizioni: {lista_modelli_s3}")

            # 7. SCALE-TO-ZERO: Il lavoro è finito, spegniamo le macchine per risparmiare!
            scale_worker_infrastructure(0)

            print(f"✅ Job {job_id} completato al 100%. Master pronto per il prossimo Job.\n")


if __name__ == "__main__":
    main()