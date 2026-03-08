import boto3
import json
import argparse
import uuid
from datetime import datetime

# Usa la tua coda FIFO
CLIENT_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/248593862537/JobRequestQueue.fifo'
AWS_REGION = 'us-east-1'

def main():
    parser = argparse.ArgumentParser(description="Client per Random Forest Distribuita")
    parser.add_argument('--dataset', type=str, required=True, choices=['higgs', 'taxi', 'airlines'], help="Il dataset da analizzare")
    parser.add_argument('--workers', type=int, required=True, help="Numero di worker da allocare")
    parser.add_argument('--trees', type=int, required=True, help="Numero totale di alberi della foresta")
    
    args = parser.parse_args()

    sqs_client = boto3.client('sqs', region_name=AWS_REGION)

    # Genera una data leggibile tipo "20240308_153022"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crea l'ID perfetto: es. "job_taxi_200trees_20240308_153022"
    job_id = f"job_{args.dataset}_{args.trees}trees_{timestamp}"

    message_body = {
        "job_id": job_id, # Lo passiamo direttamente noi!
        "dataset": args.dataset,
        "num_workers": args.workers,
        "num_trees": args.trees
    }

    try:
        response = sqs_client.send_message(
            QueueUrl=CLIENT_QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageGroupId="ML_Jobs", # Obbligatorio per code FIFO
            MessageDeduplicationId=job_id # Evita di inviare due volte per sbaglio lo stesso job
        )
        print(f" Richiesta inviata con successo!")
        print(f" Job ID: {job_id}")
        print(f" Dataset: {args.dataset} | Workers: {args.workers} | Alberi: {args.trees}")
    except Exception as e:
        print(f" Errore nell'invio: {e}")

if __name__ == "__main__":
    main()
