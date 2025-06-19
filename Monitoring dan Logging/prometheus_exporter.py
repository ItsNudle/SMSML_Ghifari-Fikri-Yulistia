from prometheus_client import start_http_server, Summary, Counter, Gauge
import requests
import time
import random
import json

INFERENCE_TIME = Summary('inference_processing_seconds', 'Time spent processing inference')
INFERENCE_COUNTER = Counter('inference_requests_total', 'Total number of inference requests')
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Model accuracy score')


MODEL_ACCURACY.set(0.92)

@INFERENCE_TIME.time()
def infer():
    INFERENCE_COUNTER.inc()
    
    payload = {
        "inputs": [[0.0] * 3000]
    }

    response = requests.post(
        "http://localhost:5000/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        print("❌ Inference failed:", response.text)
    else:
        print("✅ Inference result:", response.json())

if __name__ == "__main__":
    start_http_server(8001)
    print("🚀 Prometheus exporter running at http://localhost:8001/metrics")
    
    while True:
        infer()
        time.sleep(5)