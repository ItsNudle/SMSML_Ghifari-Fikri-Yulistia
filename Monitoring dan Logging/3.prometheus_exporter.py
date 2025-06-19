from prometheus_client import start_http_server, Gauge
import time, random

accuracy = Gauge("model_accuracy", "Akurasi model")
latency = Gauge("model_latency_seconds", "Latency inferring")
precision = Gauge("model_precision", "Precision model")
recall = Gauge("model_recall", "Recall model")
f1_score = Gauge("model_f1", "F1 Score")
requests_total = Gauge("model_requests_total", "Total inference request")

def collect_metrics():
    while True:
        accuracy.set(random.uniform(0.85, 0.99))
        latency.set(random.uniform(0.01, 0.1))
        precision.set(random.uniform(0.8, 0.95))
        recall.set(random.uniform(0.8, 0.95))
        f1_score.set(random.uniform(0.8, 0.95))
        requests_total.inc()
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(8000)
    collect_metrics()
