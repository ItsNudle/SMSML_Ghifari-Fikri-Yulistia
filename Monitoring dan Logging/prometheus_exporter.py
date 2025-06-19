from prometheus_client import start_http_server, Summary, Counter, Gauge
import requests
import time
import json
import psutil

INFERENCE_TIME = Summary('inference_processing_seconds', 'Time spent processing inference')
INFERENCE_COUNTER = Counter('inference_requests_total', 'Total number of inference requests')

CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage (%)')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage (%)')
DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage (%)')
NET_SENT = Gauge('system_network_sent_bytes', 'Bytes sent through network')
NET_RECV = Gauge('system_network_recv_bytes', 'Bytes received through network')
DISK_READ = Gauge('system_disk_read_bytes', 'Disk read in bytes')
DISK_WRITE = Gauge('system_disk_write_bytes', 'Disk write in bytes')
LOAD_AVG_1M = Gauge('system_load_1min', 'System load average over 1 minute')
UPTIME = Gauge('system_uptime_seconds', 'System uptime in seconds')

@INFERENCE_TIME.time()
def infer():
    INFERENCE_COUNTER.inc()

    with open("serving_input_example.json", "r") as f:
        payload = json.load(f)

    response = requests.post(
        "http://localhost:5000/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        print("❌ Inference failed:", response.text)
    else:
        print("✅ Inference result:", response.json())

def collect_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    NET = psutil.net_io_counters()
    NET_SENT.set(NET.bytes_sent)
    NET_RECV.set(NET.bytes_recv)
    DISK = psutil.disk_io_counters()
    DISK_READ.set(DISK.read_bytes)
    DISK_WRITE.set(DISK.write_bytes)
    try:
        LOAD_AVG_1M.set(psutil.getloadavg()[0])
    except:
        pass
    UPTIME.set(time.time() - psutil.boot_time())

if __name__ == "__main__":
    start_http_server(8001)
    print("🚀 Exporter running at http://localhost:8001/metrics")

    while True:
        collect_system_metrics()
        infer()
        time.sleep(5)