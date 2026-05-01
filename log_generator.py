import requests
import time
import random
import json

print(" Log Generator started — sending fake traffic...")

for i in range(200):
    log = {
        "dur": round(random.uniform(0.1, 50), 2),
        "proto": random.choice(["tcp", "udp", "icmp"]),
        "service": random.choice(["http", "ftp", "-", "dns"]),
        "state": random.choice(["CON", "FIN", "INT", "REQ"]),
        "spkts": random.randint(1, 100),
        "dpkts": random.randint(0, 50),
        "sbytes": random.randint(100, 10000),
        "dbytes": random.randint(0, 5000),
        "rate": round(random.uniform(0.5, 100), 2)
    }
    try:
        requests.post(
    "http://127.0.0.1:8000/ingest_log",
    json=log,
    headers={"X-API-Key": "your-secret-key"}
)
        print(f"Sent log #{i+1}")
    except:
        print("API not running yet...")
    time.sleep(1.5)  # real-time feel