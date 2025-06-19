import requests
import json

with open("serving_input_example.json", "r") as f:
    payload = json.load(f)

response = requests.post(
    "http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

if response.status_code == 200:
    print("✅ Result:", response.json())
else:
    print("❌ Error:", response.text)