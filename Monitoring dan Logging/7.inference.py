import requests
import json

with open("input_example.json", "r") as file:
    payload = json.load(file)

URL = "http://localhost:5000/invocations"

response = requests.post(
    URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

if response.status_code == 200:
    print("✅ Prediction result:")
    print(response.json())
else:
    print(f"❌ Failed with status code {response.status_code}")
    print(response.text)