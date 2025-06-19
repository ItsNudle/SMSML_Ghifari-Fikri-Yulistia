import requests
import json

with open("Monitoring dan Logging/input_example.json", "r") as f:
    data = json.load(f)

response = requests.post(
    url="http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    json=data
)

print(response.json())
