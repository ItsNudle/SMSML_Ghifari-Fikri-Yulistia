import requests
import pandas as pd

data = pd.DataFrame([[0.1, 0.5, 0.7]], columns=["feat1", "feat2", "feat3"])
response = requests.post(
    url="http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    json={"columns": list(data.columns), "data": data.values.tolist()}
)
print(response.json())
