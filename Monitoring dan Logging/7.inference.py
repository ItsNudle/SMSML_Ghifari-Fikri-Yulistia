import requests
import pandas as pd

data = {
  "columns": ["feature1", "feature2", ...],
  "data": [[...]]
}

res = requests.post("http://localhost:5000/invocations", json=data)
print(res.json())
