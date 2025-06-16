import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dagshub import dagshub_logger

os.environ["MLFLOW_TRACKING_USERNAME"] = "ghifari.fikri.yulistia"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "28a2bed8301cd660e33707a009cb925162d47426"

dagshub.init(repo_owner='ghifari.fikri.yulistia', repo_name='SMSML_Ghifari-Fikri-Yulistia', mlflow=True)

X = pd.read_csv("tfidf.csv")
y = pd.read_csv("labels.csv")["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
param_dist = {
    'C': uniform(0.01, 0.1, 1, 10),
    'solver': ['liblinear', 'lbfgs']
}
search =  GridSearchCV(model, param_distributions=param_dist, n_iter=4, cv=3, random_state=42, verbose=1)
search.fit(X_train, y_train)

y_pred = search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with mlflow.start_run():
    mlflow.log_param("best_C", search.best_params_['C'])
    mlflow.log_param("best_solver", search.best_params_['solver'])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(search.best_estimator_, "best_model.pkl")

    signature = infer_signature(X_train, search.predict(X_train))
    input_example = X_train.iloc[:1]

    mlflow.sklearn.log_model(
        search.best_estimator_,
        "model",
        signature=signature,
        input_example=input_example
    )

print("✅ Model berhasil dituning dan dicatat di DagsHub MLflow.")