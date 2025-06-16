import os
from dagshub import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

os.environ["MLFLOW_TRACKING_USERNAME"] = "ghifari.fikri.yulistia"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "28a2bed8301cd660e33707a009cb925162d47426"
dagshub.init(repo_owner='ghifari.fikri.yulistia', repo_name='SMSML_Ghifari-Fikri-Yulistia', mlflow=True)

X = pd.read_csv("tfidf.csv")
y = pd.read_csv("labels.csv")["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
model = LogisticRegression(max_iter=200)
clf = GridSearchCV(model, param_grid=params, cv=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with mlflow.start_run():
    mlflow.log_param("best_C", clf.best_params_['C'])
    mlflow.log_param("best_solver", clf.best_params_['solver'])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    joblib.dump(clf.best_estimator_, "best_model.pkl")
    mlflow.sklearn.log_model(clf.best_estimator_, "model")

print("✅ Training selesai dan dilogging ke DagsHub!")