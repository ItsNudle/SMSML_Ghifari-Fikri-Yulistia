from prometheus_client import start_http_server, Gauge
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score

g_accuracy = Gauge("model_accuracy", "Akurasi model (test)")
g_precision = Gauge("model_precision", "Precision model (test)")
g_recall = Gauge("model_recall", "Recall model (test)")
g_f1 = Gauge("model_f1", "F1 Score model (test)")
g_log_loss = Gauge("model_log_loss", "Log Loss model (test)")
g_roc_auc = Gauge("model_roc_auc", "ROC AUC Score model (test)")

def collect_and_export_metrics():
    model = joblib.load("MLProject/model_artifacts/sklearn_model/model.pkl")
    
    X_test = pd.read_csv("Membangun_model/spam_ham_emails_preprocessing/tfidf.csv").sample(200, random_state=42)
    y_test = pd.read_csv("Membangun_model/spam_ham_emails_preprocessing/labels.csv")["label"].iloc[X_test.index]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    g_accuracy.set(accuracy_score(y_test, y_pred))
    g_precision.set(precision_score(y_test, y_pred))
    g_recall.set(recall_score(y_test, y_pred))
    g_f1.set(f1_score(y_test, y_pred))
    g_log_loss.set(log_loss(y_test, y_prob))
    g_roc_auc.set(roc_auc_score(y_test, y_prob[:, 1]))

if __name__ == "__main__":
    start_http_server(8000)
    collect_and_export_metrics()
    print("🚀 Prometheus metrics served at http://localhost:8000")