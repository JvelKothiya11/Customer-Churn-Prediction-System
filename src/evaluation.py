import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def precision_recall_vs_threshold(y_true, y_proba, thresholds=np.arange(0.2,0.81,0.05)):
    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred)
        })
    df = pd.DataFrame(results)
    return df

def plot_precision_recall(df):
    plt.figure(figsize=(7,5))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Threshold")
    plt.legend()
    plt.show()

def get_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
