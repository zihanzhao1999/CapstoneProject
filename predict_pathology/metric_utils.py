import pandas as pd
import sklearn.metrics

def calculate_metric(pathology, prediction, index=["data"]):
    return pd.DataFrame({
        "accuracy": sklearn.metrics.accuracy_score(pathology, prediction),
        "precision": sklearn.metrics.precision_score(pathology, prediction, average="macro"),
        "recall": sklearn.metrics.recall_score(pathology, prediction, average="macro"),
        "f1 score": sklearn.metrics.f1_score(pathology, prediction, average="macro"),
        "balanced accuracy": sklearn.metrics.balanced_accuracy_score(pathology, prediction)
        }, index=index)