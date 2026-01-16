import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def eval_multiclass(y_true, y_pred):
    return {

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true,y_pred, average = "macro")),
        "confusion_matrix": confusion_matrix(y_true,y_pred).tolist()
    
    }


# Input: true values // Output: predicted values
def eval_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    return {
        "mae": float(mean_absolute_error(y_true,y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true,y_pred)) 
    }

def save_metrics(metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents= True, exist_ok = True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent = 2)
