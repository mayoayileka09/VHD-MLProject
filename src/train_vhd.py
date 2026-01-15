from pathlib import Path
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import PATHS
from src.data import load_excel
from src.preprocessing import split_features_targets,build_preprocessor
from src.models import get_vhd_models
from src.evaluate import eval_multiclass, save_metrics


def main():
    PATHS.METRICS_DIR.mkdir(parents = True, exist_ok= True)
    PATHS.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_excel(PATHS.DATA_DIR/"Electrocardiography_data.xlsx")

    target_col = "VHD"
    X, y = split_features_targets(df, target_col)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size= 0.30, random_state=42, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    preprocessor = build_preprocessor(X_train)
    models = get_vhd_models()

    all_results = {}

    best_name, best_val = None, -1 #VAriables for best performing model and best validation score 

    for name, model in models.items():
        pipe = Pipeline(steps =[
            ("preprocess", preprocessor),
            ("model",model)
        ])

        pipe.fit(X_train,y_train)
        val_pred = pipe.predict(X_val)

        metrics = eval_multiclass(y_val, val_pred)
        all_results[name] = {"val": metrics}

        if metrics["macro_f1"] > best_val:
            best_val = metrics["macro_f1"]
            best_name = name
            best_pipe = pipe

    test_pred = best_pipe.predict(X_test)
    test_metrics = eval_multiclass(y_test, test_pred)
    all_results[best_name]["test"] = test_metrics
    all_results["selected_model"] = best_name

    save_metrics(all_results, PATHS.METRICS_DIR / "vhd_metrics.json")
    joblib.dump(best_pipe, PATHS.MODELS_DIR / f"vhd_best_{best_name}.joblib")

    print("Selected:", best_name)
    print("Test macro_f1:", test_metrics["macro_f1"])

if __name__ == "__main__":
    main()