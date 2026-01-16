from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import PATHS
from src.data import load_excel
from src.preprocessing import split_features_targets, build_preprocessor
from src.models import get_ef_models
from src.evaluate import eval_regression, save_metrics

def main():
    PATHS.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_excel(PATHS.DATA_DIR/"Electrocardiography_data.xlsx")

    target_col = "EF"
    X,y = split_features_targets(df, target_col)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    preprocesser = build_preprocessor(X_train)
    models = get_ef_models()

    all_results = {}
    best_name,best_val = None, float("inf")

    for name, model in models.items():
        pipe = Pipeline(steps = [
            ("preprocess", preprocesser),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        metrics = eval_regression(y_val, val_pred)
        all_results[name] = {"val": metrics}

        if metrics["mae"] < best_val:
            best_val = metrics["mae"]
            best_name = name
            best_pipe = pipe

    test_pred = best_pipe.predict(X_test)
    test_metrics = eval_regression(y_test, test_pred)
    all_results[best_name]["test"] = test_metrics
    all_results["selected_model"] = best_name

    save_metrics(all_results, PATHS.METRICS_DIR / "ef_metrics.json")
    joblib.dump(best_pipe, PATHS.MODELS_DIR / f"ef_best_{best_name}.joblib")

    print("Selected:", best_name)
    print("Test MAE:", test_metrics["mae"])

if __name__ == "__main__":
    main()