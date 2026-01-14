from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor

def get_vhd_models(random_state: int = 42):
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # helps if VHD is imbalanced
            n_jobs=None
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            random_state=random_state
        )
    }