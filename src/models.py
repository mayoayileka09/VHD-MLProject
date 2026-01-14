from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier,HistGradientBoostingRegressor

def get_vhd_models(random_state: int = 42):
    return {
        "logreg": LogisticRegression(
            max_iter=2000, # Controls how many optimization steps are allowed
            class_weight="balanced",  # helps if VHD is imbalanced // automatically adjusts weights based on class freq.
            n_jobs=None 
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, # number of trees
            random_state=random_state,
            n_jobs=-1 # uses all available cpu resources
        ),
        "hgb": HistGradientBoostingClassifier(
            random_state=random_state
        )
    }

def get_ef_models(random_state: int = 42):
    return{
        "ridge": Ridge(random_state = random_state),
        "rf": RandomForestRegressor(
            n_estimators= 400,
            random_state=random_state,
            n_jobs = -1
        ),
        "hgb": HistGradientBoostingRegressor(
            random_state=random_state
        )
    }
