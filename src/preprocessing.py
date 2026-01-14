import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def split_features_targets(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]) #Features
    y = df[target_col].copy() #Target
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    #Extraction and Seperation of numeric & categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include = ["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline( steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ])
    
    categorical_pipe = Pipeline( steps = [
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore")),
        ])
    
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_pipe, numeric_cols),
            ("cat",categorical_pipe,categorical_cols),
        ],
        remainder="drop" #non specified columns are dropped
        )

    return preprocessor





