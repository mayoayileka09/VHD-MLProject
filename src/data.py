import pandas as pd
from pathlib import Path

def load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df

