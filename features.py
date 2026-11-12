import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific physical features for the AI4I 2020 dataset:
    1. temp_diff [K]: Process temperature [K] - Air temperature [K]
    2. power [W]: Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60)
    """
    df_out = df.copy()
    df_out["temp_diff"] = df_out["Process temperature [K]"] - df_out["Air temperature [K]"]
    omega = df_out["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
    df_out["power"] = df_out["Torque [Nm]"] * omega
    return df_out