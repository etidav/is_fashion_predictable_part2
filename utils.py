from typing import Dict

import pandas as pd
from config import WEEKS_IN_YEAR
from metrics import mape, mase, smape


def load_time_series_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = df.index.inferred_freq
    return df


def eval_predictions(
    ts_signal: pd.DataFrame, predictions: Dict[str, pd.DataFrame], output_path: str
) -> None:
    error_metrics = pd.DataFrame(columns=["mase", "mape", "smape"])
    for model_name, model_prediction in predictions.items():
        error_metrics.loc[model_name] = [
            mase(ts_signal, model_prediction, seasonality=WEEKS_IN_YEAR)[0].round(4),
            mape(ts_signal, model_prediction)[0].round(4),
            smape(ts_signal, model_prediction)[0].round(4)
        ]
    error_metrics.sort_values(by="mase", ascending=False).to_csv(output_path)