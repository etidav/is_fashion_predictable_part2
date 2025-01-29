import pandas as pd

from statsforecast_model import EtsForecastModel, TbatsForecastModel, StatsForecastModelType, ArimaForecastModel, \
    ThetaForecastModel
from utils import load_time_series_csv

fashion_ts = load_time_series_csv("us_female_sneakers.csv")
fashion_ts_train = fashion_ts.loc[:"2023-01-01"]

statistical_models = {
    StatsForecastModelType.ets: EtsForecastModel(),
    StatsForecastModelType.theta: ThetaForecastModel(),
    StatsForecastModelType.tbats: TbatsForecastModel(),
    StatsForecastModelType.arima: ArimaForecastModel(),
}

statistical_models_forecasts = {
    model_name: model.predict(fashion_ts_train) for model_name, model in statistical_models.items()
}

# eval_predictions(
#     ts_signal=fashion_ts,
#     predictions={
#         "Exp. Smooth.": ets_prediction,
#         "Snaive": snaive_prediction
#     },
#     output_path="error_metrics.csv"
# )