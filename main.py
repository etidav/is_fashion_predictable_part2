import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from neuralforecast_model import (
    DeepARForecastModel, NBeatsForecastModel, NeuralForecastModelType, NHitsForecastModel,
    PatchTSTForecastModel, TimeMixerForecastModel, TSMixerForecastModel
)
from prophet_model import ProphetForecastModel, ProphetForecastModelType
from snaive_model import SnaiveForecastModel, SnaiveForecastModelType
from statsforecast_model import (
    ArimaForecastModel, EtsForecastModel, StatsForecastModelType, TbatsForecastModel,
    ThetaForecastModel
)
from utils import (
    errors_distribution, eval_predictions, load_forecast, load_time_series_csv, mean_errors_ranking,
    save_forecast
)

fashion_ts = load_time_series_csv("us_female_sneakers.csv")
fashion_ts_train = fashion_ts.loc[:"2023-01-01"]
compute_prediction = False

statistical_models = {
    SnaiveForecastModelType.snaive:
        SnaiveForecastModel(),
    ProphetForecastModelType.prophet:
        ProphetForecastModel(),
    StatsForecastModelType.ets:
        EtsForecastModel(model_config="AAA", damped=True),
    StatsForecastModelType.theta:
        ThetaForecastModel(),
    StatsForecastModelType.tbats:
        TbatsForecastModel(
            use_boxcox=True, use_arma_errors=True, use_damped_trend=True, use_trend=True
        ),
    StatsForecastModelType.arima:
        ArimaForecastModel(
            d=1,
            D=1,
            start_p=1,
            max_p=1,
            start_q=1,
            max_q=1,
            start_P=0,
            max_P=0,
            start_Q=2,
            max_Q=2
        )
}

deep_models = {
    NeuralForecastModelType.deepar: DeepARForecastModel(),
    NeuralForecastModelType.nbeats: NBeatsForecastModel(),
    NeuralForecastModelType.nhits: NHitsForecastModel(),
    NeuralForecastModelType.patchtst: PatchTSTForecastModel(),
    NeuralForecastModelType.tsmixer: TSMixerForecastModel(),
    NeuralForecastModelType.timemixer: TimeMixerForecastModel(),
}

if compute_prediction:
    for model in deep_models.values():
        model.fit(fashion_ts_train)
    models_forecasts = {
        **{
            model_name: model.predict(fashion_ts_train) for model_name, model in statistical_models.items()
        },
        **{
            model_name: model.predict(fashion_ts_train) for model_name, model in deep_models.items()
        }
    }
    save_forecast(models_forecasts=models_forecasts, forecasts_dir="model_forecasts")
else:
    models_forecasts = load_forecast(forecasts_dir="model_forecasts")

error_metrics = eval_predictions(fashion_ts, models_forecasts)
mean_errors_ranking(error_metrics=error_metrics, output_path="mean_errors_ranking.csv")
errors_distribution(error_metrics=error_metrics, output_path="errors_distribution.png")