import abc
import copy
import multiprocessing
from enum import Enum
from functools import partial
from typing import Dict, Optional, Tuple

import pandas as pd
import statsforecast.models as statsforecast_models
from config import WEEK_FREQUENCY_TIMEINDEX, WEEKS_IN_YEAR
from forecast_model import ForecastModel
from statsforecast import StatsForecast
from tqdm import tqdm


class StatsForecastModelType(str, Enum):
    tbats = "tbats"
    arima = "arima"
    ets = "ets"
    theta = "theta"


STATS_FORECAST_CLASSES = {
    StatsForecastModelType.tbats: statsforecast_models.AutoTBATS,
    StatsForecastModelType.arima: statsforecast_models.AutoARIMA,
    StatsForecastModelType.ets: statsforecast_models.AutoETS,
    StatsForecastModelType.theta: statsforecast_models.AutoTheta,
}


class StatsForecastModel(ForecastModel, metaclass=abc.ABCMeta):
    """
        Abstract for forecast model using stats methods. Mostly an overlay of statsforecast library : https://nixtlaverse.nixtla.io/statsforecast/ by Nixtla.
    """

    model_type: Optional[StatsForecastModelType]
    model: Optional[StatsForecast] = None
    season_length: int = WEEKS_IN_YEAR

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model is None:
            self.init_model()

    @abc.abstractmethod
    def init_model(self):
        pass

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "StatsForecastModel":
        return cls()

    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):
        pass

    @classmethod
    def save_model(cls, path: str):
        pass

    def fit_predict_single_ts(self, ts_signal: pd.Series) -> pd.Series:
        """Fit Tbats models.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single: A pd.Series gathering the Tbats prediction.
        """
        base_model = copy.deepcopy(self.model)
        single_forecast = base_model.forecast(y=ts_signal.values, h=self.horizon)
        return single_forecast["mean"]

    def fit_predict_single_ts_multiprocess(
        self, ts_signal: Tuple[str, pd.Series]
    ) -> Tuple[str, pd.Series]:
        """Fit ETS models.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single: A pd.Series gathering the model prediction.
        """
        base_model = copy.deepcopy(self.model)
        single_forecast = base_model.forecast(y=ts_signal[1].values, h=self.horizon)
        return ts_signal[0], single_forecast["mean"]

    def fit_predict_all_ts(self, historical_data: pd.DataFrame, multiprocess: bool,
                           processes: int) -> Dict[str, pd.Series]:

        if multiprocess:
            with multiprocessing.Pool(processes=processes) as pool:
                fitted_model = list(
                    tqdm(
                        pool.imap(
                            partial(self.fit_predict_single_ts_multiprocess),
                            historical_data.T.iterrows(),
                            chunksize=5,
                        )
                    )
                )
                all_forecast = {
                    forecast_result[0]: forecast_result[1] for forecast_result in fitted_model
                }
        else:
            all_forecast = {}
            for ts_id in tqdm(
                historical_data.columns,
                desc=f"Predicting with model {self.model_type}",
                unit="Series"
            ):
                all_forecast[ts_id] = self.fit_predict_single_ts(historical_data[ts_id])

        return all_forecast

    def predict(
        self,
        historical_data: pd.DataFrame,
        time_index: str = None,
        multiprocess: bool = True,
        processes: int = 8
    ) -> pd.DataFrame:
        """Compute predictions of an statsforecast model.

            Args:
                historical_data: A pd.DataFrame gathering time series signals with date as index and time series' name as columns.
                time_index: A date in the format 'YYYY-MM-DD' where to stop the historical signals and start the
                            forecasts. If None is provided, the forecast period will start at the end of the time
                            series.

            Returns:
                forecast: A pd.DataFrame gathering the statsforecast's predictions.

        """
        print(
            f"Fit and Predict {len(historical_data.T)} time series with the {self.model_type} model."
        )
        historical_data = historical_data[:time_index] if time_index is not None else historical_data

        all_forecast = self.fit_predict_all_ts(historical_data, multiprocess, processes)
        forecast_index = pd.date_range(
            start=historical_data.index[-1],
            periods=self.horizon + 1,
            freq=WEEK_FREQUENCY_TIMEINDEX
        )[1:]
        forecast_columns = historical_data.columns
        forecast = pd.DataFrame(all_forecast, columns=forecast_columns, index=forecast_index)
        forecast[forecast < 0] = 0.
        return forecast


class TbatsForecastModel(StatsForecastModel):
    """
        Forecast model using AutoTbats method https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats
    """
    model_type = StatsForecastModelType.tbats
    use_boxcox: Optional[bool] = None
    use_trend: Optional[bool] = None
    use_damped_trend: Optional[bool] = None
    use_arma_errors: Optional[bool] = None

    def init_model(self):
        self.model = statsforecast_models.AutoTBATS(
            season_length=self.season_length,
            use_boxcox=self.use_boxcox,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors
        )


class ArimaForecastModel(StatsForecastModel):
    """
        Forecast model using AutoTbats method https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats
    """
    model_type: StatsForecastModelType = StatsForecastModelType.arima
    d: Optional[int] = None
    D: Optional[int] = None
    max_p: int = 5
    max_q: int = 5
    max_P: int = 2
    max_Q: int = 2
    start_p: int = 2
    start_q: int = 2
    start_P: int = 1
    start_Q: int = 1

    def init_model(self):
        self.model = statsforecast_models.AutoARIMA(
            season_length=self.season_length,
            d=self.d,
            D=self.D,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            start_p=self.start_p,
            start_q=self.start_q,
            start_P=self.start_P,
            start_Q=self.start_Q,
        )


class EtsForecastModel(StatsForecastModel):
    """
        Forecast model using AutoTbats method https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats
    """
    model_type: StatsForecastModelType = StatsForecastModelType.ets
    model_config: str = "ZZZ"
    damped: Optional[bool] = None
    phi: Optional[float] = None

    def init_model(self):
        self.model = statsforecast_models.AutoETS(
            season_length=self.season_length,
            model=self.model_config,
            damped=self.damped,
            phi=self.phi
        )


class ThetaForecastModel(StatsForecastModel):
    """
        Forecast model using AutoTbats method https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats
    """
    model_type: StatsForecastModelType = StatsForecastModelType.theta
    decomposition_type: str = "multiplicative"

    def init_model(self):
        self.model = statsforecast_models.AutoTheta(
            season_length=self.season_length, decomposition_type=self.decomposition_type
        )