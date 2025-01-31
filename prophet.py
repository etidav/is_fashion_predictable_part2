import multiprocessing
from enum import Enum
from functools import partial
from typing import Dict, Optional, Tuple

import pandas as pd

from config import WEEK_FREQUENCY_TIMEINDEX
from forecast_model import ForecastModel
from prophet import Prophet
from tqdm import tqdm


class ProphetForecastModelType(str, Enum):
    prophet = "prophet"


class ProphetForecastModel(ForecastModel):
    growth: str = "linear"
    yearly_seasonality = True  # Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality = False  # Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality = False  # Can be 'auto', True, False, or a number of Fourier terms to generate.
    seasonality_mode: str = "multiplicative"  # can be 'additive', or 'multiplicative'
    seasonality_prior_scale: float = 10.0
    changepoint_prior_scale: float = 0.05
    n_processes: int = 8  # number of parallelized processes for prophet running

    model_type = ProphetForecastModelType.prophet

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "StatsForecastModel":
        return cls()

    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):
        pass

    @classmethod
    def save_model(cls, path: str):
        pass

    @staticmethod
    def format_single_ts(ts: pd.Series) -> pd.DataFrame:
        return ts.reset_index(name="y").rename(columns={"date": "ds"})

    def fit_predict_single_ts(
        self, ts_signal: pd.Series, forecast_index: pd.DataFrame
    ) -> pd.Series:
        """Fit Prophet models.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single: A pd.Series gathering the Tbats prediction.
        """
        base_model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        base_model.fit(df=self.format_single_ts(ts_signal))
        formatted_predict_df = pd.DataFrame({"ds": forecast_index})
        return base_model.predict(formatted_predict_df).set_index("ds").yhat

    def fit_predict_single_ts_multiprocess(
        self, ts_signal: Tuple[str, pd.Series], forecast_index: pd.DataFrame
    ) -> Tuple[str, pd.Series]:
        """Fit Prophet models.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single: A pd.Series gathering the model prediction.
        """
        base_model = Prophet(
            growth=self.growth,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        base_model.fit(df=self.format_single_ts(ts_signal[1]))
        formatted_predict_df = pd.DataFrame({"ds": forecast_index})
        return ts_signal[0], base_model.predict(formatted_predict_df).set_index("ds").yhat

    def fit_predict_all_ts(
        self, historical_data: pd.DataFrame, forecast_index: pd.DataFrame, multiprocess: bool,
        processes: int
    ) -> Dict[str, pd.Series]:

        if multiprocess:
            with multiprocessing.Pool(processes=processes) as pool:
                fitted_model = list(
                    tqdm(
                        pool.imap(
                            partial(
                                self.fit_predict_single_ts_multiprocess,
                                forecast_index=forecast_index
                            ),
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
                all_forecast[ts_id] = self.fit_predict_single_ts(
                    historical_data[ts_id], forecast_index
                )

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
        forecast_index = pd.date_range(
            start=historical_data.index[-1],
            periods=self.horizon + 1,
            freq=WEEK_FREQUENCY_TIMEINDEX
        )[1:]
        all_forecast = self.fit_predict_all_ts(
            historical_data, forecast_index, multiprocess, processes
        )
        forecast_columns = historical_data.columns
        forecast = pd.DataFrame(all_forecast, columns=forecast_columns, index=forecast_index)
        forecast[forecast < 0] = 0.
        return forecast