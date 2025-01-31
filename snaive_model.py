from enum import Enum
from typing import Optional

import pandas as pd

from config import WEEKS_IN_YEAR, WEEK_FREQUENCY_TIMEINDEX
from forecast_model import ForecastModel


class SnaiveForecastModelType(str, Enum):
    snaive = "snaive"


class SnaiveForecastModel(ForecastModel):
    """
        Class defining a Naive model that repeats the values of the past year as its prediction.
    """
    model_type = SnaiveForecastModelType.snaive

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "SnaiveForecastModel":
        return cls()

    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):
        pass

    @classmethod
    def save_model(cls, path: str):
        pass

    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:
        """ Compute prediction of a Seasonal naive model.

            Args:
                historical_data: A TimeSeriesDataset gathering time series signals.
                time_index: A date in the format 'YYYY-MM-DD' where to stop the historical signals and start the
                            forecasts. If None is provided, the forecast period will start at the end of the time
                            series.

            Returns:
                forecast: A TimeSeriesForecast gathering the Naive predictions.

        """

        historical_data_signals = historical_data[:time_index
                                                 ] if time_index is not None else historical_data
        historical_data_past_year = historical_data_signals.tail(WEEKS_IN_YEAR)

        nb_repeat, part_repeat = divmod(self.horizon, WEEKS_IN_YEAR)
        naive_forecast = [historical_data_past_year] * nb_repeat + [
            historical_data_past_year[:part_repeat]
        ]
        naive_forecast = pd.concat(naive_forecast, ignore_index=True)

        forecast_index = pd.date_range(
            start=historical_data_signals.index[-1],
            periods=self.horizon + 1,
            freq=WEEK_FREQUENCY_TIMEINDEX
        )[1:]
        naive_forecast.index = forecast_index
        return naive_forecast