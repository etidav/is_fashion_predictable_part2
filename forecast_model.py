from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd
from config import PREDICTION_ONE_YEAR, WEEKS_IN_YEAR
from pydantic import BaseModel


class ForecastModel(BaseModel, ABC):
    """
    Abstract class for a forecasting model.
    """

    horizon: int = PREDICTION_ONE_YEAR
    scale_period: int = 2 * WEEKS_IN_YEAR
    model_type: Optional[str]

    @classmethod
    @abstractmethod
    def from_saved_model(cls, path: str, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def save_model(cls, path: str):
        pass

    @abstractmethod
    def predict(self, historical_data: pd.DataFrame, time_index: str = None) -> pd.DataFrame:
        """Return the prediction of a forecasting model.

            Args:
                historical_data: A pd.DataFrame gathering time series signals with date as index and time series' name as columns.
                time_index: A date in the format 'YYYY-MM-DD' where to stop the historical signals and start the
                 forecasts. If None is provided, the forecast period will start at the end of the time
                 series.

            Returns:
                A pd.DataFrame gathering the predictions.

        """

    @abstractmethod
    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):
        """
            Fitting method for models
        """
        pass

    def standard_scaler(self, data: Union[pd.DataFrame, pd.Series]):
        scale_factor_mean = data.iloc[:self.scale_period].mean(axis=0).values
        # For constant timeseries, std is equal to 0. In that case, we replace the std by 1 to avoid inf or nan.
        scale_factor_std = data.iloc[:self.scale_period].std(axis=0).replace(0., 1.).values
        scaled_data = (data - scale_factor_mean) / scale_factor_std
        return scaled_data, scale_factor_mean, scale_factor_std
