from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from config import PREDICTION_ONE_YEAR
from pydantic import BaseModel


class ForecastModel(BaseModel, ABC):
    """
    Abstract class for a forecasting model.
    """

    horizon: int = PREDICTION_ONE_YEAR
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
