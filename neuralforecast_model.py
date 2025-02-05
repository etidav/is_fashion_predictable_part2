import abc
import copy
import datetime
import os
from enum import Enum
from typing import List, Optional, Union

import neuralforecast.models as neuralforecast_models
import pandas as pd
from neuralforecast import NeuralForecast
from torch import load as torch_load

from config import WEEK_FREQUENCY_TIMEINDEX, WEEKS_IN_YEAR
from forecast_model import ForecastModel

DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_MAX_STEPS = 250
WEEK_FREQUENCY_INDICATOR = "W"


class NeuralForecastModelType(str, Enum):
    nhits = "nhits"
    nbeats = "nbeats"
    patchtst = "patchtst"
    deepar = "deepar"
    tft = "tft"
    tsmixer = "tsmixer"
    timemixer = "timemixer"


NEURAL_FORECAST_CLASSES = {
    NeuralForecastModelType.nhits: neuralforecast_models.NHITS,
    NeuralForecastModelType.nbeats: neuralforecast_models.NBEATS,
    NeuralForecastModelType.patchtst: neuralforecast_models.PatchTST,
    NeuralForecastModelType.deepar: neuralforecast_models.DeepAR,
    NeuralForecastModelType.tft: neuralforecast_models.TFT,
    NeuralForecastModelType.tsmixer: neuralforecast_models.TSMixer,
    NeuralForecastModelType.timemixer: neuralforecast_models.TimeMixer
}


class NeuralForecastModel(ForecastModel, metaclass=abc.ABCMeta):
    """
        Abstract for forecast model using neural methods. Mostly an overlay of neuralforecast library : https://nixtla.github.io/neuralforecast/ by Nixtla.
    """

    model_type: Optional[NeuralForecastModelType]
    model: Optional[NeuralForecast] = None
    scale_period: int = 2 * WEEKS_IN_YEAR

    max_steps: int = DEFAULT_MAX_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = 0
    num_workers_loader: int = 0

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
    def from_saved_model(cls, path: str, **kwargs) -> "NeuralForecastModel":
        if not os.path.isdir(path):
            raise ValueError(f"No model saved at following directory path : {path}")

        model = NeuralForecast.load(path=path)
        nforecast_class_name = model.models[0].__class__.__name__
        if nforecast_class_name not in [
            nforecast_class.__name__ for nforecast_class in NEURAL_FORECAST_CLASSES.values()
        ]:
            raise NotImplementedError(
                f"NeuralForecastModel type is only available for {NeuralForecastModelType.__members__.values()}"
            )
        classnames_to_type = {
            nforecast_class.__name__: type
            for type, nforecast_class in NEURAL_FORECAST_CLASSES.items()
        }
        model_type = classnames_to_type[nforecast_class_name]
        return cls(
            horizon=model.models[0].h,
            max_steps=model.models[0].max_steps,
            learning_rate=model.models[0].learning_rate,
            batch_size=model.models[0].batch_size,
            model=model,
            model_type=model_type
        )

    def load_checkpoint_hparams(self, path: str) -> dict:
        if not os.path.isdir(path):
            raise ValueError(f"No model saved at following directory path : {path}")

        checkpoints = [file for file in os.listdir(path) if file.endswith(".ckpt")]

        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoint saved at following directory path : {path}")
        if len(checkpoints) > 1:
            raise ValueError(
                f"Found {len(checkpoints)} checkpoint saved at following directory path {path}, need to have only 1 checkpoint"
            )
        if self.checkpoint_name not in checkpoints:
            raise ValueError(f"Checkpoint {self.checkpoint_name} not found in checkpoints.")

        checkpoint = torch_load(os.path.join(path, self.checkpoint_name))
        return checkpoint["hyper_parameters"]

    def save_model(self, path: str):
        if not self.model._fitted:
            raise ValueError("Model should be fitted before saving it !")

        if os.path.isdir(
            path
        ):  # checkpoint accumulates when save path already exists, deleting it !
            for filename in os.listdir(path):
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(path, filename)
                    os.unlink(file_path)
        self.model.save(
            path=path, model_index=None, overwrite=True, save_dataset=True
        )  # mandatory to save dataset in neuralforecast==1.6.4, do we use 1.6.3 ?

    @staticmethod
    def format_single_ts(ts: pd.Series, ts_index: Union[str, datetime.datetime]) -> pd.DataFrame:
        formated_ts = pd.DataFrame(columns=["unique_id", "ds", "y"], index=range(len(ts)))
        formated_ts["unique_id"] = ts_index * 1.0
        formated_ts["ds"] = pd.to_datetime(ts.index)
        formated_ts["y"] = ts.values
        return formated_ts

    @staticmethod
    def format_multiple_ts(
        data: Union[pd.DataFrame, pd.Series], time_index: str = None
    ) -> pd.DataFrame:

        formated_train_data = []
        if time_index is not None:
            data = data.loc[:time_index]
        for i, j in enumerate(data):
            formated_train_data.append(NeuralForecastModel.format_single_ts(data[j], i + 1))
        formated_train_data = pd.concat(formated_train_data, axis=0).reset_index(drop=True)
        return formated_train_data

    def standard_scaler(self, data: Union[pd.DataFrame, pd.Series]):
        scale_factor_mean = data.iloc[:self.scale_period].mean(axis=0).values
        # For constant timeseries, std is equal to 0. In that case, we replace the std by 1 to avoid inf or nan.
        scale_factor_std = data.iloc[:self.scale_period].std(axis=0).replace(0., 1.).values
        scaled_data = (data - scale_factor_mean) / scale_factor_std
        return scaled_data, scale_factor_mean, scale_factor_std

    def fit(self, historical_data: pd.DataFrame, time_index: Optional[str] = None):
        print(f"Fitting model with {len(historical_data.T)} time series ...")
        scaled_data, _, _ = self.standard_scaler(historical_data)
        y_train = self.format_multiple_ts(scaled_data, time_index=time_index)
        self.model.fit(df=y_train, val_size=self.horizon)

    def predict(
        self, historical_data: pd.DataFrame, time_index: Optional[int] = None
    ) -> pd.DataFrame:
        if time_index is not None:
            historical_data = copy.deepcopy(historical_data)
            historical_data.date_split(time_index)
        # Instead of using all the historical data, use only the latest data window of length self.scale_period to
        # compute the scaled data.
        scaled_data, scale_factor_mean, scale_factor_std = self.standard_scaler(
            historical_data.iloc[-self.scale_period:]
        )
        dataset = self.format_multiple_ts(scaled_data, time_index=time_index)
        y_pred_df = self.model.predict(df=dataset)

        nforecast_class_name = NEURAL_FORECAST_CLASSES[self.model_type].__name__
        predictions = y_pred_df[nforecast_class_name].values.reshape(-1, self.horizon).T
        predictions = predictions * scale_factor_std + scale_factor_mean
        # set negative values to 0
        predictions[predictions < 0] = 0.
        forecast_index = pd.date_range(
            start=historical_data.index[-1],
            periods=self.horizon + 1,
            freq=WEEK_FREQUENCY_TIMEINDEX
        )[1:]
        forecast = pd.DataFrame(
            predictions,
            columns=historical_data.columns,
            index=forecast_index,
        )
        return forecast

    @property
    @abc.abstractmethod
    def checkpoint_name(self) -> str:
        pass


class NHitsForecastModel(NeuralForecastModel):
    """
        Forecast model using NHits method https://nixtla.github.io/neuralforecast/models.nhits.html

        Args:
            n_blocks: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).
            mlp_units: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).
            n_freq_downsample: List[int], list with the stack’s coefficients (inverse expressivity ratios). Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
            n_pool_kernel_size: List[int], list with the size of the windows to take a max/avg over. Note that len(stack_types)=len(n_freq_downsample)=len(n_pool_kernel_size).
    """

    model_type = NeuralForecastModelType.nhits

    n_blocks: List[int] = [1, 1, 1]
    mlp_units: List[List[int]] = 3 * [[512, 512]]
    n_pool_kernel_size: List[int] = [2, 2, 1]
    n_freq_downsample: List[int] = [4, 2, 1]

    def init_model(self):
        if not len(self.n_blocks) == len(self.mlp_units) == len(self.n_pool_kernel_size
                                                               ) == len(self.n_freq_downsample):
            raise ValueError(
                "Model argument n_blocks, mlp_units n_pool_kernel_size, n_freq_downsample should have same length."
            )

        model = neuralforecast_models.NHITS(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            n_blocks=self.n_blocks,
            mlp_units=self.mlp_units,
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            stack_types=["identity"] * len(self.n_blocks),
            num_workers_loader=self.num_workers_loader
        )
        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "NHitsForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.nhits:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.nhits.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.n_blocks = h_params["n_blocks"]
            forecast_model.mlp_units = h_params["mlp_units"]
            forecast_model.n_pool_kernel_size = h_params["n_pool_kernel_size"]
            forecast_model.n_freq_downsample = h_params["n_freq_downsample"]
        return forecast_model

    @property
    def checkpoint_name(self):
        return "nhits_0.ckpt"


class NBeatsForecastModel(NeuralForecastModel):
    """
        Forecast model using NBeats method https://nixtla.github.io/neuralforecast/models.nbeats.html

        Args:
            n_harmonics: int, Number of harmonic terms for seasonality stack type. Note that it will only be used if a seasonality stack is used.
            n_polynomials: int, polynomial degree for trend stack. Note that it will only be used if a trend stack is used.
            stack_types: List[str], List of stack types. Subset from [‘seasonality’, ‘trend’, ‘identity’].
            n_blocks: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).
            mlp_units: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).
    """
    model_type = NeuralForecastModelType.nbeats

    n_harmonics: int = 2
    n_polynomials: int = 2
    stack_types: List[str] = ["identity", "trend", "seasonality"]
    n_blocks: list = [1, 1, 1]
    mlp_units: list = 3 * [[512, 512]]

    def init_model(self):
        if not len(self.stack_types) == len(self.n_blocks) == len(self.mlp_units):
            raise ValueError(
                "Model argument stack_types, n_blocks and mlp_units should have same length"
            )

        model = neuralforecast_models.NBEATS(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            n_harmonics=self.n_harmonics,
            n_polynomials=self.n_polynomials,
            n_blocks=self.n_blocks,
            mlp_units=self.mlp_units,
            stack_types=self.stack_types,
            num_workers_loader=self.num_workers_loader
        )
        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "NBeatsForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.nbeats:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.nbeats.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.n_harmonics = h_params["n_harmonics"]
            forecast_model.n_polynomials = h_params["n_polynomials"]
            forecast_model.n_blocks = h_params["n_blocks"]
            forecast_model.mlp_units = h_params["mlp_units"]
            forecast_model.stack_types = h_params["stack_types"]
        return forecast_model

    @property
    def checkpoint_name(self):
        return "nbeats_0.ckpt"


class PatchTSTForecastModel(NeuralForecastModel):
    """
        Forecast model using PatchTST method https://nixtla.github.io/neuralforecast/models.patchtst.html

        Args:
            encoder_layers: int, number of layers for encoder.
            n_heads: int, number of multi-head’s attention.
            hidden_size: int, units of embeddings and encoders.
            linear_hidden_size: int, units of linear layer.
    """
    model_type = NeuralForecastModelType.patchtst

    encoder_layers: int = 3
    n_heads: int = 16
    hidden_size: int = 128
    linear_hidden_size: int = 256
    patch_len: int = 16

    def init_model(self):
        model = neuralforecast_models.PatchTST(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            encoder_layers=self.encoder_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            linear_hidden_size=self.linear_hidden_size,
            patch_len=self.patch_len,
            num_workers_loader=self.num_workers_loader
        )
        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "PatchTSTForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.patchtst:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.patchtst.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.encoder_layers = h_params["encoder_layers"]
            forecast_model.n_heads = h_params["n_heads"]
            forecast_model.hidden_size = h_params["hidden_size"]
            forecast_model.linear_hidden_size = h_params["linear_hidden_size"]
            forecast_model.patch_len = h_params["patch_len"]
        return forecast_model

    @property
    def checkpoint_name(self):
        return "patchtst_0.ckpt"


class DeepARForecastModel(NeuralForecastModel):
    """
        Forecast model using DeepAr method : https://nixtla.github.io/neuralforecast/models.deepar.html

        Args:
            lstm_n_layers: number of LSTM layers.
            lstm_hidden_size: LSTM hidden size.
            decoder_hidden_layers: number of decoder MLP hidden layers. Default: 0 for linear layer.
            decoder_hidden_size: decoder MLP hidden size. Default: 0 for linear layer.
            trajectory_samples: int, number of Monte Carlo trajectories during inference.

    """
    model_type = NeuralForecastModelType.deepar

    lstm_n_layers: int = 2
    lstm_hidden_size: int = 128
    decoder_hidden_layers: int = 0
    decoder_hidden_size: int = 0
    trajectory_samples: int = 100

    def init_model(self):
        model = neuralforecast_models.DeepAR(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            lstm_n_layers=self.lstm_n_layers,
            lstm_hidden_size=self.lstm_hidden_size,
            decoder_hidden_layers=self.decoder_hidden_layers,
            decoder_hidden_size=self.decoder_hidden_size,
            trajectory_samples=self.trajectory_samples,
            num_workers_loader=self.num_workers_loader
        )

        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @property
    def checkpoint_name(self):
        return "deepar_0.ckpt"

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "DeepARForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.deepar:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.deepar.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.lstm_n_layers = h_params["lstm_n_layers"]
            forecast_model.lstm_hidden_size = h_params["lstm_hidden_size"]
            forecast_model.decoder_hidden_layers = h_params["decoder_hidden_layers"]
            forecast_model.decoder_hidden_size = h_params["decoder_hidden_size"]
            forecast_model.trajectory_samples = h_params["trajectory_samples"]

        return forecast_model


class TFTForecastModel(NeuralForecastModel):
    """
        Forecast model using TFT method : https://nixtlaverse.nixtla.io/neuralforecast/index.html

        Args:
            tgt_size: int = 1
            hidden_size: int = 128, units of embeddings and encoders.
            n_head: int = 4, number of attention heads in temporal fusion decoder.
    """
    model_type = NeuralForecastModelType.tft

    tgt_size: int = 1
    hidden_size: int = 128
    n_head: int = 4

    def init_model(self):
        model = neuralforecast_models.TFT(
            input_size=2 * self.horizon,
            h=self.horizon,
            tgt_size=self.tgt_size,
            hidden_size=self.hidden_size,
            n_head=self.n_head,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            num_workers_loader=self.num_workers_loader
        )

        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @property
    def checkpoint_name(self):
        return "tft_0.ckpt"

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "TFTForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.tft:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.tft.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.tgt_size = h_params["tgt_size"]
            forecast_model.hidden_size = h_params["hidden_size"]
            forecast_model.n_head = h_params["n_head"]

        return forecast_model


class TSMixerForecastModel(NeuralForecastModel):
    """
        Forecast model using TSMixer method : https://nixtlaverse.nixtla.io/neuralforecast/index.html

        Args:
            n_series: int = 100
    """
    model_type = NeuralForecastModelType.tsmixer

    n_series: int = 100

    def init_model(self):
        model = neuralforecast_models.TSMixer(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            num_workers_loader=self.num_workers_loader,
            n_series=self.n_series
        )

        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @property
    def checkpoint_name(self):
        return "tft_0.ckpt"

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "TSMixerForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.tsmixer:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.tsmixer.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.tgt_size = h_params["tgt_size"]
            forecast_model.hidden_size = h_params["hidden_size"]
            forecast_model.n_head = h_params["n_head"]

        return forecast_model


class TimeMixerForecastModel(NeuralForecastModel):
    """
        Forecast model using TimeMixer method : https://nixtlaverse.nixtla.io/neuralforecast/index.html

        Args:
            n_series: int = 100
    """
    model_type = NeuralForecastModelType.timemixer

    n_series: int = 100

    def init_model(self):
        model = neuralforecast_models.TimeMixer(
            input_size=2 * self.horizon,
            h=self.horizon,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            random_seed=self.seed,
            num_workers_loader=self.num_workers_loader,
            n_series=self.n_series
        )

        self.model = NeuralForecast(models=[model], freq=WEEK_FREQUENCY_INDICATOR)

    @property
    def checkpoint_name(self):
        return "tft_0.ckpt"

    @classmethod
    def from_saved_model(cls, path: str, **kwargs) -> "TimeMixerForecastModel":
        forecast_model = super().from_saved_model(path)

        if forecast_model.model_type != NeuralForecastModelType.timemixer:
            raise ValueError(
                f"Trying to instanciate {NeuralForecastModelType.timemixer.value} model from {forecast_model.model_type.value} saved model path."
            )
        h_params = forecast_model.load_checkpoint_hparams(path)
        if h_params:
            forecast_model.tgt_size = h_params["tgt_size"]
            forecast_model.hidden_size = h_params["hidden_size"]
            forecast_model.n_head = h_params["n_head"]

        return forecast_model