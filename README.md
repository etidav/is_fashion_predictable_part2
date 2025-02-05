# is fashion predictable? part 2 
Code repository to reproduce results of the Medium article ["Is Fashion Predictable? part 2"]()

## Code Organisation

The repository is organized as follows:

 - [forecast_model.py](model/forecast_model.py): File with the abstract class ForecastModel from which all forecast models implemented in this repo inherit
 - [statsforecast_model.py](model/statsforecast_model.py): File with all the statistical models implemented with the Nixtla package Statsforecast.
 - [neuralforecast_model.py](model/neuralforecast_model.py): File with all the deep learning models implemented with the Nixtla package Neuralforecast.
 - [prophet_model.py](model/prophet_model.py): File with Meta's Prophet model.
 - [snaive_model.py](model/snaive_model.py): File with the Snaive model.
 - [metrics.py](metrics.py): File with the three error metrics: MASE, SMAPE and MAPE.
 - [main.py](main.py): File reproducing the main experience of the media article.