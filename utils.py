import os
from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from config import WEEKS_IN_YEAR
from metrics import mape, mase, smape


def load_time_series_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = df.index.inferred_freq
    return df


def save_forecast(models_forecasts: Dict[str, pd.DataFrame], forecasts_dir: str) -> None:
    if not os.path.exists(forecasts_dir):
        os.mkdir(forecasts_dir)
    for model_name, model_predictions in models_forecasts.items():
        model_predictions.to_csv(os.path.join(forecasts_dir, f"{model_name}.csv"))


def load_forecast(forecasts_dir: str) -> Dict[str, pd.DataFrame]:
    return {
        single_model_forecast_path.replace('.csv', ''):
        pd.read_csv(os.path.join(forecasts_dir, single_model_forecast_path), index_col=0)
        for single_model_forecast_path in os.listdir(forecasts_dir)
    }


def eval_predictions(ts_signal: pd.DataFrame,
                     predictions: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    error_metrics = {}
    for model_name, model_prediction in predictions.items():
        error_metrics[model_name] = pd.DataFrame(
            columns=["mase", "mape", "smape"], index=ts_signal.columns
        )
        error_metrics[model_name]["mase"] = mase(
            ts_signal, model_prediction, seasonality=WEEKS_IN_YEAR
        ).round(4)
        error_metrics[model_name]["mape"] = mape(ts_signal, model_prediction).round(4)
        error_metrics[model_name]["smape"] = smape(ts_signal, model_prediction).round(4)
    return error_metrics


def mean_errors_ranking(error_metrics: Dict[str, pd.DataFrame], output_path: str) -> None:
    mean_result = pd.DataFrame(
        {model_name: model_metrics.mean() for model_name, model_metrics in error_metrics.items()}
    ).T.sort_values(by="mase", ascending=False)
    mean_result.to_csv(output_path)


def errors_distribution(error_metrics: Dict[str, pd.DataFrame], output_path: str) -> None:
    data = pd.DataFrame(
        {model_name: error_metrics[model_name]["mase"].values for model_name in error_metrics}
    )
    num_models = len(data.columns)
    colors = [cm.tab20(i / num_models) for i in range(num_models)]
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 8))
    boxplot = ax.boxplot(
        [data[col] for col in data.mean().sort_values().index],
        patch_artist=True,
        labels=data.columns,
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2, color='black'),
        meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=8),
        showfliers=False
    )
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)
        patch.set_linewidth(1.5)
        patch.set_linestyle('-')
    for i, median in enumerate(boxplot['medians']):
        median_value = median.get_ydata()[0]
        ax.text(
            i + 1,
            median_value,
            f'{median_value:.2f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='black'
        )
    ax.set_title('Comparison of MAPE Across Forecast Models', fontsize=18, fontweight='bold')
    ax.set_xlabel('Forecast Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(output_path, format="png")
