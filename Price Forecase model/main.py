import pandas as pd
import numpy as np
from darts.models import LinearRegressionModel, RandomForest
from pre_processing_model import PreProcessing
from forecast_model import ModelForecast
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from darts.metrics import mae, mse, rmse


start_date = '2025-02-25'
end_date = '2025-02-28'
# load data from energinet api (default resolution 1 min) and convert to Timeseries
preprocess_data = PreProcessing(start_date, end_date)
preprocess_data.fetch_data()
afrr_energy_UP_ts, afrr_energy_DOWN_ts, past_features_UP_ts, past_features_DOWN_ts, future_features_UP_ts, future_features_DOWN_ts  = preprocess_data.df_to_ts()

# define a LR Regressor model
# output_chunk_length = 48
# lr_model = LinearRegressionModel(
#     output_chunk_length=  output_chunk_length,
#     lags = list(range(-1, -120, -5)),
#     lags_past_covariates = list(range(-1, -20, -5)),
#     lags_future_covariates = list(range(-1, -20, -5))
# )

rf_model = RandomForest(
    lags = list(range(-1, -121, -5)),
    lags_past_covariates = [-1],
    lags_future_covariates = [-1],
    output_chunk_length = 48,
    n_estimators = 100,
    # criterion="absolute_error",
)

# use the forecast model to backtest on the timeseries data
forecast = ModelForecast(
    model=rf_model,
    split_size=0.4,
    forecast_horizon = 15, #minutes step
    stride = 15, #minutes
    target=afrr_energy_UP_ts,
    past_covar= past_features_UP_ts,
    future_covar=future_features_UP_ts
)

output_df, output_ts = forecast.backtest_historical_forecast()
# plot the results
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=output_df.index, y=output_df.iloc[:,0], mode='lines', name='Activation Price Up'))
fig.add_trace(go.Scatter(x=output_df.index, y=output_df.iloc[:,1], mode='lines', name='Actual Price UP'))
fig.update_layout(title='Time vs Activation Price Up & Actual Price Up',xaxis_title='Time',yaxis_title='Price (EUR/MWh)',legend_title='Legend',template='plotly_white')
fig.show()

print(f"MAE_ts: {mae(afrr_energy_UP_ts, output_ts):.2f}")
print(f"MSE_ts: {mse(afrr_energy_UP_ts, output_ts):.2f}")
print(f"RMSE_ts: {rmse(afrr_energy_UP_ts, output_ts):.2f}")

print("MAE:", mean_absolute_error(output_df.iloc[:,0], output_df.iloc[:,1]))
print("MSE:", mean_squared_error(output_df.iloc[:,0], output_df.iloc[:,1]))
print("RMSE:", root_mean_squared_error(output_df.iloc[:,0], output_df.iloc[:,1]))