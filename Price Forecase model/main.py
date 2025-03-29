import pandas as pd
import numpy as np
from darts.models import LinearRegressionModel, RandomForest
from pre_processing_model import PreProcessing
from forecast_model import ModelForecast
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from darts.metrics import mae, mse, rmse
import joblib

start_date = '2025-03-14'
end_date = '2025-03-15'
# load data from energinet api (default resolution 1 min) and convert to Timeseries
preprocess_data = PreProcessing(start_date, end_date)
preprocess_data.fetch_data()
afrr_energy_UP_ts, afrr_energy_DOWN_ts, past_features_UP_ts, past_features_DOWN_ts, future_features_UP_ts, future_features_DOWN_ts  = preprocess_data.df_to_ts()
output_chunk_length = 30 #minutes

model = 'UP'

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
    max_depth = 10
)

# use the forecast model to backtest on the timeseries data
forecast = ModelForecast(
    model=rf_model,
    split_size=0.5,
    forecast_horizon = 30, #minutes step
    stride = 30, #minutes
    target=afrr_energy_UP_ts,
    past_covar= past_features_UP_ts,
    future_covar=future_features_UP_ts
)

output_df, output_ts = forecast.backtest_historical_forecast()

# re-normalise the output dataset
if model == 'UP':
    scaler = joblib.load('scaler_UP.pkl')
    actual_ts = scaler.inverse_transform(afrr_energy_UP_ts)
    forecasted_ts = scaler.inverse_transform(output_ts)
elif model == 'DOWN':
    scaler = joblib.load('scaler_DOWN.pkl')
    actual_ts = scaler.inverse_transform(afrr_energy_DOWN_ts)
    forecasted_ts = scaler.inverse_transform(output_ts)

Y_df = pd.merge(actual_ts.pd_dataframe(), forecasted_ts.pd_dataframe(), left_index=True, right_index=True, how='right')

# error metrics
print(f"MAE_ts: {mae(actual_ts, forecasted_ts):.2f}")
print(f"MSE_ts: {mse(actual_ts, forecasted_ts):.2f}")
print(f"RMSE_ts: {rmse(actual_ts, forecasted_ts):.2f}")

print("MAE:", mean_absolute_error(Y_df['aFRR_UpActivatedPriceEUR_x'], Y_df['aFRR_UpActivatedPriceEUR_y']))
print("MSE:", mean_squared_error(Y_df['aFRR_UpActivatedPriceEUR_x'], Y_df['aFRR_UpActivatedPriceEUR_y']))
print("RMSE:", root_mean_squared_error(Y_df['aFRR_UpActivatedPriceEUR_x'], Y_df['aFRR_UpActivatedPriceEUR_y']))


# plot the results
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=Y_df.index, y=Y_df['aFRR_UpActivatedPriceEUR_y'], mode='lines', name='Forecasted Price Up'))
fig.add_trace(go.Scatter(x=Y_df.index, y=Y_df['aFRR_UpActivatedPriceEUR_x'], mode='lines', name='Actual Price UP'))
fig.update_layout(title='Forecast vs Actual price {model}',xaxis_title='Time',yaxis_title='Price (EUR/MWh)',legend_title='Legend',template='plotly_white')
fig.show()