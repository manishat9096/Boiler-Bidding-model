import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel, LinearRegressionModel
from darts.timeseries import concatenate
from darts.metrics import mae, mse, rmse

class ModelForecast:
    def __init__(self, model, target, split_size = 0.5, forecast_horizon = 100, stride = 100, past_covar = None, future_covar = None):
        self.model = model
        self.split_size = split_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.target = target
        self.target_train = None
        self.target_test = None
        self.past_covar = past_covar
        self.past_covar_train = None
        self.past_covar_test = None
        self.future_covar = future_covar
        self.future_covar_train = None
        self.future_covar_test = None

    def split_ts(self):
        self.target_train, self.target_test = self.target.split_after(self.split_size)
        if self.past_covar is not None:
            self.past_covar_train, self.past_covar_test = self.past_covar.split_after(self.split_size)
        if self.future_covar is not None:
            self.future_covar_train, self.future_covar_test = self.future_covar.split_after(self.split_size)

    def fit_model(self):
        self.model.fit(self.target_train, past_covariates = self.past_covar_train, future_covariates = self.future_covar_train)

    def backtest_historical_forecast(self):
   
        self.split_ts()
        self.fit_model()
        
        # print(f"MSE: {mae(series_air, pred):.2f}")
        # print(f"RMSE: {mae(series_air, pred):.2f}")

        gp_backtest_forecasts = self.model.historical_forecasts(
            series=concatenate([self.target_train, self.target_test], axis=0),
            past_covariates = concatenate([self.past_covar_train, self.past_covar_test], axis=0),
            future_covariates = concatenate([self.future_covar_train, self.future_covar_test], axis=0),
            start=1,
            forecast_horizon = self.forecast_horizon,
            enable_optimization =False,
            num_samples =1,
            predict_likelihood_parameters =False,    
            stride = self.stride,
            retrain =False,
            last_points_only=False,
            verbose=True
        )
        # print(f"MAE: {mae(concatenate([self.target_train, self.target_test], axis=0), concatenate(gp_backtest_forecasts, axis=0)):.2f}")
        gp_backtest_forecasts_df = pd.DataFrame()
        for idx, forecast in enumerate(gp_backtest_forecasts):
            forecast_df = forecast.pd_dataframe()
            # forecast_df['idx'] = idx
            gp_backtest_forecasts_df = pd.concat([gp_backtest_forecasts_df, forecast_df])
        gp_backtest_forecasts_df['y_actual'] = concatenate([self.target_train, self.target_test], axis=0).pd_dataframe()
    
        return gp_backtest_forecasts_df, concatenate(gp_backtest_forecasts, axis=0)
