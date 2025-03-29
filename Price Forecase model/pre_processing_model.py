import numpy as np
import pandas as pd
from fetch_api import entsoe_api, energinet_api
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
import joblib

class PreProcessing:
    def __init__(self, start_date, end_date, resolution = '1min'):
        self.start_date = start_date
        self.end_date = end_date
        self.resolution = resolution
        self.df = None
        self.pipeline = Pipeline([MissingValuesFiller()])
        self.scaler = Scaler()

    def fetch_data(self):
        # Fetch data from API and resample to a common resolution.
        activated_capacity_MW = self.powersystemrightnow() # minute res
        afrr_capacity_prices = self.aFRRcapacitymarket() # Hour res
        afrr_energy_prices = self.aFRRenergymarket() # Second res
        spot_prices = self.spotprices() # Hour res

        # resample all datasets and merge into one
        activated_capacity_MW = activated_capacity_MW.resample(self.resolution).ffill()
        afrr_capacity_prices = afrr_capacity_prices.resample(self.resolution).ffill()
        afrr_energy_prices = afrr_energy_prices.resample(self.resolution).mean()
        spot_prices = spot_prices.resample(self.resolution).ffill()
        # merge all dataframes to one dataset
        self.df = pd.concat([afrr_energy_prices, afrr_capacity_prices, spot_prices, activated_capacity_MW], axis = 1)
    
    def process_df(self, df, time_col, cols):
        if df.empty:
            print(f'No data from this period {self.start_date} to {self.end_date}')
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        return df[cols]

    def powersystemrightnow(self):
        # fetch Power System Right Now dataset
        df = energinet_api('PowerSystemRightNow', self.start_date, self.end_date)
        df = self.process_df(df, 'Minutes1UTC', ['aFRR_ActivatedDK1'])
        return df

    def aFRRcapacitymarket(self):
        # fetch aFRR Capacity market dataset
        df = energinet_api('AfrrReservesNordic', self.start_date, self.end_date)
        df = df.loc[df['PriceArea'] == 'DK1']
        df = self.process_df(df, 'HourUTC', ['aFRR_UpCapPriceEUR', 'aFRR_DownCapPriceEUR'])
        return df

    def aFRRenergymarket(self):
        # fetch aFRR Energy market dataset
        df = energinet_api('AfrrEnergyActivated', self.start_date, self.end_date)
        df = df.loc[df['PriceArea'] == 'DK1']
        df = self.process_df(df, 'ActivationTime', ['aFRR_UpActivatedPriceEUR','aFRR_DownActivatedPriceEUR'])
        df['is_ActivatedUP'] = df['aFRR_UpActivatedPriceEUR'].notna().astype(int) # Binary Feature - Activation UP ON/OFF
        df['is_ActivatedDown'] = df['aFRR_DownActivatedPriceEUR'].notna().astype(int) # Binary Feature - Activation DOWN ON/OFF
        return df

    def spotprices(self):
        # fetch Elspot prices dataset
        df = energinet_api('Elspotprices', self.start_date, self.end_date)
        df = df.loc[df['PriceArea'] == 'DK1']
        df = self.process_df(df, 'HourUTC', ['SpotPriceEUR'])
        return df

    def df_to_ts(self):
        """Convert DataFrame to TimeSeries format."""
        if self.df is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        past_features_UP_price = self.df[['aFRR_ActivatedDK1', 'is_ActivatedUP']]
        past_features_DOWN_price = self.df[['aFRR_ActivatedDK1', 'is_ActivatedDown']]

        future_features_UP_price = self.df[['aFRR_UpCapPriceEUR', 'SpotPriceEUR']]
        future_features_DOWN_price = self.df[['aFRR_DownCapPriceEUR', 'SpotPriceEUR']]

        # convert to time series to create a multi-variate time series for the features and a univariate time series for the target
        target_UP = TimeSeries.from_series(self.df['aFRR_UpActivatedPriceEUR'])
        target_DOWN = TimeSeries.from_series(self.df['aFRR_DownActivatedPriceEUR'])
        past_features_UP = TimeSeries.from_dataframe(past_features_UP_price)
        past_features_DOWN = TimeSeries.from_dataframe(past_features_DOWN_price)
        future_features_UP = TimeSeries.from_dataframe(future_features_UP_price)
        future_features_DOWN = TimeSeries.from_dataframe(future_features_DOWN_price)
        
        target_UP= self.pipeline.fit_transform(target_UP)
        target_DOWN= self.pipeline.fit_transform(target_DOWN)
        past_features_UP=self.pipeline.fit_transform(past_features_UP)
        past_features_DOWN=self.pipeline.fit_transform(past_features_DOWN)
        future_features_UP=self.pipeline.fit_transform(future_features_UP)
        future_features_DOWN=self.pipeline.fit_transform(future_features_DOWN)

        scaled_target_UP = self.scaler.fit_transform(target_UP)
        joblib.dump(self.scaler, 'scaler_UP.pkl')
        scaled_target_DOWN = self.scaler.fit_transform(target_DOWN)
        joblib.dump(self.scaler, 'scaler_DOWN.pkl')

        return (scaled_target_UP,
                scaled_target_DOWN,
                self.scaler.fit_transform(past_features_UP),
                self.scaler.fit_transform(past_features_DOWN),
                self.scaler.fit_transform(future_features_UP),
                self.scaler.fit_transform(future_features_DOWN)
                )


if __name__ == "__main__":

    start_date = '2025-02-25'
    end_date = '2025-02-28'
    
    preprocess_data = PreProcessing(start_date, end_date)
    preprocess_data.fetch_data()
    afrr_energy_price_UP_ts_scl, afrr_energy_price_DOWN_ts_scl, past_features_UP_price_ts_scl, past_features_DOWN_price_ts_scl, future_features_UP_price_ts_scl, future_features_DOWN_price_ts_scl = preprocess_data.df_to_ts()