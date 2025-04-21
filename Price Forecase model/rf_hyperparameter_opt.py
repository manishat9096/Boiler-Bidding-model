import optuna
from optuna.samplers import TPESampler
from darts.timeseries import concatenate
from darts.models import RandomForest
from pre_processing_model import PreProcessing
from forecast_model import ModelForecast
from darts.metrics import mae, mse, rmse
from plotly.io import show

def parameter_optimisation(afrr_energy_UP_ts_train, afrr_energy_UP_ts_test,
                            past_features_UP_ts_train, past_features_UP_ts_test, 
                            future_features_UP_ts_train, future_features_UP_ts_test,  n_trials = 10, output_chunk_length = 50):
    # define a objective function that returns the error after model.predict
    # run study and trial on the error to minimise
    # best hyperparameters is the optimal values of the Optimisation

    def objective(trial):
        """
        Paramters to optimize in RF: 
        lags, lags_past_covariates, lags_future_covariates, n_estimators, max_depth (deafult range 30-40),
        min_samples_split = 2, min_samples_leaf=1 (default)
        """
        #define range of all the parameters
        lags_max = trial.suggest_int('lags_max', 100, 200)
        lags_past_max = trial.suggest_int('lags_past_max', 2, 100)
        lags_future_max = trial.suggest_int('lags_future_max', 2, 100)
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 10, 30)
        # define model RF
        rf_model = RandomForest(
            lags = list(range(-1, -lags_max, -1)),
            lags_past_covariates = list(range(-1, -lags_past_max, -1)),
            lags_future_covariates = list(range(-1, -lags_future_max, -1)),
            output_chunk_length = output_chunk_length,
            n_estimators = n_estimators,
            max_depth= max_depth
        )

        try:
            rf_model.fit(afrr_energy_UP_ts_train, past_covariates = past_features_UP_ts_train, future_covariates = future_features_UP_ts_train)

            y_pred = rf_model.predict(n=len(afrr_energy_UP_ts_test),
                                      past_covariates = concatenate([past_features_UP_ts_train, past_features_UP_ts_test], axis=0),
                                      future_covariates = concatenate([future_features_UP_ts_train, future_features_UP_ts_test], axis=0))

            # calculate error
            rmse_error = rmse(afrr_energy_UP_ts_test, y_pred)
            return rmse_error
        except Exception as e:
            print("Error: ", e)
            return float('inf')

    sampler = TPESampler()
    study = optuna.create_study(sampler = sampler)
    study.optimize(objective, n_trials = n_trials, show_progress_bar = True)
    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)

    best_params = study.best_params # is a dictionary
    print(f"Best parameters for RF model: {best_params}")
    print(f"Best RMSE: {study.best_value}")

    return best_params, study

if __name__ == "__main__":

    # Fetch the timeseries for the study
    start_date = '2025-03-01'
    end_date = '2025-03-18'
    # load data from energinet api (default resolution 1 min) and convert to Timeseries
    preprocess_data = PreProcessing(start_date, end_date)
    preprocess_data.fetch_data()
    afrr_energy_UP_ts, afrr_energy_DOWN_ts, past_features_UP_ts, past_features_DOWN_ts, future_features_UP_ts, future_features_DOWN_ts  = preprocess_data.df_to_ts()
    print('Timeseries loaded from Preprocessing Module')
    
    #split the timeseries 
    split_size = 0.6
    afrr_energy_UP_ts_train, afrr_energy_UP_ts_test = afrr_energy_UP_ts.split_after(split_size)
    past_features_UP_ts_train, past_features_UP_ts_test = past_features_UP_ts.split_after(split_size)
    future_features_UP_ts_train, future_features_UP_ts_test = future_features_UP_ts.split_after(split_size)

    afrr_energy_DOWN_ts_train, afrr_energy_DOWN_ts_test = afrr_energy_DOWN_ts.split_after(split_size)
    past_features_DOWN_ts_train, past_features_DOWN_ts_test = past_features_DOWN_ts.split_after(split_size)
    future_features_DOWN_ts_train, future_features_DOWN_ts_test = future_features_DOWN_ts.split_after(split_size)

    # pass the timeseries to the optimisation fn
    n_trials = 10
    output_chunk_length = 30

    best_parameters, study = parameter_optimisation(afrr_energy_DOWN_ts_train, afrr_energy_DOWN_ts_test,
                                             past_features_DOWN_ts_train, past_features_DOWN_ts_test, 
                                             future_features_DOWN_ts_train, future_features_DOWN_ts_test, 
                                             n_trials, output_chunk_length)
    
    # now run the ModelForecast module with these paramters
    # define the rf model with the updated parameters
