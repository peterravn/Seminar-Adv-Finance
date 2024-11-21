import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from functions import *

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

# Load each
datasets = {}
for hour in range(24):
    df = pd.read_csv(f"Splits/dataset_hour_{hour}.csv")
    datasets[hour] = deseasonalize(df, "both")

regex_choice = "^(sun_|wind_|temp_)"
pca_percent = 0.7

y_train, y_test, y_train_deseason, y_test_deseason, y_train_season, y_test_season, \
exog_variables_train, scalers, exog_variables_train_stand, exog_variables_test, \
exog_variables_test_stand, pca_train, pca_test = split_data_into_series(datasets, pca_percent, regex_choice)

p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 7 

smoother_output = 0

def fit_model(hour):
    model = SARIMAX(
        y_train_deseason[hour],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        **{'smoother_output': smoother_output},
        trend='c'
    )
    results = model.fit(disp=False)

    new_results = results.append(y_test_deseason[hour], refit=False)

    predictions = new_results.predict(start=len(y_train_deseason[hour]), end=len(y_train_deseason[hour]) + len(y_test_deseason[hour]) - 1).reshape(-1, 1)

    predictions += y_test_season[hour]

    return hour, predictions

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    with Pool() as pool:
        results = pool.map(fit_model, range(24))

    # Combine results into a single dictionary
    predictions_dict = dict(results)

    combined_predictions, combined_test = combine_24_hour_data(predictions_dict, y_test)

    rmse, smape, rmae = out_of_sample_pred(combined_test, combined_predictions)

    model_5 = np.array([[rmse], [smape], [rmae]])
    combined_predictions_model_5 = combined_predictions

    print(f'SMAPE baseline (24 lags) out of sample prediction: {model_5}')

    plot_actual_vs_predicted(combined_test, combined_predictions_model_5, "Seasonal SARIMA predictions")

    np.save('Parallelized function/model_5.npy', model_5)
    np.save('Parallelized function/combined_predictions_model_5.npy', combined_predictions_model_5)
    