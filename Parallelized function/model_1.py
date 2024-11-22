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
pca_percent = 0.8

y_train, y_test, y_train_deseason, y_test_deseason, y_train_season, y_test_season, \
exog_variables_train, scalers, exog_variables_train_stand, exog_variables_test, \
exog_variables_test_stand, pca_train, pca_test = split_data_into_series(datasets, pca_percent, regex_choice)

p, d, q = 1, 0, 0 
P, D, Q, s = 0, 0, 0, 0 

smoother_output = 0

def fit_model(hour):
    predictions_dict = {}
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

    predictions_dict[hour] = predictions + y_test_season[hour]

    print(f'Hour {hour} fitted')

    return predictions_dict

if __name__ == '__main__':
    with Pool(processes=24) as pool:
        results = pool.map(fit_model, range(24))

    # Combine results into a single dictionary
    predictions_dict = {hour: result[hour] for hour, result in enumerate(results)}

    combined_predictions, combined_test = combine_24_hour_data(predictions_dict, y_test)

    rmse, smape, rmae = out_of_sample_pred(combined_test, combined_predictions)

    model_1 = np.array([[rmse], [smape], [rmae]])
    combined_predictions_model_1 = combined_predictions

    print(f'SMAPE baseline (24 lags) out of sample prediction: {model_1}')

    plot_actual_vs_predicted(combined_test, combined_predictions_model_1, "Seasonal SARIMA predictions")
