import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from functions import *

from multiprocessing import Pool

# Load each
datasets = {}
for hour in range(24):
    datasets[hour] = pd.read_csv(f"Splits/dataset_hour_{hour}.csv")

datasets_train = {hour: datasets[hour][(datasets[hour]['DATE'] >= '2014-01-01') & (datasets[hour]['DATE'] < '2023-01-01')].drop(['DATE'], axis=1).to_numpy() for hour in range(24)}
datasets_test = {hour: datasets[hour][datasets[hour]['DATE'] >= '2023-01-01'].drop(['DATE'], axis=1).to_numpy() for hour in range(24)}

y_train = {hour: datasets_train[hour][:, 0].reshape(-1, 1) for hour in range(24)}
y_test = {hour: datasets_test[hour][:, 0].reshape(-1, 1) for hour in range(24)}

weather_train = {hour: datasets_train[hour][:, 1:-1] for hour in range(24)}
weather_test = {hour: datasets_test[hour][:, 1:-1] for hour in range(24)}

pca_train = {}
pca_test = {}

for hour in range(24):
    data_dimreduc, _, _, _, _ = PCA_dimreduc(weather_train[hour], weather_train[hour], 0.85)
    pca_train[hour] = data_dimreduc

    data_dimreduc, _, _, _, _ = PCA_dimreduc(weather_train[hour], weather_test[hour], 0.85)
    pca_test[hour] = data_dimreduc

# Define your parameters
p, d, q = 1, 0, 0 
P, D, Q, s = 0, 0, 0, 0 
smoother_output = 0

def process_hour(hour):
    # Fit the SARIMAX model for the given hour
    model = SARIMAX(
        y_train[hour],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        **{'smoother_output': smoother_output}
    )
    results = model.fit(disp=False)
    
    # Append the test data and make predictions
    new_results = results.append(y_test[hour], refit=False)
    predictions = new_results.predict(
        start=len(y_train[hour]), 
        end=len(y_train[hour]) + len(y_test[hour]) - 1
    )
    
    # Return the hour and its corresponding predictions
    print(f'Hour {hour} done')

    return hour, predictions

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # Use a Pool to parallelize the process_hour function
    with Pool(6) as pool:
        print("Test")
        results = pool.map(process_hour, range(24))
    
    # Collect the predictions into a dictionary
    predictions_dict = dict(results)
    
    # Combine the predictions and test data
    combined_predictions, combined_test = combine_24_hour_data(predictions_dict, y_test)
    
    # Calculate performance metrics
    rmse, smape, rmae = out_of_sample_pred(combined_test, combined_predictions)
    model_1 = np.array([[rmse], [smape], [rmae]])
    combined_predictions_model_1 = combined_predictions
    
    # Print the results
    print(f'SMAPE baseline (24 lags) out of sample prediction: {model_1}')
    
    # Plot the actual vs predicted values
    plot_actual_vs_predicted(combined_test, combined_predictions_model_1, "Seasonal SARIMA predictions")