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


datasets_train = {hour: datasets[hour][(datasets[hour]['DATE'] >= '2014-01-01') & (datasets[hour]['DATE'] < '2023-01-01')] for hour in range(24)}
datasets_test = {hour: datasets[hour][datasets[hour]['DATE'] >= '2023-01-01'] for hour in range(24)}

y_train = {hour: datasets_train[hour]['DK2_spot'].to_numpy().reshape(-1, 1) for hour in range(24)}
y_test = {hour: datasets_test[hour]['DK2_spot'].to_numpy().reshape(-1, 1) for hour in range(24)}

y_train_deseason = {hour: datasets_train[hour]['deseasonalized'].to_numpy().reshape(-1, 1) for hour in range(24)}
y_test_deseason = {hour: datasets_test[hour]['deseasonalized'].to_numpy().reshape(-1, 1) for hour in range(24)}

y_train_season = {hour: datasets_train[hour]['seasonal_component'].to_numpy().reshape(-1, 1) for hour in range(24)}
y_test_season = {hour: datasets_test[hour]['seasonal_component'].to_numpy().reshape(-1, 1) for hour in range(24)}

exog_variables_train = {
    hour: datasets_train[hour].filter(regex="^(sun_|wind_|temp_|DK2_spot_lag_)").to_numpy()
    for hour in range(24)
}

scalers = {}
exog_variables_train_stand = {}

# Fit and transform each hour’s data
for hour in range(24):
    # Fit the scaler on the training data for the current hour
    scaler = StandardScaler().fit(exog_variables_train[hour])
    scalers[hour] = scaler  # Store the scaler for each hour
    
    # Transform the training data
    exog_variables_train_stand[hour] = scaler.transform(exog_variables_train[hour])

exog_variables_test = {
    hour: datasets_test[hour].filter(regex="^(sun_|wind_|temp_)").to_numpy()
    for hour in range(24)
}

exog_variables_test_stand = {}

# Transform each hour’s test data using the corresponding scaler
for hour in range(24):
    # Use the scaler fitted on the training data for this hour
    exog_variables_test_stand[hour] = scalers[hour].transform(exog_variables_test[hour])
    

pca_train = {}
pca_test = {}

for hour in range(24):
    data_dimreduc, _, _, _, _ = PCA_dimreduc(exog_variables_train_stand[hour], exog_variables_train_stand[hour], 0.80)
    pca_train[hour] = data_dimreduc

    data_dimreduc, _, _, _, _ = PCA_dimreduc(exog_variables_train_stand[hour], exog_variables_test_stand[hour], 0.80)
    pca_test[hour] = data_dimreduc

# Define parameters
p, d, q = 1, 0, 0
P, D, Q, s = 0, 0, 0, 0
smoother_output = 0

predictions_dict = {}

def process_hour(hour):
    model = SARIMAX(
        y_train_deseason[hour],
        exog=pca_train[hour],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        **{'smoother_output': smoother_output}
    )
    results = model.fit(disp=False,  maxiter=20000)

    new_results = results.append(y_test_deseason[hour], exog=pca_test[hour], refit=False)

    predictions = new_results.predict(start=len(y_train_deseason[hour]), end=len(y_train_deseason[hour]) + len(y_test_deseason[hour]) - 1, 
                                      exog=pca_test[hour]).reshape(-1, 1)

    predictions_dict[hour] = predictions + y_test_season[hour]

    print(f'Hour {hour} fitted')

    # Add back the seasonal component
    return hour, predictions_dict

if __name__ == '__main__':
    # Use a Pool to parallelize the process_hour function
    with Pool() as pool:
        results = pool.map(process_hour, range(24))

    # Combine the predictions and test data
    combined_predictions, combined_test = combine_24_hour_data(predictions_dict, y_test)

    # Calculate performance metrics
    rmse, smape, rmae = out_of_sample_pred(combined_test, combined_predictions)
    model_6 = np.array([[rmse], [smape], [rmae]])
    combined_predictions_model_6 = combined_predictions

    # Print the results
    print(f'SMAPE baseline (24 lags) out of sample prediction: {model_6}')

    # Plot the actual vs predicted values
    plot_actual_vs_predicted(combined_test, combined_predictions_model_6, "Seasonal SARIMA predictions")
