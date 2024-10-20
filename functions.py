import numpy as np
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt

def VARlsExog(y, p, con, tr, exog):
    # Define dependent and independent variables for VAR estimation
    T, K = y.shape
    dep = y[p:T, :]
    
    # Independent variable
    indep = lagmatrix(y, p)
    indep = indep[p:, :]
    
    if con == 1:
        indep = np.hstack([indep, np.ones((len(indep), 1))])
    
    if tr == 1:
        indep = np.hstack([indep, np.arange(1, len(indep) + 1).reshape(-1, 1)])
    
    if isinstance(exog, np.ndarray):
        indep = np.hstack([indep, exog[p:, :]])
    
    T, Kp = indep.shape
    
    # Beta estimation
    Beta = np.linalg.inv(indep.T @ indep) @ indep.T @ dep

    # Residuals
    res = dep - indep @ Beta
    
    # Covariance matrix of residuals
    SIGMA = (res.T @ res) / T
    
    # Covariance matrix of Beta
    CovBeta = np.kron(SIGMA, np.linalg.inv(indep.T @ indep))
    
    tratioBeta = Beta.reshape(-1, order='F') / np.sqrt(np.diag(CovBeta))

    SEbeta = np.sqrt(np.diag(CovBeta)).reshape(-1, order='F')

    df = T - Kp  # degrees of freedom
    Pvalue = 2 * (1 - t.cdf(np.abs(tratioBeta), df))

    m = p
    aiccrit = np.log(np.linalg.det(SIGMA)) + 2 / T * (m * K ** 2 + K)       # AIC value
    hqccrit = np.log(np.linalg.det(SIGMA)) + 2 * np.log(np.log(T)) / T * (m * K ** 2 + K)  # HQC value
    siccrit = np.log(np.linalg.det(SIGMA)) + np.log(T) / T * (m * K ** 2 + K)  # SIC value

    return Beta, SEbeta, CovBeta, Pvalue, tratioBeta, res, indep, SIGMA, aiccrit, hqccrit, siccrit

def lagmatrix(data, lags):
    T, K = data.shape
    lagged_data = np.zeros((T, K * lags))
    for lag in range(1, lags + 1):
        lagged_data[lag:, (lag - 1) * K:lag * K] = data[:T - lag, :]
    return lagged_data


def sdummy(nobs,freq):
    """
    PURPOSE: creates a matrix of seasonal dummy variables
    ---------------------------------------------------
    USAGE: y = sdummy(nobs,freq)
    where: freq = 4 for quarterly, 12 for monthly
    ---------------------------------------------------
    RETURNS:
    y = an (nobs x freq) matrix with 0's and 1's
    e.g., 1 0 0 0 (for freq=4)
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    ---------------------------------------------------
    """

    nobs_new = nobs + (freq - (nobs % freq))
    seas = np.zeros([nobs_new,freq])
    
    for i in range(1, nobs_new, freq):
        seas[i-1:i+freq-1,0:freq] = np.identity(freq)
    
    seas = seas[:nobs]
    return seas


def pfind(y, pmax):
    t, K = y.shape

    # Construct regressor matrix and dependent variable
    XMAX = np.ones((1, t - pmax))
    
    for i in range(1, pmax + 1):
        XMAX = np.vstack([XMAX, y[pmax - i:t - i, :].T])

    Y = y[pmax:t, :].T

    aiccrit = np.zeros((pmax + 1, 1))
    hqccrit = np.zeros((pmax + 1, 1))
    siccrit = np.zeros((pmax + 1, 1))

    # Evaluate criterion for p = 0,...,pmax
    for j in range(0, pmax + 1):
        m = j
        T = t - pmax
        X = XMAX[:j * K + 1, :]

        B = Y @ X.T @ np.linalg.inv(X @ X.T)

        SIGMA = (Y - B @ X) @ (Y - B @ X).T / T

        aiccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 / T * (m * K ** 2 + K)       # AIC value
        hqccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 * np.log(np.log(T)) / T * (m * K ** 2 + K)  # HQC value
        siccrit[j] = np.log(np.linalg.det(SIGMA)) + np.log(T) / T * (m * K ** 2 + K)  # SIC value

    # Rank models for p = 0,1,2,...,pmax
    aichat = np.argmin(aiccrit)
    hqchat = np.argmin(hqccrit)
    sichat = np.argmin(siccrit)

    infomat = np.hstack([siccrit, hqccrit, aiccrit])
    m = np.arange(0, pmax + 1).reshape(-1, 1)
    imat = np.hstack([m, infomat])

    LagInformationValue = pd.DataFrame(imat, columns=['Lag', 'SIC', 'HQ', 'AIC'])
    OptimalLag = pd.DataFrame([[sichat, hqchat, aichat]], columns=['SIC', 'HQ', 'AIC'])

    return LagInformationValue, OptimalLag

def smape(y_test, y_pred):
    numerator = np.abs(y_pred - y_test)
    denominator = (np.abs(y_test) + np.abs(y_pred)) / 2.0

    epsilon = 1e-32  # A small constant to avoid division by zero
    denominator = np.where(denominator == 0, epsilon, denominator)

    # Calculate sMAPE
    smape_values = numerator / denominator
    smape = np.mean(smape_values) * 100

    return smape



def VARLMtest(y, p, con, tr, exog, h):

    t, K = y.shape
    Beta, SEbeta, CovBeta, Pvalue, tratioBeta, residuals_u, indep, SIGMA_u, aiccrit, hqccrit, siccrit = VARlsExog(y, p, con, tr, exog)
    u_lags = lagmatrix(residuals_u, h)
    Beta, SEbeta, CovBeta, Pvalue, tratioBeta, residuals_e, indep, SIGMA_e, aiccrit, hqccrit, siccrit = VARlsExog(y, p, con, tr, u_lags)
    LML = (t - p)*(K-np.trace(np.linalg.inv(SIGMA_u) @ SIGMA_e));
    LMLpval = 1-chi2.cdf(LML,h*K**2);

    m = K * h
    q = 1/2 * K * m - 1
    s = ((K**2 * m**2 - 4) / (K**2 + m**2 - 5))**(1/2)
    N = (t - p) - K * p - m - (1/2) * (K - m + 1)

    FLMh = ((np.linalg.det(SIGMA_u) / np.linalg.det(SIGMA_e))**(1/s) - 1) * ((N * s - q) / (K * m))
    FLMh_pval = 1 - f.cdf(FLMh, h * K**2, N * s - q)

    
    Results = [[LML, FLMh], [LMLpval, FLMh_pval], [h, h]]

    lm_table = pd.DataFrame({
        'Measure': pd.Categorical(['Test statistic', 'p-value', 'Lag order']),
        'Breusch_Godfrey': [row[0] for row in Results],
        'Edgerton_Shukur': [row[1] for row in Results]
    })

    return Results, lm_table


def plot_actual_vs_predicted(y_test, predictions, graph_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[-168:], label='Actual', color='blue', alpha=1, linewidth=0.5)
    plt.plot(predictions[-168:], label='Predicted', color='red', alpha=1, linewidth=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(graph_name, fontsize=14)
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.show()
