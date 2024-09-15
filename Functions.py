import numpy as np
from scipy.stats import t

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
