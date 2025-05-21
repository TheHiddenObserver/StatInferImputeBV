# -*- coding: utf-8 -*-
import scipy.linalg
import numpy as np
from simulator import simulator
from fit_model import fit_impute_model, fit_linear_regression_model
from utils import get_U_hat, get_U0_hat

def error_var_estimate(U0, U_hat, Y0, Y, model_pilot ):
    '''
    Compute the estimated error variance.

    Parameters
    ----------
    U0: ndarray of shape (n,p+q)
        The features with true binary values.
    U_hat : ndarray of shape (N,p+q)
        The features with imputed binary values.
    Y0 : ndarray of shape (n,)
        The response vector for the pilot sample.
    Y : ndarray of shape (N,)
        The response vector for the whole sample.
    model_pilot : sklearn.LinearRegression
        The fitted linear regression model using the pilot sample.

    Returns
    -------
    sigma_hat_pilot : float
        The estimated error variance.

    '''
    n = len(Y0)
    coef_pilot = model_pilot.coef_    # Get the coefficient of the pilot estimator.
    # U0, Y0 = U_hat[pilot_index,:], Y[pilot_index]       # Get U0 and Y0.
    sigma_hat_pilot = np.sum((Y0 - model_pilot.predict(U0)) ** 2) / (n-len(coef_pilot)) # Compute the estimator for the error variance
    return sigma_hat_pilot

def cov_pilot_estimate(U0, U0_hat, U_hat):
    '''
    Compute some covariance estimator related to sigmau.

    Parameters
    ----------
    U0 : ndarray of shape (n,p+q)
        The features with binary values for pilot sample.
    U0_hat : ndarray of shape (n,p+q)
        The features with imputed binary values for pilot sample.
    U_hat : ndarray of shape (N,p+q)
        The features with imputed binary values.

    Returns
    -------
    sigmau_hat : ndarray of shape (p+q,p+q)
        The matrix \hat\Sigma_u.
    sigmau_tilde : ndarray of shape (p+q,p+q)
        The matrix \tilde\Sigma_u.

    '''
    
    # U0 = U_hat[pilot_index,:]
    n = len(U0)
    sigmau_hat = U0.T.dot(U0) / n
    sigmau_tilde = U0_hat.T.dot(U0_hat) / n
    return sigmau_hat, sigmau_tilde


def cov_estimate(W0, U0, U0_hat, U_hat, Y0, Y, model_pilot, p = 2):
    '''
    Compute the estimated covariance matrix for the impute estimator. 

    Parameters
    ----------
    W0 : ndarray of shape (n,r)
        The auxiliary features for pilot sample.
    U0 : ndarray of shape (n,p+q)
        The features with true binary values for pilot sample.
    U0_hat : ndarray of shape (n,p+q)
        The features with imputed binary values for pilot sample.
    U_hat : ndarray of shape (N,p+q)
        The features with imputed binary values.
    Y0 : ndarray of (n,)
        The response vector for the pilot sample.
    Y : ndarray of (N,)
        The response vector for the whole sample.
    model_pilot : sklearn.LinearRegression
        The fitted linear regression model using the pilot sample.
    p : int, optional
        The dimension of Z. The default is 2.

    Returns
    -------
    sigmaimp_hat : ndarray of shape (p+q,p+q)
        The estimated covariance matrix for the imputed estimator.
    sigma_hat_pilot : float
        The estimated error variance.

    '''
    r, q = W0.shape[1], U_hat.shape[1] - p # Get r and q.
    coef_pilot = model_pilot.coef_    # Get the coefficient of the pilot estimator.
    Z0_hat = U0_hat[:,0:p]            # Get Z0_hat.
    n, N = len(Y0), len(Y)
    # U0, Y0 = U_hat[pilot_index,:], Y[pilot_index]       # Get U0 and Y0.
    
    # Compute the asymptotic covariance matrix of \hat A. 
    matrix_list = []
    for i in range(p):
        matrix_list.append(np.linalg.inv((W0.T * (Z0_hat[:,i] * (1-Z0_hat[:,i]))).dot(W0) / n)) # Compute the information matrix.
    info_fin = scipy.linalg.block_diag(*matrix_list)  # Compute the inverse of information matrix.
    
    # Compute the estimator of \Sigma_{uw}
    beta_pilot = coef_pilot[0:p]  # Get the pilot estimator of beta.
    beta_pilotd = beta_pilot * Z0_hat * (1-Z0_hat) # Compute \beta^\top D_{ip}
    sigmauw = np.einsum("ki, kj -> kij", U0_hat, W0) # Prepare for U_i W_i^\top
    sigma_uwhat = np.mean(np.einsum('kb, kcd -> kcbd',beta_pilotd , sigmauw).reshape(n,(p+q),p*r), axis = 0) # Compute the kronecker product
    
    sigma_hat_pilot = np.sum((Y0 - model_pilot.predict(U0)) ** 2) / (n-len(coef_pilot)) # Compute the estimator for the error variance
    omega_hat = (((beta_pilot * Z0_hat * (1-Z0_hat)).dot(beta_pilot) + sigma_hat_pilot).reshape(n,1) * U0_hat).T.dot(U0_hat) / n # Compute the \hat\omega
    
    sigmau_tilde = U0_hat.T.dot(U0_hat) / n # Compute the estimator related to \Sigma_u
    sigmau_tilde_inv = np.linalg.inv(sigmau_tilde) # Compute the inverse of the estimator.
    
    sigmaimp_hat = sigmau_tilde_inv.dot(sigma_uwhat.dot(info_fin).dot(sigma_uwhat.T)/n + omega_hat / N).dot(sigmau_tilde_inv) # Compute the final covariance matrix estimator
    return sigmaimp_hat, sigma_hat_pilot

if __name__ == "__main__":
    r, p, q = 9, 2, 7
    n, N = 400, 10000
    W, Z, X, Y = simulator(42, n, N, k = 1)
    W0, Z0, X0, Y0 = W[0:n], Z[0:n], X[0:n], Y[0:n]
    
    U0 = np.hstack([Z0, X0])
    model_logit_list = fit_impute_model(W0, Z0)
    U_hat = get_U_hat(n, N, model_logit_list, W, Z, X)
    U0_hat = get_U0_hat(W0, Z0, model_logit_list, X0)
    
    model_pilot = fit_linear_regression_model(U0, Y0)
    model_impute = fit_linear_regression_model(U_hat, Y)
    sigmaimp_hat, sigma_hat_pilot = cov_estimate(W0, U0, U0_hat, U_hat, Y0, Y, model_pilot)
    
    sigmau_hat, sigmau_tilde = cov_pilot_estimate(U0, U0_hat, U_hat)
    se_impute = np.sqrt(np.diag(sigmaimp_hat))
    