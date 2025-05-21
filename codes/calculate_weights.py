# -*- coding: utf-8 -*-
import numpy as np
from simulator import simulator
from fit_model import fit_impute_model, fit_linear_regression_model
from cov_impute import cov_pilot_estimate, cov_estimate
from utils import get_U_hat, get_U0_hat
from evaluation import evaluation_mse, evaluation_mse_coef

def calculate_optimal_weights(U0, U0_hat, U_hat, sigma_hat_pilot, sigmaimp_hat):
    '''
    Estimate the optimal weights for theta.w.

    Parameters
    ----------
    U0: ndarray of shape (n,p+q)
        The features with true binary values.
    U0_hat : ndarray of shape (n,p+q)
        The features with imputed binary values for pilot sample.
    U_hat : ndarray of shape (N,p+q)
        The features with imputed binary values.
    sigma_hat_pilot : float
        The estimated error variance.
    sigmaimp_hat : ndarray of shape (p+q,p+q)
        The estimated covariance matrix for the imputed estimator.
    Returns
    -------
    w_hat : float
        The estimated optimal weight to combine pilot and imputed estimator.

    '''
    n, N = len(U0), len(U_hat)
    sigmau_hat, sigmau_tilde = cov_pilot_estimate(U0, U0_hat, U_hat) # Compute the estimator related to \Sigma_u
    
    sigmau_hat_inv = np.linalg.inv(sigmau_hat) # Compute the inverse of the estimator.
    tr_sigmau_hat_inv = np.sum(np.diag(sigmau_hat_inv)) # Compute the trace of the inverse of the estimator.
    
    sigmau_tilde_inv = np.linalg.inv(sigmau_tilde) # Compute the inverse of the estimator.
    tr_sigmau_tilde_inv = np.sum(np.diag(sigmau_tilde_inv)) # Compute the trace of the inverse of the estimator.
        
    tr_sigmaimp_hat = np.sum(np.diag(sigmaimp_hat)) # Compute the trace of the inverse of the estimator.
    
    w_hat = (tr_sigmaimp_hat - sigma_hat_pilot * tr_sigmau_tilde_inv / N ) / (sigma_hat_pilot * tr_sigmau_hat_inv /n + tr_sigmaimp_hat - 2 * sigma_hat_pilot * tr_sigmau_tilde_inv / N ) # Compute the estimated optimal weights.
    w_hat = max(min(1, w_hat),0)
    return w_hat

if __name__ == "__main__":
    r, p, q = 9, 2, 7
    n, N = 400, 10000
    W, Z, X, Y = simulator(42, n, N, k = 1)
    W0, Z0, X0, Y0 = W[0:n], Z[0:n], X[0:n], Y[0:n]
    pilot_index = np.arange(n)
    
    U0 = np.hstack([Z0, X0])
    model_logit_list = fit_impute_model(W0, Z0)
    U_hat = get_U_hat(n, N, model_logit_list, W, Z, X)
    U0_hat = get_U0_hat(W0, Z0, model_logit_list, X0)
    
    model_pilot = fit_linear_regression_model(U0, Y0)
    model_impute = fit_linear_regression_model(U_hat, Y)
    sigmaimp_hat, sigma_hat_pilot = cov_estimate(W0, U0, U0_hat, U_hat, Y0, Y, model_pilot)
    
    sigmau_hat, sigmau_tilde = cov_pilot_estimate(U0, U0_hat, U_hat)
    se_impute = np.sqrt(np.diag(sigmaimp_hat))
    
    w_hat = calculate_optimal_weights(U0, U0_hat, U_hat, sigma_hat_pilot, sigmaimp_hat)
    theta_w = model_pilot.coef_ * w_hat + model_impute.coef_ * (1-w_hat)
    
    err_pilot = evaluation_mse(model_pilot)
    err_impute = evaluation_mse(model_impute)
    err_w = evaluation_mse_coef(theta_w)
    