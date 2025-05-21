# -*- coding: utf-8 -*-

import numpy as np
from simulator import simulator
from sklearn.linear_model import LogisticRegression, LinearRegression
from utils import *

def fit_impute_model(W0, Z0):
    '''
    Fit the impute model for logistic regression.
    Parameters
    ----------
    W0 : ndarray of shape (n,r)
        The auxiliary features for pilot sample.
    Z0 : ndarray of shape (n,p)
        The binary vector for the pilot sample.

    Returns
    -------
    model_list : list
        A list of estimated logistic model.

    '''
    p = Z0.shape[1]     # The dimension of Z
    model_list = []     # Store the regression model
    for i in range(p):  
        model_logit_curr = LogisticRegression(fit_intercept=False, 
                                              multi_class='multinomial', 
                                              solver='lbfgs',
                                              tol = 1e-12,
                                              max_iter=10000, 
                                              penalty = "none") # Define the model
        model_logit_curr.fit(W0, Z0[:,i]) # Fit the regression model
        model_list.append(model_logit_curr) # Store the regression model
    return model_list

def fit_linear_regression_model(U, Y):
    '''
    Fit the linear regression model.
    Parameters
    ----------
    U : ndarray of shape (None,p+q)
        The set of covariates.
    Y : ndarray of shape (None,)
        The response vector.

    Returns
    -------
    model 
        The fitted linear regression model.

    '''
    model = LinearRegression(fit_intercept=False).fit(U,Y) # Define and fit the linear regression model
    return model

    

if __name__ == "__main__":
    n, N = 200, 10000
    W, Z, X, Y = simulator(42, n, N, k = 1)
    W0, Z0, X0, Y0 = W[0:n], Z[0:n], X[0:n], Y[0:n]
    model_logit_list = fit_impute_model(W0, Z0)
    U_hat = get_U_hat(n, N, model_logit_list, W, Z, X)
    model_impute = fit_linear_regression_model(U_hat, Y)
    