# -*- coding: utf-8 -*-
import numpy as np

def get_whole_Z(n, N, model_list, W, Z):
    '''
    Return the Z vector of the imputed whole sample. Here the first n rows are from pilot sample.
    The remaining N-n rows are from imputed sample.

    Parameters
    ----------
    n : int
        The pilot sample size.
    N : int
        The whole sample size.
    model_list : list
        A list of estimated logistic model.
    W : ndarray of shape (N,r)
        The auxiliary features.
    Z : ndarray of shape (N,p)
        The binary vector.

    Returns
    -------
    Z_new : ndarray of shape (N,p)
        The binary feature with imputed values.

    '''
    p = Z.shape[1]   # The dimension of Z.
    Z0 = Z[0:n]      # Get the pilot sample.
    Z_new = []       # To store the features combining the imputed results.
    for i in range(p):
        model_logit_curr = model_list[i] # The current logistic regression model.
        Z_hat_curr = model_logit_curr.predict_proba(W[n:N,:])[:,1] # Impute the binary feature in the imputed sample by its probability value.
        Z_new_curr = np.hstack([Z0[:,i], Z_hat_curr]) # Combine the feature with the pilot sample.
        Z_new.append(Z_new_curr) # Store the results
    Z_new = np.array(Z_new).T    # Transform the results into an array.
    return Z_new

def get_U_hat(n, N, model_list, W, Z, X):
    '''
    Return the \hat U vector of the imputed whole sample. Here the first n rows are from pilot sample.
    The remaining N-n rows are from imputed sample.
    
    Parameters
    ----------
    n : int
        The pilot sample size.
    N : int
        The whole sample size.
    model_list : list
        A list of estimated logistic model.
    W : ndarray of shape (N,r)
        The auxiliary features.
    Z : ndarray of shape (N,p)
        The binary vector.
    X : ndarray of (N,q)
        The full observed feature vector.

    Returns
    -------
    U_hat : ndarray of shape (N,p+q)
        The features with imputed binary values.

    '''
    Z_new = get_whole_Z(n, N, model_list, W, Z) # Get the Z feature of the imputed whole sample. 
    U_hat = np.hstack([Z_new, X]) # Combine the features
    return U_hat

def get_Z0_hat(W0, Z0, model_list):
    '''
    Return the predicted Z vector of the pilot sample.

    Parameters
    ----------
    W0 : ndarray of shape (n,r)
        The auxiliary features for pilot sample.
    Z0 : ndarray of shape (n,p)
        The binary vector for the pilot sample.
    model_list : list
        A list of estimated logistic model.

    Returns
    -------
    Z0_hat : ndarray of shape (n,p)
        The predicted binary vector for the pilot sample.

    '''
    p = Z0.shape[1] # The dimension of Z.
    Z0_hat = []     # To store the features with the imputed results.
    for i in range(p):
        model_logit_curr = model_list[i] # The current logistic regression model.
        Z0_hat_curr = model_logit_curr.predict_proba(W0)[:,1] # Impute the binary feature in the pilot sample by its probability value.
        Z0_hat.append(Z0_hat_curr) # Store the results
    Z0_hat = np.array(Z0_hat).T    # Transform the results into an array.
    return Z0_hat

def get_U0_hat(W0, Z0, model_list, X0):
    '''
    Return the \hat U vector of the pilot sample. Here the binary features are replaced by its imputed values.

    Parameters
    ----------
    W0 : ndarray of shape (n,r)
        The auxiliary features for pilot sample.
    Z0 : ndarray of shape (n,p)
        The binary vector for the pilot sample.
    model_list : list
        A list of estimated logistic model.
    X0 : ndarray of (n,q)
        The full observed feature vector.

    Returns
    -------
    U0_hat : ndarray of shape (n,p+q)
        The features with imputed binary values for pilot sample.

    '''
    Z0_hat = get_Z0_hat(W0, Z0, model_list) # Get the imputed Z feature of the pilot sample. 
    U0_hat = np.hstack([Z0_hat, X0]) # Combine the features
    return U0_hat
