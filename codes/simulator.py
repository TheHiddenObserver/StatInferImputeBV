# -*- coding: utf-8 -*-

import numpy as np


def logit(x):
    '''
    The logit function.
    '''
    return 1/(1+np.exp(-x))

def simulator_logit(n, N, k = 1, C = 0, t = 0):
    '''
    Generate the dataset for logistic regression.

    Parameters
    ----------
    n : int
        The pilot sample size.
    N : int
        The whole sample size.
    k : float, optional
        The strength of \|alpha_j\|. The default is 1.
    C : float, optional
        The constant controlling the maximum positive proportion. The default is 0.
    t : float, optional
        The constant controlling the imbalanceness. The default is 0.

    Returns
    -------
    W : ndarray of shape (N,r)
        The auxiliary features.
    Z : ndarray of shape (N,p)
        The binary vector.

    '''
    r, p = 9, 2 
    sigmaw = 0.25 ** abs(np.subtract.outer(np.arange(r-1), np.arange(r-1))) # The covariance matrix of \tilde W_i
    W_tilde = np.random.multivariate_normal(np.zeros(r-1),sigmaw,N,"raise") # Generate the \tilde W_i
    W = np.hstack([np.ones((N,1)), W_tilde])                                # Combine the vector W_i
    
    # Set the parameters of logit regression. 
    alphan = - C * np.log(n)    # Generate alpha_n. 
    alpha1 = np.hstack([alphan, np.array([2.5,0,0,1.5,0,0,-4,0]) * k/2]) # Compute alpha_1
    alpha2 = np.hstack([t * alphan, np.array([1,1,1,-3*np.sqrt(2)/2,1/3,0,0,0]) * k]) # Compute alpha_2
    alpha = np.vstack([alpha1, alpha2]).T  # Combine the vectors
    
    prob = logit(W.dot(alpha))  # Compute the probability P(Z_{ij} = 1).
    
    Z = np.random.binomial(1, prob) # Generate the Z vector,
    
    return W, Z

def simulator_linear(n, N, Z, sigma = 1):
    '''
    Generate the variables of the linear regression. 
    Parameters
    ----------
    N : int
        The whole sample size.
    Z : ndarray of shape (N,p)
        The binary vector generated from simulator_logit.
    sigma : float, optional
        The variance of random error. The default is 1.

    Returns
    -------
    X : ndarray of (N,q)
        The feature vector.
    Y : ndarray of (N,)
        The response vector.

    '''
    p, q = 2, 7
    beta = np.array([3,0]) # The coefficients of beta
    gamma = np.array([1,1.5,0,0,0,2,0]) # The coefficients of gamma.
    sigmax = 0.5 ** abs(np.subtract.outer(np.arange(q-1), np.arange(q-1))) # The covariance matrix of \tilde X_i.
    X_tilde = np.random.multivariate_normal(np.zeros(q-1),sigmax,N,"raise") + 1 # Generate \tilde X_i
    X = np.hstack([np.ones((N,1)),X_tilde]) # Combine the vector X_i
    eps = np.random.normal(size = N, scale = sigma) # Generate the random error.
    Y = Z.dot(beta) + X.dot(gamma) + eps # Generate Y. 
    return X, Y

def simulator(seed, n, N, k = 1, sigma = 1, C = 0, t = 0):
    '''
    Generate all the random variables. 
    Parameters
    ----------
    seed : int 
        The random seed.
    n : int
        The pilot sample size.
    N : int
        The whole sample size.
    k : float, optional
        The strength of \|alpha_j\|. The default is 1.
    sigma : float, optional
        The variance of random error. The default is 1.
    C : float, optional
        The constant controlling the maximum positive proportion. The default is 0.
    t : float, optional
        The constant controlling the imbalanceness. The default is 0.

    Returns
    -------
    W : ndarray of shape (N,r)
        The auxiliary features.
    Z : ndarray of shape (N,p)
        The binary vector.
    X : ndarray of (N,q)
        The feature vector.
    Y : ndarray of (N,)
        The response vector.

    '''
    np.random.seed(seed)     # Set the random seed.
    W, Z = simulator_logit(n, N, k, C, t)  # Generate (W,Z)
    X, Y = simulator_linear(n, N, Z, sigma) # Generate (X,Y)
    return W, Z, X, Y
    
    
if __name__ == "__main__":
    r, p = 9, 2
    n, N = 200, 10000
    W, Z, X, Y = simulator(42, n, N, k = 1)

    
    
    
