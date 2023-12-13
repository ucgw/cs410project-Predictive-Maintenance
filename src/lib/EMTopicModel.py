import sys
import numpy as np
from scipy.special import logsumexp


def n_ones_matrix(n: int) -> np.ndarray:
    """
    Create a matrix of shape (n, 1) filled with ones.

    Parameters:
    - n (int): The number of rows in the matrix.

    Returns:
    - np.ndarray: A matrix of shape (n, 1) filled with ones.
    """
    return np.full((n, 1), 1)


def find_logW(X, log_P, log_pi):
    """
    Compute the weights W from the E step of expectation maximization.


    Parameters:

    X (np.array): A numpy array of the shape (N,d) where N is the number of documents and d is the number of words.

    log_P (np.array): A numpy array of the shape (t,d) where t is the number of topics for clustering and d is the number of words.

    log_pi (np.array): A numpy array of the shape (t,1) where t is the number of topics for clustering.


    Returns:

    log_W (np.array): A numpy array of the shape (N,t) where N is the number of documents and t is the number of topics for clustering.
    """
    N, d = X.shape
    t = log_pi.shape[0]
    N_ones = n_ones_matrix(N)

    log_R_N_t = np.dot(N_ones, log_pi.T) + np.dot(X, log_P.T)
    log_S_N_t = logsumexp(log_R_N_t, axis=1, keepdims=True)

    log_W = log_R_N_t - log_S_N_t
    assert log_W.shape == (N, t)

    return log_W


def update_logP(X, log_W, eps=1e-100):
    """
    Compute the parameters log(P) from the M step of expectation maximization.


    Parameters:

    X (np.array): A numpy array of the shape (N,d) where N is the number of documents and d is the number of words.

    log_W (np.array): A numpy array of the shape (N,t) where N is the number of documents and t is the number of topics for clustering.


    Returns:

    log_P (np.array): A numpy array of the shape (t,d) where t is the number of topics for clustering and d is the number of words.
    """
    N, d = X.shape
    t = log_W.shape[1]
    assert log_W.shape[0] == N

    E_t_d = np.dot(np.exp(log_W).T, X) + eps
    log_F_t_d = logsumexp(np.log(E_t_d), axis=1, keepdims=True)

    log_P = np.log(E_t_d) - log_F_t_d
    assert log_P.shape == (t, d)

    return log_P


def update_log_pi(log_W):
    """
    Compute the prior pi from the M step of expectation maximization.


    Parameters:

    log_W (np.array): A numpy array of the shape (N,t) where N is the number of documents and t is the number of topics for clustering.


    Returns:

    log_pi (np.array): A numpy array of the shape (t,1) where t is the number of topics for clustering.
    """
    N, t = log_W.shape

    log_pi = (logsumexp(log_W, axis=0, keepdims=True) - np.log(N)).T
    assert log_pi.shape == (t,1)

    return log_pi


def run(X: np.ndarray, topics: int, iterations: int = 100, seed: int = 12345, debug: bool = False) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the expectation maximization algorithm for topic modeling.

    Parameters:
    - X (np.ndarray): A numpy array of shape (N,d) where N is the number of documents and d is the number of words.
    - topics (int): The number of topics for clustering.
    - iterations (int): The number of iterations.
    - seed (int): Seed for random generation.
    - debug (bool): Flag to print debug information.

    Returns:
    - log_pi (np.ndarray): A numpy array of shape (t,1) where t is the number of topics for clustering.
    - log_P (np.ndarray): A numpy array of shape (t,d) where t is the number of topics for clustering and d is
    the number of words.
    - log_W (np.ndarray): A numpy array of shape (N,t) where N is the number of documents and t is the number of topics
    for clustering.
    """
    N, d = X.shape

    np_rand = np.random.RandomState(seed=seed)
    pi_init = np.ones((topics, 1))/float(topics)
    log_pi = np.log(pi_init)

    P_init = np_rand.uniform(0, 1, (topics, d))
    P_init = P_init/P_init.sum(axis=1).reshape(-1, 1)
    log_P = np.log(P_init)

    log_W = None

    if debug:
        sys.stderr.write('.run started')

    for iteration in range(iterations):
        if debug:
            sys.stderr.write('.')

        # The E-Step
        log_W = find_logW(X, log_P, log_pi)

        # The M-Step
        log_P = update_logP(X, log_W)
        log_pi = update_log_pi(log_W)

    if debug:
        sys.stderr.write('run finished.\n')

    return log_pi, log_P, log_W
