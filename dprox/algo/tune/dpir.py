import numpy as np
import torch


def get_rho_sigma_admm(sigma=2.55 / 255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS * w + modelSigmaS_lin * (1 - w)) / 255.
    rhos = list(map(lambda x: lam * (sigma**2) / (x**2), sigmas))
    return rhos, sigmas


def log_descent(upper, lower, iter=24, sigma=0.255 / 255,
                w=1.0, lam=0.23, sqrt=False):
    """
    generate a list of rhos and sigmas based on given parameters using
    logarithmic descent.

    :param upper: The upper bound of the range of modelSigmaS values to be generated using a logarithmic
    scale
    :param lower: The lower bound of the range of values for modelSigmaS
    :param iter: The number of iterations or steps in the descent algorithm, defaults to 24 (optional)
    :param sigma: The standard deviation of the noise in the image
    :param w: The parameter w is a weight used to balance the logarithmic and linear scales when
    generating the sequence of modelSigmaS values. It is used to calculate the sigmas values, which are
    the squared values of the modelSigmaS values divided by 255
    :param lam: lam is a hyperparameter that controls the strength of the regularization term in the
    optimization problem. 
    :return: two lists: `rhos` and `sigmas`.
    """
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS * w + modelSigmaS_lin * (1 - w)) / 255.
    rhos = list(map(lambda x: lam * (sigma**2) / (x**2), sigmas))
    if not sqrt:
        sigmas = list(sigmas**2)
    rhos = torch.tensor(rhos).float()
    sigmas = torch.tensor(sigmas).float()
    return rhos, sigmas


def f(params): return [np.sqrt(p) * 255 for p in params]
# this can be 70.94


def log_descent2(upper, lower, iter=24, sigma=0.255 / 255, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS * w + modelSigmaS_lin * (1 - w)) / 255.
    sigmas = list((sigmas**2) / (sigma**2))
    rhos = list(map(lambda x: lam * (sigma**2) / (x**2), sigmas))
    sigmas = list(map(lambda x: 1 / x, sigmas))
    return rhos, sigmas


def log_descent_origin(upper, lower, iter=24,
                       sigma=0.255 / 255, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS * w + modelSigmaS_lin * (1 - w)) / 255.
    rhos = list(map(lambda x: lam * (sigma**2) / (x**2), sigmas))
    return rhos
