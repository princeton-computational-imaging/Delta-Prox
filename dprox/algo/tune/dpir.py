import numpy as np


def get_rho_sigma_admm(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas


def log_descent(upper, lower, iter=24,
                sigma=0.255/255, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    sigmas = list(sigmas**2)
    return rhos, sigmas


def f(params): return [np.sqrt(p)*255 for p in params]
# this can be 70.94

def log_descent2(upper, lower, iter=24,
                sigma=0.255/255, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    sigmas = list((sigmas**2)/(sigma**2))
    rhos = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    sigmas = list(map(lambda x: 1/x, sigmas))
    return rhos, sigmas

def log_descent_origin(upper, lower, iter=24,
                       sigma=0.255/255, w=1.0, lam=0.23):
    modelSigmaS = np.logspace(np.log10(upper), np.log10(lower), iter).astype(np.float32)
    modelSigmaS_lin = np.linspace(upper, lower, iter).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    return rhos
