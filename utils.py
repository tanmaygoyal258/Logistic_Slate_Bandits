import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def weighted_norm(x, A):
    return np.sqrt(np.dot(x, np.dot(A, x)))


def gaussian_sample_ellipsoid(center, design, radius):
    dim = len(center)
    sample = np.random.normal(0, 1, (dim,))
    res = np.real_if_close(center + np.linalg.solve(sqrtm(design), sample) * radius)
    return res

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def log_loss(theta , arm , reward , reward_model):
    return -reward * np.log(reward_model(np.dot(arm , theta))) - (\
        1 - reward) * np.log(1 - reward_model(np.dot(arm , theta)))

def regularized_log_loss(theta , arms , rewards , reward_model , lamda):
    loss = 0
    for arm , reward in zip(arms , rewards):
        loss += log_loss(theta , arm , reward , reward_model)
    return loss + (lamda/2)*np.linalg.norm(theta)**2