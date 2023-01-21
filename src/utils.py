import numpy as np
from scipy.special import betaln, digamma
from scipy.stats import beta, norm
from scipy.special import xlogy
from scipy.integrate import simpson


def poisson_probs(lams:np.ndarray, K:int=10):
    n, m = lams.shape
    ## expand around the mean
    mean = np.mean(lams)
    lb = int(np.maximum( (mean - K), 0  ))
    M = 2*K
    probs = np.empty((n, m, M))
    for i in range(M):
        k = i+lb
        probs[:,:,i] = np.power(lams, k) * np.exp(-lams)/np.math.factorial(k)
    return(probs)


def beta_probs(a:np.ndarray, b:np.ndarray, K:int=10):
    n,m = a.shape
    x = np.linspace(0.001,0.999,num=K)
    Y = np.empty((n, m, K))
    for i in range(K):
        Y[:,:,i] = beta.pdf(x[i], a=a, b=b)
    return(Y, x)


def beta_entropy(a:np.ndarray, b:np.ndarray):
    result = betaln(a, b) - (a - 1.)*digamma(a) - (b - 1.)*digamma(b) + (a + b - 2.)*digamma(a + b)
    return(result)

def continuous_entropy(Y:np.ndarray, x:np.ndarray, axis=-1):
    neg_ylogy = -xlogy(Y, Y)
    entr = simpson(neg_ylogy, x, axis=axis)
    return(entr)
    
def norm_probs(mu:np.ndarray, sigma:float, K:int=10):
    low = np.min(mu) - 2*sigma
    high = np.max(mu) + 2*sigma
    x = np.linspace(low, high, num=K)
    
    Y = np.stack([ norm.pdf(a, loc=mu, scale=sigma) for a in x ], axis=-1)
    return(Y, x)
