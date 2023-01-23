import abc
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import entropy
from mf_models import BayesianMFModel, BayesBernMFModel, BayesNormMFModel
from reg_models import BayesianModel, BayesBetaRegression, BayesLinearRegression, BayesLogisticRegression, BayesPoissonRegression
from utils import poisson_probs, beta_probs, continuous_entropy, beta_entropy, norm_probs
from dists import MFDistance


class MPSSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, **kwargs):
        self.n_samples = n_samples
    
    def select(self, model:BayesianModel, X:np.ndarray, **kwargs)->int:
        raise NotImplementedError





class MPSMFSelector():
    __metaclass__ = abc.ABCMeta
    def __init__(self, n_samples:int, dist:MFDistance, **kwargs):
        self.n_samples = n_samples
        self.distance = dist

    def select(self, model:BayesianMFModel, index_pairs:list, **kwargs)->int:
        raise NotImplementedError



class MPSNormMF(MPSMFSelector):
    def __init__(self, n_samples:int, dist:MFDistance, sigma:float, **kwargs):
        super().__init__(n_samples, dist, **kwargs)
        self.sigma = sigma

    def select(self, model:BayesNormMFModel, index_pairs:list, **kwargs)->int:
        W_list, V_list = model.sample(self.n_samples)

        mu = np.einsum('tik, tjk -> tij', W_list, V_list) ## n x dim1 x dim2
        
        mu_star = mu[0,:,:]
        W_star = W_list[0,:,:]
        V_star = V_list[0,:,:]
        y = mu_star + self.sigma*np.random.standard_normal(size=mu_star.shape)

        mu_rest = mu[1:,:,:]
        N, _, _ = mu_rest.shape
        W_rest = W_list[1:,:,:]
        V_rest = V_list[1:,:,:]

        dists = np.array([self.distance.distance(W_rest[i], V_rest[i], mu_rest[i], W_star, V_star, mu_star) for i in range(N)])
        log_likelihoods = 0.5*np.square((y - mu_rest)/self.sigma)

        ## Combine into groups
        m = len(index_pairs)

        ll_group = np.empty((N, m))
        for i, (ii,jj) in enumerate(index_pairs):
            ll_group[:, i] = np.sum( log_likelihoods[:, ii, jj] , axis=1)

        ll_sums = logsumexp(ll_group, axis=0, keepdims=True)
        ll_group = ll_group - ll_sums
        probs = np.exp(ll_group)

        
        scores = np.sum(probs*dists[:,np.newaxis], axis=0)
        idx = np.argmin(scores)
        return(idx)